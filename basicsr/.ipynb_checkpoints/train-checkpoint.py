import datetime
import logging
import math
import time
import torch
from os import path as osp
import os
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from basicsr.archs.SCINet_arch import switch_deploy_flag
import os

# 引入进度条库
try:
    from tqdm import tqdm
except ImportError:
    os.system('pip install tqdm')
    from tqdm import tqdm

# 指定使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置为 "0" 来使用第一个 GPU，"1" 表示使用第二个 GPU

def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, train_sampler, val_loaders = None, None, []
    num_iter_per_epoch = 0 # 记录每轮迭代数以便进度条使用
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            
            # --- [DEBUG] 物理路径扫描 ---
            gt_path = dataset_opt.get('dataroot_gt')
            if gt_path and os.path.exists(gt_path):
                img_list = [f for f in os.listdir(gt_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
                logger.info(f"[DEBUG] 物理磁盘扫描: 文件夹 {gt_path} 下共有 {len(img_list)} 张图片文件。")
            
            logger.info(f"[DEBUG] 正在构建训练数据集: {dataset_opt['name']}...")
            train_set = build_dataset(dataset_opt)
            logger.info(f"[DEBUG] 训练集构建完成，样本总数: {len(train_set)}")
            
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            
            logger.info(f"[DEBUG] 正在初始化训练 Dataloader (Workers: {dataset_opt.get('num_worker_per_gpu')})...")
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            logger.info(f"[DEBUG] 训练 Dataloader 初始化成功。")

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            logger.info(f"[DEBUG] 正在构建验证数据集: {dataset_opt['name']}...")
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    # 为了进度条，我们将 num_iter_per_epoch 也返回
    return train_loader, train_sampler, val_loaders, total_epochs, total_iters, num_iter_per_epoch


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True  # Enable cudnn optimization

    # Initialize device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load resume states if necessary
    resume_state = load_resume_state(opt)

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # Initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # Create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters, iters_per_epoch = result

    mode = False
    switch_deploy_flag(mode)

    # Create model and move it to the correct device (GPU or CPU)
    logger.info(f"[DEBUG] 正在构建模型 [{opt['model_type']}]...")
    model = build_model(opt)
    logger.info(f"[DEBUG] 模型构建完成，正在移动至设备: {device}...")
    model.to(device)  # Ensure the model is on GPU
    logger.info(f"[DEBUG] 设备移动完成。")

    # Load resume state
    if resume_state:  # Resume training
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # Create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # Dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    logger.info(f"[DEBUG] 正在初始化 Prefetcher (模式: {prefetch_mode})...")
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.')
    logger.info(f"[DEBUG] Prefetcher 初始化完成。")

    # Training loop
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        logger.info(f"[DEBUG] >>> 进入 Epoch {epoch}")
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        
        # --- 初始化进度条 ---
        pbar = tqdm(total=iters_per_epoch, unit='batch', desc=f'Epoch {epoch}', leave=False)
        
        logger.info(f"[DEBUG] 正在请求第一批训练数据 (prefetcher.next)...")
        train_data = prefetcher.next()
        
        if train_data is None:
            logger.warning(f"[DEBUG] 警告: 获取到的第一批数据为空！请检查数据路径。")

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break

            # Update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # Feed data to the model
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()

            # --- 更新进度条 ---
            pbar.update(1)
            pbar.set_postfix(iter=current_iter, loss=model.get_current_log().get('l_pix', 'N/A'))

            # Log progress
            if current_iter % opt['logger']['print_freq'] == 0:
                # logger.info(f"[DEBUG] 达到打印频率，刷新日志...")
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # Save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info(f'Saving models and training states (Iter {current_iter}).')
                model.save(epoch, current_iter)

            # Validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                logger.info(f"[DEBUG] 触发验证 (Iter {current_iter})...")
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                logger.info(f"[DEBUG] 验证完成。")

            data_timer.start()
            iter_timer.start()
            
            train_data = prefetcher.next()
        
        pbar.close() # Epoch 结束关闭进度条

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)