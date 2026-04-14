
# import logging
# import torch
# from os import path as osp
# from basicsr.data import build_dataloader, build_dataset
# from basicsr.models import build_model
# from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
# from basicsr.utils.options import dict2str, parse_options
# from basicsr.archs.SCINet_arch import switch_deploy_flag
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # GPU  指定显卡
# def test_pipeline(root_path):
#     # parse options, set distributed setting, set ramdom seed
#     opt, _ = parse_options(root_path, is_train=False)

#     torch.backends.cudnn.benchmark = True
#     # torch.backends.cudnn.deterministic = True

#     # mkdir and initialize loggers
#     make_exp_dirs(opt)
#     log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
#     logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
#     logger.info(get_env_info())
#     logger.info(dict2str(opt))

#     # create test dataset and dataloader
#     test_loaders = []
#     for _, dataset_opt in sorted(opt['datasets'].items()):
#         test_set = build_dataset(dataset_opt)
#         test_loader = build_dataloader(
#             test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
#         logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
#         test_loaders.append(test_loader)

#     mode = True
#     switch_deploy_flag(mode)
#     # create model
#     model = build_model(opt)

#     for test_loader in test_loaders:
#         test_set_name = test_loader.dataset.opt['name']
#         logger.info(f'Testing {test_set_name}...')
#         model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


# if __name__ == '__main__':
#     root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
#     test_pipeline(root_path)


import logging
import torch
import os
from os import path as osp
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from basicsr.archs.SCINet_arch import switch_deploy_flag

# 指定 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def test_pipeline(root_path):
    # 1. 解析配置
    opt, _ = parse_options(root_path, is_train=False)

    # 2. 初始化日志与环境
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    if torch.cuda.is_available():
        logger.info(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
    
    torch.backends.cudnn.benchmark = True

    # 3. 构建数据集
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # --- 关键修改：加载与转换流程 ---
    
    # 第一步：确保处于非部署模式，以便正确加载所有训练分支的权重（DBB/BN等）
    logger.info("Step 1: Setting switch_deploy_flag(False) for weight loading...")
    switch_deploy_flag(False)
    
    # 第二步：构建模型 (BasicSR 的 build_model 会根据配置文件自动 load_network)
    model = build_model(opt)
    logger.info("Step 2: Model weights loaded successfully.")

    # 第三步：执行重参数化转换
    # 注意：model 是 BasicSR 的 Model Wrapper，真正的网络在 model.net_g 中
    if hasattr(model.net_g, 'switch_to_deploy'):
        logger.info("Step 3: Found switch_to_deploy method. Converting model...")
        model.net_g.switch_to_deploy()
        # 转换完成后，同步全局标志位
        switch_deploy_flag(True)
        logger.info("Model reparameterization complete.")
    else:
        logger.warning("Step 3: switch_to_deploy method not found in model.net_g. Skipping conversion.")
    
    # ----------------------------

    # 4. 开始测试
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)




