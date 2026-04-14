from collections import OrderedDict
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SCINetDetectionModel(BaseModel, nn.Module):
    def __init__(self, opt):
        super().__init__(opt)
        nn.Module.__init__(self)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self._init_pretrained_weights()

        self.cri_pix = None
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)

        self.hm_weight = train_opt.get('hm_weight', 1.0)
        self.size_weight = train_opt.get('size_weight', 0.1)
        self.off_weight = train_opt.get('off_weight', 1.0)
        self.sr_weight = train_opt.get('sr_weight', 1.0)

        self.setup_optimizers()
        self.setup_schedulers()

    def _extract_state_dict(self, checkpoint_obj, param_key):
        """Extract state dict from a checkpoint object."""
        if isinstance(checkpoint_obj, dict):
            if param_key in checkpoint_obj:
                return checkpoint_obj[param_key]
            if 'params' in checkpoint_obj:
                return checkpoint_obj['params']
            if 'state_dict' in checkpoint_obj:
                return checkpoint_obj['state_dict']
        return checkpoint_obj

    def _load_by_key_and_shape(self, load_path, param_key='params', key_map=None, log_prefix='partial'):
        """Load checkpoint tensors by matching key names (optionally remapped) and shapes.

        Args:
            load_path (str): Path to checkpoint.
            param_key (str): Candidate root key in checkpoint dict.
            key_map (dict | None): Optional source->target key remapping.
            log_prefix (str): Prefix used in logger.
        """
        logger = get_root_logger()
        net = self.get_bare_model(self.net_g)
        current_state = net.state_dict()
        loaded = torch.load(load_path, map_location=lambda storage, loc: storage)
        loaded_state = self._extract_state_dict(loaded, param_key)
        if not isinstance(loaded_state, dict):
            logger.warning(f'[{log_prefix}] {load_path} has unsupported checkpoint format, skip.')
            return

        key_map = key_map or {}
        matched = {}
        skipped = []
        for src_k, src_v in loaded_state.items():
            tgt_k = key_map.get(src_k, src_k)
            if tgt_k in current_state and current_state[tgt_k].shape == src_v.shape:
                matched[tgt_k] = src_v
            else:
                skipped.append((src_k, tgt_k))

        current_state.update(matched)
        net.load_state_dict(current_state, strict=True)
        logger.info(f'[{log_prefix}] loaded {len(matched)} tensors from {load_path}.')
        if skipped:
            logger.info(f'[{log_prefix}] skipped {len(skipped)} tensors due to key/shape mismatch.')

    def _init_pretrained_weights(self):
        """Support staged initialization for SR + detection."""
        path_opt = self.opt.get('path', {})
        strict_load_g = path_opt.get('strict_load_g', True)

        # 1) Optional base full-network preload (same architecture recommended).
        load_path = path_opt.get('pretrain_network_g', None)
        if load_path is not None:
            param_key = path_opt.get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, strict_load_g, param_key)

        # 2) Optional SR-first staged preload (typically SCINet weights).
        sr_load_path = path_opt.get('pretrain_network_sr', None)
        if sr_load_path is not None:
            sr_param_key = path_opt.get('param_key_sr', path_opt.get('param_key_g', 'params'))
            self.load_network(self.net_g, sr_load_path, False, sr_param_key)

        # 3) Optional detection staged preload (e.g., CenterNet-style head or same-arch detector).
        det_load_path = path_opt.get('pretrain_network_det', None)
        if det_load_path is not None:
            det_param_key = path_opt.get('param_key_det', path_opt.get('param_key_g', 'params'))
            det_key_map = path_opt.get('det_key_map', None)
            self._load_by_key_and_shape(
                load_path=det_load_path,
                param_key=det_param_key,
                key_map=det_key_map,
                log_prefix='det_init')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        optional_tensors = ['gt', 'gt_heatmap', 'gt_size', 'gt_offset', 'gt_mask']
        for key in optional_tensors:
            if key in data:
                setattr(self, key, data[key].to(self.device))
            elif hasattr(self, key):
                delattr(self, key)

    def _heatmap_focal_loss(self, pred, target):
        pred = pred.sigmoid().clamp(min=1e-4, max=1 - 1e-4)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        neg_weights = (1 - target).pow(4)

        pos_loss = -(pred.log()) * (1 - pred).pow(2) * pos_inds
        neg_loss = -(1 - pred).log() * pred.pow(2) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        if num_pos == 0:
            return neg_loss.sum()
        return (pos_loss.sum() + neg_loss.sum()) / num_pos

    def _masked_l1_loss(self, pred, target, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        denom = mask.sum().clamp_min(1.0)
        return loss / denom

    def optimize_parameters(self, current_iter):
        del current_iter
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        if hasattr(self, 'gt_heatmap'):
            l_heatmap = self._heatmap_focal_loss(self.output['heatmap'], self.gt_heatmap) * self.hm_weight
            l_total += l_heatmap
            loss_dict['l_heatmap'] = l_heatmap

        if hasattr(self, 'gt_size') and hasattr(self, 'gt_mask'):
            l_size = self._masked_l1_loss(self.output['size'], self.gt_size, self.gt_mask) * self.size_weight
            l_total += l_size
            loss_dict['l_size'] = l_size

        if hasattr(self, 'gt_offset') and hasattr(self, 'gt_mask'):
            l_offset = self._masked_l1_loss(self.output['offset'], self.gt_offset, self.gt_mask) * self.off_weight
            l_total += l_offset
            loss_dict['l_offset'] = l_offset

        if self.cri_pix is not None and hasattr(self, 'gt'):
            l_sr = self.cri_pix(self.output['sr'], self.gt) * self.sr_weight
            l_total += l_sr
            loss_dict['l_sr'] = l_sr

        if not loss_dict:
            raise ValueError('No detection or SR supervision was provided.')

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        del current_iter, tb_logger
        dataset_name = dataloader.dataset.opt['name']
        use_pbar = self.opt['val'].get('pbar', False)
        pbar = tqdm(total=len(dataloader), unit='image') if use_pbar else None

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            if save_img:
                sr_img = tensor2img([visuals['result']])
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.bmp')
                imwrite(sr_img, save_img_path)

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if pbar is not None:
            pbar.close()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output['sr'].detach().cpu()
        out_dict['heatmap'] = self.output['heatmap'].sigmoid().detach().cpu()
        out_dict['size'] = self.output['size'].detach().cpu()
        out_dict['offset'] = self.output['offset'].detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
