import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


def gaussian2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(0, b1**2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(0, b2**2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(0, b3**2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    if left < 0 or right <= 0 or top < 0 or bottom <= 0:
        return heatmap

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


@DATASET_REGISTRY.register()
class SCINetDetectionDataset(data.Dataset):
    """Paired SR dataset with CenterNet-style tiny-target supervision.

    Expected annotation format (one bbox per line):
      x1 y1 x2 y2 [class_id]
    Coordinates are in GT image space.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean')
        self.std = opt.get('std')

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.ann_folder = opt['dataroot_ann']
        self.annotation_ext = opt.get('annotation_ext', '.txt')
        self.num_classes = int(opt.get('num_classes', 1))
        self.min_overlap = float(opt.get('min_overlap', 0.7))

        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif opt.get('meta_info_file'):
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def _annotation_path(self, gt_path):
        stem = Path(gt_path).stem
        return str(Path(self.ann_folder) / f'{stem}{self.annotation_ext}')

    def _load_boxes(self, ann_path, width, height):
        if not Path(ann_path).exists():
            return []
        boxes = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                x1, y1, x2, y2 = map(float, parts[:4])
                cls_id = int(parts[4]) if len(parts) > 4 else 0
                cls_id = min(max(cls_id, 0), self.num_classes - 1)

                x1 = np.clip(x1, 0, width - 1)
                y1 = np.clip(y1, 0, height - 1)
                x2 = np.clip(x2, 0, width - 1)
                y2 = np.clip(y2, 0, height - 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((x1, y1, x2, y2, cls_id))
        return boxes

    def _build_targets(self, boxes, height, width):
        heatmap = np.zeros((self.num_classes, height, width), dtype=np.float32)
        size = np.zeros((2, height, width), dtype=np.float32)
        offset = np.zeros((2, height, width), dtype=np.float32)
        mask = np.zeros((1, height, width), dtype=np.float32)

        for x1, y1, x2, y2, cls_id in boxes:
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            cx_int = int(np.clip(cx, 0, width - 1))
            cy_int = int(np.clip(cy, 0, height - 1))

            radius = gaussian_radius((math.ceil(h), math.ceil(w)), self.min_overlap)
            radius = max(0, int(radius))
            draw_umich_gaussian(heatmap[cls_id], (cx_int, cy_int), radius)

            size[0, cy_int, cx_int] = w
            size[1, cy_int, cx_int] = h
            offset[0, cy_int, cx_int] = cx - cx_int
            offset[1, cy_int, cx_int] = cy - cy_int
            mask[0, cy_int, cx_int] = 1.0

        return heatmap, size, offset, mask

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

        if self.opt.get('color') == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        gt_h, gt_w = img_gt.shape[:2]
        ann_path = self._annotation_path(gt_path)
        boxes = self._load_boxes(ann_path, gt_w, gt_h)
        gt_heatmap, gt_size, gt_offset, gt_mask = self._build_targets(boxes, gt_h, gt_w)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        gt_heatmap = torch.from_numpy(gt_heatmap)
        gt_size = torch.from_numpy(gt_size)
        gt_offset = torch.from_numpy(gt_offset)
        gt_mask = torch.from_numpy(gt_mask)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_heatmap': gt_heatmap,
            'gt_size': gt_size,
            'gt_offset': gt_offset,
            'gt_mask': gt_mask,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'ann_path': ann_path
        }

    def __len__(self):
        return len(self.paths)
