#!/usr/bin/env python3
"""Inference + decode utility for SCINet tiny-target detector.

Loads `network_g` from an option yaml and a checkpoint, runs on an input folder,
decodes CenterNet-style heatmap/size/offset to bboxes, and saves visualizations.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.archs import build_network


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=Path, required=True, help='Path to option yaml containing network_g.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to net_g checkpoint (.pth).')
    parser.add_argument('--input_dir', type=Path, required=True, help='Folder containing LR input images.')
    parser.add_argument('--output_dir', type=Path, required=True, help='Folder to save visualization images.')
    parser.add_argument('--param_key', type=str, default='params', help='Checkpoint parameter key.')
    parser.add_argument('--topk', type=int, default=100, help='Top-K keypoints before thresholding.')
    parser.add_argument('--score_thr', type=float, default=0.2, help='Detection score threshold.')
    parser.add_argument('--nms_kernel', type=int, default=3, help='Local NMS maxpool kernel size.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model(opt_path, ckpt_path, param_key, device):
    opt = yaml.safe_load(opt_path.read_text(encoding='utf-8'))
    net = build_network(opt['network_g']).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get(param_key, ckpt)
    if not isinstance(state, dict):
        raise ValueError(f'Invalid checkpoint format: key={param_key}')

    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f'Loaded checkpoint: {ckpt_path}')
    if missing:
        print(f'[Warn] missing keys: {len(missing)}')
    if unexpected:
        print(f'[Warn] unexpected keys: {len(unexpected)}')

    net.eval()
    return net


def preprocess_bgr(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    img = img[..., ::-1]  # BGR -> RGB
    tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().unsqueeze(0)
    return tensor


def pool_nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_single(heatmap, size, offset, topk=100, score_thr=0.2, nms_kernel=3):
    # tensors: [1,C,H,W], [1,2,H,W], [1,2,H,W]
    heat = torch.sigmoid(heatmap)
    heat = pool_nms(heat, kernel=nms_kernel)

    b, c, h, w = heat.shape
    scores, inds = torch.topk(heat.view(b, -1), k=min(topk, c * h * w))

    detections = []
    for score, ind in zip(scores[0], inds[0]):
        s = float(score.item())
        if s < score_thr:
            continue

        cls_id = int(ind // (h * w))
        loc = int(ind % (h * w))
        y = loc // w
        x = loc % w

        off_x = float(offset[0, 0, y, x].item())
        off_y = float(offset[0, 1, y, x].item())
        bw = float(size[0, 0, y, x].item())
        bh = float(size[0, 1, y, x].item())

        cx = x + off_x
        cy = y + off_y
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        detections.append((x1, y1, x2, y2, s, cls_id))

    return detections


def draw_boxes(img_bgr, detections):
    import cv2
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2, score, cls_id in detections:
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(out, f'c{cls_id}:{score:.2f}', (x1, max(y1 - 3, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    net = load_model(args.opt, args.checkpoint, args.param_key, device)

    image_paths = sorted([p for p in args.input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    print(f'Found images: {len(image_paths)}')

    with torch.no_grad():
        for img_path in image_paths:
            import cv2
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f'[Skip] cannot read {img_path}')
                continue

            inp = preprocess_bgr(img_bgr).to(device)
            out = net(inp)

            sr = out['sr'][0].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
            sr_bgr = (sr[..., ::-1] * 255.0).astype(np.uint8)

            dets = decode_single(
                out['heatmap'].detach().cpu(),
                out['size'].detach().cpu(),
                out['offset'].detach().cpu(),
                topk=args.topk,
                score_thr=args.score_thr,
                nms_kernel=args.nms_kernel,
            )

            vis = draw_boxes(sr_bgr, dets)
            out_path = args.output_dir / f'{img_path.stem}_det.png'
            cv2.imwrite(str(out_path), vis)

    print(f'Saved results to: {args.output_dir}')


if __name__ == '__main__':
    main()
