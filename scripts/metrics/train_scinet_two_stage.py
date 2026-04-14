"""Two-stage training helper for SCINet small-object detection + joint SR.

Stage 1 (det-pretrain):
  - Uses SCINetDetectionModel
  - Turns off SR optimization (sr_weight=0.0)
  - Optionally initializes from SR checkpoint + external detector checkpoint

Stage 2 (joint):
  - Loads Stage-1 detector result as detection init
  - Enables SR + detection joint optimization
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run two-stage SCINet training with optional detector pretrain.')
    parser.add_argument('--base_opt', type=str, default='options/train/train_SCINet_detection_x4.yml',
                        help='Base YAML config path.')
    parser.add_argument('--run_name', type=str, default='scinet_det_two_stage',
                        help='Prefix name for stage1/stage2 runs.')
    parser.add_argument('--sr_pretrain', type=str, default='',
                        help='Optional SR checkpoint for staged init.')
    parser.add_argument('--det_pretrain', type=str, default='',
                        help='Optional external detector checkpoint (e.g., CenterNet-style).')
    parser.add_argument('--det_param_key', type=str, default='state_dict',
                        help='Param key for detector checkpoint root dict.')
    parser.add_argument('--det_key_map_yaml', type=str, default='',
                        help='Optional YAML path of key mapping {src_key: tgt_key}.')
    parser.add_argument('--stage1_total_iter', type=int, default=80000)
    parser.add_argument('--stage2_total_iter', type=int, default=300000)
    parser.add_argument('--stage1_sr_weight', type=float, default=0.0)
    parser.add_argument('--stage2_sr_weight', type=float, default=1.0)
    parser.add_argument('--python_bin', type=str, default=sys.executable)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_train(python_bin: str, opt_path: Path) -> None:
    cmd = [python_bin, 'basicsr/train.py', '-opt', str(opt_path)]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    base_opt_path = Path(args.base_opt)
    base = load_yaml(base_opt_path)
    workspace = Path('experiments') / f'{args.run_name}_auto'
    temp_dir = workspace / 'generated_opts'
    temp_dir.mkdir(parents=True, exist_ok=True)

    det_key_map = None
    if args.det_key_map_yaml:
        det_key_map = load_yaml(Path(args.det_key_map_yaml))

    # -------- Stage 1: detector pretrain --------
    stage1 = copy.deepcopy(base)
    stage1['name'] = f'{args.run_name}_stage1_det'
    stage1['train']['total_iter'] = int(args.stage1_total_iter)
    stage1['train']['sr_weight'] = float(args.stage1_sr_weight)
    stage1_path = stage1.setdefault('path', {})
    stage1_path['pretrain_network_g'] = None
    if args.sr_pretrain:
        stage1_path['pretrain_network_sr'] = args.sr_pretrain
    if args.det_pretrain:
        stage1_path['pretrain_network_det'] = args.det_pretrain
        stage1_path['param_key_det'] = args.det_param_key
    if det_key_map:
        stage1_path['det_key_map'] = det_key_map

    stage1_opt_path = temp_dir / f'{args.run_name}_stage1.yml'
    dump_yaml(stage1, stage1_opt_path)
    run_train(args.python_bin, stage1_opt_path)

    # checkpoint saved by BaseModel when current_iter=-1 -> net_g_latest.pth
    stage1_ckpt = Path('experiments') / stage1['name'] / 'models' / 'net_g_latest.pth'
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f'Stage1 checkpoint not found: {stage1_ckpt}')

    # -------- Stage 2: joint training --------
    stage2 = copy.deepcopy(base)
    stage2['name'] = f'{args.run_name}_stage2_joint'
    stage2['train']['total_iter'] = int(args.stage2_total_iter)
    stage2['train']['sr_weight'] = float(args.stage2_sr_weight)
    stage2_path = stage2.setdefault('path', {})
    stage2_path['pretrain_network_g'] = None
    if args.sr_pretrain:
        stage2_path['pretrain_network_sr'] = args.sr_pretrain
    stage2_path['pretrain_network_det'] = str(stage1_ckpt)
    stage2_path['param_key_det'] = 'params'

    stage2_opt_path = temp_dir / f'{args.run_name}_stage2.yml'
    dump_yaml(stage2, stage2_opt_path)
    run_train(args.python_bin, stage2_opt_path)

    print('Two-stage training finished.')
    print(f'Stage1 ckpt: {stage1_ckpt}')
    print(f'Stage1 opt:  {stage1_opt_path}')
    print(f'Stage2 opt:  {stage2_opt_path}')


if __name__ == '__main__':
    main()
