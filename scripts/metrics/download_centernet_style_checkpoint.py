"""Download a CenterNet-style checkpoint to local disk.

This script is intentionally generic: provide a direct URL to any checkpoint
compatible with your staged-loading plan, then point train config to the saved path.

Example:
python scripts/metrics/download_centernet_style_checkpoint.py \
  --url https://your-host/path/to/centernet.pth \
  --output experiments/pretrained_models/CenterNetStyle/centernet.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download CenterNet-style checkpoint.')
    parser.add_argument('--url', type=str, required=True, help='Direct checkpoint URL.')
    parser.add_argument('--output', type=str, required=True, help='Output .pth path.')
    parser.add_argument('--force', action='store_true', help='Overwrite existing file.')
    return parser.parse_args()


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    ratio = min(downloaded / total_size, 1.0)
    print(f'\rDownloading... {ratio:.1%}', end='')


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f'Checkpoint already exists, skip: {out_path}')
        return

    print(f'Downloading from: {args.url}')
    print(f'Save to: {out_path}')
    urlretrieve(args.url, str(out_path), reporthook=_progress)
    print('\nDone.')


if __name__ == '__main__':
    main()
