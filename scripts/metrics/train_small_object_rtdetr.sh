#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/metrics/train_small_object_rtdetr.sh /path/to/dataset.yaml

DATA_YAML=${1:-}
if [[ -z "${DATA_YAML}" ]]; then
  echo "Usage: bash scripts/metrics/train_small_object_rtdetr.sh /path/to/dataset.yaml"
  exit 1
fi

python scripts/metrics/finetune_small_object_detector.py \
  --detector rtdetr \
  --model rtdetr-l.pt \
  --download_model \
  --data "${DATA_YAML}" \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --small_obj_preset \
  --project runs/train \
  --name rtdetr_small_object_ft
