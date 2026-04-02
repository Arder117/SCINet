"""Fine-tune YOLO/RT-DETR for small-object detection (template).

This script wraps Ultralytics training and exposes commonly-used knobs for
small-target scenarios (higher input size, augment controls, etc.).

Example:
python scripts/metrics/finetune_small_object_detector.py \
  --detector yolo \
  --model yolo11n.pt \
  --data /path/to/dataset.yaml \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --small_obj_preset
"""

from __future__ import annotations

import argparse
from pathlib import Path

def _load_yolo_cls():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from e
    return YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune small-object detector with Ultralytics.")
    parser.add_argument("--detector", type=str, default="yolo", choices=["yolo", "rtdetr"],
                        help="Detector type hint (both are loaded by ultralytics.YOLO).")
    parser.add_argument("--model", type=str, required=True,
                        help="Pretrained checkpoint path/name, e.g. yolo11n.pt or rtdetr-l.pt.")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset yaml path (Ultralytics format).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Larger size is usually better for tiny objects (with more GPU memory).")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="small_object_finetune")
    parser.add_argument("--optimizer", type=str, default="auto", choices=["auto", "SGD", "Adam", "AdamW", "RMSProp"])
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--freeze", type=int, default=0,
                        help="Freeze first N layers (useful for stable quick adaptation).")
    parser.add_argument("--download_model", action="store_true",
                        help="Download checkpoint by model name when local file does not exist.")

    # Augmentation knobs that often affect tiny-object performance
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy_paste", type=float, default=0.0)
    parser.add_argument("--close_mosaic", type=int, default=10)

    parser.add_argument("--small_obj_preset", action="store_true",
                        help="Apply stronger defaults for tiny objects.")
    return parser.parse_args()


def build_train_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "freeze": args.freeze,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "close_mosaic": args.close_mosaic,
    }

    if args.small_obj_preset:
        kwargs.update(
            {
                "imgsz": max(args.imgsz, 1280),
                "mosaic": 1.0,
                "mixup": 0.1,
                "copy_paste": max(args.copy_paste, 0.2),
                "translate": min(args.translate, 0.05),
                "degrees": 0.0,
                "close_mosaic": max(args.close_mosaic, 10),
            }
        )

    return kwargs


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    YOLO = _load_yolo_cls()

    model_path = Path(args.model)
    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        if not args.download_model:
            raise FileNotFoundError(
                f"Model not found locally: {args.model}. "
                "Use --download_model with an official Ultralytics model name, e.g. yolo11n.pt"
            )
        print(f"Downloading model checkpoint via Ultralytics: {args.model}")
        model = YOLO(args.model)

    train_kwargs = build_train_kwargs(args)

    print("Start fine-tuning with arguments:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    result = model.train(**train_kwargs)
    print("\nTraining done.")
    print(result)

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_path.exists():
        print(f"Best checkpoint: {best_path}")
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=args.data, imgsz=train_kwargs["imgsz"], device=args.device)
        print("\nValidation metrics for best checkpoint:")
        print(metrics)
    else:
        print(f"Warning: best checkpoint not found at {best_path}")


if __name__ == "__main__":
    main()
