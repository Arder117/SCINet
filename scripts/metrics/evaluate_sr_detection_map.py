"""Evaluate downstream detection mAP on SR outputs.

Template pipeline:
1) Run detector (YOLO / RT-DETR via Ultralytics) on SR images.
2) Convert detections to COCO result json.
3) Evaluate with COCOeval and print mAP / AP50 / AP75.

Example:
python scripts/metrics/evaluate_sr_detection_map.py \
  --detector yolo \
  --model /path/to/yolo.pt \
  --sr_dir /path/to/sr_images \
  --ann_json /path/to/instances_test.json \
  --out_json /tmp/sr_det_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detection mAP on super-resolved images.")
    parser.add_argument("--detector", type=str, default="yolo", choices=["yolo", "rtdetr"],
                        help="Detector backend. Both are loaded via ultralytics.YOLO API.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to detector checkpoint (.pt), e.g. YOLO or RT-DETR weights.")
    parser.add_argument("--sr_dir", type=str, required=True,
                        help="Directory containing SR images to evaluate.")
    parser.add_argument("--ann_json", type=str, required=True,
                        help="COCO annotation json for the evaluation split.")
    parser.add_argument("--out_json", type=str, default="./results/sr_detection_results.json",
                        help="Output COCO detection result json.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold.")
    parser.add_argument("--device", type=str, default="0", help="Device for inference, e.g. 0 or cpu.")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--class_map_json", type=str, default="",
                        help="Optional json mapping model class id -> COCO category id, e.g. {\"0\": 1}.")
    parser.add_argument("--save_empty", action="store_true",
                        help="If set, emit empty-detection images into json as [] (default COCO style is skip).")
    return parser.parse_args()


def _load_eval_deps():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise ImportError("pycocotools is required. Install with: pip install pycocotools") from e

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics is required. Install with: pip install ultralytics") from e

    return COCO, COCOeval, YOLO


def build_name_to_image_id(COCO, ann_json: Path) -> Tuple[object, Dict[str, int]]:
    coco_gt = COCO(str(ann_json))
    name_to_id: Dict[str, int] = {}
    for img in coco_gt.dataset.get("images", []):
        file_name = Path(img["file_name"]).name
        name_to_id[file_name] = int(img["id"])
    return coco_gt, name_to_id


def iter_sr_images(sr_dir: Path) -> List[Path]:
    files = [p for p in sorted(sr_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No images found in {sr_dir} with extensions: {sorted(IMAGE_EXTS)}")
    return files


def infer_and_convert_to_coco(
    model,
    image_paths: List[Path],
    name_to_img_id: Dict[str, int],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    max_det: int,
    class_map: Optional[Dict[int, int]],
    save_empty: bool,
) -> List[dict]:
    detections: List[dict] = []

    for path in image_paths:
        image_name = path.name
        if image_name not in name_to_img_id:
            # Skip SR images that are not part of the given annotation split.
            continue
        image_id = name_to_img_id[image_name]

        results = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            verbose=False,
        )
        pred = results[0]

        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            if save_empty:
                pass
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": class_map.get(int(cls_id), int(cls_id) + 1) if class_map else int(cls_id) + 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    return detections


def evaluate_coco(COCOeval, coco_gt, det_json_path: Path) -> Dict[str, float]:
    coco_dt = coco_gt.loadRes(str(det_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP@[0.50:0.95]": float(coco_eval.stats[0]),
        "AP@0.50": float(coco_eval.stats[1]),
        "AP@0.75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
    }


def main() -> None:
    args = parse_args()

    sr_dir = Path(args.sr_dir)
    ann_json = Path(args.ann_json)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    COCO, COCOeval, YOLO = _load_eval_deps()

    coco_gt, name_to_img_id = build_name_to_image_id(COCO, ann_json)
    image_paths = iter_sr_images(sr_dir)

    class_map = None
    if args.class_map_json:
        with open(args.class_map_json, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        class_map = {int(k): int(v) for k, v in raw_map.items()}

    # YOLO class in ultralytics can load both YOLO and RT-DETR *.pt weights.
    model = YOLO(args.model)
    dets = infer_and_convert_to_coco(
        model=model,
        image_paths=image_paths,
        name_to_img_id=name_to_img_id,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        class_map=class_map,
        save_empty=args.save_empty,
    )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(dets, f, ensure_ascii=False)

    print(f"Saved {len(dets)} detections to: {out_json}")
    metrics = evaluate_coco(COCOeval, coco_gt, out_json)
    print("\nDetection metrics on SR outputs:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
