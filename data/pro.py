#!/usr/bin/env python3
"""
Resize images and synchronize Pascal VOC XML annotations.
Ensures GT and LQ dimensions are perfectly matched for SR training.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def modcrop_size(width: int, height: int, scale: int) -> Tuple[int, int]:
    """计算能被 scale 整除的尺寸。"""
    return width - (width % scale), height - (height % scale)

def clip_xyxy(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    xmin = max(0.0, min(xmin, float(width)))
    ymin = max(0.0, min(ymin, float(height)))
    xmax = max(0.0, min(xmax, float(width)))
    ymax = max(0.0, min(ymax, float(height)))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步缩放图像与 VOC XML 标注。")
    parser.add_argument("--img_dir", required=True, help="输入原图(GT)目录")
    parser.add_argument("--xml_dir", required=True, help="输入 XML 目录")
    parser.add_argument("--out_lq_dir", required=True, help="输出缩小后的图(LQ)目录")
    parser.add_argument("--out_gt_dir", required=True, help="输出裁剪后的原图(GT_modcrop)目录")
    parser.add_argument("--out_xml_dir", required=True, help="输出更新后的 XML 目录")
    parser.add_argument("--scale", type=int, default=4, help="缩小倍率 (default: 4)")
    parser.add_argument("--min_box", type=float, default=1.0, help="保留框的最小尺寸(单位:LQ像素)")
    return parser.parse_args()

def update_text(node: Optional[ET.Element], value: str) -> None:
    if node is not None:
        node.text = value

def update_voc_xml(
    xml_path: Path,
    out_xml_path: Path,
    rel_img_path: Path,
    crop_size: Tuple[int, int],
    out_size: Tuple[int, int],
    scale: int,
    min_box: float,
) -> int:
    if not xml_path.exists():
        return 0
    
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    crop_w, crop_h = crop_size
    out_w, out_h = out_size

    # 更新 size 节点
    size_node = root.find("size")
    if size_node is not None:
        update_text(size_node.find("width"), str(out_w))
        update_text(size_node.find("height"), str(out_h))

    update_text(root.find("filename"), rel_img_path.name)
    update_text(root.find("path"), str(rel_img_path.as_posix()))

    objects = list(root.findall("object"))
    keep_count = 0

    for obj in objects:
        bnd = obj.find("bndbox")
        if bnd is None:
            root.remove(obj)
            continue

        def _get(name: str) -> Optional[float]:
            node = bnd.find(name)
            return float(node.text) if node is not None and node.text else None

        coords = [_get(n) for n in ["xmin", "ymin", "xmax", "ymax"]]
        if None in coords:
            root.remove(obj)
            continue

        # 1. 裁剪到 modcrop 后的尺寸
        clipped = clip_xyxy(*coords, crop_w, crop_h)
        if clipped is None:
            root.remove(obj)
            continue

        # 2. 缩放坐标
        xmin, ymin, xmax, ymax = [c / scale for c in clipped]
        
        # 3. 再次裁剪到输出图边界并检查大小
        clipped_out = clip_xyxy(xmin, ymin, xmax, ymax, out_w, out_h)
        if clipped_out is None or (clipped_out[2]-clipped_out[0]) < min_box or (clipped_out[3]-clipped_out[1]) < min_box:
            root.remove(obj)
            continue

        # 4. 写入 XML (取整)
        for i, name in enumerate(["xmin", "ymin", "xmax", "ymax"]):
            update_text(bnd.find(name), str(int(round(clipped_out[i]))))
        keep_count += 1

    out_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_xml_path), encoding="utf-8", xml_declaration=True)
    return keep_count

def process_one(
    img_path: Path,
    xml_path: Path,
    out_lq_path: Path,
    out_gt_path: Path,
    out_xml_path: Path,
    rel_path: Path,
    scale: int,
    min_box: float,
) -> int:
    with Image.open(img_path) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        
        in_w, in_h = im.size
        # 核心逻辑：强制进行 modcrop
        crop_w, crop_h = modcrop_size(in_w, in_h, scale)
        im_gt = im.crop((0, 0, crop_w, crop_h))

        # 计算输出 LQ 尺寸
        out_w, out_h = crop_w // scale, crop_h // scale
        im_lq = im_gt.resize((out_w, out_h), Image.BICUBIC)

        # 保存裁剪后的 GT
        out_gt_path.parent.mkdir(parents=True, exist_ok=True)
        im_gt.save(out_gt_path)

        # 保存缩小的 LQ
        out_lq_path.parent.mkdir(parents=True, exist_ok=True)
        im_lq.save(out_lq_path)

    # 处理 XML
    kept = update_voc_xml(
        xml_path=xml_path,
        out_xml_path=out_xml_path,
        rel_img_path=rel_path,
        crop_size=(crop_w, crop_h),
        out_size=(out_w, out_h),
        scale=scale,
        min_box=min_box,
    )
    return kept

def main() -> None:
    args = parse_args()
    img_dir, xml_dir = Path(args.img_dir), Path(args.xml_dir)
    out_lq_dir, out_gt_dir = Path(args.out_lq_dir), Path(args.out_gt_dir)
    out_xml_dir = Path(args.out_xml_dir)

    images = list(path for path in img_dir.rglob("*") if path.is_file() and is_image_file(path))
    if not images:
        print("未找到图片。")
        return

    success, skipped = 0, 0

    for idx, img_path in enumerate(images, start=1):
        rel = img_path.relative_to(img_dir)
        xml_path = (xml_dir / rel).with_suffix(".xml")
        
        # 输出路径
        out_lq_path = out_lq_dir / rel
        out_gt_path = out_gt_dir / rel
        out_xml_path = (out_xml_dir / rel).with_suffix(".xml")

        if not xml_path.exists():
            print(f"[{idx}/{len(images)}] 跳过 {img_path.name} (XML未找到)")
            continue

        kept = process_one(
            img_path, xml_path, out_lq_path, out_gt_path, out_xml_path, rel, args.scale, args.min_box
        )

        if kept > 0:
            success += 1
        else:
            # 如果没有目标，删除生成的空文件以保持训练集纯净
            if out_lq_path.exists(): out_lq_path.unlink()
            if out_gt_path.exists(): out_gt_path.unlink()
            if out_xml_path.exists(): out_xml_path.unlink()
            skipped += 1

        print(f"[{idx}/{len(images)}] 处理 {img_path.name}, 保留 {kept} 个目标")

    print(f"\n完成: {success} 成功, {skipped} 过滤(无目标), 总计 {len(images)}")

if __name__ == "__main__":
    main()