#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def modcrop_size(width: int, height: int, scale: int) -> Tuple[int, int]:
    """确保尺寸能被 scale 整除"""
    return width - (width % scale), height - (height % scale)

def clip_xyxy(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    xmin = max(0.0, min(xmin, float(width)))
    ymin = max(0.0, min(ymin, float(height)))
    xmax = max(0.0, min(xmax, float(width)))
    ymax = max(0.0, min(ymax, float(height)))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax

def update_text(node: Optional[ET.Element], value: str) -> None:
    if node is not None:
        node.text = value

def save_voc_xml(xml_path: Path, out_xml_path: Path, rel_img_path: Path, 
                 crop_size: Tuple[int, int], out_size: Tuple[int, int], 
                 scale_factor: float, min_box: float) -> int:
    """生成适配当前图像尺寸的 XML 标注"""
    if not xml_path.exists(): return 0
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    crop_w, crop_h = crop_size
    out_w, out_h = out_size

    size_node = root.find("size")
    if size_node is not None:
        update_text(size_node.find("width"), str(out_w))
        update_text(size_node.find("height"), str(out_h))

    update_text(root.find("filename"), rel_img_path.name)
    update_text(root.find("path"), str(rel_img_path.as_posix()))

    keep_count = 0
    for obj in list(root.findall("object")):
        bnd = obj.find("bndbox")
        if bnd is None: continue
        
        # 获取原始坐标
        try:
            coords = [float(bnd.find(n).text) for n in ["xmin", "ymin", "xmax", "ymax"]]
        except (ValueError, AttributeError):
            root.remove(obj)
            continue

        # 1. 裁剪到 modcrop 区域 (基于原图坐标)
        clipped = clip_xyxy(*coords, crop_w, crop_h)
        if clipped is None:
            root.remove(obj)
            continue
        
        # 2. 坐标缩放 (如果是 GT 则 scale_factor=1, 如果是 LQ 则 scale_factor=4)
        final_coords = [c / scale_factor for c in clipped]
        
        # 3. 尺寸过滤 (防止缩放后框消失)
        if (final_coords[2]-final_coords[0]) < min_box or (final_coords[3]-final_coords[1]) < min_box:
            root.remove(obj)
            continue

        # 4. 写入 XML
        for i, name in enumerate(["xmin", "ymin", "xmax", "ymax"]):
            update_text(bnd.find(name), str(int(round(final_coords[i]))))
        keep_count += 1

    out_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_xml_path), encoding="utf-8", xml_declaration=True)
    return keep_count

def main():
    parser = argparse.ArgumentParser(description="同步生成裁剪后的 GT 和 缩小的 LQ 数据集")
    parser.add_argument("--img_dir", required=True, help="原始图像目录 (train/img)")
    parser.add_argument("--xml_dir", required=True, help="原始 XML 目录 (train/xml)")
    parser.add_argument("--out_gt_img", required=True, help="高清裁剪图输出目录")
    parser.add_argument("--out_gt_xml", required=True, help="高清裁剪标注输出目录")
    parser.add_argument("--out_lq_img", required=True, help="低清缩小图输出目录")
    parser.add_argument("--out_lq_xml", required=True, help="低清缩小标注输出目录")
    parser.add_argument("--scale", type=int, default=4, help="缩放倍率")
    parser.add_argument("--min_box", type=float, default=1.0, help="保留框的最小尺寸")
    args = parser.parse_args()

    images = [p for p in Path(args.img_dir).rglob("*") if is_image_file(p)]
    print(f"找到 {len(images)} 张图片，开始处理...")

    for idx, img_path in enumerate(images, start=1):
        rel = img_path.relative_to(args.img_dir)
        xml_path = (Path(args.xml_dir) / rel).with_suffix(".xml")
        
        if not xml_path.exists():
            print(f"跳过 {rel.name}: XML 不存在")
            continue

        with Image.open(img_path) as im:
            if im.mode not in ("RGB", "L"): im = im.convert("RGB")
            
            # 1. 计算 Modcrop 尺寸并裁剪得到 GT
            cw, ch = modcrop_size(im.size[0], im.size[1], args.scale)
            im_gt = im.crop((0, 0, cw, ch))
            
            # 2. 缩放得到 LQ
            lw, lh = cw // args.scale, ch // args.scale
            im_lq = im_gt.resize((lw, lh), Image.BICUBIC)

            # 3. 保存高清图与低清图
            gt_p, lq_p = Path(args.out_gt_img)/rel, Path(args.out_lq_img)/rel
            gt_p.parent.mkdir(parents=True, exist_ok=True)
            lq_p.parent.mkdir(parents=True, exist_ok=True)
            im_gt.save(gt_p)
            im_lq.save(lq_p)

            # 4. 生成两套 XML
            # 高清版: scale_factor = 1.0 (坐标仅裁剪不缩小)
            save_voc_xml(xml_path, (Path(args.out_gt_xml)/rel).with_suffix(".xml"), rel, (cw, ch), (cw, ch), 1.0, args.min_box)
            # 低清版: scale_factor = args.scale (坐标缩小)
            kept = save_voc_xml(xml_path, (Path(args.out_lq_xml)/rel).with_suffix(".xml"), rel, (cw, ch), (lw, lh), float(args.scale), args.min_box)
            
            print(f"[{idx}/{len(images)}] 处理 {rel.name}: GT({cw}x{ch}) -> LQ({lw}x{lh}), 目标数: {kept}")

if __name__ == "__main__":
    main()