#!/usr/bin/env python3
"""
Convert VOC XML annotations to SCINetDetectionDataset txt format.
Tailored for: GT scale alignment (1:1 scale with XML)
Output: x1 y1 x2 y2 class_id (0-based)
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VOC XML to SCINet TXT")
    parser.add_argument('--xml_dir', type=Path, required=True, help='Input XML directory (e.g., gttrain/xml)')
    parser.add_argument('--out_dir', type=Path, required=True, help='Output TXT directory (e.g., train/ann)')
    parser.add_argument('--classes', type=str, default='UAV', help='Comma-separated class names (first is ID 0)')
    parser.add_argument('--skip_unknown', action='store_true', help='Skip objects not in class list')
    return parser.parse_args()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 建立类别映射。例如 'UAV' -> 0
    class_list = [c.strip() for c in args.classes.split(',') if c.strip()]
    class_to_id = {name: idx for idx, name in enumerate(class_list)}

    xml_files = sorted(args.xml_dir.glob('*.xml'))
    converted = 0
    skipped_objs = 0

    print(f"开始处理 {len(xml_files)} 个 XML 文件...")

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"警告: 无法解析 {xml_path.name}, 跳过。错误: {e}")
            continue

        # 获取图像尺寸以进行边界检查
        size_node = root.find('size')
        if size_node is not None:
            width = float(size_node.findtext('width', '0'))
            height = float(size_node.findtext('height', '0'))
        else:
            width, height = 0, 0

        rows = []
        for obj in root.findall('object'):
            name = obj.findtext('name', '').strip()

            # 类别 ID 处理
            if name in class_to_id:
                class_id = class_to_id[name]
            else:
                if args.skip_unknown:
                    skipped_objs += 1
                    continue
                class_id = 0  # 默认归为第一类

            bnd = obj.find('bndbox')
            if bnd is None:
                continue

            # 读取原始坐标 (580x324 尺度)
            xmin = float(bnd.findtext('xmin', '0'))
            ymin = float(bnd.findtext('ymin', '0'))
            xmax = float(bnd.findtext('xmax', '0'))
            ymax = float(bnd.findtext('ymax', '0'))

            # VOC 标准转换: 1-based (XML) 转为 0-based (训练使用)
            # 因为你的尺度是 1:1，不需要乘以任何倍数
            xmin -= 1.0
            ymin -= 1.0
            xmax -= 1.0
            ymax -= 1.0

            # 边界限制，防止坐标超出图片范围
            if width > 0 and height > 0:
                xmin = clamp(xmin, 0.0, width - 1.0)
                ymin = clamp(ymin, 0.0, height - 1.0)
                xmax = clamp(xmax, 0.0, width - 1.0)
                ymax = clamp(ymax, 0.0, height - 1.0)

            # 过滤无效框
            if xmax <= xmin or ymax <= ymin:
                continue

            rows.append(f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} {class_id}")

        if rows:
            # 使用 xml 的文件名作为 txt 的文件名，确保一一对应
            out_txt = args.out_dir / f"{xml_path.stem}.txt"
            out_txt.write_text('\n'.join(rows), encoding='utf-8')
            converted += 1

    print("-" * 30)
    print(f"处理完成！")
    print(f"成功转换 XML: {converted}")
    print(f"输出目录: {args.out_dir}")
    if skipped_objs > 0:
        print(f"跳过未知类别对象: {skipped_objs}")
    print(f"使用类别映射: {class_to_id}")


if __name__ == '__main__':
    main()