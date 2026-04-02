#!/usr/bin/env python3
"""Convert VOC XML annotations to SCINetDetectionDataset txt format.

Output per image txt line format:
    x1 y1 x2 y2 class_id
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', type=Path, required=True, help='Directory containing VOC XML files.')
    parser.add_argument('--out_dir', type=Path, required=True, help='Output directory for txt labels.')
    parser.add_argument('--classes', type=str, default='', help='Comma-separated class names in fixed id order.')
    parser.add_argument('--default_class_id', type=int, default=0, help='Class id if class list is not provided.')
    parser.add_argument('--skip_unknown', action='store_true', help='Skip objects with unknown class names.')
    return parser.parse_args()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {}
    if args.classes.strip():
        class_to_id = {name.strip(): idx for idx, name in enumerate(args.classes.split(',')) if name.strip()}

    xml_files = sorted(args.xml_dir.glob('*.xml'))
    converted = 0
    skipped = 0

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext('filename')
        stem = Path(filename).stem if filename else xml_path.stem

        size = root.find('size')
        width = int(size.findtext('width', default='0')) if size is not None else 0
        height = int(size.findtext('height', default='0')) if size is not None else 0

        rows = []
        for obj in root.findall('object'):
            name = obj.findtext('name', default='').strip()
            if class_to_id:
                if name not in class_to_id:
                    if args.skip_unknown:
                        skipped += 1
                        continue
                    class_id = args.default_class_id
                else:
                    class_id = class_to_id[name]
            else:
                class_id = args.default_class_id

            bnd = obj.find('bndbox')
            if bnd is None:
                continue

            xmin = float(bnd.findtext('xmin', default='0'))
            ymin = float(bnd.findtext('ymin', default='0'))
            xmax = float(bnd.findtext('xmax', default='0'))
            ymax = float(bnd.findtext('ymax', default='0'))

            # VOC boxes are usually 1-based inclusive; convert to 0-based continuous coordinates.
            xmin -= 1.0
            ymin -= 1.0
            xmax -= 1.0
            ymax -= 1.0

            if width > 0 and height > 0:
                xmin = clamp(xmin, 0.0, width - 1.0)
                ymin = clamp(ymin, 0.0, height - 1.0)
                xmax = clamp(xmax, 0.0, width - 1.0)
                ymax = clamp(ymax, 0.0, height - 1.0)

            if xmax <= xmin or ymax <= ymin:
                continue

            rows.append(f'{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} {class_id}')

        out_txt = args.out_dir / f'{stem}.txt'
        out_txt.write_text('\n'.join(rows), encoding='utf-8')
        converted += 1

    print(f'Converted XML files: {converted}')
    if class_to_id:
        print(f'Class mapping: {class_to_id}')
    if skipped:
        print(f'Skipped unknown objects: {skipped}')


if __name__ == '__main__':
    main()
