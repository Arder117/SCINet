#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image

def check_split(data_root, split_name, scale=4):
    """检查单个数据集（如 train）的 GT 和 LQ 是否匹配"""
    gt_dir = data_root / f"gt{split_name}" / "img"
    lq_dir = data_root / f"lq{split_name}" / "img"
    
    if not gt_dir.exists() or not lq_dir.exists():
        print(f"跳过 {split_name}: 文件夹未找到 ({gt_dir} 或 {lq_dir})")
        return

    print(f"\n>>> 正在检查数据集: [ {split_name.upper()} ]")
    
    gt_images = sorted([p for p in gt_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    
    match_count = 0
    error_count = 0
    missing_count = 0

    for gt_path in gt_images:
        rel_path = gt_path.relative_to(gt_dir)
        lq_path = lq_dir / rel_path

        if not lq_path.exists():
            missing_count += 1
            continue

        try:
            with Image.open(gt_path) as img_gt, Image.open(lq_path) as img_lq:
                w_gt, h_gt = img_gt.size
                w_lq, h_lq = img_lq.size

                if w_lq * scale != w_gt or h_lq * scale != h_gt:
                    print(f"  [尺寸不对] {rel_path}: GT({w_gt}x{h_gt}) vs LQ({w_lq}x{h_lq})")
                    error_count += 1
                else:
                    match_count += 1
        except Exception as e:
            print(f"  [读取失败] {rel_path}: {e}")
            error_count += 1

    print(f"    结果: ✅ 成功:{match_count} | ❌ 错误:{error_count} | ❓ 缺失:{missing_count}")

def main():
    # 自动定位到你的数据根目录
    DATA_ROOT = Path("~/SCINet/data").expanduser()
    SCALE = 4
    
    print(f"开始全局一致性检查 (Scale: {SCALE}x)")
    print("=" * 50)
    
    # 依次检查三个子集
    for split in ["train", "val", "test"]:
        check_split(DATA_ROOT, split, SCALE)
    
    print("\n" + "=" * 50)
    print("检查结束。如果所有错误均为 0，你可以放心开始训练！")

if __name__ == "__main__":
    main()