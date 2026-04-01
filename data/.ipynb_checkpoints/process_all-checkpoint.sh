#!/bin/bash

# 基础路径配置
DATA_ROOT=~/SCINet/data
PY_SCRIPT=${DATA_ROOT}/preprocess_images.py

# 确保 Python 脚本存在
if [ ! -f "$PY_SCRIPT" ]; then
    echo "错误: 找不到 $PY_SCRIPT"
    exit 1
fi

# 处理三个子集
for split in train val test; do
    echo "===================================================="
    echo "正在同步生成高清(GT)与低清(LQ)数据集: ${split}"
    
    # 输入源 (你原始存放数据的目录)
    IN_IMG=${DATA_ROOT}/${split}/img
    IN_XML=${DATA_ROOT}/${split}/xml
    
    # 输出 1: 高清裁剪版 (例如 gttrain)
    GT_DIR=${DATA_ROOT}/gt${split}
    # 输出 2: 低清缩小版 (例如 lqtrain)
    LQ_DIR=${DATA_ROOT}/lq${split}
    
    # 清理并重建目录
    rm -rf "${GT_DIR}" "${LQ_DIR}"
    mkdir -p "${GT_DIR}/img" "${GT_DIR}/xml"
    mkdir -p "${LQ_DIR}/img" "${LQ_DIR}/xml"
    
    # 调用 Python 处理
    python3 "$PY_SCRIPT" \
        --img_dir "$IN_IMG" \
        --xml_dir "$IN_XML" \
        --out_gt_img "${GT_DIR}/img" \
        --out_gt_xml "${GT_DIR}/xml" \
        --out_lq_img "${LQ_DIR}/img" \
        --out_lq_xml "${LQ_DIR}/xml" \
        --scale 4 \
        --min_box 1.0
        
    echo "子集 ${split} 处理完毕。"
done

echo "===================================================="
echo "所有任务已完成！"
echo "请检查目录: gt[train/val/test] 和 lq[train/val/test]"