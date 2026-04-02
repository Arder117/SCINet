# SCINet 小目标检测快速上手

已补齐可直接使用的数据集类与训练配置：
- 数据集类：`SCINetDetectionDataset`
- 训练配置：`options/train/train_SCINet_detection_x4.yml`

## 1) 标注格式

请在 `dataroot_ann` 下为每张图像准备同名 `.txt` 文件（例如 `0001.bmp` 对应 `0001.txt`）。

每行一个目标框，格式为：

```text
x1 y1 x2 y2 [class_id]
```

- 坐标是 **GT 图像坐标系**（像素）
- `class_id` 可省略，省略时默认 0

## 2) 训练命令

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_SCINet_detection_x4.yml
```

## 3) dataloader 输出（已在数据集类中实现）

`SCINetDetectionDataset` 会返回：

- `lq`
- `gt`
- `gt_heatmap`
- `gt_size`
- `gt_offset`
- `gt_mask`

其中检测监督为 CenterNet 风格：
- `gt_heatmap`: 目标中心高斯热图
- `gt_size`: 中心点处目标宽高
- `gt_offset`: 中心点亚像素偏移
- `gt_mask`: size/offset 的有效位置掩码

## 4) 损失组成

- `heatmap`: focal 风格损失
- `size` / `offset`: masked L1
- `sr`: 像素重建损失（L1）

总损失权重在 yaml 中可调：
- `hm_weight`
- `size_weight`
- `off_weight`
- `sr_weight`

## 5) 推荐流程

1. 先训练 SR 模型作为初始化。  
2. 在检测配置中设置 `path.pretrain_network_g`。  
3. 联合训练 SR + Detection。  

## 6) 推理后处理

模型会输出 `heatmap/size/offset`，你需要在推理脚本中完成：
1. heatmap Top-K
2. offset 校正中心点
3. size 还原 bbox
4. NMS

## 7) 注意事项

- 当前数据集实现默认不做 bbox 对齐增强（如随机裁剪/翻转）。建议先跑通训练，再按需求补充带几何变换的标注同步增强。
- 若出现 `No detection or SR supervision was provided.`，请检查 dataloader 是否正确返回上述监督键。
