# SCINet 小目标检测快速上手

本仓库已经包含联合超分与检测的网络实现：
- 检测骨干：`SCINetSmallTargetDetector`
- 训练模型：`SCINetDetectionModel`

## 1) 训练入口

使用新增配置文件：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_SCINet_detection_x4.yml
```

> 如果你只使用 `PairedImageDataset`（仅返回 `lq/gt`），模型会只训练 SR 分支。要训练检测头，必须在 dataloader 中额外提供检测监督。

## 2) dataloader 需要提供的键

`SCINetDetectionModel.feed_data()` 支持以下输入：

- `lq`: 低分辨率输入
- `gt`: 超分真值（可选，但建议保留）
- `gt_heatmap`: 中心点热力图
- `gt_size`: 中心点对应目标宽高（2 通道）
- `gt_offset`: 中心点亚像素偏移（2 通道）
- `gt_mask`: 回归有效位置掩码

## 3) 损失组成

- `heatmap`: focal 风格损失
- `size` / `offset`: masked L1
- `sr`: 像素重建损失（L1）

总损失权重由 yaml 中以下参数控制：

- `hm_weight`
- `size_weight`
- `off_weight`
- `sr_weight`

## 4) 推荐训练流程

1. 先用 SR 配置预训练 SCINet 超分模型。  
2. 在检测配置中设置 `path.pretrain_network_g` 加载预训练权重。  
3. 联合训练（SR + Detection）并调节 4 个损失权重。  

## 5) 推理后处理（需要你补充）

网络前向输出包含：
- `heatmap`（先 sigmoid）
- `size`
- `offset`

需要在推理脚本里完成：
1. `heatmap` 取 Top-K 中心点
2. 用 `offset` 修正中心坐标
3. 用 `size` 还原 bbox
4. 执行 NMS，得到最终检测结果

## 6) 常见问题

- **报错 `No detection or SR supervision was provided.`**  
  说明 dataloader 没有提供 `gt`，且也没提供检测监督键。

- **只看到 SR 指标，没有检测指标**  
  说明你的验证流程还没加入检测解码和 mAP/Recall 计算逻辑。
