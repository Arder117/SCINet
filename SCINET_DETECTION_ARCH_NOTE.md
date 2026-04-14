# SCINet 小目标检测架构解读

本文档基于当前仓库实现，说明 `SCINetSmallTargetDetector` 的网络拓扑、检测头与小目标监督方式。

## 1. 总体结构（SR + Detection 联合）

`SCINetSmallTargetDetector` 由四部分组成：

1. **SCINet 主干**（复用超分 backbone，并开启 `return_features=True`）
2. **检测前融合层**（把多路 backbone 特征与 SR 图拼接后再压缩）
3. **双维校准模块 DFCM**（通道 + 空间方向注意力）
4. **CenterNet 风格检测头**（heatmap / size / offset）

前向逻辑：

- backbone 输出 `sr/shallow/fused_lr/contrast/...`
- 将 `shallow/fused_lr/contrast` 双线性插值到 SR 分辨率
- 与 `sr` 在通道维拼接后送入融合层
- 经 DFCM 强化小目标响应
- 检测头输出 `heatmap, size, offset`

## 2. SCINet backbone 在检测任务中的作用

检测模型不是另起一个主干，而是直接复用 SCINet 的表征。

- `shallow`: 浅层纹理信息，保留边缘/高频
- `contrast`: 对比增强分支输出
- `fused_lr`: 低分辨率下融合后的语义特征
- `sr`: 最终超分结果（高分辨率）

做法本质上是：

- 用 SR 分支先“放大小目标的空间尺寸”
- 再将多尺度语义与重建图联合用于检测

这比仅在 LR 空间做检测更有利于 tiny object 定位。

## 3. 小目标增强关键：DFCM（DualDimensionalFeatureCalibration）

DFCM 用三个门控进行重标定：

- **通道门控**：全局池化 + 1x1 Conv 两层，输出 channel attention
- **高度门控**：对宽度求均值得到 `H` 方向上下文，再做 1x1 Conv 门控
- **宽度门控**：对高度求均值得到 `W` 方向上下文，再做 1x1 Conv 门控

最终是：

`calibrated = x * channel_attn * spatial_attn`

并与残差相加后用 `3x3 ConvBNGELU` 投影，兼顾稳定训练与细粒度响应。

## 4. 检测头（SmallTargetHead）

检测头是轻量双卷积 stem + 三个 1x1 分支：

- `heatmap`: 输出 `num_classes` 通道
- `size`: 输出 2 通道（w,h），并经 `softplus` 保证正值
- `offset`: 输出 2 通道（dx,dy）做亚像素中心回归

实现细节：

- `heatmap` bias 初始化为 `-2.19`，使初始 sigmoid 概率偏低，缓解正负样本极不平衡问题。
- 检测发生在 **SR 分辨率**，因此输出张量空间尺寸与 `sr` 相同。

## 5. 监督信号（数据集编码）

`SCINetDetectionDataset` 把框标注转换为 CenterNet 监督：

- `gt_heatmap`: 在目标中心画 2D 高斯核（半径由 `gaussian_radius` 与 `min_overlap` 决定）
- `gt_size`: 仅在中心点处写入目标 `w,h`
- `gt_offset`: 写入小数偏移 `(cx-floor(cx), cy-floor(cy))`
- `gt_mask`: 中心点位置置 1，用于 size/offset 的稀疏监督

因此 size/offset 并不是全图密集监督，而是“中心点回归”。

## 6. 损失函数与训练策略

`SCINetDetectionModel` 联合四个损失：

- 热图：focal 风格损失（对正负样本分别加权）
- 尺寸：masked L1（仅 `gt_mask=1` 位置参与）
- 偏移：masked L1（同上）
- 超分：像素重建损失（默认 L1）

总损失：

`L = hm_w * L_hm + size_w * L_size + off_w * L_off + sr_w * L_sr`

默认配置中 `size_weight=0.1`，其余多为 1.0，用于抑制尺寸分支过大梯度。

## 7. 推理解码（检测后处理）

推理脚本使用如下流程：

1. `sigmoid(heatmap)`
2. 局部最大池化 NMS（`pool_nms`）
3. 取 Top-K 峰值
4. 结合 `offset` 还原中心点亚像素坐标
5. 结合 `size` 还原边框 `(x1,y1,x2,y2)`
6. 阈值过滤并可视化

这是标准 CenterNet 解码流程，且直接在 SR 图上画框。

## 8. 该实现为何适配“小目标”

可归纳为三层增强：

1. **分辨率增强**：先 SR 再检测，提升小目标可见像素数
2. **特征增强**：融合 shallow + contrast + fused_lr + sr，多路互补
3. **监督增强**：中心热图 + 亚像素偏移，使微小目标定位更稳定

另外，DFCM 对通道和 H/W 方向分别门控，能让弱小响应在复杂背景下更容易被凸显。

## 9. 关键可调超参数建议

- `detector_feat`: 检测分支宽度（默认 96）
- `num_classes`: 类别数
- `min_overlap`: 高斯半径计算超参数
- `topk/score_thr/nms_kernel`: 推理后处理参数
- `hm_weight/size_weight/off_weight/sr_weight`: 多任务损失平衡参数

