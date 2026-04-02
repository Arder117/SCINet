import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.SCINet_arch import SCINet
from basicsr.utils.registry import ARCH_REGISTRY


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class DualDimensionalFeatureCalibration(nn.Module):
    """Calibrate features along channel and spatial dimensions for tiny target responses."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )
        self.height_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.width_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.proj = ConvBNAct(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        channel_attn = self.channel_gate(x)
        h_context = torch.mean(x, dim=3, keepdim=True)
        w_context = torch.mean(x, dim=2, keepdim=True)
        spatial_attn = self.height_gate(h_context) * self.width_gate(w_context)
        calibrated = x * channel_attn * spatial_attn
        return self.proj(calibrated + x)


class SmallTargetHead(nn.Module):
    def __init__(self, in_channels, feat_channels, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, feat_channels, kernel_size=3, padding=1),
            ConvBNAct(feat_channels, feat_channels, kernel_size=3, padding=1)
        )
        self.heatmap = nn.Conv2d(feat_channels, num_classes, kernel_size=1)
        self.size = nn.Conv2d(feat_channels, 2, kernel_size=1)
        self.offset = nn.Conv2d(feat_channels, 2, kernel_size=1)
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, x):
        feat = self.stem(x)
        return {
            'heatmap': self.heatmap(feat),
            'size': F.softplus(self.size(feat)),
            'offset': self.offset(feat),
            'det_features': feat
        }


@ARCH_REGISTRY.register()
class SCINetSmallTargetDetector(nn.Module):
    """
    Joint SR + tiny-target detector.
    The detector reuses SCINet features and predicts CenterNet-style heatmap/size/offset maps.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 num_block=8,
                 num_out_ch=3,
                 upscale=4,
                 conv='BSConvU',
                 upsampler='pixelshuffledirect',
                 p=0.25,
                 detector_feat=96,
                 num_classes=1):
        super().__init__()
        self.backbone = SCINet(
            num_in_ch=num_in_ch,
            num_feat=num_feat,
            num_block=num_block,
            num_out_ch=num_out_ch,
            upscale=upscale,
            conv=conv,
            upsampler=upsampler,
            p=p,
            return_features=True
        )
        fusion_channels = num_feat * 3 + num_out_ch
        self.fusion = nn.Sequential(
            ConvBNAct(fusion_channels, detector_feat, kernel_size=3, padding=1),
            ConvBNAct(detector_feat, detector_feat, kernel_size=3, padding=1)
        )
        self.dfcm = DualDimensionalFeatureCalibration(detector_feat)
        self.head = SmallTargetHead(detector_feat, detector_feat, num_classes)

    def forward(self, x):
        features = self.backbone(x, return_features=True)
        sr = features['sr']
        target_size = sr.shape[-2:]

        shallow = F.interpolate(features['shallow'], size=target_size, mode='bilinear', align_corners=False)
        fused_lr = F.interpolate(features['fused_lr'], size=target_size, mode='bilinear', align_corners=False)
        contrast = F.interpolate(features['contrast'], size=target_size, mode='bilinear', align_corners=False)

        det_input = torch.cat([shallow, fused_lr, contrast, sr], dim=1)
        det_feat = self.fusion(det_input)
        det_feat = self.dfcm(det_feat)
        pred = self.head(det_feat)
        pred['sr'] = sr
        pred['backbone_features'] = features
        return pred
