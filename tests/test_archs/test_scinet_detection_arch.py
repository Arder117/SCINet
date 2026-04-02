import torch

from basicsr.archs import build_network


def test_scinet_backbone_can_return_features():
    net = build_network(
        dict(
            type='SCINet',
            num_in_ch=3,
            num_feat=32,
            num_block=8,
            num_out_ch=3,
            upscale=4,
            return_features=True
        )
    )
    net.eval()

    with torch.no_grad():
        out = net(torch.rand(1, 3, 32, 40))

    assert isinstance(out, dict)
    assert out['sr'].shape == (1, 3, 128, 160)
    assert out['shallow'].shape[2:] == (32, 40)
    assert out['fused_lr'].shape[2:] == (32, 40)


def test_scinet_small_target_detector_output_shapes():
    net = build_network(
        dict(
            type='SCINetSmallTargetDetector',
            num_in_ch=3,
            num_feat=32,
            num_block=8,
            num_out_ch=3,
            upscale=4,
            detector_feat=48,
            num_classes=1
        )
    )
    net.eval()

    with torch.no_grad():
        out = net(torch.rand(2, 3, 32, 40))

    assert out['sr'].shape == (2, 3, 128, 160)
    assert out['heatmap'].shape == (2, 1, 128, 160)
    assert out['size'].shape == (2, 2, 128, 160)
    assert out['offset'].shape == (2, 2, 128, 160)
