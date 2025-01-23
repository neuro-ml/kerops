import torch
from torch import nn

from kerops.ops.avgpool import AvgPoolCeilStats, AvgPoolCeilStatsBackward
from kerops.utils import allclose_two_stage


def test_avgpool_ceil_stats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x_def = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda')
    x = x_def.to(memory_format=torch.channels_last_3d)
    ap = nn.AvgPool3d(2, ceil_mode=True)

    res = ap(x_def)
    res_fp32 = res.to(torch.float32)
    mean = res_fp32.mean(dim=(0, 2, 3, 4))
    sqmean = (res_fp32**2).mean(dim=(0, 2, 3, 4))

    res_, mean_, sqmean_ = AvgPoolCeilStats(x)

    assert torch.allclose(mean, mean_, atol=1e-3, rtol=1e-5)
    assert torch.allclose(sqmean, sqmean_, atol=1e-3, rtol=1e-5)
    if res.numel() < 100:
        assert torch.allclose(res, res_, rtol=1e-5, atol=5e-5)
    else:
        assert allclose_two_stage(
            res,
            res_,
            rtol_strict=1e-5,
            atol_strict=5e-5,
            rtol_narrow=1e-4,
            atol_narrow=1e-3,
            precision=torch.float32,
            debug_info='print',
        )


def test_avgpool_ceil_stats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16)
    x.requires_grad_(True)
    o = nn.AvgPool3d(2, ceil_mode=True)(x)
    o_fp32 = o.to(torch.float32)
    mean = o_fp32.mean(dim=(0, 2, 3, 4))
    sqmean = (o_fp32**2).mean(dim=(0, 2, 3, 4))

    upstr_pool = torch.randn_like(o)
    upstr_mean = torch.randn_like(mean)
    upstr_sqmean = torch.randn_like(sqmean)

    loss = (upstr_pool * o).sum() + (upstr_mean * mean).sum() + (upstr_sqmean * sqmean).sum()
    loss.backward()
    grad = x.grad

    grad_check = AvgPoolCeilStatsBackward(
        upstr_pool.to(memory_format=torch.channels_last_3d),
        upstr_mean,
        upstr_sqmean,
        o.to(memory_format=torch.channels_last_3d),
        x.shape,
    )

    if grad.numel() < 100:
        assert torch.allclose(grad, grad_check, atol=2e-3, rtol=1e-4)
    else:
        assert allclose_two_stage(
            grad,
            grad_check,
            rtol_strict=1e-4,
            atol_strict=5e-4,
            rtol_narrow=1e-4,
            atol_narrow=2e-3,
            debug_info='print',
        )
