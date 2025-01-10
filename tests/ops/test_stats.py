import torch

from kerops.ops.stats import Stats, StatsBackward
from kerops.utils import allclose_two_stage


def test_stats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    x_fp32 = x.to(torch.float32)
    dim = (0, 2, 3, 4)
    mean = x_fp32.mean(dim=dim)
    sqmean = (x_fp32**2).mean(dim=dim)
    mean_, sqmean_ = Stats(x)

    assert torch.allclose(mean, mean_, atol=1e-4, rtol=1e-4)
    assert torch.allclose(sqmean, sqmean_, atol=1e-4, rtol=1e-4)


def test_stats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    x.requires_grad_(True)

    upstr_sqmean = torch.randn(channels, device='cuda', dtype=torch.float32) / 3
    upstr_mean = torch.randn(channels, device='cuda', dtype=torch.float32) / 3

    x_fp32 = x.to(torch.float32)
    mean = x_fp32.mean(dim=(0, 2, 3, 4))
    sqmean = (x_fp32**2).mean(dim=(0, 2, 3, 4))

    loss = (sqmean * upstr_sqmean).sum() + (mean * upstr_mean).sum()
    loss.backward()
    grad = x.grad

    grad_to_check = StatsBackward(x, upstr_mean, upstr_sqmean)
    if x.numel() // channels >= 100:
        assert allclose_two_stage(
            grad, grad_to_check, rtol_strict=1e-5, atol_strict=1e-4, rtol_narrow=1e-4, atol_narrow=1e-3
        )
    else:
        assert torch.allclose(grad, grad_to_check, rtol=1e-5, atol=5e-4)
