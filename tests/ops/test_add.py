import torch

from kerops.ops.addition import AddStats, AddStatsBackward
from kerops.utils import allclose_two_stage


def test_addstats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    y = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    r = x + y
    r_fp32 = r.to(torch.float32)
    dim = (0, 2, 3, 4)
    mean = r_fp32.mean(dim=dim)
    sqmean = (r_fp32**2).mean(dim=dim)
    r_, mean_, sqmean_ = AddStats(x, y)

    assert torch.allclose(r, r_)
    assert torch.allclose(mean, mean_, atol=1e-4, rtol=1e-4)
    assert torch.allclose(sqmean, sqmean_, atol=1e-4, rtol=1e-4)


def test_addstats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    y = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    x.requires_grad_(True)
    y.requires_grad_(True)

    upstr_sqmean = torch.randn(channels, device='cuda', dtype=torch.float32) / 3
    upstr_mean = torch.randn(channels, device='cuda', dtype=torch.float32) / 3
    upstr_add = torch.randn_like(x, dtype=torch.float16, device='cuda') / 3

    z = x + y
    z_fp32 = z.to(torch.float32)
    mean = z_fp32.mean(dim=(0, 2, 3, 4))
    sqmean = (z_fp32**2).mean(dim=(0, 2, 3, 4))

    loss = (sqmean * upstr_sqmean).sum() + (mean * upstr_mean).sum() + (z_fp32 * upstr_add).sum()
    loss.backward()
    grad = x.grad

    grad_to_check = AddStatsBackward(upstr_add, upstr_mean, upstr_sqmean, z)
    if x.numel() // channels >= 100:
        assert allclose_two_stage(
            grad, grad_to_check, rtol_strict=1e-5, atol_strict=1e-4, rtol_narrow=1e-4, atol_narrow=1e-3
        )
    else:
        assert torch.allclose(grad, grad_to_check, rtol=1e-5, atol=5e-4)
