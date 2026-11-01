import torch

from kerops.ops.stats import Stats, StatsBackward
from kerops.utils import assert_close

DIMS = (0, 2, 3, 4)


def test_stats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    mean, sqmean = Stats(x)

    assert_close(mean, x.float().mean(DIMS), max_error=1e-5, mean_error=3e-6)
    assert_close(sqmean, x.float().square().mean(DIMS), max_error=1e-5, mean_error=3e-6)


def test_stats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    # Scaled by the voxel count, as a loss scale keeps fp16 gradients of the order of 1 out of the subnormal range.
    voxels = x.numel() // channels
    mean_grad = torch.randn(channels, device='cuda') * voxels / 3
    sqmean_grad = torch.randn(channels, device='cuda') * voxels / 3

    grad = StatsBackward(x, mean_grad, sqmean_grad)

    x_fp32 = x.float().requires_grad_()
    ((x_fp32.mean(DIMS) * mean_grad).sum() + (x_fp32.square().mean(DIMS) * sqmean_grad).sum()).backward()
    assert_close(grad, x_fp32.grad, max_error=1e-5, mean_error=1e-7)
