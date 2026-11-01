import torch

from kerops.ops.stats import Stats, StatsBackward
from kerops.utils import assert_close, assert_exact, small_integers

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


def test_stats_exact(channels, power_of_2_shape):
    bsize, *sides = power_of_2_shape
    torch.manual_seed(322)
    x = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)

    mean, sqmean = Stats(x)

    assert_exact(mean, x.double().mean(DIMS))
    assert_exact(sqmean, x.double().square().mean(DIMS))


def test_stats_backward_exact(channels, power_of_2_shape):
    bsize, *sides = power_of_2_shape
    torch.manual_seed(322)
    x = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    # Multiples of the voxel count, so that the gradient is of integers.
    voxels = x.numel() // channels
    mean_grad = small_integers(channels, dtype=torch.float32) * voxels
    sqmean_grad = small_integers(channels, dtype=torch.float32) * voxels / 2

    grad = StatsBackward(x, mean_grad, sqmean_grad)

    x_fp64 = x.double().requires_grad_()
    ((x_fp64.mean(DIMS) * mean_grad).sum() + (x_fp64.square().mean(DIMS) * sqmean_grad).sum()).backward()
    assert_exact(grad, x_fp64.grad)
