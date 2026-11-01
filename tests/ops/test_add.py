import torch

from kerops.ops.addition import AddStats, AddStatsBackward
from kerops.utils import assert_close, assert_exact, small_integers

DIMS = (0, 2, 3, 4)


def test_addstats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    y = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    out, mean, sqmean = AddStats(x, y)

    assert_close(out, x.float() + y.float(), max_error=1e-5, mean_error=1e-7)
    # The stats are of the fp16 sum the op returns.
    assert_close(mean, out.float().mean(DIMS), max_error=5e-6, mean_error=2e-6)
    assert_close(sqmean, out.float().square().mean(DIMS), max_error=5e-6, mean_error=2e-6)


def test_addstats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    out = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    out_grad = torch.randn_like(out) / 3
    # Scaled by the voxel count, so that the stats add to the gradient as much as out_grad does.
    voxels = out.numel() // channels
    mean_grad = torch.randn(channels, device='cuda') * voxels / 3
    sqmean_grad = torch.randn(channels, device='cuda') * voxels / 3

    grad = AddStatsBackward(out_grad, mean_grad, sqmean_grad, out)

    out_fp32 = out.float().requires_grad_()
    loss = (out_fp32 * out_grad).sum()
    loss = loss + (out_fp32.mean(DIMS) * mean_grad).sum() + (out_fp32.square().mean(DIMS) * sqmean_grad).sum()
    loss.backward()
    assert_close(grad, out_fp32.grad, max_error=1e-5, mean_error=1e-7)


def test_addstats_exact(channels, power_of_2_shape):
    bsize, *sides = power_of_2_shape
    torch.manual_seed(322)
    x = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    y = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)

    out, mean, sqmean = AddStats(x, y)

    reference = x.double() + y.double()
    assert_exact(out, reference)
    assert_exact(mean, reference.mean(DIMS))
    assert_exact(sqmean, reference.square().mean(DIMS))


def test_addstats_backward_exact(channels, power_of_2_shape):
    bsize, *sides = power_of_2_shape
    torch.manual_seed(322)
    out = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    out_grad = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    # Multiples of the voxel count, so that the stats add integers to the gradient.
    voxels = out.numel() // channels
    mean_grad = small_integers(channels, dtype=torch.float32) * voxels
    sqmean_grad = small_integers(channels, dtype=torch.float32) * voxels / 2

    grad = AddStatsBackward(out_grad, mean_grad, sqmean_grad, out)

    out_fp64 = out.double().requires_grad_()
    loss = (out_fp64 * out_grad).sum()
    loss = loss + (out_fp64.mean(DIMS) * mean_grad).sum() + (out_fp64.square().mean(DIMS) * sqmean_grad).sum()
    loss.backward()
    assert_exact(grad, out_fp64.grad)
