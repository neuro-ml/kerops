import torch
from torch.nn import functional as F

from kerops.ops.avgpool import AvgPoolCeilStats, AvgPoolCeilStatsBackward
from kerops.utils import assert_close

DIMS = (0, 2, 3, 4)


def test_avgpool_ceil_stats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    out, mean, sqmean = AvgPoolCeilStats(x)

    reference = F.avg_pool3d(x.float(), 2, ceil_mode=True)
    assert_close(out, reference, max_error=1e-5, mean_error=1e-7)
    assert_close(mean, reference.mean(DIMS), max_error=1e-4, mean_error=5e-5)
    assert_close(sqmean, reference.square().mean(DIMS), max_error=1e-4, mean_error=5e-5)


def test_avgpool_ceil_stats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    out = F.avg_pool3d(x, 2, ceil_mode=True).contiguous(memory_format=torch.channels_last_3d)
    out_grad = torch.randn_like(out)
    # Scaled by the voxel count of the output, so that the stats add to the gradient as much as out_grad does.
    voxels = out.numel() // channels
    mean_grad = torch.randn(channels, device='cuda') * voxels
    sqmean_grad = torch.randn(channels, device='cuda') * voxels

    grad = AvgPoolCeilStatsBackward(out_grad, mean_grad, sqmean_grad, out, x.shape)

    x_fp32 = x.float().requires_grad_()
    out_fp32 = F.avg_pool3d(x_fp32, 2, ceil_mode=True)
    # The op gets the forward output in fp16: the reference takes those values and keeps the fp32 gradients.
    out_fp32 = out_fp32 + (out.float() - out_fp32).detach()
    loss = (out_fp32 * out_grad).sum()
    loss = loss + (out_fp32.mean(DIMS) * mean_grad).sum() + (out_fp32.square().mean(DIMS) * sqmean_grad).sum()
    loss.backward()
    assert_close(grad, x_fp32.grad, max_error=1e-5, mean_error=1e-7)
