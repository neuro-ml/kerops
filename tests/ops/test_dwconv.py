import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.conv import DWConv, DWConvWGRAD
from kerops.ops.conv.dwconv_wgrad import generate_inputs_dwconv_wgrad
from kerops.utils import assert_close, assert_wgrad_close


def test_dwconv(bsize, channels, other_1, other_2, other_3):
    if channels < 8:
        pytest.skip('tuned only for 8+ channels')

    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )
    weight = torch.empty(channels, 1, 3, 3, 3, device='cuda', dtype=torch.float16)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    out = DWConv(x, weight[:, 0].permute(1, 2, 3, 0).contiguous())

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        reference = F.conv3d(x.float(), weight.float(), padding=1, groups=channels)
    assert_close(out, reference, max_error=1e-2, mean_error=1e-4)

    torch.cuda.empty_cache()


def assert_dwconv_wgrad_close(x, grad):
    weight = torch.zeros(x.shape[1], 1, 3, 3, 3, device='cuda', requires_grad=True)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        F.conv3d(x.float(), weight, padding=1, groups=x.shape[1]).backward(grad.float())
    assert_wgrad_close(DWConvWGRAD(x, grad), weight.grad[:, 0].permute(1, 2, 3, 0), cos_error=1e-6, norm_error=5e-4)


def test_dwconv_wgrad(bsize, channels, other_1, other_2, other_3):
    if channels < 8 or other_1 > 53 or other_2 > 53 or other_3 > 53:
        pytest.skip('tuned only for 8+ channels; the depthwise reference of cuDNN is slow past 53')

    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).relu()
    grad = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16)
    assert_dwconv_wgrad_close(x.to(memory_format=torch.channels_last_3d), grad.to(memory_format=torch.channels_last_3d))

    torch.cuda.empty_cache()


# The bench sizes, 128³ to 256³ voxels: the error of long accumulation chains shows only there.
@pytest.mark.parametrize('channels', [8, 16, 32, 64, 128])
def test_dwconv_wgrad_bench_sizes(channels):
    torch.manual_seed(322)
    x, grad = generate_inputs_dwconv_wgrad({'channels': channels})
    assert_dwconv_wgrad_close(x.relu(), grad)

    torch.cuda.empty_cache()
