import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.conv import DWConv, DWConvWGRAD
from kerops.ops.conv.dwconv_wgrad import generate_inputs_dwconv_wgrad
from kerops.utils import assert_close, assert_exact, assert_wgrad_close, small_integers, sparse_signs

# Sides of the large exact tests: past 2M voxels the tolerance check of a weight gradient no longer sees one voxel lost.
LARGE = (128, 128, 128)


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


def dwconv_wgrad_reference(x, grad):
    """[3, 3, 3, C] weight gradient of the depthwise convolution by cuDNN, in the dtype of `x`."""
    weight = torch.zeros(x.shape[1], 1, 3, 3, 3, device='cuda', dtype=x.dtype, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        F.conv3d(x, weight, padding=1, groups=x.shape[1]).backward(grad)
    return weight.grad[:, 0].permute(1, 2, 3, 0)


def assert_dwconv_wgrad_close(x, grad):
    reference = dwconv_wgrad_reference(x.float(), grad.float())
    assert_wgrad_close(DWConvWGRAD(x, grad), reference, cos_error=1e-6, norm_error=5e-4)


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


@pytest.mark.parametrize('channels', [8, 16, 32, 64, 128])
def test_dwconv_exact(channels, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    weight = small_integers(3, 3, 3, channels)

    reference = F.conv3d(x.double(), weight.double().permute(3, 0, 1, 2)[:, None], padding=1, groups=channels)
    assert_exact(DWConv(x, weight), reference)


@pytest.mark.parametrize('channels', [8, 16, 32, 64, 128])
def test_dwconv_wgrad_exact(channels, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    grad = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)

    assert_exact(DWConvWGRAD(x, grad), dwconv_wgrad_reference(x.double(), grad.double()))


@pytest.mark.parametrize('channels', [8, 16, 32, 64, 128])
def test_dwconv_large_exact(channels):
    torch.manual_seed(322)
    x = small_integers(1, channels, *LARGE).to(memory_format=torch.channels_last_3d)
    weight = small_integers(3, 3, 3, channels)

    reference = F.conv3d(x.double(), weight.double().permute(3, 0, 1, 2)[:, None], padding=1, groups=channels)
    assert_exact(DWConv(x, weight), reference)
    torch.cuda.empty_cache()


# A weight gradient sums over every voxel, which sparse signs keep within the integers fp16 holds.
@pytest.mark.parametrize('channels', [8, 16, 32, 64, 128])
def test_dwconv_wgrad_large_exact(channels):
    torch.manual_seed(322)
    voxels = LARGE[0] * LARGE[1] * LARGE[2]
    x = sparse_signs(1, channels, *LARGE, sum_length=voxels).to(memory_format=torch.channels_last_3d)
    grad = sparse_signs(1, channels, *LARGE, sum_length=voxels).to(memory_format=torch.channels_last_3d)

    assert_exact(DWConvWGRAD(x, grad), dwconv_wgrad_reference(x.double(), grad.double()))
    torch.cuda.empty_cache()
