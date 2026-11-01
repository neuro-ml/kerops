import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.conv import Conv3d, Conv3dWgrad
from kerops.ops.conv.conv import channel_pairs
from kerops.ops.conv.conv_wgrad import cudnn_wgrad, generate_inputs_conv_wgrad
from kerops.utils import assert_close, assert_exact, assert_wgrad_close, small_integers, sparse_signs

# Sides of the large exact tests: past 2M voxels the tolerance check of a weight gradient no longer sees one voxel lost.
LARGE = (128, 128, 128)


def pair_id(pair):
    return f'{pair["in_channels"]}-{pair["out_channels"]}'


# TODO: small shapes causes recompilations for some reasons regardless of configuration
def test_conv(bsize, conv_in_channels, conv_out_channels, other_1, other_2, other_3):
    if not (
        conv_in_channels == conv_out_channels
        or conv_in_channels * 2 == conv_out_channels
        or conv_in_channels == 2 * conv_out_channels
    ):
        pytest.skip('tuned only for cin == cout, 2 * cin == cout and cin == 2 * cout')

    torch.manual_seed(322)
    x = (
        torch.randn(bsize, conv_in_channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16)
        .relu()
        .to(memory_format=torch.channels_last_3d)
    )

    weight = torch.empty(conv_out_channels, conv_in_channels, 3, 3, 3, device='cuda', dtype=torch.float16)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    weight_kernel = weight.permute(2, 3, 4, 1, 0).contiguous()

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        out_standard = F.conv3d(x.float(), weight.float(), None, padding=(1, 1, 1), stride=(1, 1, 1))
    out_check = Conv3d(x, weight_kernel)

    assert_close(out_check, out_standard, max_error=5e-3, mean_error=2e-4)

    torch.cuda.empty_cache()


def assert_conv_wgrad_close(grad, x):
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        reference = cudnn_wgrad(grad.float(), x.float())
    assert_wgrad_close(Conv3dWgrad(grad, x), reference, cos_error=1e-6, norm_error=2e-4)


def test_conv_wgrad(bsize, conv_in_channels, conv_out_channels, other_1, other_2, other_3):
    if not (
        conv_in_channels == conv_out_channels
        or conv_in_channels * 2 == conv_out_channels
        or conv_in_channels == 2 * conv_out_channels
    ):
        pytest.skip('tuned only for cin == cout, 2 * cin == cout and cin == 2 * cout')

    torch.manual_seed(1337)
    x = (
        torch.randn(bsize, conv_in_channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16)
        .relu()
        .to(memory_format=torch.channels_last_3d)
    )
    grad = torch.randn(bsize, conv_out_channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )
    assert_conv_wgrad_close(grad, x)
    torch.cuda.empty_cache()


# The bench sizes, 128³ to 256³ voxels: the error of long accumulation chains shows only there.
@pytest.mark.parametrize('pair', channel_pairs(), ids=pair_id)
def test_conv_wgrad_bench_sizes(pair):
    torch.manual_seed(1337)
    grad, x = generate_inputs_conv_wgrad(pair)
    assert_conv_wgrad_close(grad, x.relu())
    torch.cuda.empty_cache()


@pytest.mark.parametrize('pair', channel_pairs(), ids=pair_id)
def test_conv_exact(pair, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x = small_integers(bsize, pair['in_channels'], *sides).to(memory_format=torch.channels_last_3d)
    weight = small_integers(3, 3, 3, pair['in_channels'], pair['out_channels'])

    reference = F.conv3d(x.double(), weight.double().permute(4, 3, 0, 1, 2), padding=1)
    assert_exact(Conv3d(x, weight), reference)


@pytest.mark.parametrize('pair', channel_pairs(), ids=pair_id)
def test_conv_wgrad_exact(pair, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x = small_integers(bsize, pair['in_channels'], *sides).to(memory_format=torch.channels_last_3d)
    grad = small_integers(bsize, pair['out_channels'], *sides).to(memory_format=torch.channels_last_3d)

    assert_exact(Conv3dWgrad(grad, x), cudnn_wgrad(grad.double(), x.double()))


# The bench sizes: an output sums 27 * in_channels products whatever the size, so dense small integers stay exact.
@pytest.mark.parametrize('pair', channel_pairs(), ids=pair_id)
def test_conv_large_exact(pair):
    torch.manual_seed(322)
    x = small_integers(1, pair['in_channels'], *LARGE).to(memory_format=torch.channels_last_3d)
    weight = small_integers(3, 3, 3, pair['in_channels'], pair['out_channels'])

    reference = F.conv3d(x.double(), weight.double().permute(4, 3, 0, 1, 2), padding=1)
    assert_exact(Conv3d(x, weight), reference)
    torch.cuda.empty_cache()


# A weight gradient sums over every voxel, which sparse signs keep within the integers fp16 holds.
@pytest.mark.parametrize('pair', channel_pairs(), ids=pair_id)
def test_conv_wgrad_large_exact(pair):
    torch.manual_seed(322)
    voxels = LARGE[0] * LARGE[1] * LARGE[2]
    x = sparse_signs(1, pair['in_channels'], *LARGE, sum_length=voxels).to(memory_format=torch.channels_last_3d)
    grad = sparse_signs(1, pair['out_channels'], *LARGE, sum_length=voxels).to(memory_format=torch.channels_last_3d)

    assert_exact(Conv3dWgrad(grad, x), cudnn_wgrad(grad.double(), x.double()))
    torch.cuda.empty_cache()
