import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.conv import Conv3d, Conv3dWgrad
from kerops.ops.conv.conv import channel_pairs
from kerops.ops.conv.conv_wgrad import cudnn_wgrad, generate_inputs_conv_wgrad
from kerops.utils import wgrad_errors

# Error budget of Conv3d (fp16 partial sums), relative to the output std of an fp32 reference.
CONV_MAX_ERROR = 5e-3
CONV_MEAN_ERROR = 4e-4

# Error budget of Conv3dWgrad against an fp32 reference: 1 - cosine similarity and the relative difference of norms.
WGRAD_COS_ERROR = 2e-4
WGRAD_NORM_ERROR = 1e-3


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

    error = (out_check.float() - out_standard).abs() / out_standard.std()
    assert (
        error.max() <= CONV_MAX_ERROR and error.mean() <= CONV_MEAN_ERROR
    ), f'max {error.max():.1e}, mean {error.mean():.1e}'

    torch.cuda.empty_cache()


def assert_wgrad_within_budget(grad, x):
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        reference = cudnn_wgrad(grad.float(), x.float())
    cos_error, norm_error = wgrad_errors(Conv3dWgrad(grad, x), reference)
    assert (
        cos_error <= WGRAD_COS_ERROR and norm_error <= WGRAD_NORM_ERROR
    ), f'cos {cos_error:.1e}, norm {norm_error:.1e}'


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
    assert_wgrad_within_budget(grad, x)
    torch.cuda.empty_cache()


# The bench sizes, 128³ to 256³ voxels: the error of long accumulation chains shows only there.
@pytest.mark.parametrize('pair', channel_pairs(), ids=lambda pair: f'{pair["in_channels"]}-{pair["out_channels"]}')
def test_conv_wgrad_bench_sizes(pair):
    torch.manual_seed(1337)
    grad, x = generate_inputs_conv_wgrad(pair)
    assert_wgrad_within_budget(grad, x.relu())
    torch.cuda.empty_cache()
