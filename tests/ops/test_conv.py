import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.conv import Conv3d, Conv3dWgrad
from kerops.utils import weight_grad_similarity


# Error budget of Conv3d (fp16 partial sums), relative to the output std of an fp32 reference.
CONV_MAX_ERROR = 5e-3
CONV_MEAN_ERROR = 4e-4


# TODO: small shapes causes recompilations for some reasons regardless of configuration
def test_conv(bsize, conv_in_channels, conv_out_channels, other_1, other_2, other_3):
    if not (conv_in_channels == conv_out_channels or conv_in_channels * 2 == conv_out_channels or conv_in_channels == 2 * conv_out_channels):
        pytest.skip('tuned only for cin == cout, 2 * cin == cout and cin == 2 * cout')
    
    torch.manual_seed(322)
    x = torch.randn(bsize, conv_in_channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).relu().to(
        memory_format=torch.channels_last_3d
    )

    weight = torch.empty(conv_out_channels, conv_in_channels, 3, 3, 3, device='cuda', dtype=torch.float16)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    weight_kernel = weight.permute(2, 3, 4, 1, 0).contiguous()

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        out_standard = F.conv3d(x.float(), weight.float(), None, padding=(1, 1, 1), stride=(1, 1, 1))
    out_check = Conv3d(x, weight_kernel)

    error = (out_check.float() - out_standard).abs() / out_standard.std()
    assert error.max() <= CONV_MAX_ERROR and error.mean() <= CONV_MEAN_ERROR, f'max {error.max():.1e}, mean {error.mean():.1e}'

    torch.cuda.empty_cache()


def test_conv_dwgrad(bsize, conv_in_channels, conv_out_channels, other_1, other_2, other_3):
    if not (conv_in_channels == conv_out_channels or conv_in_channels * 2 == conv_out_channels or conv_in_channels == 2 * conv_out_channels):
        pytest.skip('tuned only for cin == cout, 2 * cin == cout and cin == 2 * cout')
    
    torch.manual_seed(1337)
    x = torch.randn(bsize, conv_in_channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )
    grad = torch.randn(bsize, conv_out_channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )

    weight = torch.empty(conv_out_channels, conv_in_channels, 3, 3, 3, device='cuda', dtype=torch.float16)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    _, wgrad_standard, _ = torch.ops.aten.convolution_backward(
        grad,
        x,
        weight,
        [0],  # bias_sizes
        [1, 1, 1],  # stride
        [1, 1, 1],  # padding
        [1, 1, 1],  # dilation
        False,  # transposed
        [0, 0, 0],  # output padding
        1,  # groups!
        [False, True, False],  # output_mask - grad_inpt, grad_weight, grad_bias
    )

    wgrad_check = Conv3dWgrad(grad, x)
    wgrad_check = wgrad_check.permute(-1, -2, 0, 1, 2)

    assert weight_grad_similarity(wgrad_standard.float(), wgrad_check.float())
    torch.cuda.empty_cache()
