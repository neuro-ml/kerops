import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.linear import LinBReLULinAdd, LinBReLULinBackward, ReLULinearAdd, ReLULinearBackward
from kerops.utils import assert_close, assert_exact, assert_wgrad_close, small_integers


def conv1x1(x, weight, bias=None):
    return F.conv3d(x, weight.permute(1, 0)[:, :, None, None, None], bias)


def kaiming(rows, columns):
    weight = torch.empty(rows, columns, device='cuda')
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    return weight.half()


def test_relu_linear_add(bsize, other_1, other_2, other_3, channels_out):
    torch.manual_seed(322)
    x = torch.randn(bsize, 2 * channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    other = torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = kaiming(2 * channels_out, channels_out)

    out = ReLULinearAdd(x, weight, other)

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        reference = conv1x1(F.relu(x).float(), weight.float()) + other.float()
    assert_close(out, reference, max_error=2e-3, mean_error=2e-5)


def test_relu_linear_add_backward(bsize, other_1, other_2, other_3, channels_out):
    torch.manual_seed(322)
    x = torch.randn(bsize, 2 * channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    grad = torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = kaiming(2 * channels_out, channels_out)

    x_grad, weight_grad = ReLULinearBackward(x, grad, weight)

    x_fp32, weight_fp32 = x.float().requires_grad_(), weight.float().requires_grad_()
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        conv1x1(F.relu(x_fp32), weight_fp32).backward(grad.float())
    assert_close(x_grad, x_fp32.grad, max_error=1e-5, mean_error=1e-7)
    assert_wgrad_close(weight_grad, weight_fp32.grad, cos_error=2e-7, norm_error=2e-4)


def test_linbrelulinadd(bsize, channels, other_1, other_2, other_3):
    if channels not in [16, 32, 64]:
        pytest.skip('tuned only for 16, 32 and 64 channels')

    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    other = torch.randn_like(x)
    weight_up, weight_down = kaiming(channels, 2 * channels), kaiming(2 * channels, channels)
    bias = torch.randn(2 * channels, dtype=torch.float16, device='cuda')

    out = LinBReLULinAdd(x, weight_up, weight_down, bias, other)

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        hidden = conv1x1(x.float(), weight_up.float(), bias.float()).relu()
        reference = conv1x1(hidden, weight_down.float()) + other.float()
    assert_close(out, reference, max_error=3e-3, mean_error=5e-5)


def test_linbrelulin_backward(bsize, channels, other_1, other_2, other_3):
    if channels not in [16, 32]:
        pytest.skip('checked only for 16 and 32 channels')

    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    grad = torch.randn_like(x)
    weight_up, weight_down = kaiming(channels, 2 * channels), kaiming(2 * channels, channels)
    bias = torch.randn(2 * channels, dtype=torch.float16, device='cuda')

    x_grad, weight_up_grad, weight_down_grad, bias_grad = LinBReLULinBackward(x, grad, weight_up, weight_down, bias)

    x_fp32, weight_up_fp32, weight_down_fp32, bias_fp32 = (
        tensor.float().requires_grad_() for tensor in (x, weight_up, weight_down, bias)
    )
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        hidden = conv1x1(x_fp32, weight_up_fp32, bias_fp32)
        # The op rounds the hidden layer to fp16 before the ReLU, as torch under autocast does: the reference takes
        # those values, and so the same ReLU mask, and keeps the fp32 gradients.
        rounded = (conv1x1(x.float(), weight_up.float()).half() + bias.view(1, -1, 1, 1, 1)).float()
        hidden = hidden + (rounded - hidden).detach()
        conv1x1(hidden.relu(), weight_down_fp32).backward(grad.float())
    # A hidden value within fp16 rounding of zero may still get the other sign in the op, which sums in another order:
    # x_grad is compared at the voxels whose hidden values are all clear of zero.
    clear = (rounded.abs() > 4e-3).all(dim=1)
    channels_last = (0, 2, 3, 4, 1)
    assert_close(
        x_grad.permute(channels_last)[clear], x_fp32.grad.permute(channels_last)[clear], max_error=5e-3, mean_error=5e-5
    )
    assert_wgrad_close(weight_up_grad, weight_up_fp32.grad, cos_error=2e-6, norm_error=3e-4)
    assert_wgrad_close(weight_down_grad, weight_down_fp32.grad, cos_error=2e-7, norm_error=5e-4)
    assert_wgrad_close(bias_grad, bias_fp32.grad, cos_error=2e-6, norm_error=2e-3)


def channels_last_integers(bsize, channels, sides, magnitude=2):
    return small_integers(bsize, channels, *sides, magnitude=magnitude).to(memory_format=torch.channels_last_3d)


def test_relu_linear_add_exact(channels_out, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x, other = channels_last_integers(bsize, 2 * channels_out, sides), channels_last_integers(
        bsize, channels_out, sides
    )
    weight = small_integers(2 * channels_out, channels_out)

    reference = conv1x1(F.relu(x.double()), weight.double()) + other.double()
    assert_exact(ReLULinearAdd(x, weight, other), reference)


def test_relu_linear_add_backward_exact(channels_out, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x, grad = channels_last_integers(bsize, 2 * channels_out, sides), channels_last_integers(bsize, channels_out, sides)
    weight = small_integers(2 * channels_out, channels_out)

    x_grad, weight_grad = ReLULinearBackward(x, grad, weight)

    x_fp64, weight_fp64 = x.double().requires_grad_(), weight.double().requires_grad_()
    conv1x1(F.relu(x_fp64), weight_fp64).backward(grad.double())
    assert_exact(x_grad, x_fp64.grad)
    assert_exact(weight_grad, weight_fp64.grad)


@pytest.mark.parametrize('channels', [16, 32, 64])
def test_linbrelulinadd_exact(channels, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x, other = channels_last_integers(bsize, channels, sides), channels_last_integers(bsize, channels, sides)
    weight_up, weight_down = small_integers(channels, 2 * channels), small_integers(2 * channels, channels)
    bias = small_integers(2 * channels)

    hidden = conv1x1(x.double(), weight_up.double(), bias.double()).relu()
    reference = conv1x1(hidden, weight_down.double()) + other.double()
    assert_exact(LinBReLULinAdd(x, weight_up, weight_down, bias, other), reference)


# Integers of magnitude 1: the op sums the weight gradients of its programs in fp16, which holds integers up to 2048.
@pytest.mark.parametrize('channels', [16, 32])
def test_linbrelulin_backward_exact(channels, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x, grad = channels_last_integers(bsize, channels, sides, 1), channels_last_integers(bsize, channels, sides, 1)
    weight_up, weight_down = small_integers(channels, 2 * channels, magnitude=1), small_integers(
        2 * channels, channels, magnitude=1
    )
    bias = small_integers(2 * channels, magnitude=1)

    x_grad, weight_up_grad, weight_down_grad, bias_grad = LinBReLULinBackward(x, grad, weight_up, weight_down, bias)

    x_fp64, weight_up_fp64, weight_down_fp64, bias_fp64 = (
        tensor.double().requires_grad_() for tensor in (x, weight_up, weight_down, bias)
    )
    conv1x1(conv1x1(x_fp64, weight_up_fp64, bias_fp64).relu(), weight_down_fp64).backward(grad.double())
    assert_exact(x_grad, x_fp64.grad)
    assert_exact(weight_up_grad, weight_up_fp64.grad)
    assert_exact(weight_down_grad, weight_down_fp64.grad)
    assert_exact(bias_grad, bias_fp64.grad)
