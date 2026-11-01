import torch

from kerops.ops.bnrelu import ApplyBNReLU, ApplyBNReLUBackward
from kerops.utils import assert_close, assert_exact, assert_wgrad_close, small_integers


def test_apply_bn_relu(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = torch.randn(channels, device='cuda') + 0.5
    bias = torch.randn(channels, device='cuda') * 2

    out = ApplyBNReLU(x, weight, bias)

    reference = torch.relu(x.float() * weight.view(1, -1, 1, 1, 1) + bias.view(1, -1, 1, 1, 1))
    assert_close(out, reference, max_error=1e-5, mean_error=1e-7)


def test_apply_bn_relu_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = torch.randn(1, channels, 1, 1, 1, device='cuda') + 0.5
    bias = torch.randn(1, channels, 1, 1, 1, device='cuda') * 2
    out_grad = torch.randn_like(x) / 10

    grad, weight_grad, bias_grad = ApplyBNReLUBackward(x, weight, bias, out_grad)

    x_fp32, weight, bias = x.float().requires_grad_(), weight.requires_grad_(), bias.requires_grad_()
    torch.relu(x_fp32 * weight + bias).backward(out_grad.float())
    assert_close(grad, x_fp32.grad, max_error=1e-5, mean_error=1e-7)
    assert_wgrad_close(weight_grad, weight.grad, cos_error=1e-10, norm_error=5e-5)
    assert_wgrad_close(bias_grad, bias.grad, cos_error=1e-10, norm_error=1e-5)


def test_apply_bn_relu_exact(channels, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    weight, bias = small_integers(channels, dtype=torch.float32), small_integers(channels, dtype=torch.float32)

    reference = torch.relu(x.double() * weight.view(1, -1, 1, 1, 1) + bias.view(1, -1, 1, 1, 1))
    assert_exact(ApplyBNReLU(x, weight, bias), reference)


def test_apply_bn_relu_backward_exact(channels, exact_shape):
    bsize, *sides = exact_shape
    torch.manual_seed(322)
    x = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    out_grad = small_integers(bsize, channels, *sides).to(memory_format=torch.channels_last_3d)
    weight = small_integers(1, channels, 1, 1, 1, dtype=torch.float32)
    bias = small_integers(1, channels, 1, 1, 1, dtype=torch.float32)

    grad, weight_grad, bias_grad = ApplyBNReLUBackward(x, weight, bias, out_grad)

    x_fp64, weight_fp64, bias_fp64 = (tensor.double().requires_grad_() for tensor in (x, weight, bias))
    torch.relu(x_fp64 * weight_fp64 + bias_fp64).backward(out_grad.double())
    assert_exact(grad, x_fp64.grad)
    assert_exact(weight_grad, weight_fp64.grad)
    assert_exact(bias_grad, bias_fp64.grad)
