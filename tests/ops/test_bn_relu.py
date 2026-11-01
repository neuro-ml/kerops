import torch

from kerops.ops.bnrelu import ApplyBNReLU, ApplyBNReLUBackward
from kerops.utils import assert_close, assert_wgrad_close


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
