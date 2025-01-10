import torch
from torch import nn

from kerops.ops.addition import AddStats
from kerops.ops.bnrelu import ApplyBNReLU, ApplyBNReLUBackward
from kerops.utils import allclose_two_stage


def test_apply_bn_relu(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x_def = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda')
    x = x_def.to(memory_format=torch.channels_last_3d)
    y_def = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda')
    y = y_def.to(memory_format=torch.channels_last_3d)
    weight = torch.randn(channels, device='cuda') + 0.5
    bias = torch.randn(channels, device='cuda') * 2

    bn = nn.BatchNorm3d(channels, device='cuda')
    bn.weight = nn.Parameter(weight)
    bn.bias = nn.Parameter(bias)
    z = x_def + y_def

    with torch.cuda.amp.autocast():
        out = nn.functional.relu(bn(z))

        sum, mean, sqmean = AddStats(x, y)
        rvareps = torch.rsqrt(sqmean - mean**2 + 1e-5)
        W = weight * rvareps
        B = bias - mean * rvareps * weight
        out_check = ApplyBNReLU(sum, W, B)
    assert allclose_two_stage(
        out, out_check, rtol_strict=1e-5, atol_strict=1e-4, percentile_strict=0.995, rtol_narrow=1e-3, atol_narrow=1e-3
    )


def test_apply_bn_relu_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = torch.randn(1, channels, 1, 1, 1, device='cuda', dtype=torch.float32) + 0.5
    bias = torch.randn(1, channels, 1, 1, 1, device='cuda', dtype=torch.float32) * 2

    x.requires_grad_(True)
    weight.requires_grad_(True)
    bias.requires_grad_(True)

    upstream_grad = torch.randn_like(x, dtype=torch.float16) / 10

    x_fp32 = x.to(torch.float32)
    y_fp32 = nn.functional.relu(x_fp32 * weight + bias)

    y_fp32.backward(upstream_grad)

    outgrad, weight_grad, bias_grad = x.grad, weight.grad, bias.grad

    outgrad_check, weight_grad_check, bias_grad_check = ApplyBNReLUBackward(x, weight, bias, upstream_grad)
    assert torch.allclose(outgrad, outgrad_check, rtol=1e-5, atol=1e-4)
    assert torch.allclose(weight_grad, weight_grad_check, rtol=1e-4, atol=1e-3)
    assert torch.allclose(bias_grad, bias_grad_check, rtol=1e-4, atol=5e-4)
