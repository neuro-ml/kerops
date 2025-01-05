import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.addition import AddStats, AddStatsBackward
from kerops.ops.avgpool import AvgPoolCeilStats, AvgPoolCeilStatsBackward
from kerops.ops.bnrelu import ApplyBNReLU, ApplyBNReLUBackward
from kerops.ops.conv import DWConv, DWConvWGRAD
from kerops.ops.linear import ReLULinearAdd, ReLULinearBackward
from kerops.ops.stats import Stats, StatsBackward
from kerops.utils import allclose_two_stage


@pytest.fixture(params=[2, 4, 8, 16, 32])
def channels(request):
    return request.param


@pytest.fixture(params=[16, 32])
def channels_out(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def bsize(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 7, 13, 16, 32, 37, 53, 111, 128])
def other_1(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 7, 13, 16, 32, 37, 53, 111, 128])
def other_2(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 7, 13, 16, 32, 37, 53, 111, 128])
def other_3(request):
    return request.param


def test_stats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    x_fp32 = x.to(torch.float32)
    dim = (0, 2, 3, 4)
    mean = x_fp32.mean(dim=dim)
    sqmean = (x_fp32**2).mean(dim=dim)
    mean_, sqmean_ = Stats(x)

    assert torch.allclose(mean, mean_, atol=1e-4, rtol=1e-4)
    assert torch.allclose(sqmean, sqmean_, atol=1e-4, rtol=1e-4)


def test_stats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    x.requires_grad_(True)

    upstr_sqmean = torch.randn(channels, device='cuda', dtype=torch.float32) / 3
    upstr_mean = torch.randn(channels, device='cuda', dtype=torch.float32) / 3

    x_fp32 = x.to(torch.float32)
    mean = x_fp32.mean(dim=(0, 2, 3, 4))
    sqmean = (x_fp32**2).mean(dim=(0, 2, 3, 4))

    loss = (sqmean * upstr_sqmean).sum() + (mean * upstr_mean).sum()
    loss.backward()
    grad = x.grad

    grad_to_check = StatsBackward(x, upstr_mean, upstr_sqmean)
    if x.numel() // channels >= 100:
        assert allclose_two_stage(
            grad, grad_to_check, rtol_strict=1e-5, atol_strict=1e-4, rtol_narrow=1e-4, atol_narrow=1e-3
        )
    else:
        assert torch.allclose(grad, grad_to_check, rtol=1e-5, atol=5e-4)


def test_addstats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    y = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    r = x + y
    r_fp32 = r.to(torch.float32)
    dim = (0, 2, 3, 4)
    mean = r_fp32.mean(dim=dim)
    sqmean = (r_fp32**2).mean(dim=dim)
    r_, mean_, sqmean_ = AddStats(x, y)

    assert torch.allclose(r, r_)
    assert torch.allclose(mean, mean_, atol=1e-4, rtol=1e-4)
    assert torch.allclose(sqmean, sqmean_, atol=1e-4, rtol=1e-4)


def test_addstats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    y = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    x.requires_grad_(True)
    y.requires_grad_(True)

    upstr_sqmean = torch.randn(channels, device='cuda', dtype=torch.float32) / 3
    upstr_mean = torch.randn(channels, device='cuda', dtype=torch.float32) / 3
    upstr_add = torch.randn_like(x, dtype=torch.float16, device='cuda') / 3

    z = x + y
    z_fp32 = z.to(torch.float32)
    mean = z_fp32.mean(dim=(0, 2, 3, 4))
    sqmean = (z_fp32**2).mean(dim=(0, 2, 3, 4))

    loss = (sqmean * upstr_sqmean).sum() + (mean * upstr_mean).sum() + (z_fp32 * upstr_add).sum()
    loss.backward()
    grad = x.grad

    grad_to_check = AddStatsBackward(upstr_add, upstr_mean, upstr_sqmean, z)
    if x.numel() // channels >= 100:
        assert allclose_two_stage(
            grad, grad_to_check, rtol_strict=1e-5, atol_strict=1e-4, rtol_narrow=1e-4, atol_narrow=1e-3
        )
    else:
        assert torch.allclose(grad, grad_to_check, rtol=1e-5, atol=5e-4)


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


def test_avgpool_ceil_stats(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x_def = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda')
    x = x_def.to(memory_format=torch.channels_last_3d)
    ap = nn.AvgPool3d(2, ceil_mode=True)

    res = ap(x_def)
    res_fp32 = res.to(torch.float32)
    mean = res_fp32.mean(dim=(0, 2, 3, 4))
    sqmean = (res_fp32**2).mean(dim=(0, 2, 3, 4))

    res_, mean_, sqmean_ = AvgPoolCeilStats(x)

    assert torch.allclose(mean, mean_, atol=1e-3, rtol=1e-5)
    assert torch.allclose(sqmean, sqmean_, atol=1e-3, rtol=1e-5)
    if res.numel() < 100:
        assert torch.allclose(res, res_, rtol=1e-5, atol=5e-5)
    else:
        assert allclose_two_stage(
            res, res_, rtol_strict=1e-5, atol_strict=5e-5, rtol_narrow=1e-4, atol_narrow=1e-3, precision=torch.float32
        )


def test_avgpool_ceil_stats_backward(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16)
    x.requires_grad_(True)
    o = nn.AvgPool3d(2, ceil_mode=True)(x)
    o_fp32 = o.to(torch.float32)
    mean = o_fp32.mean(dim=(0, 2, 3, 4))
    sqmean = (o_fp32**2).mean(dim=(0, 2, 3, 4))

    upstr_pool = torch.randn_like(o)
    upstr_mean = torch.randn_like(mean)
    upstr_sqmean = torch.randn_like(sqmean)

    loss = (upstr_pool * o).sum() + (upstr_mean * mean).sum() + (upstr_sqmean * sqmean).sum()
    loss.backward()
    grad = x.grad

    grad_check = AvgPoolCeilStatsBackward(
        upstr_pool.to(memory_format=torch.channels_last_3d),
        upstr_mean,
        upstr_sqmean,
        o.to(memory_format=torch.channels_last_3d),
        x.shape,
    )

    if grad.numel() < 100:
        assert torch.allclose(grad, grad_check, atol=2e-3, rtol=1e-4)
    else:
        if not allclose_two_stage(
            grad, grad_check, rtol_strict=1e-4, atol_strict=5e-4, rtol_narrow=1e-4, atol_narrow=2e-3
        ):
            assert allclose_two_stage(
                grad,
                grad_check,
                rtol_strict=1e-4,
                atol_strict=5e-4,
                rtol_narrow=1e-4,
                atol_narrow=2e-3,
                debug_info='print',
            )


def test_dwconv(bsize, channels, other_1, other_2, other_3):
    if channels >= 8:
        torch.manual_seed(322)
        x = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
            memory_format=torch.channels_last_3d
        )

        weight_standard = nn.Conv3d(channels, channels, 3, groups=channels).to('cuda').weight.to(torch.float16)
        weight_check = weight_standard[:, 0].permute(1, 2, 3, 0).contiguous()

        with torch.inference_mode():
            out_standard = F.conv3d(x, weight_standard, None, padding=(1, 1, 1), stride=(1, 1, 1), groups=channels)
            out_check = DWConv(x, weight_check)

        assert allclose_two_stage(
            out_standard,
            out_check,
            rtol_strict=1e-4,
            atol_strict=1e-3,
            rtol_narrow=1e-4,
            atol_narrow=2e-3,
            debug_info='print',
        )

        torch.cuda.empty_cache()


# TODO: how to sample to make weight's gradient adequate?
def test_dwconv_wgrad(bsize, channels, other_1, other_2, other_3):
    if channels >= 8 and other_1 < 53 and other_2 < 53 and other_3 < 53:
        torch.manual_seed(322)
        x = (
            torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
                memory_format=torch.channels_last_3d
            )
            / 5
        )
        grad = (
            torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
                memory_format=torch.channels_last_3d
            )
            / 5
        )
        weight_standard = torch.randn(channels, 1, 3, 3, 3, device='cuda', dtype=torch.float16)

        with torch.inference_mode():
            _, grad_w, _ = torch.ops.aten.convolution_backward(
                grad,
                x,
                weight_standard,
                [0],  # bias_sizes
                [1, 1, 1],  # stride
                [1, 1, 1],  # padding
                [1, 1, 1],  # dilation
                False,  # transposed
                [0, 0, 0],  # output padding
                channels,  # groups!
                [False, True, False],  # output_mask - grad_inpt, grad_weight, grad_bias
            )
            grad_w = grad_w[:, 0].permute(1, 2, 3, 0).contiguous()

            grad_w_check = DWConvWGRAD(x, grad)

        assert allclose_two_stage(
            grad_w,
            grad_w_check,
            rtol_strict=1e-2,
            atol_strict=bsize * 1e-2,
            rtol_narrow=1e-2,
            atol_narrow=bsize * 2e-2,
            debug_info='print',
        )

        torch.cuda.empty_cache()


def test_relu_linear_add(bsize, other_1, other_2, other_3, channels_out):
    torch.manual_seed(322)
    x = torch.randn(bsize, 32, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    other = torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = nn.Conv3d(32, channels_out, 1, bias=False).weight.data.to('cuda').to(torch.float16)

    with torch.inference_mode():
        out_1 = F.conv3d(F.relu(x), weight, None, stride=(1, 1, 1), padding=(0, 0, 0)) + other
        out_2 = ReLULinearAdd(x, weight[:, :, 0, 0, 0].permute(1, 0).contiguous(), other)

    assert allclose_two_stage(
        out_1, out_2, atol_strict=1e-4, rtol_strict=1e-4, rtol_narrow=1e-3, atol_narrow=1e-3, debug_info='print'
    )


def test_relu_linear_add_backward(bsize, other_1, other_2, other_3, channels_out):
    if other_1 < 53 and other_2 < 53 and other_3 < 53:
        torch.manual_seed(322)
        x = (
            torch.randn(bsize, 32, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
                memory_format=torch.channels_last_3d
            )
            / 5
        )
        weight_x = nn.Conv3d(32, channels_out, 1, bias=False).weight.data.to('cuda').to(torch.float16)
        x.requires_grad_(True)
        weight_x.requires_grad_(True)

        grad = (
            torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
                memory_format=torch.channels_last_3d
            )
            / 5
        )

        out_1 = F.conv3d(F.relu(x), weight_x, None, stride=(1, 1, 1), padding=(0, 0, 0))
        out_1.backward(grad)

        grad_x_check, grad_weight_x_check = ReLULinearBackward(
            x, grad, weight_x[:, :, 0, 0, 0].permute(1, 0).contiguous()
        )
        grad_weight_x_check = grad_weight_x_check.permute(1, 0).contiguous()[:, :, None, None, None]

        assert allclose_two_stage(
            x.grad,
            grad_x_check,
            atol_strict=1e-4,
            rtol_strict=1e-4,
            rtol_narrow=1e-3,
            atol_narrow=1e-3,
            debug_info='print',
        )
        assert allclose_two_stage(
            weight_x.grad,
            grad_weight_x_check,
            atol_strict=1e-2,
            rtol_strict=bsize * 1e-2,
            rtol_narrow=1e-2,
            atol_narrow=bsize * 2e-2,
            debug_info='print',
        )
