import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.linear import LinBReLULinAdd, LinBReLULinBackward, ReLULinearAdd, ReLULinearBackward
from kerops.utils import allclose_two_stage, weight_grad_similarity


# Error budget of the forward linear ops (fp16 intermediates), relative to the output std of an fp32 reference.
LINEAR_MAX_ERROR = 5e-3
LINEAR_MEAN_ERROR = 4e-4


def conv1x1(x, weight, bias=None):
    return F.conv3d(x, weight.permute(1, 0)[:, :, None, None, None], bias)


def assert_linear_within_budget(out, reference):
    error = (out.float() - reference).abs() / reference.std()
    message = f'max {error.max():.1e}, mean {error.mean():.1e}'
    assert error.max() <= LINEAR_MAX_ERROR and error.mean() <= LINEAR_MEAN_ERROR, message


def test_relu_linear_add(bsize, other_1, other_2, other_3, channels_out):
    torch.manual_seed(322)
    x = torch.randn(bsize, 2 * channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    other = torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = torch.empty(2 * channels_out, channels_out, device='cuda', dtype=torch.float16)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        reference = conv1x1(F.relu(x).float(), weight.float()) + other.float()
    assert_linear_within_budget(ReLULinearAdd(x, weight, other), reference)


def test_relu_linear_add_backward(bsize, other_1, other_2, other_3, channels_out):
    torch.manual_seed(322)
    x = torch.randn(bsize, 2 * channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    x.requires_grad_(True)

    weight_x = torch.empty(2 * channels_out, channels_out, device='cuda', dtype=torch.float32)
    nn.init.kaiming_uniform_(weight_x, a=math.sqrt(5))
    weight_x.requires_grad_(True)

    grad = torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    with torch.amp.autocast('cuda'):
        out_1 = F.conv3d(
            F.relu(x), weight_x.permute(1, 0)[:, :, None, None, None], None, stride=(1, 1, 1), padding=(0, 0, 0)
        )

    out_1.backward(grad)

    grad_x_check, grad_weight_x_check = ReLULinearBackward(x, grad, weight_x.to(torch.float16))
    grad_weight_x_check = grad_weight_x_check.to(torch.float32)

    assert allclose_two_stage(
        x.grad,
        grad_x_check,
        atol_strict=1e-4,
        rtol_strict=1e-4,
        rtol_narrow=1e-3,
        atol_narrow=1e-3,
        debug_info='print',
    )

    assert weight_grad_similarity(
        weight_x.grad,
        grad_weight_x_check,
        rtol_cos=1e-4,
        atol_cos=1e-4,
        rtol_len=1e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )


def test_linbrelulinadd(bsize, channels, other_1, other_2, other_3):
    if channels not in [16, 32, 64]:
        pytest.skip('tuned only for 16, 32 and 64 channels')

    torch.manual_seed(322)

    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    add_other = torch.randn_like(x)

    weight_up = torch.randn(channels, channels * 2, dtype=torch.float32, device='cuda')
    nn.init.kaiming_uniform_(weight_up, a=math.sqrt(5))

    weight_down = torch.randn(channels * 2, channels, dtype=torch.float32, device='cuda')
    nn.init.kaiming_uniform_(weight_down, a=math.sqrt(5))

    bias = torch.randn(2 * channels, dtype=torch.float32, device='cuda')

    weight_up, weight_down, bias = weight_up.half(), weight_down.half(), bias.half()
    out = LinBReLULinAdd(x, weight_up, weight_down, bias, add_other)

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        hidden = conv1x1(x.float(), weight_up.float(), bias.float()).relu()
        reference = conv1x1(hidden, weight_down.float()) + add_other.float()
    assert_linear_within_budget(out, reference)


def test_linbrelulin_backward(bsize, channels, other_1, other_2, other_3):
    if channels not in [16, 32]:
        pytest.skip('checked only for 16 and 32 channels')

    torch.manual_seed(322)

    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    grad = torch.randn_like(x)
    x.requires_grad_(True)

    weight_up = torch.randn(channels, channels * 2, dtype=torch.float32, device='cuda')
    nn.init.kaiming_uniform_(weight_up, a=math.sqrt(5))
    weight_up.requires_grad_(True)

    weight_down = torch.randn(channels * 2, channels, dtype=torch.float32, device='cuda')
    nn.init.kaiming_uniform_(weight_down, a=math.sqrt(5))
    weight_down.requires_grad_(True)

    bias = torch.randn(2 * channels, dtype=torch.float32, device='cuda')
    bias.requires_grad_(True)

    input_grad, weight_up_grad, weight_down_grad, bias_grad = LinBReLULinBackward(
        x,
        grad,
        weight_up.to(torch.float16),
        weight_down.to(torch.float16),
        bias.to(torch.float16),
    )

    weight_up_grad = weight_up_grad.to(torch.float32)
    weight_down_grad = weight_down_grad.to(torch.float32)
    bias_grad = bias_grad.to(torch.float32)

    with torch.amp.autocast('cuda'):
        x_ = F.conv3d(x, weight_up.permute(1, 0)[:, :, None, None, None], bias, stride=1, padding=0)
        x_ = F.relu(x_)
        x_ = F.conv3d(x_, weight_down.permute(1, 0)[:, :, None, None, None], None, stride=1, padding=0)

    x_.backward(grad)

    assert weight_grad_similarity(
        weight_down.grad,
        weight_down_grad,
        rtol_cos=1e-4,
        atol_cos=1e-4,
        rtol_len=1e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )

    assert weight_grad_similarity(
        weight_up.grad,
        weight_up_grad,
        rtol_cos=1e-4,
        atol_cos=1e-4,
        rtol_len=2e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )

    assert weight_grad_similarity(
        bias.grad,
        bias_grad,
        rtol_cos=1e-3,
        atol_cos=1e-3,
        rtol_len=2e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )

    allclose = allclose_two_stage(
        x.grad,
        input_grad,
        rtol_strict=1e-4,
        atol_strict=1e-4,
        rtol_narrow=1e-3,
        atol_narrow=1e-3,
        debug_info='print',
    )

    # Because of relu non-differentiability some voxels may significatnly differ
    # Check these voxels manually
    if not allclose:
        soft_map = torch.abs(x.grad - input_grad) < 1e-3 + 1e-3 * torch.abs(input_grad)
        not_soft_map_vox = torch.all((~soft_map), dim=1)

        b, h, w, d = torch.where(not_soft_map_vox)

        with torch.cuda.amp.autocast(), torch.inference_mode():
            for inp, grd, inp_grad in zip(x[b, :, h, w, d], grad[b, :, h, w, d], input_grad[b, :, h, w, d]):
                linup = inp @ weight_up + bias

                linup_relu_grad = weight_down @ grd
                linup_grad = linup_relu_grad * (linup > 0)
                base_grd = weight_up @ linup_grad

                assert torch.allclose(base_grd, inp_grad, atol=1e-3, rtol=1e-3)
