import torch
from triton import language as tl, next_power_of_2

from ..assets import ASSETS_ROOT
from ...kernels.conv_fused import _ApplyBNReLUConvStats_cl3d_impl
from ...settings import autotune, TableKernelConfig, ConfiguredFunction
from ...utils import cdiv
from .conv import generate_inputs_conv


bnreluconv3d_config = TableKernelConfig(
    args_to_problem_sizes=lambda x, bn_weight, bn_bias, weight: {
        'in_channels': weight.shape[-2],
        'out_channels': weight.shape[-1],
    },
    toml_path=ASSETS_ROOT / 'BNReLUConv3d.toml'
)


@ConfiguredFunction.configure(bnreluconv3d_config)
def ApplyBNReLUConv3dStats(x, bn_weight, bn_bias, weight, /, *, num_warps, D_BLOCK, CIN_BLOCK):
    assert x.device == weight.device == bn_weight.device == bn_bias.device
    assert x.is_cuda

    assert x.ndim == 5
    assert weight.ndim == 5
    assert bn_weight.ndim == bn_bias.ndim == 1
    bsize, x_channels, H, W, D = x.shape
    in_channels, out_channels = weight.shape[-2:]
    assert x_channels == next_power_of_2(x_channels)
    assert list(weight.shape) == [3, 3, 3, in_channels, out_channels]  # 3, 3, 3, in_channels, out_channels
    assert in_channels == x_channels == bn_weight.numel() == bn_bias.numel()
    assert out_channels == next_power_of_2(out_channels)

    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert weight.is_contiguous()

    assert x.dtype == weight.dtype == torch.float16
    assert bn_weight.dtype == bn_bias.dtype == torch.float32
    
    assert D_BLOCK == next_power_of_2(D_BLOCK)
    assert CIN_BLOCK == next_power_of_2(CIN_BLOCK)
    assert CIN_BLOCK <= in_channels

    ACCTYPE = tl.float32
    output = torch.empty([bsize, H, W, D, out_channels], device=x.device, dtype=x.dtype, layout=x.layout).permute(0, -1, 1, 2, 3)
    grid = (cdiv(W, 2), cdiv(H, 2), cdiv(D, D_BLOCK))
    grid_vol = grid[0] * grid[1] * grid[2]
    mean = torch.zeros([bsize, grid_vol, out_channels], device=x.device, dtype=torch.float32)
    sqmean = torch.zeros([bsize, grid_vol, out_channels], device=x.device, dtype=torch.float32)

    numel_no_channels = bsize * H * W * D

    for unbatched_x, unbatched_y, unbatched_mean, unbatched_sqmean in zip(x, output, mean, sqmean):
        _ApplyBNReLUConvStats_cl3d_impl[grid](
            unbatched_x,
            bn_weight,
            bn_bias,
            weight,
            unbatched_y,
            unbatched_mean,
            unbatched_sqmean,
            H,
            W,
            D,
            D_BLOCK=D_BLOCK,
            ACCTYPE=ACCTYPE,
            IN_CHANNELS=in_channels,
            OUT_CHANNELS=out_channels,
            CIN_BLOCK=CIN_BLOCK,
            num_warps=num_warps,
        )

    return output, mean.sum(dim=(0, 1)) / numel_no_channels, sqmean.sum(dim=(0, 1)) / numel_no_channels


def generate_inputs_bnreluconv(problem_sizes, device='cuda'):
    x, w = generate_inputs_conv(problem_sizes, device)

    in_channels = problem_sizes['in_channels']
    bn_weight = torch.randn(in_channels, device=device, dtype=torch.float32)
    bn_bias = torch.randn(in_channels, device=device, dtype=torch.float32)

    return x, bn_weight, bn_bias, w


def pruning_rule(problem_size, named_config):
    D_BLOCK = named_config['D_BLOCK']
    CIN_BLOCK = named_config['CIN_BLOCK']

    in_channels, out_channels = problem_size['in_channels'], problem_size['out_channels']

    if in_channels >= 32 and out_channels >= 32 and D_BLOCK > 32:
        return False

    if (in_channels >= 128 or out_channels >= 128) and D_BLOCK > 16:
        return False

    if CIN_BLOCK > in_channels:
        return False

    return True


def autotune_bnreluconv(toml_path, **autotune_kwargs):
    channels = [2 ** i for i in range(4, 8)]
    problem_sizes = [
        {'in_channels': cin, 'out_channels': cout}
        for cin in channels
        for cout in channels
        if (cin == 2 * cout) or (cin * 2 == cout) or (cin == cout)
    ]

    autotune(
        getattr(ApplyBNReLUConv3dStats, 'function', ApplyBNReLUConv3dStats),
        generate_inputs_bnreluconv,
        problem_sizes,
        pruning_rule,
        toml_path,
        **autotune_kwargs,
        num_warps=[2, 4],
        D_BLOCK=[16, 32, 64],
        CIN_BLOCK=[16, 32, 64],
    )
