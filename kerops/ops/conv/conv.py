import itertools

import torch
from torch.nn import functional as F
from triton import next_power_of_2

from ...kernels.conv_fprop import _Conv_cl3d_tiled_halo
from ...settings import ConfiguredFunction, TableKernelConfig, autotune
from ...utils import cdiv
from ..assets import ASSETS_ROOT


def check_activation(x):
    """Asserts the activation layout the kernels expect: channels_last_3d fp16 on CUDA, power-of-2 channels."""
    assert x.is_cuda and x.ndim == 5 and x.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert x.shape[1] == next_power_of_2(x.shape[1])


def conv_shapes(x, weight):
    """(batch, H, W, D, in channels, out channels), after checking the layout Conv3d expects."""
    check_activation(x)
    assert x.device == weight.device and weight.ndim == 5
    bsize, in_channels, H, W, D = x.shape
    out_channels = weight.shape[-1]
    assert list(weight.shape) == [3, 3, 3, in_channels, out_channels]
    assert out_channels == next_power_of_2(out_channels)
    assert weight.is_contiguous() and weight.dtype == torch.float16
    assert H * W * D * max(in_channels, out_channels) < 2**31, 'offsets inside one sample are int32'
    return bsize, H, W, D, in_channels, out_channels


def channels_last_output(x, out_channels):
    bsize, _, H, W, D = x.shape
    return torch.empty([bsize, H, W, D, out_channels], device=x.device, dtype=x.dtype).permute(0, 4, 1, 2, 3)


def _problem_sizes(x, weight):
    return {'in_channels': weight.shape[-2], 'out_channels': weight.shape[-1]}


conv3d_config = TableKernelConfig(args_to_problem_sizes=_problem_sizes, toml_path=ASSETS_ROOT / 'Conv3d.toml')


@ConfiguredFunction.configure(conv3d_config)
def Conv3d(
    x,
    weight,
    /,
    *,
    BLOCK_K,
    TH,
    TW,
    TD,
    NUM_STAGES,
    WARPS_M,
    WARPS_N,
    POSITION_MAJOR,
    HALO_VEC,
    WEIGHT_VEC,
):
    """3x3x3 convolution, padding 1, of channels_last_3d fp16 `x` with `weight` [3, 3, 3, in, out]."""
    bsize, H, W, D, in_channels, out_channels = conv_shapes(x, weight)
    output = channels_last_output(x, out_channels)
    grid = (cdiv(W, TW) * cdiv(H, TH) * cdiv(D, TD) * bsize,)
    _Conv_cl3d_tiled_halo[grid](
        x,
        weight,
        output,
        H,
        W,
        D,
        in_channels,
        out_channels,
        BLOCK_K,
        TH,
        TW,
        TD,
        NUM_STAGES,
        WARPS_M,
        WARPS_N,
        POSITION_MAJOR,
        HALO_VEC,
        WEIGHT_VEC,
        num_warps=WARPS_M * WARPS_N,
    )
    return output


def generate_inputs_conv(problem_sizes, device='cuda'):
    in_channels, out_channels = problem_sizes['in_channels'], problem_sizes['out_channels']
    channels = max(in_channels, out_channels)
    size = 256 if channels <= 32 else 192 if channels <= 64 else 128

    x = torch.randn(1, in_channels, size, size, size, device=device, dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )
    w = torch.randn(3, 3, 3, in_channels, out_channels, device=device, dtype=torch.float16) / (27 * in_channels) ** 0.5

    return x, w


def cudnn_conv(x, weight):
    return F.conv3d(x, weight.permute(4, 3, 0, 1, 2), padding=1)


# Error budget (max 5e-3, mean 4e-4 of the output std, see test_conv): an fp16 partial may span at most 64 products
# (4 mma).
# The search space keeps what won on the RTX 4090: one stage and BLOCK_K 16.


def _fits_registers_and_warps(problem_size, positions, rows, warps_m, warps_n):
    """mma warps tile `rows` x out channels; fp32 accumulators of all positions take at most 128 registers a thread."""
    out_channels = problem_size['out_channels']
    accumulators = positions * rows * out_channels // (32 * warps_m * warps_n)
    return 16 * warps_m <= rows and 8 * warps_n <= out_channels and accumulators <= 128


def pruning_rule_tiled_halo(problem_size, config):
    planes = (config['TH'] + 2) * (config['TW'] + 2)
    shared = 2 * config['NUM_STAGES'] * config['BLOCK_K'] * (planes * config['TD'] + 9 * problem_size['out_channels'])
    return (
        config['BLOCK_K'] <= problem_size['in_channels']
        and 3 * config['BLOCK_K'] <= 64
        and shared <= 99 * 1024
        and _fits_registers_and_warps(
            problem_size, config['TH'] * config['TW'], config['TD'], config['WARPS_M'], config['WARPS_N']
        )
    )


def channel_pairs():
    channels = [2**i for i in range(4, 8)]
    return [
        {'in_channels': cin, 'out_channels': cout}
        for cin, cout in itertools.product(channels, channels)
        if (cin == 2 * cout) or (cin * 2 == cout) or (cin == cout)
    ]


def autotune_conv(toml_path, **autotune_kwargs):
    autotune(
        Conv3d.function,
        generate_inputs_conv,
        channel_pairs(),
        pruning_rule_tiled_halo,
        toml_path,
        comparator=cudnn_conv,
        **autotune_kwargs,
        BLOCK_K=[16],
        TH=[2, 4, 8],
        TW=[2, 4, 8],
        TD=[16, 32, 64],
        NUM_STAGES=[1],
        WARPS_M=[1, 2, 4],
        WARPS_N=[1, 2, 4],
        POSITION_MAJOR=[False, True],
        HALO_VEC=[4, 8],
        WEIGHT_VEC=[4, 8],
    )
