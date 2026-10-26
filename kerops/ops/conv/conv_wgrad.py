import torch

from ..assets import ASSETS_ROOT
from ...kernels.conv_wgrad import _Conv_wgrad_cl3d_halo
from ...settings import autotune, TableKernelConfig, ConfiguredFunction
from ...utils import cdiv
from .conv import channel_pairs, check_activation, generate_inputs_conv


def _problem_sizes(grad, x):
    return {'in_channels': x.shape[1], 'out_channels': grad.shape[1]}


conv3d_wgrad_config = TableKernelConfig(
    args_to_problem_sizes=_problem_sizes,
    toml_path=ASSETS_ROOT / 'Conv3dWgrad.toml'
)


@ConfiguredFunction.configure(conv3d_wgrad_config)
def Conv3dWgrad(
    grad, x, /, *, P_BLOCK, Q_BLOCK, TH, TW, TD, GH, GW, GD, NUM_STAGES, WARPS_M, WARPS_N, SWAP, PROGRAMS,
):
    """Weight gradient [3, 3, 3, in, out] of the 3x3x3 convolution, padding 1, of `x` whose output gradient is `grad`.
    P is the operand read with an H/W halo: x, or grad when SWAP. Programs: about PROGRAMS, each for GH x GW x GD taps,
    a P_BLOCK x Q_BLOCK block of channels and a share of the TH x TW x TD tiles."""
    check_activation(x)
    check_activation(grad)
    assert grad.device == x.device and grad.shape[0] == x.shape[0] and grad.shape[2:] == x.shape[2:]
    bsize, in_channels, H, W, D = x.shape
    assert H * W * D * max(in_channels, grad.shape[1]) < 2**31, 'offsets inside one sample are int32'

    p, q = (grad, x) if SWAP else (x, grad)
    weight_grad = torch.zeros(3, 3, 3, in_channels, grad.shape[1], device=x.device, dtype=torch.float32)
    n_tiles = bsize * cdiv(H, TH) * cdiv(W, TW) * cdiv(D, TD)
    work = (3 // GH) * (3 // GW) * (3 // GD) * (p.shape[1] // P_BLOCK) * (q.shape[1] // Q_BLOCK)
    grid = (work * max(1, PROGRAMS // work),)
    _Conv_wgrad_cl3d_halo[grid](
        p, q, weight_grad, H, W, D, n_tiles, p.shape[1], q.shape[1], P_BLOCK, Q_BLOCK, TH, TW, TD, GH, GW, GD,
        NUM_STAGES, WARPS_M, WARPS_N, SWAP, num_warps=WARPS_M * WARPS_N,
    )
    return weight_grad.to(torch.float16)


def generate_inputs_conv_wgrad(problem_sizes):
    x, _ = generate_inputs_conv(problem_sizes)
    grad = torch.randn(1, problem_sizes['out_channels'], *x.shape[2:], device=x.device, dtype=x.dtype)
    return grad.to(memory_format=torch.channels_last_3d), x


def cudnn_wgrad(grad, x):
    """Weight gradient by cuDNN in the [3, 3, 3, in, out] layout of Conv3dWgrad."""
    weight = torch.empty(grad.shape[1], x.shape[1], 3, 3, 3, device=x.device, dtype=x.dtype)
    _, weight_grad, _ = torch.ops.aten.convolution_backward(
        grad, x, weight, [0], [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1, [False, True, False]
    )
    return weight_grad.permute(2, 3, 4, 1, 0)


def pruning_rule(problem_size, config):
    """Channel blocks of at least half the channels, mma warps tiling the block, fp32 accumulators of 32 to 128
    registers a thread, shared memory for two programs on an SM, and 96 to 1024 mma a warp per tile: fewer do not
    pay for the loads of a tile, more unroll into minutes of compilation. SWAP only where it changes which operand
    has the halo."""
    in_channels, out_channels = problem_size['in_channels'], problem_size['out_channels']
    p_channels, q_channels = (out_channels, in_channels) if config['SWAP'] else (in_channels, out_channels)
    p_block, q_block, warps_m, warps_n = config['P_BLOCK'], config['Q_BLOCK'], config['WARPS_M'], config['WARPS_N']
    taps = config['GH'] * config['GW'] * config['GD']
    p_planes = (config['TH'] + config['GH'] - 1) * (config['TW'] + config['GW'] - 1)
    q_planes = config['GD'] * config['TH'] * config['TW']
    shared = 2 * config['NUM_STAGES'] * config['TD'] * (p_planes * p_block + q_planes * q_block)
    mma = taps * config['TH'] * config['TW'] * config['TD'] // 16 * (p_block // (16 * warps_m)) * (q_block // (8 * warps_n))
    return (
        p_channels // 2 <= p_block <= p_channels and q_channels // 2 <= q_block <= q_channels
        and 16 * warps_m <= p_block and 8 * warps_n <= q_block
        and 32 <= taps * p_block * q_block // (32 * warps_m * warps_n) <= 128
        and shared <= 48 * 1024
        and 96 <= mma <= 1024
        and not (config['SWAP'] and in_channels == out_channels)
    )


SEARCH_SPACE = dict(
    P_BLOCK=[16, 32, 64, 128], Q_BLOCK=[16, 32, 64, 128], TH=[1, 2, 4], TW=[2, 4], TD=[16, 32], GH=[1, 3], GW=[1, 3],
    GD=[1, 3], NUM_STAGES=[1], WARPS_M=[1, 2], WARPS_N=[1, 2, 4], SWAP=[False, True], PROGRAMS=[256],
)


def autotune_conv_wgrad(toml_path, **autotune_kwargs):
    autotune(
        Conv3dWgrad.function, generate_inputs_conv_wgrad, channel_pairs(), pruning_rule, toml_path,
        comparator=cudnn_wgrad, rep_ms=50, **autotune_kwargs, **SEARCH_SPACE,
    )
