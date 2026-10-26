from functools import cache

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


@cache
def winograd_output(device):
    """Gᵀ of Winograd F(3, 2): the 4 positions the kernel sums along D into the 3 kd taps. Its last column carries the
    sign of the last row of A, which the kernel leaves out."""
    return torch.tensor([[1, 0.5, 0.5, 0], [0, 0.5, -0.5, 0], [0, 0.5, 0.5, -1]], device=device)


@ConfiguredFunction.configure(conv3d_wgrad_config)
def Conv3dWgrad(
    grad, x, /, *, P_BLOCK, Q_BLOCK, TH, TW, TD, GH, GW, NUM_STAGES, WARPS_M, WARPS_N, SWAP, PROGRAMS, P_VEC, Q_VEC,
):
    """Weight gradient [3, 3, 3, in, out] of the 3x3x3 convolution, padding 1, of `x` whose output gradient is `grad`.
    P is the operand read with an H/W halo: x, or grad when SWAP, which gives the transposed gradient of the mirrored
    taps. Programs: about PROGRAMS, each for GH x GW (kh, kw) taps, a P_BLOCK x Q_BLOCK block of channels and a share
    of the TH x TW x TD tiles. P_VEC, Q_VEC: fp16 per cp.async of P and Q; 4 (8 bytes) goes through L1, 8 bypasses it."""
    check_activation(x)
    check_activation(grad)
    assert grad.device == x.device and grad.shape[0] == x.shape[0] and grad.shape[2:] == x.shape[2:]
    bsize, in_channels, H, W, D = x.shape
    assert H * W * D * max(in_channels, grad.shape[1]) < 2**31, 'offsets inside one sample are int32'

    p, q = (grad, x) if SWAP else (x, grad)
    winograd_sums = torch.zeros(3, 3, 4, p.shape[1], q.shape[1], device=x.device, dtype=torch.float32)
    n_tiles = bsize * cdiv(H, TH) * cdiv(W, TW) * cdiv(D, TD)
    work = (3 // GH) * (3 // GW) * (p.shape[1] // P_BLOCK) * (q.shape[1] // Q_BLOCK)
    grid = (work * max(1, PROGRAMS // work),)
    _Conv_wgrad_cl3d_halo[grid](
        p, q, winograd_sums, H, W, D, n_tiles, p.shape[1], q.shape[1], P_BLOCK, Q_BLOCK, TH, TW, TD, GH, GW,
        NUM_STAGES, WARPS_M, WARPS_N, P_VEC, Q_VEC, num_warps=WARPS_M * WARPS_N,
    )
    weight_grad = torch.einsum('ka,hwapq->hwkpq', winograd_output(x.device), winograd_sums)
    if SWAP:
        weight_grad = weight_grad.flip(0, 1, 2).transpose(3, 4)
    return weight_grad.to(torch.float16, memory_format=torch.contiguous_format)


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
    registers a thread, at most 48 KB of shared memory a stage, and 32 to 1024 mma a warp per tile: fewer do not
    pay for the loads of a tile, more unroll into minutes of compilation. SWAP only where it changes which operand
    has the halo, 8-byte copies only for blocks of at most 32 channels, the only ones where they won."""
    in_channels, out_channels = problem_size['in_channels'], problem_size['out_channels']
    p_channels, q_channels = (out_channels, in_channels) if config['SWAP'] else (in_channels, out_channels)
    p_block, q_block, warps_m, warps_n = config['P_BLOCK'], config['Q_BLOCK'], config['WARPS_M'], config['WARPS_N']
    th, tw, td = config['TH'], config['TW'], config['TD']
    sums = config['GH'] * config['GW'] * 4
    p_planes = (th + config['GH'] - 1) * (tw + config['GW'] - 1)
    shared = config['NUM_STAGES'] * td * (4 * p_planes * p_block + 2 * th * tw * q_block)
    mma = sums * th * tw * td // 32 * (p_block // (16 * warps_m)) * (q_block // (8 * warps_n))
    return (
        p_channels // 2 <= p_block <= p_channels and q_channels // 2 <= q_block <= q_channels
        and 16 * warps_m <= p_block and 8 * warps_n <= q_block
        and 32 <= sums * p_block * q_block // (32 * warps_m * warps_n) <= 128
        and shared <= 48 * 1024 * config['NUM_STAGES']
        and 32 <= mma <= 1024
        and not (config['SWAP'] and in_channels == out_channels)
        and (config['P_VEC'] == 8 or p_block <= 32) and (config['Q_VEC'] == 8 or q_block <= 32)
    )


SEARCH_SPACE = dict(
    P_BLOCK=[16, 32, 64, 128], Q_BLOCK=[16, 32, 64, 128], TH=[1, 2, 4], TW=[2, 4], TD=[32, 64], GH=[1, 3], GW=[1, 3],
    NUM_STAGES=[1, 2], WARPS_M=[1, 2], WARPS_N=[1, 2, 4], SWAP=[False, True], PROGRAMS=[256], P_VEC=[4, 8], Q_VEC=[4, 8],
)


def autotune_conv_wgrad(toml_path, **autotune_kwargs):
    autotune(
        Conv3dWgrad.function, generate_inputs_conv_wgrad, channel_pairs(), pruning_rule, toml_path,
        comparator=cudnn_wgrad, rep_ms=50, **autotune_kwargs, **SEARCH_SPACE,
    )
