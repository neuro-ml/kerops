import torch

from ...kernels.conv_wgrad import _Conv_wgrad_cl3d_halo
from ...settings import ConfiguredFunction, TableKernelConfig, autotune
from ...utils import cdiv
from ..assets import ASSETS_ROOT
from .conv import channel_pairs, check_activation, generate_inputs_conv


def _problem_sizes(grad, x):
    return {'in_channels': x.shape[1], 'out_channels': grad.shape[1]}


conv3d_wgrad_config = TableKernelConfig(
    args_to_problem_sizes=_problem_sizes, toml_path=ASSETS_ROOT / 'Conv3dWgrad.toml'
)


def wgrad_shapes(grad, x):
    """(batch, H, W, D, in channels, out channels), after checking the layout Conv3dWgrad expects."""
    check_activation(x)
    check_activation(grad)
    assert grad.device == x.device and grad.shape[0] == x.shape[0] and grad.shape[2:] == x.shape[2:]
    bsize, in_channels, H, W, D = x.shape
    out_channels = grad.shape[1]
    assert H * W * D * max(in_channels, out_channels) < 2**31, 'offsets inside one sample are int32'
    return bsize, H, W, D, in_channels, out_channels


@ConfiguredFunction.configure(conv3d_wgrad_config)
def Conv3dWgrad(
    grad,
    x,
    /,
    *,
    P_BLOCK,
    Q_BLOCK,
    TH,
    TW,
    TD,
    GH,
    GW,
    WINOGRAD,
    NUM_STAGES,
    WARPS_M,
    WARPS_N,
    SWAP,
    PROGRAMS,
    P_VEC,
    Q_VEC,
):
    """Weight gradient [3, 3, 3, in, out] of the 3x3x3 padding-1 convolution of `x` with output gradient `grad`."""
    bsize, H, W, D, in_channels, out_channels = wgrad_shapes(grad, x)
    # P is read with an H/W halo; swapping x and grad gives the transposed gradient of the mirrored taps.
    p, q = (grad, x) if SWAP else (x, grad)
    weight_grad = torch.zeros(3, 3, 3, in_channels, out_channels, device=x.device, dtype=torch.float32)
    n_tiles = bsize * cdiv(H, TH) * cdiv(W, TW) * cdiv(D, TD)
    work = (3 // GH) * (3 // GW) * (p.shape[1] // P_BLOCK) * (q.shape[1] // Q_BLOCK)
    grid = (work * max(1, PROGRAMS // work),)
    _Conv_wgrad_cl3d_halo[grid](
        p,
        q,
        weight_grad,
        H,
        W,
        D,
        n_tiles,
        p.shape[1],
        q.shape[1],
        P_BLOCK,
        Q_BLOCK,
        TH,
        TW,
        TD,
        GH,
        GW,
        WINOGRAD,
        NUM_STAGES,
        WARPS_M,
        WARPS_N,
        SWAP,
        P_VEC,
        Q_VEC,
        num_warps=WARPS_M * WARPS_N,
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
    """Configs that fit registers and shared memory, compile in seconds and are of a kind that ever won."""
    in_channels, out_channels = problem_size['in_channels'], problem_size['out_channels']
    p_channels, q_channels = (out_channels, in_channels) if config['SWAP'] else (in_channels, out_channels)
    p_block, q_block, warps_m, warps_n = config['P_BLOCK'], config['Q_BLOCK'], config['WARPS_M'], config['WARPS_N']
    th, tw, winograd = config['TH'], config['TW'], config['WINOGRAD']
    rows, p_sets, q_sets = (config['TD'] // 2, 4, 2) if winograd else (config['TD'], 1, 4)
    sums = config['GH'] * config['GW'] * (4 if winograd else 3)
    p_planes = (th + config['GH'] - 1) * (tw + config['GW'] - 1)
    shared = 2 * config['NUM_STAGES'] * rows * (p_sets * p_planes * p_block + q_sets * th * tw * q_block)
    mma = sums * th * tw * rows // 16 * (p_block // (16 * warps_m)) * (q_block // (8 * warps_n))
    return (
        rows >= 16
        and p_channels // 2 <= p_block <= p_channels
        and q_channels // 2 <= q_block <= q_channels
        and 16 * warps_m <= p_block
        and 8 * warps_n <= q_block
        and 32 <= sums * p_block * q_block // (32 * warps_m * warps_n) <= 128  # fp32 sums, registers a thread
        and shared <= 48 * 1024 * config['NUM_STAGES']
        and 32 <= mma <= 1024  # mma a warp per tile: fewer do not pay for the loads, more unroll into minutes
        and not (config['SWAP'] and in_channels == out_channels)
        # Won only there:
        and (winograd or max(in_channels, out_channels) <= 16)
        and (config['PROGRAMS'] == 256 or warps_m * warps_n <= 2)
        and (config['P_VEC'] == 8 or p_block <= 32)
        and (config['Q_VEC'] == 8 or q_block <= 32)
    )


SEARCH_SPACE = {
    'P_BLOCK': [16, 32, 64, 128],
    'Q_BLOCK': [16, 32, 64, 128],
    'TH': [1, 2, 4],
    'TW': [2, 4],
    'TD': [16, 32, 64],
    'GH': [1, 3],
    'GW': [1, 3],
    'WINOGRAD': [False, True],
    'NUM_STAGES': [1, 2],
    'WARPS_M': [1, 2],
    'WARPS_N': [1, 2, 4],
    'SWAP': [False, True],
    'PROGRAMS': [256, 512],
    'P_VEC': [4, 8],
    'Q_VEC': [4, 8],
}


def autotune_conv_wgrad(toml_path, **autotune_kwargs):
    autotune(
        Conv3dWgrad.function,
        generate_inputs_conv_wgrad,
        channel_pairs(),
        pruning_rule,
        toml_path,
        comparator=cudnn_wgrad,
        rep_ms=25,
        **autotune_kwargs,
        **SEARCH_SPACE,
    )
