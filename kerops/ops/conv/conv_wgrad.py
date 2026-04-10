import torch
from triton import language as tl, next_power_of_2

from ..assets import ASSETS_ROOT
from ...kernels.conv import _Conv_wgrad_cl3d_impl_V2, _Conv_wgrad_cl3d_splitKonH_impl
from ...settings import autotune, ConfArg, TableKernelConfig, ConfiguredFunction
from ...utils import cdiv


conv3d_wgrad_config = TableKernelConfig(
    problem_size_names=['in_channels', 'out_channels'],
    confarg_names=['num_warps', 'D_BLOCK', 'REDUCTION_FACTOR', 'CIN_BLOCK', 'COUT_BLOCK'],
    args_to_problem_sizes=lambda grad, x: (x.shape[1], grad.shape[1]),
    toml_path=ASSETS_ROOT / 'Conv3dWgrad.toml'
)


@ConfiguredFunction.configure(conv3d_wgrad_config)
def Conv3dWgrad(
    grad,
    x,
    *,
    num_warps: ConfArg,
    D_BLOCK: ConfArg,
    REDUCTION_FACTOR: ConfArg,
    CIN_BLOCK: ConfArg,
    COUT_BLOCK: ConfArg,
):
    assert x.device == grad.device
    assert x.is_cuda

    assert x.ndim == grad.ndim == 5
    xbsize, in_channels, xH, xW, xD = x.shape
    gbsize, out_channels, gH, gW, gD = grad.shape
    assert in_channels == next_power_of_2(in_channels)
    assert out_channels == next_power_of_2(out_channels)
    assert [xbsize, xH, xW, xD] == [gbsize, gH, gW, gD]

    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)

    assert x.dtype == grad.dtype == torch.float16

    assert D_BLOCK == next_power_of_2(D_BLOCK)
    assert CIN_BLOCK == next_power_of_2(CIN_BLOCK)
    assert CIN_BLOCK <= in_channels
    assert COUT_BLOCK == next_power_of_2(COUT_BLOCK)
    assert COUT_BLOCK <= out_channels
    assert isinstance(REDUCTION_FACTOR, int)

    ACCTYPE = tl.float32
    num_buffers = cdiv(xW, REDUCTION_FACTOR * 2) * cdiv(xH, REDUCTION_FACTOR * 2) * cdiv(xD, D_BLOCK * REDUCTION_FACTOR) * xbsize
    weight_grad = torch.zeros([num_buffers, 3, 3, 3, in_channels, out_channels], device=x.device, dtype=torch.float32)
    grid = (cdiv(xW, 2) * cdiv(out_channels, COUT_BLOCK), cdiv(xH, 2), cdiv(xD, D_BLOCK) * xbsize)

    _Conv_wgrad_cl3d_impl_V2[grid](
        grad,
        x,
        weight_grad,
        xH,
        xW,
        xD,
        num_buffers,
        IN_CHANNELS=in_channels,
        OUT_CHANNELS=out_channels,
        ACCTYPE=ACCTYPE,
        D_BLOCK=D_BLOCK,
        CIN_BLOCK=CIN_BLOCK,
        COUT_BLOCK=COUT_BLOCK,
        num_warps=num_warps,
    )
    
    weight_grad = weight_grad.sum(dim=0).to(torch.float16)

    return weight_grad


conv3d_wgrad_splitk_config = TableKernelConfig(
    problem_size_names=['in_channels', 'out_channels'],
    confarg_names=['num_warps', 'H_BLOCK', 'WD_BLOCK', 'CIN_BLOCK', 'COUT_BLOCK', 'SPLIT_K'],
    args_to_problem_sizes=lambda grad, x: (x.shape[1], grad.shape[1]),
    toml_path=ASSETS_ROOT / 'Conv3dWgradSplitk.toml'
)


@ConfiguredFunction.configure(conv3d_wgrad_splitk_config)
def Conv3dWgrad_splitKonH(
    grad,
    x,
    *,
    num_warps: ConfArg,
    H_BLOCK: ConfArg,
    WD_BLOCK: ConfArg,
    CIN_BLOCK: ConfArg, 
    COUT_BLOCK: ConfArg,
    SPLIT_K: ConfArg,
):
    assert x.device == grad.device
    assert x.is_cuda

    assert x.ndim == grad.ndim == 5
    xbsize, in_channels, xH, xW, xD = x.shape
    gbsize, out_channels, gH, gW, gD = grad.shape
    assert in_channels == next_power_of_2(in_channels)
    assert out_channels == next_power_of_2(out_channels)
    assert [xbsize, xH, xW, xD] == [gbsize, gH, gW, gD]

    assert xbsize == gbsize == 1

    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)

    assert x.dtype == grad.dtype == torch.float16

    assert H_BLOCK == next_power_of_2(H_BLOCK)
    assert WD_BLOCK == next_power_of_2(WD_BLOCK)
    assert CIN_BLOCK == next_power_of_2(CIN_BLOCK)
    assert CIN_BLOCK <= in_channels
    assert COUT_BLOCK == next_power_of_2(COUT_BLOCK)
    assert COUT_BLOCK <= out_channels

    ACCTYPE = tl.float32
    weight_grad = torch.zeros([3, 3, 3, in_channels, out_channels], device=x.device, dtype=torch.float32)

    grid = (
        27 * SPLIT_K,
        cdiv(in_channels, CIN_BLOCK),
        cdiv(out_channels, COUT_BLOCK)
    )

    _Conv_wgrad_cl3d_splitKonH_impl[grid](
        grad,
        x,
        weight_grad,
        xH, xW, xD,
        ACCTYPE=ACCTYPE,
        H_BLOCK=H_BLOCK, W_BLOCK=WD_BLOCK, D_BLOCK=WD_BLOCK,
        IN_CHANNELS=in_channels, OUT_CHANNELS=out_channels,
        CIN_BLOCK=CIN_BLOCK, COUT_BLOCK=COUT_BLOCK,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
    )
    
    weight_grad = weight_grad.to(torch.float16)

    return weight_grad


def pruning_rule(problem_size, named_config):
    D_BLOCK = named_config['D_BLOCK']
    CIN_BLOCK = named_config['CIN_BLOCK']
    COUT_BLOCK = named_config['COUT_BLOCK']

    in_channels, out_channels = problem_size['in_channels'], problem_size['out_channels']

    if in_channels >= 32 and out_channels >= 32 and D_BLOCK > 32:
        return False

    if (in_channels >= 128 or out_channels >= 128) and D_BLOCK > 16:
        return False

    if CIN_BLOCK > in_channels or COUT_BLOCK > out_channels:
        return False

    return True


def pruning_rule_splitKonH(problem_size, named_config):
    H_BLOCK = named_config['H_BLOCK']
    WD_BLOCK = named_config['WD_BLOCK']
    CIN_BLOCK = named_config['CIN_BLOCK']
    COUT_BLOCK = named_config['COUT_BLOCK']

    in_channels, out_channels = problem_size['in_channels'], problem_size['out_channels']

    if in_channels < CIN_BLOCK or out_channels < COUT_BLOCK:
        return False

    if in_channels // CIN_BLOCK > 4 or out_channels // COUT_BLOCK > 4:
        return False

    if H_BLOCK * WD_BLOCK * WD_BLOCK < 16:
        return False

    if H_BLOCK * WD_BLOCK * WD_BLOCK > 64:
        return False

    return True


def generate_inputs_conv_wgrad(problem_sizes):
    in_channels, out_channels = problem_sizes['in_channels'], problem_sizes['out_channels']

    if in_channels <= 32 and out_channels <= 32:
        base = 128
    elif in_channels <= 64 and out_channels <= 64:
        base = 96
    else:
        base = 64

    x = torch.randn(1, in_channels, base, base, base, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last_3d)
    grad = torch.randn(1, out_channels, base, base, base, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last_3d)

    return grad, x


def comparator(grad, x):
    cout = grad.shape[1]
    cin = x.shape[1]
    dummy_weight = torch.empty(cout, cin, 3, 3, 3, device='cuda', dtype=torch.float16)

    torch.ops.aten.convolution_backward(
        grad,
        x,
        dummy_weight,
        [0],  # bias_sizes
        [1, 1, 1],  # stride
        [1, 1, 1],  # padding
        [1, 1, 1],  # dilation
        False,  # transposed
        [0, 0, 0],  # output padding
        1,  # groups
        [False, True, False],  # output_mask - grad_inpt, grad_weight, grad_bias
    )


def autotune_conv_wgrad(toml_path, **autotune_kwargs):
    channels = [2 ** i for i in range(4, 8)]
    problem_sizes = [
        {'in_channels': cin, 'out_channels': cout}
        for cin in channels
        for cout in channels
        if (cin == 2 * cout) or (cin * 2 == cout) or (cin == cout)
    ]

    autotune(
        getattr(Conv3dWgrad, 'function', Conv3dWgrad),
        generate_inputs_conv_wgrad,
        problem_sizes,
        pruning_rule,
        toml_path,
        comparator=comparator,
        **autotune_kwargs,
        num_warps=[1, 2, 4],
        D_BLOCK=[16, 32],
        REDUCTION_FACTOR=[32],
        CIN_BLOCK=channels,
        COUT_BLOCK=channels,
    )


def autotune_conv_wgrad_splitKonH(toml_path, **autotune_kwargs):
    channels = [2 ** i for i in range(4, 8)]
    problem_sizes = [
        {'in_channels': cin, 'out_channels': cout}
        for cin in channels
        for cout in channels
        if (cin == 2 * cout) or (cin * 2 == cout) or (cin == cout)
    ]

    autotune(
        getattr(Conv3dWgrad_splitKonH, 'function', Conv3dWgrad_splitKonH),
        generate_inputs_conv_wgrad,
        problem_sizes,
        pruning_rule_splitKonH,
        toml_path,
        comparator=comparator,
        **autotune_kwargs,
        num_warps=[1, 2, 4],
        H_BLOCK=[2, 4, 8, 16],
        WD_BLOCK=[2, 4, 8, 16],
        CIN_BLOCK=channels,
        COUT_BLOCK=channels,
        SPLIT_K=[4, 8, 16],
        n_iters=85
    )
