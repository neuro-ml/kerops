import torch
from triton import next_power_of_2

from ..assets import ASSETS_ROOT
from ...kernels.linear import _ReLULinearBackward
from ...settings import autotune, ConfArg, TableKernelConfig, ConfiguredFunction
from ...utils import cdiv


relu_lin_backward_config = TableKernelConfig(
    problem_size_names=['in_channels'],
    confarg_names=['num_warps', 'D_BLOCK', 'ILP'],
    args_to_problem_sizes=lambda x: (x.shape[1], ),
    toml_path=ASSETS_ROOT / 'ReLULinBackward.toml'
)


@ConfiguredFunction.configure(relu_lin_backward_config)
def ReLULinearBackward(
    x,
    grad,
    weight,
    *,
    num_warps: ConfArg,
    D_BLOCK: ConfArg,
    ILP: ConfArg,
):
    in_channels = x.shape[1]
    out_channels = grad.shape[1]
    numel = grad.numel()

    assert grad.ndim == x.ndim == 5
    assert list(grad.shape[2:]) == list(x.shape[2:])
    assert grad.shape[0] == x.shape[0]
    assert in_channels == next_power_of_2(in_channels)
    assert out_channels == next_power_of_2(out_channels)
    assert in_channels == 2 * out_channels
    assert list(weight.shape) == [in_channels, out_channels]
    assert x.dtype == grad.dtype == weight.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // out_channels

    grid_size = cdiv(numel_no_channels, D_BLOCK * ILP)

    bsize, _, H, W, D = grad.shape
    x_grad = torch.empty_like(x)
    weight_grad = torch.zeros([grid_size, in_channels, out_channels], dtype=torch.float16, device='cuda')

    _ReLULinearBackward[(grid_size,)](
        x,
        grad,
        x_grad,
        weight,
        weight_grad,
        numel_no_channels,
        in_channels,
        out_channels,
        D_BLOCK,
        ILP,
        num_warps=num_warps,
    )

    return x_grad, weight_grad.sum(dim=0)


def generate_inputs_relu_lin_backward(problem_sizes):
    in_channels = problem_sizes['in_channels']

    if in_channels <= 32:
        base = 128
    elif in_channels <= 64:
        base = 96
    else:
        base = 64

    x = torch.randn(1, in_channels, base, base, base, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last_3d)
    grad = torch.randn(1, in_channels // 2, base, base, base, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last_3d)
    weight = torch.randn(in_channels, in_channels // 2, device='cuda', dtype=torch.float16)

    return x, grad, weight


def autotune_relu_lin_backward(toml_path, **autotune_kwargs):
    channels = [32, 64, 128]
    problem_sizes = [{'in_channels': cin} for cin in channels]

    autotune(
        getattr(ReLULinearBackward, 'function', ReLULinearBackward),
        generate_inputs_relu_lin_backward,
        problem_sizes,
        pruning_rule=None,
        toml_path=toml_path,
        **autotune_kwargs,
        num_warps=[1, 2, 4, 8],
        D_BLOCK=[16, 32, 64],
        ILP=[1, 2, 3, 4, 8, 16]
    )
