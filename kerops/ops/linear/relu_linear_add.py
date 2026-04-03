import torch
from triton import next_power_of_2

from ..assets import ASSETS_ROOT
from ...kernels.linear import _ReLULinearAdd
from ...settings import autotune, ConfArg, TableKernelConfig, ConfiguredFunction
from ...utils import cdiv


relu_lin_add_config = TableKernelConfig(
    problem_size_names=['in_channels'],
    confarg_names=['num_warps', 'D_BLOCK', 'ILP'],
    args_to_problem_sizes=lambda x: (x.shape[1], ),
    toml_path=ASSETS_ROOT / 'ReLULinAdd.toml'
)


@ConfiguredFunction.configure(relu_lin_add_config)
def ReLULinearAdd(
    x,
    weight,
    add_other,
    *,
    num_warps: ConfArg,
    D_BLOCK: ConfArg,
    ILP: ConfArg,
):
    in_channels = x.shape[1]
    out_channels = add_other.shape[1]
    numel = x.numel()

    assert x.ndim == add_other.ndim == 5
    assert list(x.shape[2:]) == list(add_other.shape[2:])
    assert x.shape[0] == add_other.shape[0]
    assert in_channels == next_power_of_2(in_channels)
    assert out_channels == next_power_of_2(out_channels)
    assert in_channels == out_channels * 2
    assert list(weight.shape) == [in_channels, out_channels], ([in_channels, out_channels], weight.shape)
    assert x.dtype == weight.dtype == add_other.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert add_other.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // in_channels
    grid_size = cdiv(numel_no_channels, D_BLOCK * ILP)

    output = torch.empty_like(add_other)

    _ReLULinearAdd[(grid_size,)](
        x, weight, add_other, output, numel_no_channels, in_channels, out_channels, D_BLOCK, ILP, num_warps=num_warps
    )

    return output


def generate_inputs_relu_lin_add(problem_sizes):
    in_channels = problem_sizes['in_channels']

    if in_channels <= 32:
        base = 128
    elif in_channels <= 64:
        base = 96
    else:
        base = 64

    x = torch.randn(1, in_channels, base, base, base, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last_3d)
    weight = torch.randn(in_channels, in_channels // 2, device='cuda', dtype=torch.float16)
    add_other = torch.randn(1, in_channels // 2, base, base, base, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last_3d)

    return x, weight, add_other


def autotune_relu_lin_add(toml_path, **autotune_kwargs):
    channels = [16, 32, 64, 128]
    problem_sizes = [{'in_channels': cin} for cin in channels]

    autotune(
        getattr(ReLULinearAdd, 'function', ReLULinearAdd),
        generate_inputs_relu_lin_add,
        problem_sizes,
        pruning_rule=None,
        toml_path=toml_path,
        **autotune_kwargs,
        num_warps=[1, 2, 4],
        D_BLOCK=[16, 32, 64],
        ILP=[1, 2, 3, 4, 8, 16]
    )
