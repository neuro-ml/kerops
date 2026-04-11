import torch
from triton import next_power_of_2

from ..assets import ASSETS_ROOT
from ...kernels.linear import _LinBReLULinAdd
from ...settings import autotune, ConfArg, TableKernelConfig, ConfiguredFunction
from ...utils import cdiv


lin_bn_relu_lin_add_config = TableKernelConfig(
    problem_size_names=['in_channels'],
    confarg_names=['num_warps', 'D_BLOCK', 'ILP'],
    args_to_problem_sizes=lambda x: (x.shape[1], ),
    toml_path=ASSETS_ROOT / 'LinBReLULinAdd.toml'
)


@ConfiguredFunction.configure(lin_bn_relu_lin_add_config)
def LinBReLULinAdd(
    x,
    weight_up,
    weight_down,
    bias,
    add_other,
    *,
    num_warps: ConfArg,
    D_BLOCK: ConfArg,
    ILP: ConfArg,
):
    in_channels = x.shape[1]
    hidden_channels = 2 * in_channels
    numel = x.numel()

    assert x.ndim == add_other.ndim == 5
    assert list(x.shape) == list(add_other.shape)
    assert in_channels == next_power_of_2(in_channels)
    assert list(weight_up.shape) == [in_channels, hidden_channels]
    assert list(weight_down.shape) == [hidden_channels, in_channels]
    assert list(bias.shape) == [hidden_channels]
    assert x.dtype == weight_up.dtype == weight_down.dtype == bias.dtype == add_other.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert add_other.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // in_channels
    grid_size = cdiv(numel_no_channels, D_BLOCK * ILP)

    output = torch.empty_like(x)

    _LinBReLULinAdd[(grid_size,)](
        x,
        weight_up,
        weight_down,
        bias,
        add_other,
        output,
        numel_no_channels,
        in_channels,
        hidden_channels,
        D_BLOCK,
        ILP,
        num_warps=num_warps,
    )

    return output


def pruning_rule(problem_size, named_config):
    D_BLOCK = named_config['D_BLOCK']

    in_channels = problem_size['in_channels']

    if in_channels >= 32 and D_BLOCK > 32:
        return False

    return True


def generate_inputs_lin_bn_relu_lin_add(problem_sizes):
    in_channels = problem_sizes['in_channels']

    if in_channels <= 32:
        base = 128
    elif in_channels <= 64:
        base = 96
    else:
        base = 64

    x = torch.randn(1, in_channels, base, base, base, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last_3d)
    weight_up = torch.randn(in_channels, 2 * in_channels, device='cuda', dtype=torch.float16)
    weight_down = torch.randn(2 * in_channels, in_channels, device='cuda', dtype=torch.float16)
    bias = torch.randn(2 * in_channels, device='cuda', dtype=torch.float16)
    add_other = torch.randn(1, in_channels, base, base, base, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last_3d)

    return x, weight_up, weight_down, bias, add_other


def autotune_lin_bn_relu_lin_add(toml_path, **autotune_kwargs):
    channels = [16, 32, 64]
    problem_sizes = [{'in_channels': cin} for cin in channels]

    autotune(
        getattr(LinBReLULinAdd, 'function', LinBReLULinAdd),
        generate_inputs_lin_bn_relu_lin_add,
        problem_sizes,
        pruning_rule,
        toml_path,
        **autotune_kwargs,
        num_warps=[1, 2, 4],
        D_BLOCK=[16, 32, 64],
        ILP=[1, 2, 3, 4]
    )
