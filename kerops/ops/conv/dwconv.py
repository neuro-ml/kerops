import torch
from triton import language as tl, next_power_of_2

from ..assets import ASSETS_ROOT
from ...kernels.dw_conv import _DWConv_cl3d_impl
from ...settings import autotune, ConfArg, TableKernelConfig, ConfiguredFunction
from ...utils import cdiv


dwconv_config = TableKernelConfig(
    problem_size_names=['channels'],
    confarg_names=['num_warps', 'D_BLOCK'],
    args_to_problem_sizes=lambda weight: (weight.shape[-1], ),
    toml_path=ASSETS_ROOT / 'DWConv.toml'
)


@ConfiguredFunction.configure(dwconv_config)
def DWConv(x, weight, *, num_warps: ConfArg, D_BLOCK: ConfArg):
    channels = x.shape[1]

    assert x.ndim == 5
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert x.dtype == weight.dtype == torch.float16
    assert channels == next_power_of_2(channels)
    assert list(weight.shape) == [3, 3, 3, channels]
    assert D_BLOCK == next_power_of_2(D_BLOCK)

    ACCTYPE = tl.float32
    bsize, _, H, W, D = x.shape
    batch_stride, _, H_stride, W_stride, _ = x.stride()

    output = torch.empty_like(x)

    H_grid = cdiv(H, 2)
    W_grid = cdiv(W, 2)
    D_grid = cdiv(D, D_BLOCK)
    grid = (H_grid, W_grid, D_grid)

    for unbatched_x, unbatched_y in zip(x, output):
        _DWConv_cl3d_impl[grid](
            unbatched_x,
            weight,
            unbatched_y,
            H,
            W,
            D,
            H_stride,
            W_stride,
            ACCTYPE,
            channels,
            D_BLOCK,
            num_warps=num_warps,
        )

    return output


def generate_inputs_dwconv(problem_sizes, device='cuda'):
    channels = problem_sizes['channels']

    if channels <= 32:
        base = 256
    elif channels <= 64:
        base = 192
    else:
        base = 128

    x = torch.randn(1, channels, base, base, base, device=device, dtype=torch.float16).to(memory_format=torch.channels_last_3d)
    w = torch.randn(3, 3, 3, channels, device=device, dtype=torch.float16)

    return x, w


def pruning_rule(problem_size, named_config):
    D_BLOCK = named_config['D_BLOCK']

    channels = problem_size['channels']

    if channels >= 32 and D_BLOCK > 32:
        return False

    if channels >= 128 and D_BLOCK > 16:
        return False

    return True


def autotune_dwconv(toml_path, **autotune_kwargs):
    problem_sizes = [{'channels': channels} for channels in [2 ** i for i in range(3, 8)]]

    autotune(
        getattr(DWConv, 'function', DWConv),
        generate_inputs_dwconv,
        problem_sizes,
        pruning_rule,
        toml_path,
        **autotune_kwargs,
        num_warps=[1, 2, 4],
        D_BLOCK=[8, 16, 32, 64],
    )
