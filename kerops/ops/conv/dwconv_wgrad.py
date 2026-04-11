import torch
from triton import language as tl, next_power_of_2

from ..assets import ASSETS_ROOT
from ...kernels.dw_conv import _DWConv_wgrad_cl3d_impl
from ...settings import autotune, ConfArg, TableKernelConfig, ConfiguredFunction
from ...utils import cdiv


dwconv_wgrad_config = TableKernelConfig(
    problem_size_names=['channels'],
    confarg_names=['num_warps', 'D_BLOCK', 'ILP'],
    args_to_problem_sizes=lambda x: (x.shape[1], ),
    toml_path=ASSETS_ROOT / 'DWConvWGRAD.toml'
)


@ConfiguredFunction.configure(dwconv_wgrad_config)
def DWConvWGRAD(
    x, grad, *, num_warps: ConfArg, D_BLOCK: ConfArg, ILP: ConfArg
):
    channels = x.shape[1]

    assert x.ndim == grad.ndim == 5
    assert x.shape == grad.shape
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)
    assert x.dtype == grad.dtype == torch.float16
    assert channels == next_power_of_2(channels)
    assert D_BLOCK == next_power_of_2(D_BLOCK)

    ACCTYPE =tl.float32
    bsize, _, H, W, D = x.shape
    batch_stride, _, H_stride, W_stride, _ = x.stride()

    H_grid = cdiv(H, 2 * ILP)
    W_grid = cdiv(W, 2)
    D_grid = cdiv(D, D_BLOCK)
    grid = (H_grid, W_grid, D_grid)

    grad_w = torch.zeros([bsize, H_grid * W_grid * D_grid, 3, 3, 3, channels], device=x.device, dtype=torch.float16)
    WD_grid = W_grid * D_grid  # TODO: mb implement in another way

    for unbatched_x, unbatched_grad, unbatched_grad_w in zip(x, grad, grad_w):
        _DWConv_wgrad_cl3d_impl[grid](
            unbatched_grad,
            unbatched_x,
            unbatched_grad_w,
            H,
            W,
            D,
            H_stride,
            W_stride,
            ACCTYPE,
            channels,
            D_BLOCK,
            WD_grid,
            D_grid,
            H_grid,
            ILP,
            num_warps=num_warps,
        )

    grad_w = grad_w.sum(dim=(0, 1))

    return grad_w


def generate_inputs_dwconv_wgrad(problem_sizes, device='cuda'):
    channels = problem_sizes['channels']

    if channels <= 32:
        base = 256
    elif channels <= 64:
        base = 192
    else:
        base = 128

    x = torch.randn(1, channels, base, base, base, device=device, dtype=torch.float16).to(memory_format=torch.channels_last_3d)
    grad = torch.randn(1, channels, base, base, base, device=device, dtype=torch.float16).to(memory_format=torch.channels_last_3d)

    return x, grad


def pruning_rule(problem_size, named_config):
    D_BLOCK = named_config['D_BLOCK']

    channels = problem_size['channels']

    if channels >= 32 and D_BLOCK > 32:
        return False

    if channels >= 128 and D_BLOCK > 16:
        return False

    return True


def autotune_dwconv_wgrad(toml_path, **autotune_kwargs):
    problem_sizes = [{'channels': channels} for channels in [2 ** i for i in range(3, 8)]]

    autotune(
        getattr(DWConvWGRAD, 'function', DWConvWGRAD),
        generate_inputs_dwconv_wgrad,
        problem_sizes,
        pruning_rule,
        toml_path,
        **autotune_kwargs,
        num_warps=[1, 2, 4],
        D_BLOCK=[8, 16, 32, 64],
        ILP=[1, 2, 3, 4]
    )
