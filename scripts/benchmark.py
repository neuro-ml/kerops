"""Times every kerops op against the torch code it replaces over channels and input sizes (1, C, S, S, S), fp16,
channels_last_3d, and prints a markdown table per op under a header naming the GPU and the kerops version."""
import argparse

import torch
import triton
from torch.nn import functional as F
from triton.testing import do_bench

from kerops.__version__ import __version__
from kerops.ops.addition import AddStats, AddStatsBackward
from kerops.ops.avgpool import AvgPoolCeilStats, AvgPoolCeilStatsBackward
from kerops.ops.bnrelu import ApplyBNReLU, ApplyBNReLUBackward
from kerops.ops.conv import Conv3d, Conv3dWgrad, DWConv, DWConvWGRAD
from kerops.ops.conv.conv import channel_pairs
from kerops.ops.conv.conv_wgrad import cudnn_wgrad
from kerops.ops.linear import LinBReLULinAdd, LinBReLULinBackward, ReLULinearAdd, ReLULinearBackward
from kerops.ops.quantization import DequantUint8Window, QuantUint8Window
from kerops.ops.stats import Stats, StatsBackward


SIZES = [32, 64, 128, 256]
# One activation of at most 1 GiB in fp16 leaves room for torch's fp32 intermediates on a 24 GB GPU.
MAX_ELEMENTS = 2**29
CHANNELS = [16, 32, 64, 128]
CONV_PAIRS = [(pair['in_channels'], pair['out_channels']) for pair in channel_pairs()]
DIMS = (0, 2, 3, 4)
WINDOW = 4.0


def activation(channels, size):
    x = torch.randn(1, channels, size, size, size, device='cuda', dtype=torch.float16)
    return x.to(memory_format=torch.channels_last_3d)


def vector(channels):
    return torch.randn(channels, device='cuda')


def flat(x):
    """[voxels, channels] view of a channels_last_3d activation."""
    return x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])


def per_channel(v):
    return v.view(1, -1, 1, 1, 1)


def stats(r):
    r = r.float()
    return r.mean(DIMS), r.square().mean(DIMS)


def stats_grad(r, mean_grad, sqmean_grad):
    n = r.numel() // r.shape[1]
    return per_channel(mean_grad / n) + r * per_channel(2 * sqmean_grad / n)


# Each op below takes its channels and the input size and returns (kerops call, torch call).

def conv3d(channels, size):
    in_channels, out_channels = channels
    x = activation(in_channels, size)
    weight = torch.randn(3, 3, 3, in_channels, out_channels, device='cuda', dtype=torch.float16)
    weight /= (27 * in_channels) ** 0.5
    weight_torch = weight.permute(4, 3, 0, 1, 2).contiguous()
    return lambda: Conv3d(x, weight), lambda: F.conv3d(x, weight_torch, padding=1)


def conv3d_wgrad(channels, size):
    in_channels, out_channels = channels
    grad, x = activation(out_channels, size), activation(in_channels, size)
    return lambda: Conv3dWgrad(grad, x), lambda: cudnn_wgrad(grad, x)


def dwconv(c, s):
    x, weight = activation(c, s), torch.randn(3, 3, 3, c, device='cuda', dtype=torch.float16)
    weight_torch = weight.permute(3, 0, 1, 2)[:, None].contiguous()
    return lambda: DWConv(x, weight), lambda: F.conv3d(x, weight_torch, padding=1, groups=c)


def dwconv_wgrad(c, s):
    x, grad = activation(c, s), activation(c, s)
    weight_torch = torch.empty(c, 1, 3, 3, 3, device='cuda', dtype=torch.float16)
    return lambda: DWConvWGRAD(x, grad), lambda: torch.ops.aten.convolution_backward(
        grad, x, weight_torch, None, [1] * 3, [1] * 3, [1] * 3, False, [0] * 3, c, [False, True, False])


def relu_linear_add(c, s):
    x, other = activation(c, s), activation(c // 2, s)
    weight = torch.randn(c, c // 2, device='cuda', dtype=torch.float16) / c ** 0.5
    return lambda: ReLULinearAdd(x, weight, other), lambda: torch.addmm(flat(other), flat(x).relu(), weight)


def relu_linear_backward(c, s):
    x, grad = activation(c, s), activation(c // 2, s)
    weight = torch.randn(c, c // 2, device='cuda', dtype=torch.float16) / c ** 0.5

    def reference():
        fx, fg = flat(x), flat(grad)
        return (fg @ weight.T) * (fx > 0), fx.relu().T @ fg
    return lambda: ReLULinearBackward(x, grad, weight), reference


def linear_weights(c):
    """Up and down weights and the bias of a c -> 2c -> c block."""
    return (torch.randn(c, 2 * c, device='cuda', dtype=torch.float16) / c ** 0.5,
            torch.randn(2 * c, c, device='cuda', dtype=torch.float16) / (2 * c) ** 0.5,
            torch.randn(2 * c, device='cuda', dtype=torch.float16))


def lin_bias_relu_lin_add(c, s):
    x, other = activation(c, s), activation(c, s)
    weight_up, weight_down, bias = linear_weights(c)
    return (lambda: LinBReLULinAdd(x, weight_up, weight_down, bias, other),
            lambda: torch.addmm(flat(other), torch.addmm(bias, flat(x), weight_up).relu(), weight_down))


def lin_bias_relu_lin_backward(c, s):
    x, grad = activation(c, s), activation(c, s)
    weight_up, weight_down, bias = linear_weights(c)

    def reference():
        fx, fg = flat(x), flat(grad)
        hidden = torch.addmm(bias, fx, weight_up)
        hidden_grad = (fg @ weight_down.T) * (hidden > 0)
        return hidden_grad @ weight_up.T, fx.T @ hidden_grad, hidden.relu().T @ fg, hidden_grad.sum(0)
    return lambda: LinBReLULinBackward(x, grad, weight_up, weight_down, bias), reference


def add_stats(c, s):
    x, y = activation(c, s), activation(c, s)

    def reference():
        r = x + y
        return r, *stats(r)
    return lambda: AddStats(x, y), reference


def add_stats_backward(c, s):
    add_grad, r, mean_grad, sqmean_grad = activation(c, s), activation(c, s), vector(c), vector(c)
    return (lambda: AddStatsBackward(add_grad, mean_grad, sqmean_grad, r),
            lambda: (add_grad + stats_grad(r, mean_grad, sqmean_grad)).half())


def avgpool_stats(c, s):
    x = activation(c, s)

    def reference():
        pooled = F.avg_pool3d(x, 2, ceil_mode=True)
        return pooled, *stats(pooled)
    return lambda: AvgPoolCeilStats(x), reference


def avgpool_stats_backward(c, s):
    x = activation(c, s)
    pooled = F.avg_pool3d(x, 2, ceil_mode=True).contiguous(memory_format=torch.channels_last_3d)
    pooled_grad, mean_grad, sqmean_grad = torch.randn_like(pooled), vector(c), vector(c)

    def reference():
        grad = (pooled_grad + stats_grad(pooled, mean_grad, sqmean_grad)).half()
        return torch.ops.aten.avg_pool3d_backward(grad, x, [2] * 3, [2] * 3, [0] * 3, True, True, None)
    return lambda: AvgPoolCeilStatsBackward(pooled_grad, mean_grad, sqmean_grad, pooled, x.shape), reference


def bn_relu(c, s):
    x, weight, bias = activation(c, s), vector(c), vector(c)
    weight_torch, bias_torch = per_channel(weight.half()), per_channel(bias.half())
    return lambda: ApplyBNReLU(x, weight, bias), lambda: torch.addcmul(bias_torch, x, weight_torch).relu_()


def bn_relu_backward(c, s):
    x, grad, weight, bias = activation(c, s), activation(c, s), per_channel(vector(c)), per_channel(vector(c))

    def reference():
        masked = grad * (x * weight + bias > 0)
        return (masked * weight).half(), (masked * x).sum(DIMS), masked.sum(DIMS)
    return lambda: ApplyBNReLUBackward(x, weight, bias, grad), reference


def stats_forward(c, s):
    x = activation(c, s)
    return lambda: Stats(x), lambda: stats(x)


def stats_backward(c, s):
    x, mean_grad, sqmean_grad = activation(c, s), vector(c), vector(c)
    return lambda: StatsBackward(x, mean_grad, sqmean_grad), lambda: stats_grad(x, mean_grad, sqmean_grad).half()


def quant(c, s):
    x = activation(c, s)
    return (lambda: QuantUint8Window(x, WINDOW),
            lambda: ((x.float().clamp(-WINDOW, WINDOW) + WINDOW) * (255 / (2 * WINDOW))).to(torch.uint8))


def dequant(c, s):
    x = torch.randint(0, 256, (1, c, s, s, s), device='cuda', dtype=torch.uint8)
    return (lambda: DequantUint8Window(x, torch.float16, WINDOW),
            lambda: (x.float() * (2 * WINDOW / 255) - WINDOW).half())


# name: (op, channels, what the torch call is)
OPS = {
    'Conv3d': (conv3d, CONV_PAIRS, 'cuDNN'),
    'Conv3dWgrad': (conv3d_wgrad, CONV_PAIRS, 'cuDNN'),
    'DWConv': (dwconv, [8, *CHANNELS], 'cuDNN'),
    'DWConvWGRAD': (dwconv_wgrad, [8, *CHANNELS], 'cuDNN'),
    'ReLULinearAdd': (relu_linear_add, CHANNELS, 'cuBLAS addmm'),
    'ReLULinearBackward': (relu_linear_backward, [32, 64, 128], 'cuBLAS matmuls'),
    'LinBReLULinAdd': (lin_bias_relu_lin_add, [16, 32, 64], 'cuBLAS addmm'),
    'LinBReLULinBackward': (lin_bias_relu_lin_backward, [16, 32, 64], 'cuBLAS matmuls'),
    'AddStats': (add_stats, CHANNELS, 'torch eager'),
    'AddStatsBackward': (add_stats_backward, CHANNELS, 'torch eager'),
    'AvgPoolCeilStats': (avgpool_stats, CHANNELS, 'torch eager'),
    'AvgPoolCeilStatsBackward': (avgpool_stats_backward, CHANNELS, 'torch eager'),
    'ApplyBNReLU': (bn_relu, CHANNELS, 'torch eager'),
    'ApplyBNReLUBackward': (bn_relu_backward, CHANNELS, 'torch eager'),
    'Stats': (stats_forward, CHANNELS, 'torch eager'),
    'StatsBackward': (stats_backward, CHANNELS, 'torch eager'),
    'QuantUint8Window': (quant, CHANNELS, 'torch eager'),
    'DequantUint8Window': (dequant, CHANNELS, 'torch eager'),
}


def speedup(op, channels, counts, size):
    """Cell of the table: torch time over kerops time, or a dash for inputs past MAX_ELEMENTS."""
    if max(counts) * size**3 > MAX_ELEMENTS:
        return '—'
    kerops, reference = op(channels, size)
    torch_ms, kerops_ms = (do_bench(f, warmup=25, rep=100, return_mode='median') for f in (reference, kerops))
    del kerops, reference
    torch.cuda.empty_cache()
    return f'{torch_ms / kerops_ms:.2f}x'


def print_table(name):
    op, channels_list, reference_name = OPS[name]
    print(f'\n## {name}: speedup over {reference_name}\n')
    print('| channels | ' + ' | '.join(f'{size}³' for size in SIZES) + ' |')
    print('|:--------:|' + '------:|' * len(SIZES))
    for channels in channels_list:
        counts = channels if isinstance(channels, tuple) else (channels,)
        cells = [speedup(op, channels, counts, size) for size in SIZES]
        print(f'| {" → ".join(map(str, counts))} | ' + ' | '.join(cells) + ' |', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ops', nargs='*', default=list(OPS), help=f'any of {list(OPS)}')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    print(f'# kerops {__version__} on {torch.cuda.get_device_name()}\n')
    print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, '
          f'triton {triton.__version__}. Input (1, C, S, S, S), fp16, channels_last_3d; a cell is torch time over '
          f'kerops time, medians of triton do_bench; — is an input over {2 * MAX_ELEMENTS / 2**30:.0f} GiB in fp16.')
    for name in args.ops:
        print_table(name)
