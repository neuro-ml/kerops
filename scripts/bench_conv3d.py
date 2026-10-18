"""Times Conv3d implementations with their tuned configs against cuDNN; errors are relative to an fp32 reference."""
import argparse
import threading

import torch
from torch.nn import functional as F
from triton.testing import do_bench

from kerops.ops.conv import Conv3d
from kerops.ops.conv.conv import channel_pairs

# Each implementation takes the input, the torch-layout weight and the kerops-layout weight.
IMPLEMENTATIONS = {
    'cudnn': lambda x, weight, weight_kernel: F.conv3d(x, weight, padding=1),
    'kerops': lambda x, weight, weight_kernel: Conv3d(x, weight_kernel),
}

INPUTS = {
    'randn': lambda x: x,
    'relu': torch.relu,
}


def spatial_size(in_channels: int, out_channels: int) -> int:
    channels = max(in_channels, out_channels)
    return 256 if channels <= 32 else 192 if channels <= 64 else 128


def make_inputs(in_channels: int, out_channels: int, size: int, kind: str = 'randn'):
    torch.manual_seed(0)
    x = INPUTS[kind](torch.randn(1, in_channels, size, size, size, device='cuda', dtype=torch.float16))
    weight = torch.randn(out_channels, in_channels, 3, 3, 3, device='cuda', dtype=torch.float16) / (27 * in_channels) ** 0.5
    return x.to(memory_format=torch.channels_last_3d), weight


def fp32_reference(x, weight):
    torch.backends.cudnn.allow_tf32 = False
    return F.conv3d(x.float(), weight.float(), padding=1)


def bench_with_clock(fn) -> tuple[float, float]:
    """Median ms of `fn` and the median SM clock in MHz sampled while it runs."""
    clocks = []
    done = threading.Event()

    def sample_clock():
        while not done.wait(0.01):
            clocks.append(torch.cuda.clock_rate())

    sampler = threading.Thread(target=sample_clock)
    sampler.start()
    ms = do_bench(fn, warmup=50, rep=300, return_mode='median')
    done.set()
    sampler.join()
    return ms, sorted(clocks)[len(clocks) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('implementations', nargs='*', default=['cudnn', 'kerops'], help=f'any of {list(IMPLEMENTATIONS)}')
    parser.add_argument('--inputs', default='randn', choices=list(INPUTS))
    args = parser.parse_args()

    print(f'inputs: {args.inputs}')
    print(f'{"cin->cout":>10} {"size":>5} {"impl":>14} {"ms":>8} {"vs cudnn":>8} {"MHz":>5} {"max err":>8} {"mean err":>9}')
    for pair in channel_pairs():
        in_channels, out_channels = pair['in_channels'], pair['out_channels']
        size = spatial_size(in_channels, out_channels)
        x, weight = make_inputs(in_channels, out_channels, size, args.inputs)
        weight_kernel = weight.permute(2, 3, 4, 1, 0).contiguous()
        reference = fp32_reference(x, weight)
        scale = reference.std()

        cudnn_ms = None
        for name in ['cudnn'] + [name for name in args.implementations if name != 'cudnn']:
            conv = IMPLEMENTATIONS[name]
            try:
                error = (conv(x, weight, weight_kernel).float() - reference).abs()
            except Exception as e:
                print(f'{f"{in_channels}->{out_channels}":>10} {size:>4}³ {name:>14} failed: {str(e).splitlines()[0][:80]}')
                continue
            ms, mhz = bench_with_clock(lambda: conv(x, weight, weight_kernel))
            cudnn_ms = cudnn_ms or ms
            if name in args.implementations:
                print(
                    f'{f"{in_channels}->{out_channels}":>10} {size:>4}³ {name:>14} {ms:8.3f} {cudnn_ms / ms:7.2f}x {mhz:5.0f}'
                    f' {error.max() / scale:8.1e} {error.mean() / scale:9.1e}'
                )
        del x, weight, weight_kernel, reference
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
