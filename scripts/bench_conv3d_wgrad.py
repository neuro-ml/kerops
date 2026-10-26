"""Times Conv3dWgrad implementations with their tuned configs against cuDNN; errors are relative to an fp32 reference."""
import argparse

import torch

from bench_conv3d import bench_with_clock
from kerops.ops.conv.conv import channel_pairs
from kerops.ops.conv.conv_wgrad import (
    Conv3dWgrad,
    Conv3dWgrad_grad_based,
    Conv3dWgrad_splitk,
    Conv3dWgrad_splitk_3acc,
    cudnn_wgrad,
    generate_inputs_conv_wgrad,
)
from kerops.utils import wgrad_errors

IMPLEMENTATIONS = {
    'cudnn': cudnn_wgrad,
    'kerops': Conv3dWgrad,
    'grad_based': Conv3dWgrad_grad_based,
    'splitk': Conv3dWgrad_splitk,
    'splitk_3acc': Conv3dWgrad_splitk_3acc,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('implementations', nargs='*', default=['cudnn', 'kerops'], help=f'any of {list(IMPLEMENTATIONS)}')
    args = parser.parse_args()

    print(f'{"cin->cout":>10} {"size":>5} {"impl":>12} {"ms":>8} {"vs cudnn":>8} {"MHz":>5} {"cos err":>8} {"norm err":>9}')
    for pair in channel_pairs():
        in_channels, out_channels = pair['in_channels'], pair['out_channels']
        grad, x = generate_inputs_conv_wgrad(pair)
        size = x.shape[-1]
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            reference = cudnn_wgrad(grad.float(), x.float())

        cudnn_ms = None
        for name in ['cudnn'] + [name for name in args.implementations if name != 'cudnn']:
            wgrad = IMPLEMENTATIONS[name]
            try:
                cos_error, norm_error = wgrad_errors(wgrad(grad, x), reference)
            except Exception as e:
                print(f'{f"{in_channels}->{out_channels}":>10} {size:>4}³ {name:>12} failed: {str(e).splitlines()[0][:80]}')
                continue
            ms, mhz = bench_with_clock(lambda: wgrad(grad, x))
            cudnn_ms = cudnn_ms or ms
            if name in args.implementations:
                print(
                    f'{f"{in_channels}->{out_channels}":>10} {size:>4}³ {name:>12} {ms:8.3f} {cudnn_ms / ms:7.2f}x {mhz:5.0f}'
                    f' {cos_error:8.1e} {norm_error:9.1e}'
                )
        del grad, x, reference
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
