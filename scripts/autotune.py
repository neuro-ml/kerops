"""Autotunes ops for the current GPU and writes the best configs into kerops/ops/assets, keeping other GPUs' configs."""
import argparse

from kerops.ops.assets import ASSETS_ROOT
from kerops.ops.conv.conv import autotune_bnreluconv, autotune_conv
from kerops.ops.conv.conv_wgrad import (
    autotune_conv_wgrad,
    autotune_conv_wgrad_grad_based,
    autotune_conv_wgrad_splitk,
    autotune_conv_wgrad_splitk_3acc,
)
from kerops.ops.conv.dwconv import autotune_dwconv
from kerops.ops.conv.dwconv_wgrad import autotune_dwconv_wgrad
from kerops.ops.linear.linear_bias_relu_linear_add import autotune_lin_bn_relu_lin_add
from kerops.ops.linear.linear_bias_relu_linear_backward import autotune_lin_bn_relu_lin_backward
from kerops.ops.linear.relu_linear_add import autotune_relu_lin_add
from kerops.ops.linear.relu_linear_backward import autotune_relu_lin_backward

TARGETS = {
    'conv': (autotune_conv, 'Conv3d.toml'),
    'bnreluconv': (autotune_bnreluconv, 'BNReLUConv3d.toml'),
    'conv_wgrad': (autotune_conv_wgrad, 'Conv3dWgrad.toml'),
    'conv_wgrad_grad_based': (autotune_conv_wgrad_grad_based, 'Conv3dWgrad_grad_based.toml'),
    'conv_wgrad_splitk': (autotune_conv_wgrad_splitk, 'Conv3dWgrad_splitk.toml'),
    'conv_wgrad_splitk_3acc': (autotune_conv_wgrad_splitk_3acc, 'Conv3dWgrad_splitk_3acc.toml'),
    'dwconv': (autotune_dwconv, 'DWConv.toml'),
    'dwconv_wgrad': (autotune_dwconv_wgrad, 'DWConvWGRAD.toml'),
    'lin_bn_relu_lin_add': (autotune_lin_bn_relu_lin_add, 'LinBReLULinAdd.toml'),
    'lin_bn_relu_lin_backward': (autotune_lin_bn_relu_lin_backward, 'LinBReLULinBackward.toml'),
    'relu_lin_add': (autotune_relu_lin_add, 'ReLULinAdd.toml'),
    'relu_lin_backward': (autotune_relu_lin_backward, 'ReLULinBackward.toml'),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('targets', nargs='+', choices=TARGETS)
    parser.add_argument('--processes', type=int, default=8, help='compiling processes')
    parser.add_argument('--threads', type=int, default=4, help='compiling threads per process')
    args = parser.parse_args()

    for target in args.targets:
        autotune_target, toml_name = TARGETS[target]
        print(f'{target} -> kerops/ops/assets/{toml_name}')
        autotune_target(ASSETS_ROOT / toml_name, n_processes=args.processes, n_threads=args.threads)
