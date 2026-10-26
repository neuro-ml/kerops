# Kerops
Fast algorithms for GPU

# Install
```shell
pip install kerops
```

# How fast is it?
Time comparison (ms) for NVidia RTX 3090. Input is an array of size (1, channels, 350, 350, 128); float16; <b>channels_last_3d</b>. Compared to usual 3d convolution from torch (kernel_size=3, padding=1, stride=1, bias=False, in_channels=channels, out_channels=channels). Slowdown compared to copying is shown in parentheses.

| channels             |torch.clone|  kerops.ops.DWConv   |torch.nn.Conv3d(C->C)|
|:--------------------:|:---------:|:--------------------:|:-------------------:|
| 8                    |   0.61    |         0.79 (x1.30) |     2.45 (x4.00)    |
| 16                   |   1.21    |         1.41 (x1.17) |     4.48 (x3.70)    |
| 32                   |   2.40    |         2.99 (x1.25) |     15.3 (x6.38)    |
| 64                   |   4.78    |         6.29 (x1.32) |     52.0 (x10.89)   |
| 128                  |   9.55    |         12.8 (x1.34) |     195.0 (x20.44)  |


| channels             |torch.clone|kerops.ops.DWConvWGRAD|torch.nn.Conv3d(C->C)|
|:--------------------:|:---------:|:--------------------:|:-------------------:|
| 8                    |   0.61    |         2.55 (x4.18) |     7.14 (x11.70)   |
| 16                   |   1.21    |         3.01 (x2.49) |     12.1 (x10.00)   |
| 32                   |   2.40    |         4.80 (x2.00) |     24.6 (x10.25)   |
| 64                   |   4.78    |         8.72 (x1.82) |     71.3 (x14.91)   |
| 128                  |   9.55    |         17.9 (x1.87) |     245.0 (x25.65)  |


Time comparison (ms) for NVidia RTX 4090: `kerops.ops.conv.Conv3d` against cuDNN (`torch.nn.functional.conv3d`), kernel_size=3, padding=1, bias=False. Input is an array of size (1, in_channels, size, size, size); float16; <b>channels_last_3d</b>. Speedup over cuDNN is shown in parentheses. Partial sums are accumulated in float16, so the error is up to 5e-3 (max) and 4e-4 (mean) of the output std.

| channels in → out | size |   cuDNN   | kerops.ops.conv.Conv3d |
|:-----------------:|:----:|:---------:|:----------------------:|
| 16 → 16           | 256  |   2.12    |      1.27 (x1.67)      |
| 16 → 32           | 256  |   3.44    |      2.16 (x1.59)      |
| 32 → 16           | 256  |   4.42    |      2.70 (x1.64)      |
| 32 → 32           | 256  |   6.75    |      4.15 (x1.63)      |
| 32 → 64           | 192  |   5.13    |      3.19 (x1.61)      |
| 64 → 32           | 192  |   5.56    |      3.34 (x1.66)      |
| 64 → 64           | 192  |   9.68    |      6.34 (x1.53)      |
| 64 → 128          | 128  |   5.63    |      3.84 (x1.47)      |
| 128 → 64          | 128  |   5.56    |      3.70 (x1.50)      |
| 128 → 128         | 128  |   10.93   |      7.79 (x1.40)      |

The same for the weight gradient: `kerops.ops.conv.Conv3dWgrad` against cuDNN (`torch.ops.aten.convolution_backward`), output gradient of out_channels, float16, <b>channels_last_3d</b>. Winograd F(3, 2) along the last axis does 4 products per pair of voxels instead of 6; partial sums are accumulated in float16 and flushed into float32. Against a float32 reference, 1 - cosine similarity is below 5e-7 and the norm differs by less than 1e-4.

| channels in → out | size |   cuDNN   | kerops.ops.conv.Conv3dWgrad |
|:-----------------:|:----:|:---------:|:---------------------------:|
| 16 → 16           | 256  |   5.50    |        1.44 (x3.84)         |
| 16 → 32           | 256  |   5.64    |        2.02 (x2.79)         |
| 32 → 16           | 256  |   8.38    |        1.98 (x4.23)         |
| 32 → 32           | 256  |   10.36   |        3.38 (x3.07)         |
| 32 → 64           | 192  |   6.52    |        2.89 (x2.26)         |
| 64 → 32           | 192  |   8.85    |        2.89 (x3.07)         |
| 64 → 64           | 192  |   13.25   |        5.83 (x2.27)         |
| 64 → 128          | 128  |   6.10    |        3.33 (x1.83)         |
| 128 → 64          | 128  |   7.83    |        3.33 (x2.35)         |
| 128 → 128         | 128  |   13.67   |        7.71 (x1.77)         |

# How ops are configured
Every op is a `ConfiguredFunction`: tensors are positional-only arguments, kernel launch parameters are keyword-only.

```text
scripts/autotune.py ── best params of this GPU ──▶ kerops/ops/assets/<Op>.toml
                                                         │ read at import
                                                         ▼
op(*tensors, **overrides) ──▶ ConfiguredFunction ◀── TableKernelConfig: params for
                                      │                (GPU of the tensors, args_to_problem_sizes(*tensors))
                                      ▼
                       kernel launch with the params, overrides win
```

- An op without a table takes its launch parameters from their defaults.
- `op.can_be_configured(*tensors)` tells whether the table has params for this GPU and these sizes.
- At import, the TOML params must equal the op's keyword-only arguments, and the keys returned by
  `args_to_problem_sizes` must equal the TOML's `problem_size_names`.
- A GPU missing from a table raises an error; `python scripts/autotune.py <target>` (e.g. `conv`) on that GPU
  adds it and keeps the others.
