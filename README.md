# Kerops
Fast algorithms for GPU

# Install
*pip is not available right now*
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
