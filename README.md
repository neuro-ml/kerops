# Kerops
Efficient and fast algorithms on the GPU

# Install
*pip is not available right now*
```shell
pip install kerops
```

# How fast is it?
Time comparison (ms) for NVidia RTX 3090. Input is an array of size (1, channels, 350, 350, 128); float16; <b>channels_last_3d</b>. Compared to usual 3d convolution from torch (kernel_size=3, padding=1, stride=1, bias=False, in_channels=channels, out_channels=channels). Slowdown compared to copying is shown in parentheses.

| channels             |torch.clone|  kerops.ops.DWConv   |torch.nn.Conv3d(C->C)|
|:--------------------:|:---------:|:--------------------:|:-------------------:|
| 8                    |   0.61    |         0.81 (x1.32) |     2.45 (x4.00)    |
| 16                   |   1.21    |         1.27 (1.27)  |     4.48 (x3.70)    |
| 32                   |   2.40    |         3.12 (1.30)  |     15.3 (x6.38)    |
| 64                   |   4.78    |         6.29 (1.32)  |     52.0 (x10.89)   |
| 128                  |   9.55    |         13.2 (1.38)  |     195.0 (x20.44)  |
