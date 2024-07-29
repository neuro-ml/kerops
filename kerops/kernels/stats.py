import triton
import triton.language as tl


@triton.jit
def _Stats_cl3d_impl(
    X_ptr, Mean_ptr, Sqmean_ptr, numel_no_channels, num_channels: tl.constexpr, block_other: tl.constexpr
):
    pid = tl.program_id(0)
    X_ptr += pid * block_other * num_channels

    channels_offset = tl.arange(0, num_channels)
    other_offset = tl.arange(0, block_other)

    offset = other_offset[:, None] * num_channels + channels_offset[None, :]
    mask = other_offset[:, None] < numel_no_channels - pid * block_other

    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / numel_no_channels
    sqmean = tl.sum(x * x, axis=0) / numel_no_channels

    tl.atomic_add(Mean_ptr + channels_offset, mean)
    tl.atomic_add(Sqmean_ptr + channels_offset, sqmean)


@triton.jit
def _Stats_cl3d_backward_impl(
    X_ptr,
    Meangrad_ptr,
    Sqmeangrad_ptr,
    Outputgrad_ptr,
    numel_no_channels,
    num_channels: tl.constexpr,
    block_other: tl.constexpr,
):
    pid = tl.program_id(0)
    X_ptr += pid * num_channels * block_other
    Outputgrad_ptr += pid * num_channels * block_other

    channels_offset = tl.arange(0, num_channels)
    other_offset = tl.arange(0, block_other)

    offset = other_offset[:, None] * num_channels + channels_offset[None, :]
    mask = other_offset[:, None] < numel_no_channels - pid * block_other

    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    mean_grad = tl.load(Meangrad_ptr + channels_offset)
    sqmean_grad = tl.load(Sqmeangrad_ptr + channels_offset)

    grad = (2 * x * sqmean_grad / numel_no_channels) + (mean_grad / numel_no_channels)
    tl.store(Outputgrad_ptr + offset, grad, mask=mask)
