import triton
import triton.language as tl


@triton.jit
def _AddStats_cl3d_impl(
    X_ptr,
    Y_ptr,
    Out_ptr,
    Mean_ptr,
    Sqmean_ptr,
    numel,
    numel_no_channels,
    BLOCK_SIZE: tl.constexpr,
    num_channels: tl.constexpr,
    block_other: tl.constexpr,
):
    pid = tl.program_id(0)
    X_ptr += pid * BLOCK_SIZE
    Y_ptr += pid * BLOCK_SIZE
    Out_ptr += pid * BLOCK_SIZE

    channels_offset = tl.arange(0, num_channels)
    other_offset = tl.arange(0, block_other)
    offset = channels_offset[None, :] + other_offset[:, None] * num_channels
    mask = (other_offset < numel_no_channels - pid * block_other)[:, None]

    x = tl.load(X_ptr + offset, mask=mask, other=0)
    y = tl.load(Y_ptr + offset, mask=mask, other=0)
    output = (x + y).to(tl.float32)
    tl.store(Out_ptr + offset, output, mask=mask)

    mean = tl.sum(output, axis=0) / numel_no_channels
    sqmean = tl.sum(output * output, axis=0) / numel_no_channels

    tl.atomic_add(Mean_ptr + channels_offset, mean)
    tl.atomic_add(Sqmean_ptr + channels_offset, sqmean)


@triton.jit
def _AddStats_cl3d_backward_impl(
    Addgrad_ptr,
    Meangrad_ptr,
    Sqmeangrad_ptr,
    Sum_ptr,
    Outputgrad_ptr,
    numel,
    numel_no_channels,
    BLOCK_SIZE: tl.constexpr,
    num_channels: tl.constexpr,
    block_other: tl.constexpr,
):
    pid = tl.program_id(0)
    Addgrad_ptr += pid * BLOCK_SIZE
    Sum_ptr += pid * BLOCK_SIZE
    Outputgrad_ptr += pid * BLOCK_SIZE

    channels_offset = tl.arange(0, num_channels)
    other_offset = tl.arange(0, block_other)
    offset = channels_offset[None, :] + other_offset[:, None] * num_channels

    mask = (other_offset < numel_no_channels - pid * block_other)[:, None]

    sum = tl.load(Sum_ptr + offset, mask=mask, other=0.0)
    add_grad = tl.load(Addgrad_ptr + offset, mask=mask, other=0.0)
    mean_grad = tl.load(Meangrad_ptr + channels_offset[None, :])
    sqmean_grad = tl.load(Sqmeangrad_ptr + channels_offset[None, :])

    sqmean_grad_part = 2 * sum.to(tl.float32) * sqmean_grad / numel_no_channels
    mean_grad_part = mean_grad / numel_no_channels

    grad = add_grad + sqmean_grad_part + mean_grad_part
    grad = grad.to(tl.float16)

    tl.store(Outputgrad_ptr + offset, grad, mask=mask)
