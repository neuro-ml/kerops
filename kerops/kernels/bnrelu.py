import triton
import triton.language as tl


@triton.jit
def _ApplyBNReLU_cl3d_impl(
    X_ptr,
    Out_ptr,
    Weight_ptr,
    Bias_ptr,
    numel_no_channels,
    BLOCK_SIZE: tl.constexpr,
    num_channels: tl.constexpr,
    block_other: tl.constexpr,
):
    pid = tl.program_id(0)
    X_ptr += pid * BLOCK_SIZE
    Out_ptr += pid * BLOCK_SIZE

    channels_offset = tl.arange(0, num_channels)
    other_offset = tl.arange(0, block_other)
    offset = channels_offset[None, :] + other_offset[:, None] * num_channels
    mask = (other_offset < numel_no_channels - pid * block_other)[:, None]

    x = tl.load(X_ptr + offset, mask=mask, other=0).to(tl.float32)
    weight = tl.load(Weight_ptr + channels_offset[None, :])
    bias = tl.load(Bias_ptr + channels_offset[None, :])

    output = x * weight + bias
    output = tl.maximum(output, 0.0)
    tl.store(Out_ptr + offset, output, mask=mask)


@triton.jit
def _ApplyBNReLU_cl3d_backward_impl(
    Input_ptr,
    Weight_ptr,
    Bias_ptr,
    Grad_ptr,
    Outgrad_ptr,
    Weight_outgrad_ptr,
    Bias_outgrad_ptr,
    numel_no_channels,
    BLOCK_SIZE: tl.constexpr,
    num_channels: tl.constexpr,
    block_other: tl.constexpr,
):
    pid = tl.program_id(0)
    Input_ptr += pid * BLOCK_SIZE
    Grad_ptr += pid * BLOCK_SIZE
    Outgrad_ptr += pid * BLOCK_SIZE

    channels_offset = tl.arange(0, num_channels)
    other_offset = tl.arange(0, block_other)
    offset = channels_offset[None, :] + other_offset[:, None] * num_channels
    mask = (other_offset < numel_no_channels - pid * block_other)[:, None]

    weight = tl.load(Weight_ptr + channels_offset[None, :])
    bias = tl.load(Bias_ptr + channels_offset[None, :])
    input = tl.load(Input_ptr + offset, mask=mask, other=0).to(tl.float32)
    grad = tl.load(Grad_ptr + offset, mask=mask, other=0).to(tl.float32)

    grad = grad * (input * weight > -bias)

    b_grad = tl.sum(grad, axis=0)
    w_grad = tl.sum(input * grad, axis=0)
    x_grad = weight * grad

    tl.store(Outgrad_ptr + offset, x_grad, mask=mask)
    tl.atomic_add(Bias_outgrad_ptr + channels_offset, b_grad)
    tl.atomic_add(Weight_outgrad_ptr + channels_offset, w_grad)
