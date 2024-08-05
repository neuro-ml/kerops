import triton
import triton.language as tl


@triton.jit
def _ReLULinearAdd(
    input_ptr,
    weight_ptr,
    add_ptr,
    output_ptr,
    numel_no_channels,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    D_block: tl.constexpr,
    _ILP: tl.constexpr,
):
    pid = tl.program_id(0)
    input_ptr += pid * _ILP * in_channels * D_block
    add_ptr += pid * _ILP * out_channels * D_block
    output_ptr += pid * _ILP * out_channels * D_block

    in_channels_offset = tl.arange(0, in_channels)
    out_channels_offset = tl.arange(0, out_channels)
    d_offset = tl.arange(0, D_block)

    in_offset = d_offset[:, None] * in_channels + in_channels_offset[None, :]
    out_offset = d_offset[:, None] * out_channels + out_channels_offset[None, :]
    weight_offset = in_channels_offset[:, None] * out_channels + out_channels_offset[None, :]

    weight = tl.load(weight_ptr + weight_offset)

    for i in tl.static_range(0, _ILP):
        mask = d_offset[:, None] < numel_no_channels - (pid * _ILP + i) * D_block

        x = tl.load(input_ptr + in_offset, mask=mask, other=0)
        add = tl.load(add_ptr + out_offset, mask=mask, other=0)

        x = tl.maximum(x, 0.0).to(tl.float16)
        output = tl.dot(x, weight, out_dtype=tl.float32, allow_tf32=True).to(tl.float16) + add

        tl.store(output_ptr + out_offset, output, mask=mask)

        input_ptr += in_channels * D_block
        output_ptr += out_channels * D_block
        add_ptr += out_channels * D_block


@triton.jit
def _ReLULinearAddBackward(
    input_ptr,
    grad_ptr,
    input_grad_ptr,
    weight_ptr,
    weight_grad_ptr,
    numel_no_channels,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    D_block: tl.constexpr,
    _ILP: tl.constexpr,
):
    pid = tl.program_id(0)

    input_ptr += pid * _ILP * in_channels * D_block
    grad_ptr += pid * _ILP * out_channels * D_block
    input_grad_ptr += pid * _ILP * in_channels * D_block
    weight_grad_ptr += pid * in_channels * out_channels

    in_channels_offset = tl.arange(0, in_channels)
    out_channels_offset = tl.arange(0, out_channels)
    d_offset = tl.arange(0, D_block)

    input_offset = d_offset[:, None] * in_channels + in_channels_offset[None, :]
    output_offset = d_offset[:, None] * out_channels + out_channels_offset[None, :]
    weight_offset = out_channels_offset[:, None] + in_channels_offset[None, :] * out_channels
    weight_grad_offset = in_channels_offset[:, None] * out_channels + out_channels_offset[None, :]

    weight = tl.load(weight_ptr + weight_offset)

    weight_grad = tl.zeros([in_channels, out_channels], dtype=tl.float32)

    for i in tl.static_range(0, _ILP):
        mask = d_offset[:, None] < numel_no_channels - (pid * _ILP + i) * D_block

        input = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        grad = tl.load(grad_ptr + output_offset, mask=mask, other=0.0)

        weight_grad += tl.dot(
            tl.trans(tl.maximum(input, 0.0).to(tl.float16)), grad, out_dtype=tl.float32, allow_tf32=True
        )
        input_grad = tl.dot(grad, weight, out_dtype=tl.float32, allow_tf32=True).to(tl.float16) * (input > 0)

        tl.store(input_grad_ptr + input_offset, input_grad, mask=mask)

        grad_ptr += out_channels * D_block
        input_grad_ptr += in_channels * D_block
        input_ptr += in_channels * D_block

    tl.store(weight_grad_ptr + weight_grad_offset, weight_grad)
