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


@triton.jit
def _LinBReLULinAdd(
    input_ptr,
    weight_up_ptr,
    weight_down_ptr,
    bias_ptr,
    add_ptr,
    output_ptr,
    numel_no_channels,
    in_channels: tl.constexpr,
    hidden_channels: tl.constexpr,
    D_block: tl.constexpr,
    _ILP: tl.constexpr,
):
    pid = tl.program_id(0)
    input_ptr += pid * _ILP * in_channels * D_block
    add_ptr += pid * _ILP * in_channels * D_block
    output_ptr += pid * _ILP * in_channels * D_block

    in_channels_offset = tl.arange(0, in_channels)
    hidden_channels_offset = tl.arange(0, hidden_channels)
    d_offset = tl.arange(0, D_block)

    offset = d_offset[:, None] * in_channels + in_channels_offset[None, :]
    weight_up_offset = in_channels_offset[:, None] * hidden_channels + hidden_channels_offset[None, :]
    weight_down_offset = hidden_channels_offset[:, None] * in_channels + in_channels_offset[None, :]

    weight_up = tl.load(weight_up_ptr + weight_up_offset)
    weight_down = tl.load(weight_down_ptr + weight_down_offset)
    bias = tl.load(bias_ptr + hidden_channels_offset)[None]

    for i in tl.static_range(0, _ILP):
        mask = d_offset[:, None] < numel_no_channels - (pid * _ILP + i) * D_block

        x = tl.load(input_ptr + offset, mask=mask)#, other=0)
        add = tl.load(add_ptr + offset, mask=mask)#, other=0)

        hidden = tl.dot(x, weight_up, out_dtype=tl.float32, allow_tf32=True).to(tl.float16) + bias
        hidden = tl.maximum(hidden, 0.0).to(tl.float16)
        output = tl.dot(hidden, weight_down, out_dtype=tl.float32, allow_tf32=True).to(tl.float16) + add

        tl.store(output_ptr + offset, output, mask=mask)

        input_ptr += in_channels * D_block
        output_ptr += in_channels * D_block
        add_ptr += in_channels * D_block


@triton.jit
def _LinBReLULinBackward(
    input_ptr,
    grad_ptr,
    input_grad_ptr,
    weight_up_ptr,
    weight_down_ptr,
    bias_ptr,
    weight_up_grad_ptr,
    weight_down_grad_ptr,
    bias_grad_ptr,
    numel_no_channels,
    in_channels: tl.constexpr,
    hidden_channels: tl.constexpr,
    D_block: tl.constexpr,
    _ILP: tl.constexpr,
):
    pid = tl.program_id(0)

    input_ptr += pid * _ILP * in_channels * D_block
    grad_ptr += pid * _ILP * in_channels * D_block
    input_grad_ptr += pid * _ILP * in_channels * D_block
    weight_up_grad_ptr += pid * in_channels * hidden_channels
    weight_down_grad_ptr += pid * in_channels * hidden_channels
    bias_grad_ptr += pid * hidden_channels

    in_channels_offset = tl.arange(0, in_channels)
    hidden_channels_offset = tl.arange(0, hidden_channels)
    d_offset = tl.arange(0, D_block)

    offset = d_offset[:, None] * in_channels + in_channels_offset[None, :]
    weight_up_offset = in_channels_offset[:, None] * hidden_channels + hidden_channels_offset[None, :]
    weight_down_offset = hidden_channels_offset[:, None] * in_channels + in_channels_offset[None, :]

    weight_up = tl.load(weight_up_ptr + weight_up_offset)
    weight_down = tl.load(weight_down_ptr + weight_down_offset)
    bias = tl.load(bias_ptr + hidden_channels_offset)[None]

    weight_up_grad = tl.zeros([hidden_channels, in_channels], dtype=tl.float32)
    weight_down_grad = tl.zeros([in_channels, hidden_channels], dtype=tl.float32)
    bias_grad = tl.zeros([hidden_channels], dtype=tl.float32)

    out_offset = in_channels_offset[:, None] + d_offset[None, :] * in_channels

    weight_up_grad_offset = hidden_channels_offset[:, None] + in_channels_offset[None, :] * hidden_channels
    weight_down_grad_offset = in_channels_offset[:, None] + hidden_channels_offset[None, :] * in_channels

    for i in tl.static_range(0, _ILP):
        mask = d_offset[:, None] < numel_no_channels - (pid * _ILP + i) * D_block
        out_mask = d_offset[None, :] < numel_no_channels - (pid * _ILP + i) * D_block

        input = tl.load(input_ptr + offset, mask=mask, other=0.0)  # [D_block, in_channels]
        grad = tl.load(grad_ptr + offset, mask=mask, other=0.0)  # [D_block, in_channels]
        gradT = tl.trans(grad)  # [in_channels, D_block]

        linup = (
            tl.dot(input, weight_up, out_dtype=tl.float32, allow_tf32=True).to(tl.float16) + bias
        )  # [D_block, hidden_channels]
        linup_relu = tl.maximum(linup, 0.0).to(tl.float16)  # [D_block, hidden_channels]

        weight_down_grad += tl.dot(
            gradT, linup_relu, out_dtype=tl.float32, allow_tf32=True
        )  # [in_channels, hidden_channels]

        linup_gradT = tl.trans(linup > 0) * tl.dot(weight_down, gradT, out_dtype=tl.float32, allow_tf32=True).to(
            tl.float16
        )  # [hidden_channels, D_block]
        weight_up_grad += tl.dot(
            linup_gradT, input, out_dtype=tl.float32, allow_tf32=True
        )  # [hidden_channels, in_channels]
        bias_grad += tl.sum(linup_gradT, axis=1)

        input_gradT = tl.dot(weight_up, linup_gradT)
        tl.store(input_grad_ptr + out_offset, input_gradT, mask=out_mask)

        grad_ptr += in_channels * D_block
        input_grad_ptr += in_channels * D_block
        input_ptr += in_channels * D_block

    tl.store(weight_up_grad_ptr + weight_up_grad_offset, weight_up_grad)
    tl.store(weight_down_grad_ptr + weight_down_grad_offset, weight_down_grad)
    tl.store(bias_grad_ptr + hidden_channels_offset, bias_grad)
