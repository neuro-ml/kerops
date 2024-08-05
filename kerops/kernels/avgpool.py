import triton
import triton.language as tl


@triton.jit
def _AvgPoolCeilStats_cl3d_impl(
    X_ptr,
    Out_ptr,
    Mean_ptr,
    Sqmean_ptr,
    h_input,
    w_input,
    d_input,
    d_output,
    batch_stride_input,
    H_stride_input,
    W_stride_input,
    batch_stride_output,
    H_stride_output,
    W_stride_output,
    numel_no_channels_output,
    num_channels: tl.constexpr,
    almost_half_d: tl.constexpr,
):
    batch = tl.program_id(0)  # output indexing
    H = tl.program_id(1)
    W = tl.program_id(2)

    Out_ptr += batch * batch_stride_output + H * H_stride_output + W * W_stride_output

    output = tl.zeros([almost_half_d, num_channels], dtype=tl.float32)

    pair_offset = tl.arange(0, 2)
    channels_offset = tl.arange(0, num_channels)
    d_offset = tl.arange(0, almost_half_d)
    offset = (
        d_offset[:, None, None] * (2 * num_channels)
        + channels_offset[None, :, None]
        + pair_offset[None, None, :] * num_channels
    )
    output_offset = d_offset[:, None] * num_channels + channels_offset[None, :]

    mask_input = offset < d_input * num_channels
    output_mask = output_offset < d_output * num_channels
    norm_step = tl.sum(mask_input.to(tl.float32), axis=2).to(tl.float32)
    norm_step = tl.where(norm_step != 0, norm_step, 1.0)
    num_norm = 1

    # first step
    Temp_ptr = X_ptr + batch * batch_stride_input + 2 * H * H_stride_input + 2 * W * W_stride_input
    x = tl.load(Temp_ptr + offset, mask=mask_input, other=0.0).to(tl.float32)
    x = tl.sum(x, axis=2)
    output += x

    # second step
    W_skip = False
    if 2 * (W + 1) > w_input:
        W_skip = True
    else:
        Temp_ptr = X_ptr + batch * batch_stride_input + 2 * H * H_stride_input + (2 * W + 1) * W_stride_input
        x = tl.load(Temp_ptr + offset, mask=mask_input, other=0.0).to(tl.float32)
        x = tl.sum(x, axis=2)
        output += x
        num_norm += 1

    # third step
    H_skip = False
    if 2 * (H + 1) > h_input:
        H_skip = True
    else:
        Temp_ptr = X_ptr + batch * batch_stride_input + (2 * H + 1) * H_stride_input + 2 * W * W_stride_input
        x = tl.load(Temp_ptr + offset, mask=mask_input, other=0.0).to(tl.float32)
        x = tl.sum(x, axis=2)
        output += x
        num_norm += 1

    # fourth step
    if not H_skip and not W_skip:
        Temp_ptr = X_ptr + batch * batch_stride_input + (2 * H + 1) * H_stride_input + (2 * W + 1) * W_stride_input
        x = tl.load(Temp_ptr + offset, mask=mask_input, other=0.0).to(tl.float32)
        x = tl.sum(x, axis=2)
        output += x
        num_norm += 1

    # normalization step
    output = output / (norm_step * num_norm)
    tl.store(Out_ptr + output_offset, output, mask=output_mask)

    output = tl.trans(output)
    mean = tl.sum(output, axis=1) / numel_no_channels_output
    sqmean = tl.sum(output * output, axis=1) / numel_no_channels_output
    tl.atomic_add(Mean_ptr + channels_offset, mean)
    tl.atomic_add(Sqmean_ptr + channels_offset, sqmean)


@triton.jit
def _AvgPoolCeilStats_cl3d_backward_impl(
    Inpgrad_ptr,
    Outgrad_ptr,
    Output_ptr,
    Meangrad_ptr,
    Sqmeangrad_ptr,
    h_outgrad,
    w_outgrad,
    d_outgrad,
    d_inpgrad,
    batch_stride_outgrad,
    H_stride_outgrad,
    W_stride_outgrad,
    batch_stride_inpgrad,
    H_stride_inpgrad,
    W_stride_inpgrad,
    numel_no_channels_inpgrad,
    num_channels: tl.constexpr,
    almost_half_d: tl.constexpr,
):
    batch = tl.program_id(0)  # inpgrad indexing
    H = tl.program_id(1)
    W = tl.program_id(2)

    Inpgrad_ptr += batch * batch_stride_inpgrad + H * H_stride_inpgrad + W * W_stride_inpgrad
    Output_ptr += batch * batch_stride_inpgrad + H * H_stride_inpgrad + W * W_stride_inpgrad

    pair_offset = tl.arange(0, 2)
    channels_offset = tl.arange(0, num_channels)
    d_offset = tl.arange(0, almost_half_d)

    inpgrad_offset = d_offset[:, None, None] * num_channels + channels_offset[None, :, None]
    outgrad_offset = (
        d_offset[:, None, None] * (2 * num_channels)
        + channels_offset[None, :, None]
        + pair_offset[None, None, :] * num_channels
    )

    inpgrad_mask = d_offset[:, None, None] < d_inpgrad
    outgrad_mask = d_offset[:, None, None] * 2 + pair_offset[None, None, :] < d_outgrad

    inpgrad = tl.load(Inpgrad_ptr + inpgrad_offset, mask=inpgrad_mask, other=0.0)
    output = tl.load(Output_ptr + inpgrad_offset, mask=inpgrad_mask, other=0.0)

    meangrad = tl.load(Meangrad_ptr + channels_offset)[None, :, None]
    sqmeangrad = tl.load(Sqmeangrad_ptr + channels_offset)[None, :, None]

    normalizer = tl.sum(outgrad_mask.to(tl.float16), axis=2)[:, :, None].to(tl.float16)

    W_skip = False
    if 2 * (W + 1) > w_outgrad:
        W_skip = True
    else:
        normalizer *= 2

    H_skip = False
    if 2 * (H + 1) > h_outgrad:
        H_skip = True
    else:
        normalizer *= 2

    meangrad = meangrad / numel_no_channels_inpgrad
    sqmeangrad = 2 * output.to(tl.float32) * sqmeangrad / numel_no_channels_inpgrad
    grad = (inpgrad + meangrad + sqmeangrad) / normalizer

    # first
    Tmp_ptr = Outgrad_ptr + batch * batch_stride_outgrad + (2 * H) * H_stride_outgrad + (2 * W) * W_stride_outgrad
    tl.store(Tmp_ptr + outgrad_offset, grad, mask=outgrad_mask)

    # second
    if not W_skip:
        Tmp_ptr = (
            Outgrad_ptr + batch * batch_stride_outgrad + (2 * H) * H_stride_outgrad + (2 * W + 1) * W_stride_outgrad
        )
        tl.store(Tmp_ptr + outgrad_offset, grad, mask=outgrad_mask)

    # third
    if not H_skip:
        Tmp_ptr = (
            Outgrad_ptr + batch * batch_stride_outgrad + (2 * H + 1) * H_stride_outgrad + (2 * W) * W_stride_outgrad
        )
        tl.store(Tmp_ptr + outgrad_offset, grad, mask=outgrad_mask)

    # fourth
    if not H_skip and not W_skip:
        Tmp_ptr = (
            Outgrad_ptr + batch * batch_stride_outgrad + (2 * H + 1) * H_stride_outgrad + (2 * W + 1) * W_stride_outgrad
        )
        tl.store(Tmp_ptr + outgrad_offset, grad, mask=outgrad_mask)
