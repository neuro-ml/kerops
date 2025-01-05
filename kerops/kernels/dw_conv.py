import triton
import triton.language as tl


@triton.jit
def _DWConv_cl3d_impl(
    input_ptr,
    weight_ptr,
    output_ptr,
    H,
    W,
    D,
    H_stride,
    W_stride,
    ACCTYPE: tl.constexpr,
    channels: tl.constexpr,
    D_block: tl.constexpr,
):
    H_cell = tl.program_id(0)
    W_cell = tl.program_id(1)
    D_cell = tl.program_id(2)

    output_ptr += D_cell * D_block * channels
    input_ptr += D_cell * D_block * channels

    channels_offset = tl.arange(0, channels)
    channels_offset = tl.max_contiguous(tl.multiple_of(channels_offset, channels), channels)
    d_offset = tl.arange(0, D_block)
    near_offset = tl.arange(0, 4) - 1

    offset = d_offset[:, None, None] * channels + channels_offset[None, :, None] + near_offset[None, None, :] * channels
    mask = d_offset[:, None, None] + near_offset[None, None, :] < D - D_block * D_cell
    mask = mask and (d_offset[:, None, None] + near_offset[None, None, :] >= 0 - D_block * D_cell)
    mask = mask and (near_offset[None, None, :] != 2)

    weight_offset = channels_offset[None, :, None] + tl.arange(0, 4)[None, None, :] * channels
    weight_mask = tl.arange(0, 4)[None, None, :] != 3

    weight_h0_w0 = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
    weight_h0_w1 = tl.load((weight_ptr + 3 * channels) + weight_offset, mask=weight_mask, other=0.0)
    weight_h0_w2 = tl.load((weight_ptr + 6 * channels) + weight_offset, mask=weight_mask, other=0.0)
    weight_h1_w0 = tl.load((weight_ptr + 9 * channels) + weight_offset, mask=weight_mask, other=0.0)
    weight_h1_w1 = tl.load((weight_ptr + 12 * channels) + weight_offset, mask=weight_mask, other=0.0)
    weight_h1_w2 = tl.load((weight_ptr + 15 * channels) + weight_offset, mask=weight_mask, other=0.0)
    weight_h2_w0 = tl.load((weight_ptr + 18 * channels) + weight_offset, mask=weight_mask, other=0.0)
    weight_h2_w1 = tl.load((weight_ptr + 21 * channels) + weight_offset, mask=weight_mask, other=0.0)
    weight_h2_w2 = tl.load((weight_ptr + 24 * channels) + weight_offset, mask=weight_mask, other=0.0)

    h0_w0 = tl.zeros([D_block, channels], dtype=ACCTYPE)
    h0_w1 = tl.zeros([D_block, channels], dtype=ACCTYPE)
    h1_w0 = tl.zeros([D_block, channels], dtype=ACCTYPE)
    h1_w1 = tl.zeros([D_block, channels], dtype=ACCTYPE)

    out_mask = d_offset[:, None] < D - D_block * D_cell
    out_offset = d_offset[:, None] * channels + channels_offset[None, :]

    H1_store = 2 * H_cell + 1 < H
    W1_store = 2 * W_cell + 1 < W

    load_all = (H_cell > 0 and H_cell < tl.cdiv(H, 2) - 1) and (W_cell > 0 and W_cell < tl.cdiv(W, 2) - 1)

    i = -1
    j = -1
    load_next = (2 * H_cell + i < H and 2 * H_cell + i >= 0) and (2 * W_cell + j < W and 2 * W_cell + j >= 0)
    tmp_input_ptr = input_ptr + (2 * H_cell + i) * H_stride + (2 * W_cell + j) * W_stride

    x = tl.load(tmp_input_ptr + offset, mask=(load_all or load_next) and mask)

    for k in tl.static_range(0, 16):
        if k == 0:
            h0_w0 += tl.sum(x * weight_h0_w0, axis=2)
        elif k == 1:
            h0_w0 += tl.sum(x * weight_h1_w0, axis=2)
            h1_w0 += tl.sum(x * weight_h0_w0, axis=2)
        elif k == 2:
            h0_w0 += tl.sum(x * weight_h2_w0, axis=2)
            h1_w0 += tl.sum(x * weight_h1_w0, axis=2)
        elif k == 3:
            h1_w0 += tl.sum(x * weight_h2_w0, axis=2)
        elif k == 4:
            h0_w0 += tl.sum(x * weight_h0_w1, axis=2)
            h0_w1 += tl.sum(x * weight_h0_w0, axis=2)
        elif k == 5:
            h0_w0 += tl.sum(x * weight_h1_w1, axis=2)
            h0_w1 += tl.sum(x * weight_h1_w0, axis=2)
            h1_w0 += tl.sum(x * weight_h0_w1, axis=2)
            h1_w1 += tl.sum(x * weight_h0_w0, axis=2)
        elif k == 6:
            h0_w0 += tl.sum(x * weight_h2_w1, axis=2)
            h0_w1 += tl.sum(x * weight_h2_w0, axis=2)
            h1_w0 += tl.sum(x * weight_h1_w1, axis=2)
            h1_w1 += tl.sum(x * weight_h1_w0, axis=2)
        elif k == 7:
            h1_w0 += tl.sum(x * weight_h2_w1, axis=2)
            h1_w1 += tl.sum(x * weight_h2_w0, axis=2)
        elif k == 8:
            h0_w0 += tl.sum(x * weight_h0_w2, axis=2)
            h0_w1 += tl.sum(x * weight_h0_w1, axis=2)
        elif k == 9:
            h0_w0 += tl.sum(x * weight_h1_w2, axis=2)
            h0_w1 += tl.sum(x * weight_h1_w1, axis=2)
            h1_w0 += tl.sum(x * weight_h0_w2, axis=2)
            h1_w1 += tl.sum(x * weight_h0_w1, axis=2)
        elif k == 10:
            h0_w0 += tl.sum(x * weight_h2_w2, axis=2)
            h0_w1 += tl.sum(x * weight_h2_w1, axis=2)
            h1_w0 += tl.sum(x * weight_h1_w2, axis=2)
            h1_w1 += tl.sum(x * weight_h1_w1, axis=2)
        elif k == 11:
            h1_w0 += tl.sum(x * weight_h2_w2, axis=2)
            h1_w1 += tl.sum(x * weight_h2_w1, axis=2)
        elif k == 12:
            h0_w1 += tl.sum(x * weight_h0_w2, axis=2)
        elif k == 13:
            h0_w1 += tl.sum(x * weight_h1_w2, axis=2)
            h1_w1 += tl.sum(x * weight_h0_w2, axis=2)
        elif k == 14:
            h0_w1 += tl.sum(x * weight_h2_w2, axis=2)
            h1_w1 += tl.sum(x * weight_h1_w2, axis=2)
        else:
            h1_w1 += tl.sum(x * weight_h2_w2, axis=2)

        k_ = k + 1
        i = (k_ % 4) - 1
        j = (k_ // 4) - 1

        load_next = (2 * H_cell + i < H and 2 * H_cell + i >= 0) and (2 * W_cell + j < W and 2 * W_cell + j >= 0)
        tmp_input_ptr = input_ptr + (2 * H_cell + i) * H_stride + (2 * W_cell + j) * W_stride

        x = tl.load(tmp_input_ptr + offset, mask=(load_all or load_next) and mask)

    tmp_output_ptr = output_ptr + (2 * H_cell) * H_stride + (2 * W_cell) * W_stride
    tl.store(tmp_output_ptr + out_offset, h0_w0, mask=out_mask)

    tmp_output_ptr = output_ptr + (2 * H_cell) * H_stride + (2 * W_cell + 1) * W_stride
    tl.store(tmp_output_ptr + out_offset, h0_w1, mask=out_mask and W1_store)

    tmp_output_ptr = output_ptr + (2 * H_cell + 1) * H_stride + (2 * W_cell) * W_stride
    tl.store(tmp_output_ptr + out_offset, h1_w0, mask=out_mask and H1_store)

    tmp_output_ptr = output_ptr + (2 * H_cell + 1) * H_stride + (2 * W_cell + 1) * W_stride
    tl.store(tmp_output_ptr + out_offset, h1_w1, mask=out_mask and (H1_store and W1_store))


# TODO: single kernel for both grad_X and grad_W
@triton.jit
def _DWConv_wgrad_cl3d_impl(
    grad_ptr,
    input_ptr,
    weight_grad_ptr,
    H,
    W,
    D,
    H_stride,
    W_stride,
    ACCTYPE: tl.constexpr,
    channels: tl.constexpr,
    D_block: tl.constexpr,
    WD_grid,
    D_grid,
    delta_H_grid,
    ILP: tl.constexpr,
):
    H_cell = tl.program_id(0)
    W_cell = tl.program_id(1)
    D_cell = tl.program_id(2)

    input_ptr += D_cell * D_block * channels
    grad_ptr += D_cell * D_block * channels
    weight_grad_ptr += (H_cell * WD_grid + W_cell * D_grid + D_cell) * 27 * channels

    channels_offset = tl.arange(0, channels)
    channels_offset = tl.max_contiguous(tl.multiple_of(channels_offset, channels), channels)
    d_offset = tl.arange(0, D_block)
    near_offset = tl.arange(0, 4) - 1

    offset = d_offset[None, None, :] * channels + channels_offset[None, :, None] + near_offset[:, None, None] * channels
    mask = d_offset[None, None, :] + near_offset[:, None, None] < D - D_block * D_cell
    mask = mask and (d_offset[None, None, :] + near_offset[:, None, None] >= 0 - D_block * D_cell)
    mask = mask and (near_offset[:, None, None] != 2)

    in_offset = d_offset[None, :] * channels + channels_offset[:, None]
    in_mask = d_offset[None, :] < D - D_block * D_cell

    h0_w0 = tl.zeros([4, channels], dtype=ACCTYPE)
    h0_w1 = tl.zeros([4, channels], dtype=ACCTYPE)
    h0_w2 = tl.zeros([4, channels], dtype=ACCTYPE)
    h1_w0 = tl.zeros([4, channels], dtype=ACCTYPE)
    h1_w1 = tl.zeros([4, channels], dtype=ACCTYPE)
    h1_w2 = tl.zeros([4, channels], dtype=ACCTYPE)
    h2_w0 = tl.zeros([4, channels], dtype=ACCTYPE)
    h2_w1 = tl.zeros([4, channels], dtype=ACCTYPE)
    h2_w2 = tl.zeros([4, channels], dtype=ACCTYPE)

    gradw_offset = tl.arange(0, 4)[:, None] * channels + channels_offset[None, :]
    gradw_mask = near_offset[:, None] != 2

    for ilp in tl.static_range(0, ILP):
        H0_load = 2 * H_cell < H
        H1_load = 2 * H_cell + 1 < H
        W1_load = 2 * W_cell + 1 < W
        
        tmp_input_ptr = input_ptr + 2 * H_cell * H_stride + 2 * W_cell * W_stride
        x_h0_w0 = tl.load(tmp_input_ptr + offset, mask=mask and H0_load)
        
        tmp_input_ptr = input_ptr + (2 * H_cell + 1) * H_stride + 2 * W_cell * W_stride
        x_h1_w0 = tl.load(tmp_input_ptr + offset, mask=mask and H1_load)
        
        tmp_input_ptr = input_ptr + 2 * H_cell * H_stride + (2 * W_cell + 1) * W_stride
        x_h0_w1 = tl.load(tmp_input_ptr + offset, mask=mask and (W1_load and H0_load))
        
        tmp_input_ptr = input_ptr + (2 * H_cell + 1) * H_stride + (2 * W_cell + 1) * W_stride
        x_h1_w1 = tl.load(tmp_input_ptr + offset, mask=mask and (W1_load and H1_load))

        #grad = tl.zeros([channels, D_block], dtype=tl.float16)[None]
    
        for k in tl.static_range(0, 16):
            i = (k % 4) - 1
            j = (k // 4) - 1
            load_next = (2 * H_cell + i < H and 2 * H_cell + i >= 0) and (2 * W_cell + j < W and 2 * W_cell + j >= 0)
            tmp_grad_ptr = grad_ptr + (2 * H_cell + i) * H_stride + (2 * W_cell + j) * W_stride                

            if load_next:
                grad = tl.load(tmp_grad_ptr + in_offset, mask=in_mask, other=0.)[None]
                
                if i == -1 and j == -1:
                    h2_w2 += tl.sum(grad * x_h0_w0, axis=2)
                elif i == -1 and j == 0:
                    h2_w1 += tl.sum(grad * x_h0_w0, axis=2)
                    h2_w2 += tl.sum(grad * x_h0_w1, axis=2)
                elif i == -1 and j == 1:
                    h2_w0 += tl.sum(grad * x_h0_w0, axis=2)
                    h2_w1 += tl.sum(grad * x_h0_w1, axis=2)
                elif i == -1 and j == 2:
                    h2_w0 += tl.sum(grad * x_h0_w1, axis=2)
                elif i == 0 and j == -1:
                    h1_w2 += tl.sum(grad * x_h0_w0, axis=2)
                    h2_w2 += tl.sum(grad * x_h1_w0, axis=2)
                elif i == 0 and j == 0:
                    h1_w1 += tl.sum(grad * x_h0_w0, axis=2)
                    h2_w1 += tl.sum(grad * x_h1_w0, axis=2)
                    h1_w2 += tl.sum(grad * x_h0_w1, axis=2)
                    h2_w2 += tl.sum(grad * x_h1_w1, axis=2)
                elif i == 0 and j == 1:
                    h1_w0 += tl.sum(grad * x_h0_w0, axis=2)
                    h2_w0 += tl.sum(grad * x_h1_w0, axis=2)
                    h1_w1 += tl.sum(grad * x_h0_w1, axis=2)
                    h2_w1 += tl.sum(grad * x_h1_w1, axis=2)
                elif i == 0 and j == 2:
                    h1_w0 += tl.sum(grad * x_h0_w1, axis=2)
                    h2_w0 += tl.sum(grad * x_h1_w1, axis=2)
                elif i == 1 and j == -1:
                    h0_w2 += tl.sum(grad * x_h0_w0, axis=2)
                    h1_w2 += tl.sum(grad * x_h1_w0, axis=2)
                elif i == 1 and j == 0:
                    h0_w1 += tl.sum(grad * x_h0_w0, axis=2)
                    h1_w1 += tl.sum(grad * x_h1_w0, axis=2)
                    h0_w2 += tl.sum(grad * x_h0_w1, axis=2)
                    h1_w2 += tl.sum(grad * x_h1_w1, axis=2)
                elif i == 1 and j == 1:
                    h0_w0 += tl.sum(grad * x_h0_w0, axis=2)
                    h1_w0 += tl.sum(grad * x_h1_w0, axis=2)
                    h0_w1 += tl.sum(grad * x_h0_w1, axis=2)
                    h1_w1 += tl.sum(grad * x_h1_w1, axis=2)
                elif i == 1 and j == 2:
                    h0_w0 += tl.sum(grad * x_h0_w1, axis=2)
                    h1_w0 += tl.sum(grad * x_h1_w1, axis=2)
                elif i == 2 and j == -1:
                    h0_w2 += tl.sum(grad * x_h1_w0, axis=2)
                elif i == 2 and j == 0:
                    h0_w1 += tl.sum(grad * x_h1_w0, axis=2)
                    h0_w2 += tl.sum(grad * x_h1_w1, axis=2)
                elif i == 2 and j == 1:
                    h0_w0 += tl.sum(grad * x_h1_w0, axis=2)
                    h0_w1 += tl.sum(grad * x_h1_w1, axis=2)
                else:
                    h0_w0 += tl.sum(grad * x_h1_w1, axis=2)

        H_cell += delta_H_grid

    tl.store(weight_grad_ptr + gradw_offset, h0_w0, mask=gradw_mask)
    tl.store((weight_grad_ptr + 3 * channels) + gradw_offset, h0_w1, mask=gradw_mask)
    tl.store((weight_grad_ptr + 6 * channels) + gradw_offset, h0_w2, mask=gradw_mask)
    tl.store((weight_grad_ptr + 9 * channels) + gradw_offset, h1_w0, mask=gradw_mask)
    tl.store((weight_grad_ptr + 12 * channels) + gradw_offset, h1_w1, mask=gradw_mask)
    tl.store((weight_grad_ptr + 15 * channels) + gradw_offset, h1_w2, mask=gradw_mask)
    tl.store((weight_grad_ptr + 18 * channels) + gradw_offset, h2_w0, mask=gradw_mask)
    tl.store((weight_grad_ptr + 21 * channels) + gradw_offset, h2_w1, mask=gradw_mask)
    tl.store((weight_grad_ptr + 24 * channels) + gradw_offset, h2_w2, mask=gradw_mask)
