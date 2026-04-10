import triton
from triton import language as tl

"""
- Conv3dV2: D_cell -> W_cell -> H_cell changed to W_cell -> H_cell -> D_cell
- Conv3dV3: `tl.max_contiguous(tl.multiple_of(channels_offset, CHANNELS), CHANNELS)`
- Conv3dV4: Input channels BLOCKCING
- Conv3dV5: output blocking by HW: 2x2 tile
- Conv3dV6: loading order choice and weight-major alorithm choice

No impact:
 - `x @ w` and `w @ x` orientation via ORDER parameter
 - computations of multiple h, w in one block (WHEN THEY ARE FAR AWAY FROM EACH OTHER)
 - Boundary checks outside the loops
 - INTERIOR and BOUNDARY masking inside one kernel
 - `x: [D_BLOCK, CHANNELS] -> x: [NEAR, D_BLOCK, CHANNELS]` where NEAR=4 represents neighbours; `w: [4, CHANNELS_IN, CHANNELS_OUT]`
 - Output channels BLOCKCING (why?)
 - Output channels BLOCKCING via expanding grid
 - TMA (why???)
 - tl.swizzle2d and a X-major tile with sizes 2 and 4
"""

@triton.jit
def _Conv_cl3d_impl_V6(
    input_ptr,
    weight_ptr,
    output_ptr,
    H,
    W,
    D,
    D_BLOCK: tl.constexpr,
    ACCTYPE: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    CIN_BLOCK: tl.constexpr,
    WEIGHT_MAJOR: tl.constexpr,
    LOAD_WEIGHT_FIRST: tl.constexpr,
):
    W_cell = tl.program_id(0)
    H_cell = tl.program_id(1)
    BD_cell = tl.program_id(2)

    B_cell = BD_cell // tl.cdiv(D, D_BLOCK)
    D_cell = BD_cell % tl.cdiv(D, D_BLOCK)

    CIN_STEPS: tl.constexpr = IN_CHANNELS // CIN_BLOCK
    
    in_channels_offset = tl.arange(0, CIN_BLOCK)
    out_channels_offset = tl.arange(0, OUT_CHANNELS)
    d_offset = tl.arange(0, D_BLOCK)
    d_offset_shifted = d_offset[:, None] + D_cell * D_BLOCK

    input_offset = d_offset[:, None] * IN_CHANNELS + tl.max_contiguous(tl.multiple_of(in_channels_offset, CIN_BLOCK), CIN_BLOCK)[None, :]
    output_offset = d_offset[:, None] * OUT_CHANNELS + tl.max_contiguous(tl.multiple_of(out_channels_offset, OUT_CHANNELS), OUT_CHANNELS)[None, :]
    weight_offset = in_channels_offset[:, None] * OUT_CHANNELS + tl.max_contiguous(tl.multiple_of(out_channels_offset, OUT_CHANNELS), OUT_CHANNELS)[None, :]

    input_ptr += D_cell * D_BLOCK * IN_CHANNELS
    input_ptr += W_cell * 2 * IN_CHANNELS * D
    input_ptr += H_cell * 2 * IN_CHANNELS * D * W
    input_ptr += B_cell * H * W * D * IN_CHANNELS

    output_ptr += D_cell * D_BLOCK * OUT_CHANNELS
    output_ptr += W_cell * 2 * OUT_CHANNELS * D
    output_ptr += H_cell * 2 * OUT_CHANNELS * D * W
    output_ptr += B_cell * H * W * D * OUT_CHANNELS

    acc00 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc01 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc10 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc11 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)

    if WEIGHT_MAJOR:
        for weight_h in tl.static_range(0, 3):
            for weight_w in tl.static_range(0, 3):
                for cin in tl.static_range(0, CIN_STEPS):
                    for weight_d in tl.static_range(0, 3):
                        w_ptr = (
                            weight_ptr
                            + weight_d * IN_CHANNELS * OUT_CHANNELS
                            + weight_w * IN_CHANNELS * OUT_CHANNELS * 3
                            + weight_h * IN_CHANNELS * OUT_CHANNELS * 9
                            + cin * CIN_BLOCK * OUT_CHANNELS
                        )

                        if LOAD_WEIGHT_FIRST:
                            weight = tl.load(w_ptr + weight_offset)

                        i_ptr = (
                            input_ptr
                            + (weight_h - 1) * IN_CHANNELS * D * W
                            + (weight_w - 1) * IN_CHANNELS * D
                            + (weight_d - 1) * IN_CHANNELS
                            + cin * CIN_BLOCK
                        )

                        mask = ((d_offset_shifted + weight_d - 1) < D) & ((d_offset_shifted + weight_d - 1) >= 0)

                        m00 = ((H_cell * 2 + weight_h - 1) < H) & ((H_cell * 2 + weight_h - 1) >= 0) & ((W_cell * 2 + weight_w - 1) < W) & ((W_cell * 2 + weight_w - 1) >= 0)
                        m01 = ((H_cell * 2 + weight_h - 1) < H) & ((H_cell * 2 + weight_h - 1) >= 0) & ((W_cell * 2 + weight_w) < W) & ((W_cell * 2 + weight_w) >= 0)
                        m10 = ((H_cell * 2 + weight_h) < H) & ((H_cell * 2 + weight_h) >= 0) & ((W_cell * 2 + weight_w - 1) < W) & ((W_cell * 2 + weight_w - 1) >= 0)
                        m11 = ((H_cell * 2 + weight_h) < H) & ((H_cell * 2 + weight_h) >= 0) & ((W_cell * 2 + weight_w) < W) & ((W_cell * 2 + weight_w) >= 0)

                        xs = [
                            [
                                tl.load(i_ptr + input_offset, mask=mask & m00, other=0.0),
                                tl.load(i_ptr + input_offset + IN_CHANNELS * D, mask=mask & m01, other=0.0)
                            ],
                            [
                                tl.load(i_ptr + input_offset + IN_CHANNELS * D * W, mask=mask & m10, other=0.0),
                                tl.load(i_ptr + input_offset + IN_CHANNELS * D + IN_CHANNELS * D * W, mask=mask & m11, other=0.0)
                            ]
                        ]

                        if not LOAD_WEIGHT_FIRST:
                            weight = tl.load(w_ptr + weight_offset)

                        acc00 += tl.dot(xs[0][0], weight)
                        acc01 += tl.dot(xs[0][1], weight)
                        acc10 += tl.dot(xs[1][0], weight)
                        acc11 += tl.dot(xs[1][1], weight)
    else:
        for h_block in tl.static_range(0, 2):
            for w_block in tl.static_range(0, 2):
                for cin in tl.static_range(0, CIN_STEPS):
                    for dd in tl.static_range(-1, 2):  # MB other order?
                        w_ptr = (
                            weight_ptr
                            + (dd + 1) * IN_CHANNELS * OUT_CHANNELS
                            + w_block * IN_CHANNELS * OUT_CHANNELS * 3
                            + h_block * IN_CHANNELS * OUT_CHANNELS * 9
                            + cin * CIN_BLOCK * OUT_CHANNELS
                        )

                        if LOAD_WEIGHT_FIRST:
                            weights = [
                                [
                                    tl.load(w_ptr + weight_offset),
                                    tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 3),
                                ],
                                [
                                    tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 9),
                                    tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 12),
                                ]
                            ]
                        
                        i_ptr = (
                            input_ptr
                            + (h_block * 2 - 1) * IN_CHANNELS * D * W
                            + (w_block * 2 - 1) * IN_CHANNELS * D
                            + dd * IN_CHANNELS
                            + cin * CIN_BLOCK
                        )
                        mask = ((d_offset_shifted + dd) < D) & ((d_offset_shifted + dd) >= 0)

                        m00 = ((H_cell * 2 + h_block * 2 - 1) < H) & ((H_cell * 2 + h_block * 2 - 1) >= 0) & ((W_cell * 2 + w_block * 2 - 1) < W) & ((W_cell * 2 + w_block * 2 - 1) >= 0)
                        m01 = ((H_cell * 2 + h_block * 2 - 1) < H) & ((H_cell * 2 + h_block * 2 - 1) >= 0) & ((W_cell * 2 + w_block * 2) < W) & ((W_cell * 2 + w_block * 2) >= 0)
                        m10 = ((H_cell * 2 + h_block * 2) < H) & ((H_cell * 2 + h_block * 2) >= 0) & ((W_cell * 2 + w_block * 2 - 1) < W) & ((W_cell * 2 + w_block * 2 - 1) >= 0)
                        m11 = ((H_cell * 2 + h_block * 2) < H) & ((H_cell * 2 + h_block * 2) >= 0) & ((W_cell * 2 + w_block * 2) < W) & ((W_cell * 2 + w_block * 2) >= 0)
                        
                        xs = [
                            [
                                tl.load(i_ptr + input_offset, mask=mask & m00, other=0.0),
                                tl.load(i_ptr + input_offset + IN_CHANNELS * D, mask=mask & m01, other=0.0)
                            ],
                            [
                                tl.load(i_ptr + input_offset + IN_CHANNELS * D * W, mask=mask & m10, other=0.0),
                                tl.load(i_ptr + input_offset + IN_CHANNELS * D + IN_CHANNELS * D * W, mask=mask & m11, other=0.0)
                            ]
                        ]

                        if not LOAD_WEIGHT_FIRST:
                            weights = [
                                [
                                    tl.load(w_ptr + weight_offset),
                                    tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 3),
                                ],
                                [
                                    tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 9),
                                    tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 12),
                                ]
                            ]

                        for h in tl.static_range(0, 2):
                            for w in tl.static_range(0, 2):
                                # h_weight_idx = 2 * h_block + h - acc_abs_h + 1 - h_block <-- weights window shift
                                #                <---x_h------->           ^-- +1 since weight indexed from 0
                                
                                # acc00
                                if ((h_block * 2 + h) < 3) & ((w_block * 2 + w) < 3):
                                    acc00 += tl.dot(xs[h][w], weights[h_block + h][w_block + w])

                                # acc01
                                if ((h_block * 2 + h) < 3) & ((w_block * 2 + w) >  0):
                                    acc01 += tl.dot(xs[h][w], weights[h_block + h][w_block + w - 1])

                                # acc10
                                if ((h_block * 2 + h) > 0) & ((w_block * 2 + w) <  3):
                                    acc10 += tl.dot(xs[h][w], weights[h_block + h - 1][w_block + w])

                                # acc11
                                if ((h_block * 2 + h) > 0) & ((w_block * 2 + w) > 0):
                                    acc11 += tl.dot(xs[h][w], weights[h_block + h - 1][w_block + w - 1])

    omask = d_offset_shifted < D
    tl.store(output_ptr + output_offset, acc00, mask=omask & ((W_cell * 2) < W) & ((H_cell * 2) < H))
    tl.store(output_ptr + output_offset + OUT_CHANNELS * D, acc01, mask=omask & ((W_cell * 2 + 1) < W) & ((H_cell * 2) < H))
    tl.store(output_ptr + output_offset + OUT_CHANNELS * D * W, acc10, mask=omask & ((W_cell * 2) < W) & ((H_cell * 2 + 1) < H))
    tl.store(output_ptr + output_offset + OUT_CHANNELS * D * W + OUT_CHANNELS * D, acc11, mask=omask & ((W_cell * 2 + 1) < W) & ((H_cell * 2 + 1) < H))


"""
- Conv3dV2: grad and x tiling added
"""

@triton.jit
def _Conv_wgrad_cl3d_impl_V2(
    grad_ptr,
    input_ptr,
    weight_grad_ptr,
    H,
    W,
    D,
    num_buffers,
    ACCTYPE: tl.constexpr,
    D_BLOCK: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    CIN_BLOCK: tl.constexpr,
    COUT_BLOCK: tl.constexpr,
):
    WCOUT_pid = tl.program_id(0)
    H_pid = tl.program_id(1)
    BD_pid = tl.program_id(2)

    W_pid = WCOUT_pid // tl.cdiv(OUT_CHANNELS, COUT_BLOCK)
    COUT_pid = WCOUT_pid % tl.cdiv(OUT_CHANNELS, COUT_BLOCK)

    B_pid = BD_pid // tl.cdiv(D, D_BLOCK)
    D_pid = BD_pid % tl.cdiv(D, D_BLOCK)

    CIN_STEPS: tl.constexpr = IN_CHANNELS // CIN_BLOCK

    linear = WCOUT_pid + H_pid * tl.num_programs(0) + BD_pid * tl.num_programs(0) * tl.num_programs(1)
    buffer_idx = linear % num_buffers
    
    in_channels_offset = tl.arange(0, CIN_BLOCK)
    out_channels_offset = tl.arange(0, COUT_BLOCK)
    d_offset = tl.arange(0, D_BLOCK)

    grad_offset = d_offset[:, None] * OUT_CHANNELS + out_channels_offset[None, :]
    input_offset = d_offset[None, :] * IN_CHANNELS + in_channels_offset[:, None]
    weight_grad_offset = in_channels_offset[:, None] * OUT_CHANNELS + out_channels_offset[None, :]

    grad_ptr += COUT_pid * COUT_BLOCK
    grad_ptr += D_pid * D_BLOCK * OUT_CHANNELS
    grad_ptr += W_pid * 2 * D * OUT_CHANNELS
    grad_ptr += H_pid * 2 * D * W * OUT_CHANNELS
    grad_ptr += B_pid * H * W * D * OUT_CHANNELS

    input_ptr += D_pid * D_BLOCK * IN_CHANNELS
    input_ptr += W_pid * 2 * D * IN_CHANNELS
    input_ptr += H_pid * 2 * D * W * IN_CHANNELS
    input_ptr += B_pid * H * W * D * IN_CHANNELS

    weight_grad_ptr += COUT_pid * COUT_BLOCK

    g01 = ((W_pid * 2 + 1) < W)
    g10 = ((H_pid * 2 + 1) < H)
    g11 = ((W_pid * 2 + 1) < W) & ((H_pid * 2 + 1) < H)

    gmask = (d_offset < (D - D_pid * D_BLOCK))
    gmask = gmask[:, None]

    grads = [
        [
            tl.load(grad_ptr + grad_offset, other=0, mask=gmask),
            tl.load(grad_ptr + grad_offset + OUT_CHANNELS * D, other=0, mask=gmask & g01),
        ],
        [
            tl.load(grad_ptr + grad_offset + OUT_CHANNELS * D * W, other=0, mask=gmask & g10),
            tl.load(grad_ptr + grad_offset + OUT_CHANNELS * D * W + OUT_CHANNELS * D, other=0, mask=gmask & g11)
        ]
    ]

    for h in tl.static_range(-1, 2):
        for w in tl.static_range(-1, 2):
            for d in tl.static_range(-1, 2):
                for cin in tl.static_range(0, CIN_STEPS):
                    wgrad = tl.zeros([CIN_BLOCK, COUT_BLOCK], dtype=ACCTYPE)
                    
                    x_ptr = (
                        input_ptr
                        + h * IN_CHANNELS * D * W
                        + w * IN_CHANNELS * D
                        + d * IN_CHANNELS
                        + cin * CIN_BLOCK
                    )
    
                    xmask = (d_offset < (D - D_pid * D_BLOCK - d)) & (d_offset >= (- D_pid * D_BLOCK - d))
                    xmask = xmask[None, :]

                    x00 = ((H_pid * 2 + h) < H) & ((H_pid * 2 + h) >= 0) & ((W_pid * 2 + w) < W) & ((W_pid * 2 + w) >= 0)
                    x01 = ((H_pid * 2 + h) < H) & ((H_pid * 2 + h) >= 0) & ((W_pid * 2 + w + 1) < W) & ((W_pid * 2 + w + 1) >= 0)
                    x10 = ((H_pid * 2 + h + 1) < H) & ((H_pid * 2 + h + 1) >= 0) & ((W_pid * 2 + w) < W) & ((W_pid * 2 + w) >= 0)
                    x11 = ((H_pid * 2 + h + 1) < H) & ((H_pid * 2 + h + 1) >= 0) & ((W_pid * 2 + w + 1) < W) & ((W_pid * 2 + w + 1) >= 0)

                    xs = [
                        [
                            tl.load(x_ptr + input_offset, other=0, mask=xmask & x00),
                            tl.load(x_ptr + input_offset + IN_CHANNELS * D, other=0, mask=xmask & x01),
                        ],
                        [
                            tl.load(x_ptr + input_offset + IN_CHANNELS * D * W, other=0, mask=xmask & x10),
                            tl.load(x_ptr + input_offset + IN_CHANNELS * D * W + IN_CHANNELS * D, other=0, mask=xmask & x11),
                        ]
                    ]
    
                    for kh in tl.static_range(0, 2):
                        for kw in tl.static_range(0, 2):
                            wgrad += tl.dot(xs[kh][kw], grads[kh][kw])

    
                    w_ptr = (
                        weight_grad_ptr
                        + buffer_idx * IN_CHANNELS * OUT_CHANNELS * 3 * 3 * 3
                        + (h + 1) * IN_CHANNELS * OUT_CHANNELS * 3 * 3
                        + (w + 1) * IN_CHANNELS * OUT_CHANNELS * 3
                        + (d + 1) * IN_CHANNELS * OUT_CHANNELS
                        + cin * CIN_BLOCK * OUT_CHANNELS
                    )
                    tl.atomic_add(w_ptr + weight_grad_offset, wgrad, sem='relaxed')


@triton.jit
def make_offset(h_str, w_str, d_str, H_BLOCK: tl.constexpr, W_BLOCK: tl.constexpr, D_BLOCK: tl.constexpr):
    d_off = tl.arange(0, D_BLOCK)
    w_off = tl.arange(0, W_BLOCK)
    h_off = tl.arange(0, H_BLOCK)

    offset = h_off[:, None, None] * h_str + w_off[None, :, None] * w_str + d_off[None, None, :] * d_str
    offset = offset.reshape((H_BLOCK * W_BLOCK * D_BLOCK))

    return offset


@triton.jit
def make_mask(curr_h, curr_w, curr_d, H, W, D, H_BLOCK: tl.constexpr, W_BLOCK: tl.constexpr, D_BLOCK: tl.constexpr):
    mask_d = ((tl.arange(0, D_BLOCK) + curr_d) >= 0) & ((tl.arange(0, D_BLOCK) + curr_d) < D)
    mask_w = ((tl.arange(0, W_BLOCK) + curr_w) >= 0) & ((tl.arange(0, W_BLOCK) + curr_w) < W)
    mask_h = ((tl.arange(0, H_BLOCK) + curr_h) >= 0) & ((tl.arange(0, H_BLOCK) + curr_h) < H)

    mask = mask_h[:, None, None] & mask_w[None, :, None] & mask_d[None, None, :]
    mask = mask.reshape((H_BLOCK * W_BLOCK * D_BLOCK))

    return mask


@triton.jit
def _Conv_wgrad_cl3d_splitk_impl(
    grad_ptr,
    input_ptr,
    weight_grad_ptr,
    H,
    W,
    D,
    ACCTYPE: tl.constexpr,
    H_BLOCK: tl.constexpr, W_BLOCK: tl.constexpr, D_BLOCK: tl.constexpr,
    IN_CHANNELS, OUT_CHANNELS,
    CIN_BLOCK: tl.constexpr, COUT_BLOCK: tl.constexpr,
    SPLIT_K,
):
    khwd_pid = tl.program_id(0)
    cin_pid = tl.program_id(1)
    cout_batch_pid = tl.program_id(2)

    k_pid = khwd_pid // 27
    hwd_pid = khwd_pid % 27

    block_d = hwd_pid % 3
    hwd_pid = hwd_pid // 3
    block_w = hwd_pid % 3
    block_h = hwd_pid // 3

    cout_pid = cout_batch_pid % tl.cdiv(OUT_CHANNELS, COUT_BLOCK)
    batch_pid = cout_batch_pid // tl.cdiv(OUT_CHANNELS, COUT_BLOCK)

    cin_offset = tl.arange(0, CIN_BLOCK)
    cout_offset = tl.arange(0, COUT_BLOCK)
    geom_offset = make_offset(W * D, D, 1, H_BLOCK, W_BLOCK, D_BLOCK)

    grad_offset = geom_offset[:, None] * OUT_CHANNELS + cout_offset[None, :]
    input_offset = geom_offset[None, :] * IN_CHANNELS + cin_offset[:, None]
    weight_grad_offset = cin_offset[:, None] * OUT_CHANNELS + cout_offset[None, :]

    grad_ptr += cout_pid * COUT_BLOCK
    grad_ptr += k_pid * H_BLOCK * OUT_CHANNELS * D * W
    grad_ptr += batch_pid * H * W * D * OUT_CHANNELS
    input_ptr += cin_pid * CIN_BLOCK
    input_ptr += (block_d - 1) * IN_CHANNELS
    input_ptr += k_pid * H_BLOCK * IN_CHANNELS * D * W
    input_ptr += (block_w - 1) * IN_CHANNELS * D
    input_ptr += (block_h - 1) * IN_CHANNELS * D * W
    input_ptr += batch_pid * H * W * D * IN_CHANNELS
    weight_grad_ptr += cin_pid * CIN_BLOCK * OUT_CHANNELS + cout_pid * COUT_BLOCK
    weight_grad_ptr += block_d * IN_CHANNELS * OUT_CHANNELS
    weight_grad_ptr += block_w * IN_CHANNELS * OUT_CHANNELS * 3
    weight_grad_ptr += block_h * IN_CHANNELS * OUT_CHANNELS * 9
    weight_grad_ptr += batch_pid * IN_CHANNELS * OUT_CHANNELS * 27

    weight_grad = tl.zeros((CIN_BLOCK, COUT_BLOCK), dtype=ACCTYPE)

    for grad_h in range(0, tl.cdiv(H, H_BLOCK * SPLIT_K)):
        if grad_h * H_BLOCK * SPLIT_K + k_pid * H_BLOCK < H:
            for grad_w in range(0, tl.cdiv(W, W_BLOCK)):
                for grad_d in range(0, tl.cdiv(D, D_BLOCK)):
                    grad_mask = make_mask(grad_h * H_BLOCK * SPLIT_K + k_pid * H_BLOCK, grad_w * W_BLOCK, grad_d * D_BLOCK, H, W, D, H_BLOCK, W_BLOCK, D_BLOCK)
                    grad_iter_ptr = (
                        grad_ptr
                        + grad_h * H_BLOCK * W * D * OUT_CHANNELS * SPLIT_K
                        + grad_w * W_BLOCK * D * OUT_CHANNELS
                        + grad_d * D_BLOCK * OUT_CHANNELS
                    )
                    grad = tl.load(grad_iter_ptr + grad_offset, mask=grad_mask[:, None], other=0)

                    x_mask = make_mask(grad_h * H_BLOCK * SPLIT_K + k_pid * H_BLOCK + block_h - 1, grad_w * W_BLOCK + block_w - 1, grad_d * D_BLOCK + block_d - 1, H, W, D, H_BLOCK, W_BLOCK, D_BLOCK)
                    x_iter_ptr = (
                        input_ptr
                        + grad_h * H_BLOCK * W * D * IN_CHANNELS * SPLIT_K
                        + grad_w * W_BLOCK * D * IN_CHANNELS
                        + grad_d * D_BLOCK * IN_CHANNELS
                    )
                    x = tl.load(x_iter_ptr + input_offset, mask=x_mask[None, :], other=0)

                    weight_grad += tl.dot(x, grad)

    tl.atomic_add(weight_grad_ptr + weight_grad_offset, weight_grad, sem='relaxed')


@triton.jit
def _ApplyBNReLUConvStats_cl3d_impl(
    input_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    weight_ptr,
    output_ptr,
    mean_ptr,
    sqmean_ptr,
    H,
    W,
    D,
    D_BLOCK: tl.constexpr,
    ACCTYPE: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    CIN_BLOCK: tl.constexpr,
):
    W_cell = tl.program_id(0)
    H_cell = tl.program_id(1)
    D_cell = tl.program_id(2)

    CIN_STEPS: tl.constexpr = IN_CHANNELS // CIN_BLOCK
    
    in_channels_offset = tl.arange(0, CIN_BLOCK)
    out_channels_offset = tl.arange(0, OUT_CHANNELS)
    d_offset = tl.arange(0, D_BLOCK)
    d_offset_shifted = d_offset[:, None] + D_cell * D_BLOCK

    input_offset = d_offset[:, None] * IN_CHANNELS + tl.max_contiguous(tl.multiple_of(in_channels_offset, CIN_BLOCK), CIN_BLOCK)[None, :]
    output_offset = d_offset[:, None] * OUT_CHANNELS + tl.max_contiguous(tl.multiple_of(out_channels_offset, OUT_CHANNELS), OUT_CHANNELS)[None, :]
    weight_offset = in_channels_offset[:, None] * OUT_CHANNELS + tl.max_contiguous(tl.multiple_of(out_channels_offset, OUT_CHANNELS), OUT_CHANNELS)[None, :]

    input_ptr += D_cell * D_BLOCK * IN_CHANNELS
    input_ptr += W_cell * 2 * IN_CHANNELS * D
    input_ptr += H_cell * 2 * IN_CHANNELS * D * W

    output_ptr += D_cell * D_BLOCK * OUT_CHANNELS
    output_ptr += W_cell * 2 * OUT_CHANNELS * D
    output_ptr += H_cell * 2 * OUT_CHANNELS * D * W

    mean_ptr += (W_cell + tl.num_programs(0) * H_cell + tl.num_programs(0) * tl.num_programs(1) * D_cell) * OUT_CHANNELS
    sqmean_ptr += (W_cell + tl.num_programs(0) * H_cell + tl.num_programs(0) * tl.num_programs(1) * D_cell) * OUT_CHANNELS

    acc00 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc01 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc10 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc11 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)

    mean = tl.zeros([OUT_CHANNELS], dtype=ACCTYPE)
    sqmean = tl.zeros([OUT_CHANNELS], dtype=ACCTYPE)

    zero = tl.zeros([1], dtype=tl.float16)

    for h_block in tl.static_range(0, 2):
        for w_block in tl.static_range(0, 2):
            for cin in tl.static_range(0, CIN_STEPS):
                bn_weight = tl.load(bn_weight_ptr + in_channels_offset + cin * CIN_BLOCK)[None, :]
                bn_bias = tl.load(bn_bias_ptr + in_channels_offset + cin * CIN_BLOCK)[None, :]

                for dd in tl.static_range(-1, 2):  # MB other order?
                    w_ptr = (
                        weight_ptr
                        + (dd + 1) * IN_CHANNELS * OUT_CHANNELS
                        + w_block * IN_CHANNELS * OUT_CHANNELS * 3
                        + h_block * IN_CHANNELS * OUT_CHANNELS * 9
                        + cin * CIN_BLOCK * OUT_CHANNELS
                    )

                    weights = [
                        [
                            tl.load(w_ptr + weight_offset),
                            tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 3),
                        ],
                        [
                            tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 9),
                            tl.load(w_ptr + weight_offset + IN_CHANNELS * OUT_CHANNELS * 12),
                        ]
                    ]
                    
                    i_ptr = (
                        input_ptr
                        + (h_block * 2 - 1) * IN_CHANNELS * D * W
                        + (w_block * 2 - 1) * IN_CHANNELS * D
                        + dd * IN_CHANNELS
                        + cin * CIN_BLOCK
                    )
                    mask = ((d_offset_shifted + dd) < D) & ((d_offset_shifted + dd) >= 0)

                    m00 = ((H_cell * 2 + h_block * 2 - 1) < H) & ((H_cell * 2 + h_block * 2 - 1) >= 0) & ((W_cell * 2 + w_block * 2 - 1) < W) & ((W_cell * 2 + w_block * 2 - 1) >= 0)
                    m01 = ((H_cell * 2 + h_block * 2 - 1) < H) & ((H_cell * 2 + h_block * 2 - 1) >= 0) & ((W_cell * 2 + w_block * 2) < W) & ((W_cell * 2 + w_block * 2) >= 0)
                    m10 = ((H_cell * 2 + h_block * 2) < H) & ((H_cell * 2 + h_block * 2) >= 0) & ((W_cell * 2 + w_block * 2 - 1) < W) & ((W_cell * 2 + w_block * 2 - 1) >= 0)
                    m11 = ((H_cell * 2 + h_block * 2) < H) & ((H_cell * 2 + h_block * 2) >= 0) & ((W_cell * 2 + w_block * 2) < W) & ((W_cell * 2 + w_block * 2) >= 0)
                    
                    # other=0.0 is NOT necessary since it is filtrated via tl.where(mask & mXX, x, zero) <-> ReLU(x)
                    xs = [
                        [
                            tl.load(i_ptr + input_offset, mask=mask & m00),
                            tl.load(i_ptr + input_offset + IN_CHANNELS * D, mask=mask & m01)
                        ],
                        [
                            tl.load(i_ptr + input_offset + IN_CHANNELS * D * W, mask=mask & m10),
                            tl.load(i_ptr + input_offset + IN_CHANNELS * D + IN_CHANNELS * D * W, mask=mask & m11)
                        ]
                    ]

                    for h in tl.static_range(0, 2):
                        for w in tl.static_range(0, 2):
                            # h_weight_idx = 2 * h_block + h - acc_abs_h + 1 - h_block <-- weights window shift
                            #                <---x_h------->           ^-- +1 since weight indexed from 0

                            if h == 0 and w == 0:
                                valid = mask & m00
                            elif h == 0 and w == 1:
                                valid = mask & m01
                            elif h == 1 and w == 0:
                                valid = mask & m10
                            else:
                                valid = mask & m11

                            x = xs[h][w].to(tl.float32)
                            x = x * bn_weight + bn_bias
                            x = x.to(tl.float16)
                            x = tl.maximum(x, zero)
                            x = tl.where(valid, x, zero)

                            # acc00
                            if ((h_block * 2 + h) < 3) & ((w_block * 2 + w) < 3):
                                acc00 += tl.dot(x, weights[h_block + h][w_block + w])

                            # acc01
                            if ((h_block * 2 + h) < 3) & ((w_block * 2 + w) >  0):
                                acc01 += tl.dot(x, weights[h_block + h][w_block + w - 1])

                            # acc10
                            if ((h_block * 2 + h) > 0) & ((w_block * 2 + w) <  3):
                                acc10 += tl.dot(x, weights[h_block + h - 1][w_block + w])

                            # acc11
                            if ((h_block * 2 + h) > 0) & ((w_block * 2 + w) > 0):
                                acc11 += tl.dot(x, weights[h_block + h - 1][w_block + w - 1])

    omask = d_offset_shifted < D
    tl.store(output_ptr + output_offset, acc00, mask=omask & ((W_cell * 2) < W) & ((H_cell * 2) < H))
    tl.store(output_ptr + output_offset + OUT_CHANNELS * D, acc01, mask=omask & ((W_cell * 2 + 1) < W) & ((H_cell * 2) < H))
    tl.store(output_ptr + output_offset + OUT_CHANNELS * D * W, acc10, mask=omask & ((W_cell * 2) < W) & ((H_cell * 2 + 1) < H))
    tl.store(output_ptr + output_offset + OUT_CHANNELS * D * W + OUT_CHANNELS * D, acc11, mask=omask & ((W_cell * 2 + 1) < W) & ((H_cell * 2 + 1) < H))
    
    mean = tl.sum(
        tl.where(omask & ((W_cell * 2) < W) & ((H_cell * 2) < H), acc00, 0.0)
        + tl.where(omask & ((W_cell * 2 + 1) < W) & ((H_cell * 2) < H),acc01, 0.0)
        + tl.where(omask & ((W_cell * 2) < W) & ((H_cell * 2 + 1) < H), acc10, 0.0)
        + tl.where(omask & ((W_cell * 2 + 1) < W) & ((H_cell * 2 + 1) < H), acc11, 0.0),
        axis=0
    )

    sqmean = tl.sum(
        tl.where(omask & ((W_cell * 2) < W) & ((H_cell * 2) < H), acc00 * acc00, 0.0)
        + tl.where(omask & ((W_cell * 2 + 1) < W) & ((H_cell * 2) < H), acc01 * acc01, 0.0)
        + tl.where(omask & ((W_cell * 2) < W) & ((H_cell * 2 + 1) < H), acc10 * acc10, 0.0)
        + tl.where(omask & ((W_cell * 2 + 1) < W) & ((H_cell * 2 + 1) < H), acc11 * acc11, 0.0),
        axis=0
    )

    tl.store(mean_ptr + out_channels_offset, mean)
    tl.store(sqmean_ptr + out_channels_offset, sqmean)
