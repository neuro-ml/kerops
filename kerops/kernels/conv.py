import triton
from triton import language as tl

"""
- Conv3dV2: D_cell -> W_cell -> H_cell changed to W_cell -> H_cell -> D_cell
- Conv3dV3: `tl.max_contiguous(tl.multiple_of(channels_offset, CHANNELS), CHANNELS)`
- Conv3dV4: Input channels BLOCKCING
- Conv3dV5: output blocking by HW: 2x2 tile

No impact:
 - `x @ w` and `w @ x` orientation via ORDER parameter
 - computations of multiple h, w in one block (WHEN THEY ARE FAR AWAY FROM EACH OTHER)
 - Boundary checks outside the loops
 - INTERIOR and BOUNDARY masking inside one kernel
 - `x: [D_BLOCK, CHANNELS] -> x: [NEAR, D_BLOCK, CHANNELS]` where NEAR=4 represents neighbours; `w: [4, CHANNELS_IN, CHANNELS_OUT]`
 - Output channels BLOCKCING (why?)
 - Output channels BLOCKCING via expanding grid
"""

@triton.jit
def _Conv_cl3d_impl_V5(
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

    acc00 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc01 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc10 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc11 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)

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


@triton.jit
def _ApplyBNReLUConv_cl3d_impl(
    input_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
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

    acc00 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc01 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc10 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)
    acc11 = tl.zeros([D_BLOCK, OUT_CHANNELS], dtype=ACCTYPE)

    for h_block in tl.static_range(0, 2):
        for w_block in tl.static_range(0, 2):
            for cin in tl.static_range(0, CIN_STEPS):
                bn_weight = tl.load(bn_weight_ptr + in_channels_offset + cin * CIN_BLOCK)[None, :]
                bn_bias = tl.load(bn_bias_ptr + in_channels_offset + cin * CIN_BLOCK)[None, :]
                zero = tl.zeros([1], dtype=tl.float16)

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
                    
                    xs = [
                        [
                            tl.maximum(tl.fma(tl.load(i_ptr + input_offset, mask=mask & m00, other=0.0), bn_weight, bn_bias), tl.zeros([1], dtype=tl.float16)),
                            tl.maximum(tl.fma(tl.load(i_ptr + input_offset + IN_CHANNELS * D, mask=mask & m01, other=0.0), bn_weight, bn_bias), tl.zeros([1], dtype=tl.float16))
                        ],
                        [
                            tl.maximum(tl.fma(tl.load(i_ptr + input_offset + IN_CHANNELS * D * W, mask=mask & m10, other=0.0), bn_weight, bn_bias), tl.zeros([1], dtype=tl.float16)),
                            tl.maximum(tl.fma(tl.load(i_ptr + input_offset + IN_CHANNELS * D + IN_CHANNELS * D * W, mask=mask & m11, other=0.0), bn_weight, bn_bias), tl.zeros([1], dtype=tl.float16))
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
