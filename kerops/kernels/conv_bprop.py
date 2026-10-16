import triton
from triton import language as tl

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
def _Conv_wgrad_cl3d_splitk_3acc_impl(
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
    khw_pid = tl.program_id(0)
    cin_pid = tl.program_id(1)
    cout_batch_pid = tl.program_id(2)

    k_pid = khw_pid // 9
    hw_pid = khw_pid % 9

    block_w = hw_pid % 3
    block_h = hw_pid // 3

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
    input_ptr += k_pid * H_BLOCK * IN_CHANNELS * D * W
    input_ptr += (block_w - 1) * IN_CHANNELS * D
    input_ptr += (block_h - 1) * IN_CHANNELS * D * W
    input_ptr += batch_pid * H * W * D * IN_CHANNELS
    weight_grad_ptr += cin_pid * CIN_BLOCK * OUT_CHANNELS + cout_pid * COUT_BLOCK
    weight_grad_ptr += block_w * IN_CHANNELS * OUT_CHANNELS * 3
    weight_grad_ptr += block_h * IN_CHANNELS * OUT_CHANNELS * 9
    weight_grad_ptr += batch_pid * IN_CHANNELS * OUT_CHANNELS * 27

    weight_grad0 = tl.zeros((CIN_BLOCK, COUT_BLOCK), dtype=ACCTYPE)
    weight_grad1 = tl.zeros((CIN_BLOCK, COUT_BLOCK), dtype=ACCTYPE)
    weight_grad2 = tl.zeros((CIN_BLOCK, COUT_BLOCK), dtype=ACCTYPE)

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

                    x_mask = make_mask(grad_h * H_BLOCK * SPLIT_K + k_pid * H_BLOCK + block_h - 1, grad_w * W_BLOCK + block_w - 1, grad_d * D_BLOCK - 1, H, W, D, H_BLOCK, W_BLOCK, D_BLOCK)
                    x_iter_ptr = (
                        input_ptr
                        + grad_h * H_BLOCK * W * D * IN_CHANNELS * SPLIT_K
                        + grad_w * W_BLOCK * D * IN_CHANNELS
                        + grad_d * D_BLOCK * IN_CHANNELS - IN_CHANNELS
                    )
                    x = tl.load(x_iter_ptr + input_offset, mask=x_mask[None, :], other=0)

                    weight_grad0 += tl.dot(x, grad)

                    x_mask = make_mask(grad_h * H_BLOCK * SPLIT_K + k_pid * H_BLOCK + block_h - 1, grad_w * W_BLOCK + block_w - 1, grad_d * D_BLOCK, H, W, D, H_BLOCK, W_BLOCK, D_BLOCK)
                    x_iter_ptr = (
                        input_ptr
                        + grad_h * H_BLOCK * W * D * IN_CHANNELS * SPLIT_K
                        + grad_w * W_BLOCK * D * IN_CHANNELS
                        + grad_d * D_BLOCK * IN_CHANNELS
                    )
                    x = tl.load(x_iter_ptr + input_offset, mask=x_mask[None, :], other=0)

                    weight_grad1 += tl.dot(x, grad)

                    x_mask = make_mask(grad_h * H_BLOCK * SPLIT_K + k_pid * H_BLOCK + block_h - 1, grad_w * W_BLOCK + block_w - 1, grad_d * D_BLOCK + 1, H, W, D, H_BLOCK, W_BLOCK, D_BLOCK)
                    x_iter_ptr = (
                        input_ptr
                        + grad_h * H_BLOCK * W * D * IN_CHANNELS * SPLIT_K
                        + grad_w * W_BLOCK * D * IN_CHANNELS
                        + grad_d * D_BLOCK * IN_CHANNELS + IN_CHANNELS
                    )
                    x = tl.load(x_iter_ptr + input_offset, mask=x_mask[None, :], other=0)

                    weight_grad2 += tl.dot(x, grad)

    tl.atomic_add(weight_grad_ptr + weight_grad_offset, weight_grad0, sem='relaxed')
    tl.atomic_add(weight_grad_ptr + IN_CHANNELS * OUT_CHANNELS + weight_grad_offset, weight_grad1, sem='relaxed')
    tl.atomic_add(weight_grad_ptr + 2 * IN_CHANNELS * OUT_CHANNELS + weight_grad_offset, weight_grad2, sem='relaxed')
