"""Conv3d weight-gradient kernel in gluon (sm_80+): cp.async into shared memory, mma_v2 into fp16 partials, fp32 sums."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2

from .conv_fprop import _mma_layout, _rows_layout, _swizzled_layout, _with_item


@gluon.jit
def _tile_origin(tile, H, W, D, TH: gl.constexpr, TW: gl.constexpr, TD: gl.constexpr):
    """(sample, h0, w0, d0) of `tile`; D blocks run first, so programs on neighbouring tiles share halo rows in L2."""
    d_blocks = gl.cdiv(D, TD)
    w_tiles = gl.cdiv(W, TW)
    h_tiles = gl.cdiv(H, TH)
    d0 = tile % d_blocks * TD
    w0 = tile // d_blocks % w_tiles * TW
    h0 = tile // (d_blocks * w_tiles) % h_tiles * TH
    sample = tile // (d_blocks * w_tiles * h_tiles)
    return sample, h0, w0, d0


@gluon.jit
def _issue_tile_loads(
    i, split, splits, n_tiles, p_ptr, q_ptr, p_planes, q_planes, H, W, D, kh0, kw0, kd0, rows_p, k_p, rows_q, k_q,
    P_CHANNELS: gl.constexpr, Q_CHANNELS: gl.constexpr, TH: gl.constexpr, TW: gl.constexpr, TD: gl.constexpr,
    GH: gl.constexpr, GW: gl.constexpr, GD: gl.constexpr, NUM_STAGES: gl.constexpr,
):
    # The i-th tile of this program: the H/W halo planes of P that its taps reach, and its Q planes shifted by each kd.
    # Indices past the end wrap to a valid tile loaded into a buffer nobody reads, which keeps the loop free of tails.
    sample, h0, w0, d0 = _tile_origin((split + i * splits) % n_tiles, H, W, D, TH, TW, TD)
    stage = i % NUM_STAGES
    p_base = p_ptr + sample.to(gl.int64) * H * W * D * P_CHANNELS
    q_base = q_ptr + sample.to(gl.int64) * H * W * D * Q_CHANNELS
    PW: gl.constexpr = TW + GW - 1
    d_p = d0 + rows_p
    for plane in gl.static_range((TH + GH - 1) * PW):
        h = h0 + kh0 + plane // PW - 1
        w = w0 + kw0 + plane % PW - 1
        inside = (d_p < D) & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        offsets = (((h * W + w) * D + d_p) * P_CHANNELS)[:, None] + k_p[None, :]
        cp.async_copy_global_to_shared(p_planes[plane].index(stage), p_base + offsets, mask=inside[:, None])
    for shift in gl.static_range(GD):
        d_q = d0 + rows_q - kd0 - shift + 1
        for position in gl.static_range(TH * TW):
            h = h0 + position // TW
            w = w0 + position % TW
            inside = (d_q >= 0) & (d_q < D) & (h < H) & (w < W)
            offsets = (((h * W + w) * D + d_q) * Q_CHANNELS)[:, None] + k_q[None, :]
            cp.async_copy_global_to_shared(q_planes[shift * TH * TW + position].index(stage), q_base + offsets, mask=inside[:, None])
    cp.commit_group()


@gluon.jit
def _Conv_wgrad_cl3d_halo(
    p_ptr, q_ptr, out_ptr, H, W, D, n_tiles,
    P_CHANNELS: gl.constexpr, Q_CHANNELS: gl.constexpr, P_BLOCK: gl.constexpr, Q_BLOCK: gl.constexpr,
    TH: gl.constexpr, TW: gl.constexpr, TD: gl.constexpr, GH: gl.constexpr, GW: gl.constexpr, GD: gl.constexpr,
    NUM_STAGES: gl.constexpr, WARPS_M: gl.constexpr, WARPS_N: gl.constexpr, SWAP: gl.constexpr,
):
    """out[tap] += Σ_v P[v + tap - 1]ᵀ · Q[v] over the tiles of this program, for its GH x GW x GD taps and channel
    blocks; P is x and Q is grad, or the reverse when SWAP, and out is the fp32 [3, 3, 3, in, out] weight gradient.
    Each tap sums a tile in an fp16 partial flushed into fp32."""
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    PW: gl.constexpr = TW + GW - 1
    P_PLANES: gl.constexpr = (TH + GH - 1) * PW
    TAPS: gl.constexpr = GH * GW * GD
    mma_layout: gl.constexpr = _mma_layout(WARPS_M, WARPS_N)
    p_layout: gl.constexpr = _rows_layout(P_BLOCK, NUM_WARPS, 8)
    q_layout: gl.constexpr = _rows_layout(Q_BLOCK, NUM_WARPS, 8)
    a_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)

    # Programs next to each other take the same tiles for other taps and channel blocks, and find them in L2.
    groups = (3 // GH) * (3 // GW) * (3 // GD)
    p_blocks = P_CHANNELS // P_BLOCK
    work = groups * p_blocks * (Q_CHANNELS // Q_BLOCK)
    program = gl.program_id(0)
    group = program % groups
    p_block = program // groups % p_blocks
    q_block = program // (groups * p_blocks) % (Q_CHANNELS // Q_BLOCK)
    split = program // work
    splits = gl.num_programs(0) // work
    kh0 = group // ((3 // GW) * (3 // GD)) * GH
    kw0 = group // (3 // GD) % (3 // GW) * GW
    kd0 = group % (3 // GD) * GD

    p_ptr += p_block * P_BLOCK
    q_ptr += q_block * Q_BLOCK
    rows_p = gl.arange(0, TD, layout=gl.SliceLayout(1, p_layout))
    k_p = gl.arange(0, P_BLOCK, layout=gl.SliceLayout(0, p_layout))
    rows_q = gl.arange(0, TD, layout=gl.SliceLayout(1, q_layout))
    k_q = gl.arange(0, Q_BLOCK, layout=gl.SliceLayout(0, q_layout))

    p_planes = ()
    for plane in gl.static_range(P_PLANES):
        p_planes = p_planes + (gl.allocate_shared_memory(gl.float16, [NUM_STAGES, TD, P_BLOCK], _swizzled_layout(P_BLOCK)),)
    q_planes = ()
    for plane in gl.static_range(GD * TH * TW):
        q_planes = q_planes + (gl.allocate_shared_memory(gl.float16, [NUM_STAGES, TD, Q_BLOCK], _swizzled_layout(Q_BLOCK)),)

    for i in gl.static_range(NUM_STAGES - 1):
        _issue_tile_loads(
            i, split, splits, n_tiles, p_ptr, q_ptr, p_planes, q_planes, H, W, D, kh0, kw0, kd0, rows_p, k_p, rows_q, k_q,
            P_CHANNELS, Q_CHANNELS, TH, TW, TD, GH, GW, GD, NUM_STAGES,
        )

    accs = (gl.zeros([P_BLOCK, Q_BLOCK], gl.float32, mma_layout),) * TAPS
    for i in range(gl.cdiv(n_tiles - split, splits)):
        _issue_tile_loads(
            i + NUM_STAGES - 1, split, splits, n_tiles, p_ptr, q_ptr, p_planes, q_planes, H, W, D, kh0, kw0, kd0,
            rows_p, k_p, rows_q, k_q, P_CHANNELS, Q_CHANNELS, TH, TW, TD, GH, GW, GD, NUM_STAGES,
        )
        cp.wait_group(NUM_STAGES - 1)
        stage = i % NUM_STAGES
        for a_h in gl.static_range(GH):
            for a_w in gl.static_range(GW):
                for shift in gl.static_range(GD):
                    part = gl.zeros([P_BLOCK, Q_BLOCK], gl.float16, mma_layout)
                    for position in gl.static_range(TH * TW):
                        p_plane = p_planes[(position // TW + a_h) * PW + position % TW + a_w].index(stage)
                        q_plane = q_planes[shift * TH * TW + position].index(stage)
                        for k in gl.static_range(TD // 16):
                            a = p_plane.slice(k * 16, 16).permute([1, 0]).load(a_dot_layout)
                            b = q_plane.slice(k * 16, 16).load(b_dot_layout)
                            part = mma_v2(a, b, part)
                    accs = _with_item(
                        accs, (a_h * GW + a_w) * GD + shift, accs[(a_h * GW + a_w) * GD + shift] + part.to(gl.float32)
                    )
    cp.wait_group(0)

    m = p_block * P_BLOCK + gl.arange(0, P_BLOCK, layout=gl.SliceLayout(1, mma_layout))
    n = q_block * Q_BLOCK + gl.arange(0, Q_BLOCK, layout=gl.SliceLayout(0, mma_layout))
    for tap in gl.static_range(TAPS):
        kh = kh0 + tap // (GW * GD)
        kw = kw0 + tap // GD % GW
        kd = kd0 + tap % GD
        if SWAP:
            # P is grad and Q is x: out[tap][co, ci] is the weight gradient of the mirrored tap.
            offsets = (((2 - kh) * 3 + 2 - kw) * 3 + 2 - kd) * P_CHANNELS * Q_CHANNELS + n[None, :] * P_CHANNELS + m[:, None]
        else:
            offsets = ((kh * 3 + kw) * 3 + kd) * P_CHANNELS * Q_CHANNELS + m[:, None] * Q_CHANNELS + n[None, :]
        gl.atomic_add(out_ptr + offsets, accs[tap], sem='relaxed')
