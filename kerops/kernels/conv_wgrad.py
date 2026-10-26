"""Conv3d weight-gradient kernel in gluon (sm_80+): cp.async into shared memory, Winograd F(3, 2) along D on mma
operands in registers, mma_v2 into fp16 partials, fp32 sums."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2

from .conv_fprop import _mma_layout, _swizzled_layout, _with_item


@gluon.constexpr_function
def _halo_plane(position, tap, TW, GW):
    """P plane that `tap` of the group reads for `position` of the tile."""
    return (position // TW + tap // GW) * (TW + GW - 1) + position % TW + tap % GW


@gluon.constexpr_function
def _copy_layout(sets, rows, columns, num_warps, vec):
    """Layout of a [sets, rows, columns] copy with `vec` fp16 a thread along columns, which repeats no element where
    the copy has one for every thread."""
    threads_along_columns = min(32, columns // vec)
    threads_along_rows = min(32 // threads_along_columns, rows)
    warps_along_rows = min(num_warps, max(1, rows // threads_along_rows))
    return gl.BlockedLayout(
        [1, 1, vec], [32 // (threads_along_columns * threads_along_rows), threads_along_rows, threads_along_columns],
        [num_warps // warps_along_rows, warps_along_rows, 1], [2, 1, 0],
    )


@gluon.jit
def _row_offsets(sets: gl.constexpr, rows: gl.constexpr, columns: gl.constexpr, first_row: gl.constexpr,
                 channels: gl.constexpr, layout: gl.constexpr):
    """D rows 2t + s + first_row of the sets s and pairs t of a copy, and their offsets with the columns in a plane."""
    s = gl.arange(0, sets, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    t = gl.arange(0, rows, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    c = gl.arange(0, columns, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    d = 2 * t[None, :] + s[:, None] + first_row
    return d, (d * channels)[:, :, None] + c[None, None, :]


@gluon.jit
def _operands(buffer, count: gl.constexpr, stage, k: gl.constexpr, transposed: gl.constexpr, layout: gl.constexpr):
    """The k-th 16 D pairs of the `count` row sets of `buffer` as mma operands; transposed for A, whose K runs along
    rows."""
    stages: gl.constexpr = buffer.shape[0]
    columns: gl.constexpr = buffer.shape[3]
    # The same bytes as [stages * sets] matrices, each with the swizzle of an mma operand.
    matrices = buffer._reinterpret(gl.float16, [stages * count, buffer.shape[2], columns], _swizzled_layout(columns))
    operands = ()
    for index in gl.static_range(count):
        rows = matrices.index(stage * count + index).slice(k * 16, 16)
        if transposed:
            rows = rows.permute([1, 0])
        operands = operands + (rows.load(layout),)
    return operands


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
    i, split, splits, n_tiles, p_ptr, q_ptr, p_planes, q_planes, H, W, D, kh0, kw0, p_d, p_offsets, q_d, q_offsets,
    P_CHANNELS: gl.constexpr, Q_CHANNELS: gl.constexpr, TH: gl.constexpr, TW: gl.constexpr, TD: gl.constexpr,
    GH: gl.constexpr, GW: gl.constexpr, NUM_STAGES: gl.constexpr,
):
    # The i-th tile of this program, as rows of the D pairs t: of each H/W halo plane of P that its taps reach, the rows
    # d0 + 2t + r - 1 for r < 4; of each Q plane, the rows d0 + 2t + r for r < 2; one copy a plane. Indices past the end
    # wrap to a valid tile loaded into a buffer nobody reads, which keeps the loop free of tails.
    sample, h0, w0, d0 = _tile_origin((split + i * splits) % n_tiles, H, W, D, TH, TW, TD)
    stage = i % NUM_STAGES
    p_base = p_ptr + sample.to(gl.int64) * H * W * D * P_CHANNELS
    q_base = q_ptr + sample.to(gl.int64) * H * W * D * Q_CHANNELS
    PW: gl.constexpr = TW + GW - 1
    p_rows_inside = (d0 + p_d >= 0) & (d0 + p_d < D)
    for plane in gl.static_range((TH + GH - 1) * PW):
        h = h0 + kh0 + plane // PW - 1
        w = w0 + kw0 + plane % PW - 1
        inside = p_rows_inside & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        offsets = ((h * W + w) * D + d0) * P_CHANNELS + p_offsets
        cp.async_copy_global_to_shared(p_planes[plane].index(stage), p_base + offsets, mask=inside[:, :, None])
    q_rows_inside = d0 + q_d < D
    for position in gl.static_range(TH * TW):
        h = h0 + position // TW
        w = w0 + position % TW
        inside = q_rows_inside & (h < H) & (w < W)
        offsets = ((h * W + w) * D + d0) * Q_CHANNELS + q_offsets
        cp.async_copy_global_to_shared(q_planes[position].index(stage), q_base + offsets, mask=inside[:, :, None])
    cp.commit_group()


@gluon.jit
def _Conv_wgrad_cl3d_halo(
    p_ptr, q_ptr, out_ptr, H, W, D, n_tiles,
    P_CHANNELS: gl.constexpr, Q_CHANNELS: gl.constexpr, P_BLOCK: gl.constexpr, Q_BLOCK: gl.constexpr,
    TH: gl.constexpr, TW: gl.constexpr, TD: gl.constexpr, GH: gl.constexpr, GW: gl.constexpr,
    NUM_STAGES: gl.constexpr, WARPS_M: gl.constexpr, WARPS_N: gl.constexpr, P_VEC: gl.constexpr, Q_VEC: gl.constexpr,
):
    """out[kh, kw, a] += Σ_t (Bᵀ P_t)[a]ᵀ · (A Q_t)[a] over the D pairs t of the tiles of this program, for its GH x GW
    (kh, kw) taps and channel blocks: Winograd F(3, 2) along D, whose 4 positions a give the 3 kd taps through Gᵀ
    afterwards. P_t is the 4 rows of P around the pair, shifted by the tap in H/W, and Q_t the pair's 2 rows of Q;
    out is fp32 [3, 3, 4, P channels, Q channels]. Each position sums a tile in an fp16 partial flushed into fp32.
    P_VEC, Q_VEC: fp16 per cp.async of P and Q."""
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    PW: gl.constexpr = TW + GW - 1
    P_PLANES: gl.constexpr = (TH + GH - 1) * PW
    TAPS: gl.constexpr = GH * GW
    mma_layout: gl.constexpr = _mma_layout(WARPS_M, WARPS_N)
    p_layout: gl.constexpr = _copy_layout(4, TD // 2, P_BLOCK, NUM_WARPS, P_VEC)
    q_layout: gl.constexpr = _copy_layout(2, TD // 2, Q_BLOCK, NUM_WARPS, Q_VEC)
    a_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)

    # Programs next to each other take the same tiles for other taps and channel blocks, and find them in L2.
    groups = (3 // GH) * (3 // GW)
    p_blocks = P_CHANNELS // P_BLOCK
    work = groups * p_blocks * (Q_CHANNELS // Q_BLOCK)
    program = gl.program_id(0)
    group = program % groups
    p_block = program // groups % p_blocks
    q_block = program // (groups * p_blocks) % (Q_CHANNELS // Q_BLOCK)
    split = program // work
    splits = gl.num_programs(0) // work
    kh0 = group // (3 // GW) * GH
    kw0 = group % (3 // GW) * GW

    p_ptr += p_block * P_BLOCK
    q_ptr += q_block * Q_BLOCK
    p_d, p_offsets = _row_offsets(4, TD // 2, P_BLOCK, -1, P_CHANNELS, p_layout)
    q_d, q_offsets = _row_offsets(2, TD // 2, Q_BLOCK, 0, Q_CHANNELS, q_layout)

    p_planes = ()
    for plane in gl.static_range(P_PLANES):
        p_planes = p_planes + (gl.allocate_shared_memory(gl.float16, [NUM_STAGES, 4, TD // 2, P_BLOCK], _swizzled_layout(P_BLOCK, 3)),)
    q_planes = ()
    for position in gl.static_range(TH * TW):
        q_planes = q_planes + (gl.allocate_shared_memory(gl.float16, [NUM_STAGES, 2, TD // 2, Q_BLOCK], _swizzled_layout(Q_BLOCK, 3)),)

    for i in gl.static_range(NUM_STAGES - 1):
        _issue_tile_loads(
            i, split, splits, n_tiles, p_ptr, q_ptr, p_planes, q_planes, H, W, D, kh0, kw0, p_d, p_offsets, q_d, q_offsets,
            P_CHANNELS, Q_CHANNELS, TH, TW, TD, GH, GW, NUM_STAGES,
        )

    accs = (gl.zeros([P_BLOCK, Q_BLOCK], gl.float32, mma_layout),) * (TAPS * 4)
    for i in range(gl.cdiv(n_tiles - split, splits)):
        _issue_tile_loads(
            i + NUM_STAGES - 1, split, splits, n_tiles, p_ptr, q_ptr, p_planes, q_planes, H, W, D, kh0, kw0,
            p_d, p_offsets, q_d, q_offsets, P_CHANNELS, Q_CHANNELS, TH, TW, TD, GH, GW, NUM_STAGES,
        )
        cp.wait_group(NUM_STAGES - 1)
        stage = i % NUM_STAGES
        for tap in gl.static_range(TAPS):
            parts = (gl.zeros([P_BLOCK, Q_BLOCK], gl.float16, mma_layout),) * 4
            for position in gl.static_range(TH * TW):
                for k in gl.static_range(TD // 32):
                    r0, r1, r2, r3 = _operands(p_planes[_halo_plane(position, tap, TW, GW)], 4, stage, k, True, a_dot_layout)
                    s0, s1 = _operands(q_planes[position], 2, stage, k, False, b_dot_layout)
                    # Bᵀ on P and A on Q; the sign of A's last row moves into Gᵀ.
                    parts = (
                        mma_v2(r0 - r2, s0, parts[0]),
                        mma_v2(r1 + r2, s0 + s1, parts[1]),
                        mma_v2(r2 - r1, s0 - s1, parts[2]),
                        mma_v2(r1 - r3, s1, parts[3]),
                    )
            for a in gl.static_range(4):
                accs = _with_item(accs, tap * 4 + a, accs[tap * 4 + a] + parts[a].to(gl.float32))
    cp.wait_group(0)

    m = p_block * P_BLOCK + gl.arange(0, P_BLOCK, layout=gl.SliceLayout(1, mma_layout))
    n = q_block * Q_BLOCK + gl.arange(0, Q_BLOCK, layout=gl.SliceLayout(0, mma_layout))
    for index in gl.static_range(TAPS * 4):
        kh = kh0 + index // 4 // GW
        kw = kw0 + index // 4 % GW
        offsets = ((kh * 3 + kw) * 4 + index % 4) * P_CHANNELS * Q_CHANNELS + m[:, None] * Q_CHANNELS + n[None, :]
        gl.atomic_add(out_ptr + offsets, accs[index], sem='relaxed')
