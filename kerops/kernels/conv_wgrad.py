"""Conv3d weight-gradient kernel in gluon (sm_80+): cp.async to shared memory, mma_v2 into fp16 partials, fp32 sums."""

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
    """Layout of a [sets, rows, columns] copy, `vec` fp16 a thread, without repeated elements where possible."""
    threads_along_columns = min(32, columns // vec)
    threads_along_rows = min(32 // threads_along_columns, rows)
    warps_along_rows = min(num_warps, max(1, rows // threads_along_rows))
    return gl.BlockedLayout(
        [1, 1, vec],
        [32 // (threads_along_columns * threads_along_rows), threads_along_rows, threads_along_columns],
        [num_warps // warps_along_rows, warps_along_rows, 1],
        [2, 1, 0],
    )


@gluon.jit
def _row_offsets(
    sets: gl.constexpr,
    used_sets: gl.constexpr,
    rows: gl.constexpr,
    columns: gl.constexpr,
    row_stride: gl.constexpr,
    set_step: gl.constexpr,
    first_row: gl.constexpr,
    channels: gl.constexpr,
    layout: gl.constexpr,
):
    """D rows row_stride * t + set_step * s + first_row of a copy, their offsets in a plane, and the used sets."""
    s = gl.arange(0, sets, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    t = gl.arange(0, rows, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    c = gl.arange(0, columns, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    d = row_stride * t[None, :] + set_step * s[:, None] + first_row
    return d, (d * channels)[:, :, None] + c[None, None, :], (s < used_sets)[:, None]


@gluon.jit
def _operands(buffer, count: gl.constexpr, stage, k: gl.constexpr, transposed: gl.constexpr, layout: gl.constexpr):
    """The k-th 16 rows of the first `count` row sets of `buffer` as mma operands."""
    stages: gl.constexpr = buffer.shape[0]
    sets: gl.constexpr = buffer.shape[1]
    columns: gl.constexpr = buffer.shape[3]
    # A slice of a multi-set swizzled buffer cannot be loaded; the same bytes as separate matrices can.
    matrices = buffer._reinterpret(gl.float16, [stages * sets, buffer.shape[2], columns], _swizzled_layout(columns))
    operands = ()
    for index in gl.static_range(count):
        rows = matrices.index(stage * sets + index).slice(k * 16, 16)
        if transposed:
            rows = rows.permute([1, 0])
        operands = operands + (rows.load(layout),)
    return operands


@gluon.jit
def _accumulate(parts, p, q, WINOGRAD: gl.constexpr):
    """P and Q row sets into the partials: per kd tap, or per position of Winograd F(3, 2) after Bᵀ and A."""
    if WINOGRAD:
        p = (p[0] - p[2], p[1] + p[2], p[2] - p[1], p[1] - p[3])
        # The sign of the last row of A moves into Gᵀ.
        q = (q[0], q[0] + q[1], q[0] - q[1], q[1])
    for index in gl.static_range(len(parts)):
        parts = _with_item(parts, index, mma_v2(p[index if WINOGRAD else 0], q[index], parts[index]))
    return parts


@gluon.jit
def _taps(accs, first: gl.constexpr, WINOGRAD: gl.constexpr):
    """The 3 kd taps of the sums from `first`: Gᵀ of Winograd F(3, 2), or the sums themselves."""
    if WINOGRAD:
        middle = 0.5 * (accs[first + 1] + accs[first + 2])
        return accs[first] + middle, 0.5 * (accs[first + 1] - accs[first + 2]), middle - accs[first + 3]
    return accs[first], accs[first + 1], accs[first + 2]


@gluon.jit
def _tile_origin(tile, H, W, D, TH: gl.constexpr, TW: gl.constexpr, TD: gl.constexpr):
    """(sample, h0, w0, d0) of `tile`."""
    # D blocks first: programs on neighbouring tiles share halo rows in L2.
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
    i,
    split,
    splits,
    n_tiles,
    p_ptr,
    q_ptr,
    p_planes,
    q_planes,
    H,
    W,
    D,
    kh0,
    kw0,
    p_rows,
    q_rows,
    P_CHANNELS: gl.constexpr,
    Q_CHANNELS: gl.constexpr,
    TH: gl.constexpr,
    TW: gl.constexpr,
    TD: gl.constexpr,
    GH: gl.constexpr,
    GW: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    # Tiles past the end wrap to a valid one loaded into a buffer nobody reads: the loop needs no tail.
    sample, h0, w0, d0 = _tile_origin((split + i * splits) % n_tiles, H, W, D, TH, TW, TD)
    stage = i % NUM_STAGES
    p_base = p_ptr + sample.to(gl.int64) * H * W * D * P_CHANNELS
    q_base = q_ptr + sample.to(gl.int64) * H * W * D * Q_CHANNELS
    PW: gl.constexpr = TW + GW - 1
    p_d, p_offsets, p_used = p_rows
    p_rows_inside = p_used & (d0 + p_d >= 0) & (d0 + p_d < D)
    for plane in gl.static_range((TH + GH - 1) * PW):
        h = h0 + kh0 + plane // PW - 1
        w = w0 + kw0 + plane % PW - 1
        inside = p_rows_inside & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        offsets = ((h * W + w) * D + d0) * P_CHANNELS + p_offsets
        cp.async_copy_global_to_shared(p_planes[plane].index(stage), p_base + offsets, mask=inside[:, :, None])
    q_d, q_offsets, q_used = q_rows
    q_rows_inside = q_used & (d0 + q_d >= 0) & (d0 + q_d < D)
    for position in gl.static_range(TH * TW):
        h = h0 + position // TW
        w = w0 + position % TW
        inside = q_rows_inside & (h < H) & (w < W)
        offsets = ((h * W + w) * D + d0) * Q_CHANNELS + q_offsets
        cp.async_copy_global_to_shared(q_planes[position].index(stage), q_base + offsets, mask=inside[:, :, None])
    cp.commit_group()


@gluon.jit
def _Conv_wgrad_cl3d_halo(
    p_ptr,
    q_ptr,
    out_ptr,
    H,
    W,
    D,
    n_tiles,
    P_CHANNELS: gl.constexpr,
    Q_CHANNELS: gl.constexpr,
    P_BLOCK: gl.constexpr,
    Q_BLOCK: gl.constexpr,
    TH: gl.constexpr,
    TW: gl.constexpr,
    TD: gl.constexpr,
    GH: gl.constexpr,
    GW: gl.constexpr,
    WINOGRAD: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    SWAP: gl.constexpr,
    P_VEC: gl.constexpr,
    Q_VEC: gl.constexpr,
):
    """Adds Σ P[v + tap - 1]ᵀ Q[v] over this program's tiles, for its taps and channel blocks, to fp32 `out`."""
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    PW: gl.constexpr = TW + GW - 1
    P_PLANES: gl.constexpr = (TH + GH - 1) * PW
    TAPS: gl.constexpr = GH * GW
    SUMS: gl.constexpr = 4 if WINOGRAD else 3
    P_SETS: gl.constexpr = 4 if WINOGRAD else 1
    # Power-of-2 copies: the direct 3 shifted Q copies take 4 sets, the last one masked off.
    Q_SETS: gl.constexpr = 2 if WINOGRAD else 4
    ROWS: gl.constexpr = TD // 2 if WINOGRAD else TD
    mma_layout: gl.constexpr = _mma_layout(WARPS_M, WARPS_N)
    p_layout: gl.constexpr = _copy_layout(P_SETS, ROWS, P_BLOCK, NUM_WARPS, P_VEC)
    q_layout: gl.constexpr = _copy_layout(Q_SETS, ROWS, Q_BLOCK, NUM_WARPS, Q_VEC)
    a_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)

    # Neighbouring programs take the same tiles for other taps and channel blocks, and find them in L2.
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
    # Winograd: P rows 2t + s - 1, Q rows 2t + s. Direct: P rows t, Q rows t - kd + 1.
    if WINOGRAD:
        p_rows = _row_offsets(P_SETS, P_SETS, ROWS, P_BLOCK, 2, 1, -1, P_CHANNELS, p_layout)
        q_rows = _row_offsets(Q_SETS, Q_SETS, ROWS, Q_BLOCK, 2, 1, 0, Q_CHANNELS, q_layout)
    else:
        p_rows = _row_offsets(P_SETS, P_SETS, ROWS, P_BLOCK, 1, 0, 0, P_CHANNELS, p_layout)
        q_rows = _row_offsets(Q_SETS, 3, ROWS, Q_BLOCK, 1, -1, 1, Q_CHANNELS, q_layout)

    p_planes = ()
    for plane in gl.static_range(P_PLANES):
        p_planes = p_planes + (
            gl.allocate_shared_memory(gl.float16, [NUM_STAGES, P_SETS, ROWS, P_BLOCK], _swizzled_layout(P_BLOCK, 3)),
        )
    q_planes = ()
    for position in gl.static_range(TH * TW):
        q_planes = q_planes + (
            gl.allocate_shared_memory(gl.float16, [NUM_STAGES, Q_SETS, ROWS, Q_BLOCK], _swizzled_layout(Q_BLOCK, 3)),
        )

    for i in gl.static_range(NUM_STAGES - 1):
        _issue_tile_loads(
            i,
            split,
            splits,
            n_tiles,
            p_ptr,
            q_ptr,
            p_planes,
            q_planes,
            H,
            W,
            D,
            kh0,
            kw0,
            p_rows,
            q_rows,
            P_CHANNELS,
            Q_CHANNELS,
            TH,
            TW,
            TD,
            GH,
            GW,
            NUM_STAGES,
        )

    accs = (gl.zeros([P_BLOCK, Q_BLOCK], gl.float32, mma_layout),) * (TAPS * SUMS)
    for i in range(gl.cdiv(n_tiles - split, splits)):
        _issue_tile_loads(
            i + NUM_STAGES - 1,
            split,
            splits,
            n_tiles,
            p_ptr,
            q_ptr,
            p_planes,
            q_planes,
            H,
            W,
            D,
            kh0,
            kw0,
            p_rows,
            q_rows,
            P_CHANNELS,
            Q_CHANNELS,
            TH,
            TW,
            TD,
            GH,
            GW,
            NUM_STAGES,
        )
        cp.wait_group(NUM_STAGES - 1)
        stage = i % NUM_STAGES
        for tap in gl.static_range(TAPS):
            # fp16 partials for a tile, fp32 sums across tiles: long chains inside fp32 mma shrink the result.
            parts = (gl.zeros([P_BLOCK, Q_BLOCK], gl.float16, mma_layout),) * SUMS
            for position in gl.static_range(TH * TW):
                for k in gl.static_range(ROWS // 16):
                    p = _operands(p_planes[_halo_plane(position, tap, TW, GW)], P_SETS, stage, k, True, a_dot_layout)
                    q = _operands(q_planes[position], 2 if WINOGRAD else 3, stage, k, False, b_dot_layout)
                    parts = _accumulate(parts, p, q, WINOGRAD)
            for a in gl.static_range(SUMS):
                accs = _with_item(accs, tap * SUMS + a, accs[tap * SUMS + a] + parts[a].to(gl.float32))
    cp.wait_group(0)

    m = p_block * P_BLOCK + gl.arange(0, P_BLOCK, layout=gl.SliceLayout(1, mma_layout))
    n = q_block * Q_BLOCK + gl.arange(0, Q_BLOCK, layout=gl.SliceLayout(0, mma_layout))
    for tap in gl.static_range(TAPS):
        kh = kh0 + tap // GW
        kw = kw0 + tap % GW
        kd_taps = _taps(accs, tap * SUMS, WINOGRAD)
        for kd in gl.static_range(3):
            if SWAP:
                # P is grad and Q is x: the sum is the transposed gradient of the mirrored tap.
                offsets = (
                    (((2 - kh) * 3 + 2 - kw) * 3 + 2 - kd) * P_CHANNELS * Q_CHANNELS
                    + n[None, :] * P_CHANNELS
                    + m[:, None]
                )
            else:
                offsets = ((kh * 3 + kw) * 3 + kd) * P_CHANNELS * Q_CHANNELS + m[:, None] * Q_CHANNELS + n[None, :]
            gl.atomic_add(out_ptr + offsets, kd_taps[kd], sem='relaxed')
