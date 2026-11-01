"""Conv3d forward kernel in gluon (sm_80+): cp.async into shared memory, mma_v2, layouts chosen explicitly."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2


@gluon.constexpr_function
def _rows_layout(row, num_warps, vec):
    """Each thread holds `vec` contiguous fp16 of a row-major tile with `row` columns. Triton lowers 16-byte cp.async
    (vec 8) to `cp.async.cg`, which bypasses L1; vec 4 gives 8-byte `cp.async.ca`, cached in L1."""
    threads_along_row = min(32, row // vec)
    return gl.BlockedLayout([1, vec], [32 // threads_along_row, threads_along_row], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def _swizzled_layout(row, rank=2):
    """The swizzle triton picks for mma_v2 operands with `row` fp16 columns, for buffers of `rank` + 1 dimensions."""
    per_phase = max(1, 128 // (2 * row))
    return gl.SwizzledSharedLayout(
        vec=8, per_phase=per_phase, max_phase=8 // per_phase, order=list(range(rank - 1, -1, -1))
    )


@gluon.constexpr_function
def _mma_layout(warps_m, warps_n):
    return gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[warps_m, warps_n], instr_shape=[16, 8])


@gluon.jit
def _with_item(items, index: gl.constexpr, value):
    return items[:index] + (value,) + items[index + 1 :]


@gluon.jit
def _store_position(out_ptr, acc, h, w, d0, H, W, D, COUT: gl.constexpr, TD: gl.constexpr, out_layout: gl.constexpr):
    d = d0 + gl.arange(0, TD, layout=gl.SliceLayout(1, out_layout))
    n = gl.arange(0, COUT, layout=gl.SliceLayout(0, out_layout))
    inside = (d < D) & (h < H) & (w < W)
    out_offsets = (h * W + w) * D * COUT + (d * COUT)[:, None] + n[None, :]
    gl.store(out_ptr + out_offsets, gl.convert_layout(acc.to(gl.float16), out_layout), mask=inside[:, None])


@gluon.jit
def _add_in_loop(a, b):
    # An add the compiler cannot hoist out of the step loop: hoisted, each of the (TH+2)*(TW+2) plane offsets would
    # hold a 64-bit register for the whole loop.
    return gl.inline_asm_elementwise('add.s32 $0, $1, $2;', '=r,r,r', [a, b], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _issue_halo_loads(
    step,
    x_ptr,
    w_ptr,
    a_planes,
    w_taps,
    row_starts,
    column_starts,
    h0,
    w0,
    d,
    H,
    W,
    D,
    k_a,
    k_w,
    n,
    CIN: gl.constexpr,
    COUT: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    TH: gl.constexpr,
    TW: gl.constexpr,
):
    # Step (cin block, kd): the (TH+2) x (TW+2) H/W halo planes of the input rows shifted by kd, and the 9 (kh, kw)
    # weights of kd. Steps past the end reload a valid step into a buffer nobody reads, which keeps the loop free of
    # tails.
    PLANES: gl.constexpr = (TH + 2) * (TW + 2)
    wrapped = step % (3 * (CIN // BLOCK_K))
    cin_offset = wrapped // 3 * BLOCK_K
    kd = wrapped % 3
    rows = d + kd - 1
    a_cols = (rows * CIN + cin_offset)[:, None] + k_a[None, :]
    rows_inside = (rows >= 0) & (rows < D)
    stage = step % NUM_STAGES
    for plane in gl.static_range(PLANES):
        h = h0 + plane // (TW + 2) - 1
        w = w0 + plane % (TW + 2) - 1
        inside = rows_inside & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        plane_offset = _add_in_loop(row_starts[plane // (TW + 2)], column_starts[plane % (TW + 2)]) * 16
        cp.async_copy_global_to_shared(
            a_planes[plane].index(stage), x_ptr + (a_cols + plane_offset), mask=inside[:, None]
        )
    for tap_hw in gl.static_range(9):
        w_rows = (tap_hw * 3 + kd) * CIN + cin_offset + k_w
        cp.async_copy_global_to_shared(w_taps[tap_hw].index(stage), w_ptr + w_rows[:, None] * COUT + n[None, :])
    cp.commit_group()


@gluon.jit
def _Conv_cl3d_tiled_halo(
    x_ptr,
    w_ptr,
    out_ptr,
    H,
    W,
    D,
    CIN: gl.constexpr,
    COUT: gl.constexpr,
    BLOCK_K: gl.constexpr,
    TH: gl.constexpr,
    TW: gl.constexpr,
    TD: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    POSITION_MAJOR: gl.constexpr,
    HALO_VEC: gl.constexpr,
    WEIGHT_VEC: gl.constexpr,
):
    """TH x TW positions in H/W times TD along D per program. Each pipeline step (cin block, kd) brings the halo planes
    and 9 weights into shared memory; taps index planes, so no plane is copied twice. The 3 taps of one kh row
    accumulate in an fp16 partial flushed into fp32. POSITION_MAJOR runs all taps of one position at a time, keeping
    one partial live. HALO_VEC and WEIGHT_VEC: fp16 per cp.async of the halo and the weights."""
    PLANES: gl.constexpr = (TH + 2) * (TW + 2)
    POSITIONS: gl.constexpr = TH * TW
    STEPS: gl.constexpr = 3 * (CIN // BLOCK_K)
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    mma_layout: gl.constexpr = _mma_layout(WARPS_M, WARPS_N)
    a_layout: gl.constexpr = _rows_layout(BLOCK_K, NUM_WARPS, HALO_VEC)
    w_layout: gl.constexpr = _rows_layout(COUT, NUM_WARPS, WEIGHT_VEC)
    out_layout: gl.constexpr = _rows_layout(COUT, NUM_WARPS, 8)
    a_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)

    w_tiles = gl.cdiv(W, TW)
    h_tiles = gl.cdiv(H, TH)
    d_blocks = gl.cdiv(D, TD)
    # D blocks first: neighbours along D share two halo rows, and running together they find them in L2.
    program = gl.program_id(0)
    d_block = program % d_blocks
    w_tile = program // d_blocks % w_tiles
    h_tile = program // (d_blocks * w_tiles) % h_tiles
    sample = program // (w_tiles * h_tiles * d_blocks)
    x_ptr += sample.to(gl.int64) * H * W * D * CIN
    out_ptr += sample.to(gl.int64) * H * W * D * COUT
    h0 = h_tile * TH
    w0 = w_tile * TW
    d0 = d_block * TD
    d = d0 + gl.arange(0, TD, layout=gl.SliceLayout(1, a_layout))
    k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_layout))
    k_w = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, w_layout))
    n = gl.arange(0, COUT, layout=gl.SliceLayout(0, w_layout))
    # Offsets of the halo rows and columns in units of 16 elements, so that the plane offset built from them in the loop
    # keeps the 16-element alignment cp.async needs.
    row_starts = ()
    for i in gl.static_range(TH + 2):
        row_starts = row_starts + ((h0 + i - 1) * W * D * (CIN // 16),)
    column_starts = ()
    for j in gl.static_range(TW + 2):
        column_starts = column_starts + ((w0 + j - 1) * D * (CIN // 16),)

    # One allocation per plane and per tap: the barrier pass treats an allocation as a single buffer and would put a
    # barrier before every copy into it.
    a_planes = ()
    for plane in gl.static_range(PLANES):
        a_planes = a_planes + (
            gl.allocate_shared_memory(gl.float16, [NUM_STAGES, TD, BLOCK_K], _swizzled_layout(BLOCK_K)),
        )
    w_taps = ()
    for tap_hw in gl.static_range(9):
        w_taps = w_taps + (gl.allocate_shared_memory(gl.float16, [NUM_STAGES, BLOCK_K, COUT], _swizzled_layout(COUT)),)

    for step in gl.static_range(NUM_STAGES - 1):
        _issue_halo_loads(
            step,
            x_ptr,
            w_ptr,
            a_planes,
            w_taps,
            row_starts,
            column_starts,
            h0,
            w0,
            d,
            H,
            W,
            D,
            k_a,
            k_w,
            n,
            CIN,
            COUT,
            BLOCK_K,
            NUM_STAGES,
            TH,
            TW,
        )

    accs = (gl.zeros([TD, COUT], gl.float32, mma_layout),) * POSITIONS
    for step in range(STEPS):
        _issue_halo_loads(
            step + NUM_STAGES - 1,
            x_ptr,
            w_ptr,
            a_planes,
            w_taps,
            row_starts,
            column_starts,
            h0,
            w0,
            d,
            H,
            W,
            D,
            k_a,
            k_w,
            n,
            CIN,
            COUT,
            BLOCK_K,
            NUM_STAGES,
            TH,
            TW,
        )
        cp.wait_group(NUM_STAGES - 1)
        stage = step % NUM_STAGES
        if POSITION_MAJOR:
            for position in gl.static_range(POSITIONS):
                for kh in gl.static_range(3):
                    part = gl.zeros([TD, COUT], gl.float16, mma_layout)
                    for kw in gl.static_range(3):
                        b = w_taps[kh * 3 + kw].index(stage).load(b_dot_layout)
                        a = (
                            a_planes[(position // TW + kh) * (TW + 2) + position % TW + kw]
                            .index(stage)
                            .load(a_dot_layout)
                        )
                        part = mma_v2(a, b, part)
                    accs = _with_item(accs, position, accs[position] + part.to(gl.float32))
        else:
            for kh in gl.static_range(3):
                parts = (gl.zeros([TD, COUT], gl.float16, mma_layout),) * POSITIONS
                for kw in gl.static_range(3):
                    b = w_taps[kh * 3 + kw].index(stage).load(b_dot_layout)
                    for position in gl.static_range(POSITIONS):
                        a = (
                            a_planes[(position // TW + kh) * (TW + 2) + position % TW + kw]
                            .index(stage)
                            .load(a_dot_layout)
                        )
                        parts = _with_item(parts, position, mma_v2(a, b, parts[position]))
                for position in gl.static_range(POSITIONS):
                    accs = _with_item(accs, position, accs[position] + parts[position].to(gl.float32))
    cp.wait_group(0)

    for position in gl.static_range(POSITIONS):
        _store_position(
            out_ptr, accs[position], h0 + position // TW, w0 + position % TW, d0, H, W, D, COUT, TD, out_layout
        )
