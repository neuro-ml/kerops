"""Conv3d forward kernels in gluon (sm_80+): cp.async into shared memory, mma_v2, layouts chosen explicitly."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2

TAPS = gl.constexpr(27)


@gluon.constexpr_function
def _rows_layout(row, num_warps, vec):
    """Each thread holds `vec` contiguous fp16 of a row-major tile with `row` columns. Triton lowers 16-byte cp.async
    (vec 8) to `cp.async.cg`, which bypasses L1; vec 4 gives 8-byte `cp.async.ca`, cached in L1."""
    threads_along_row = min(32, row // vec)
    return gl.BlockedLayout([1, vec], [32 // threads_along_row, threads_along_row], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def _swizzled_layout(row):
    """The swizzle triton picks for mma_v2 operands with `row` fp16 columns."""
    per_phase = max(1, 128 // (2 * row))
    return gl.SwizzledSharedLayout(vec=8, per_phase=per_phase, max_phase=8 // per_phase, order=[1, 0])


@gluon.constexpr_function
def _mma_layout(warps_m, warps_n):
    return gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[warps_m, warps_n], instr_shape=[16, 8])


@gluon.jit
def _with_item(items, index: gl.constexpr, value):
    return items[:index] + (value,) + items[index + 1:]


@gluon.jit
def _store_position(out_ptr, acc, h, w, d0, H, W, D, COUT: gl.constexpr, TD: gl.constexpr, out_layout: gl.constexpr):
    d = d0 + gl.arange(0, TD, layout=gl.SliceLayout(1, out_layout))
    n = gl.arange(0, COUT, layout=gl.SliceLayout(0, out_layout))
    inside = (d < D) & (h < H) & (w < W)
    out_offsets = (h * W + w) * D * COUT + (d * COUT)[:, None] + n[None, :]
    gl.store(out_ptr + out_offsets, gl.convert_layout(acc.to(gl.float16), out_layout), mask=inside[:, None])


@gluon.jit
def _issue_halo_loads(
    step, x_ptr, w_ptr, a_smem, w_smem, h0, w0, d, H, W, D, k_a, k_w, n,
    CIN: gl.constexpr, COUT: gl.constexpr, BLOCK_K: gl.constexpr, NUM_STAGES: gl.constexpr, TH: gl.constexpr, TW: gl.constexpr,
):
    # Step (cin block, kd): the (TH+2) x (TW+2) H/W halo planes of the input rows shifted by kd, and the 9 (kh, kw)
    # weights of kd. Steps past the end reload a valid step into a buffer nobody reads, which keeps the loop free of tails.
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
        cp.async_copy_global_to_shared(a_smem.index(stage * PLANES + plane), x_ptr + (h * W + w) * D * CIN + a_cols, mask=inside[:, None])
    for tap_hw in gl.static_range(9):
        w_rows = (tap_hw * 3 + kd) * CIN + cin_offset + k_w
        cp.async_copy_global_to_shared(w_smem.index(stage * 9 + tap_hw), w_ptr + w_rows[:, None] * COUT + n[None, :])
    cp.commit_group()


@gluon.jit
def _Conv_cl3d_tiled_halo(
    x_ptr, w_ptr, out_ptr, H, W, D,
    CIN: gl.constexpr, COUT: gl.constexpr, BLOCK_K: gl.constexpr, TH: gl.constexpr, TW: gl.constexpr, TD: gl.constexpr,
    NUM_STAGES: gl.constexpr, WARPS_M: gl.constexpr, WARPS_N: gl.constexpr, POSITION_MAJOR: gl.constexpr,
):
    """TH x TW positions in H/W times TD along D per program. Each pipeline step (cin block, kd) brings the halo planes
    and 9 weights into shared memory; taps index planes, so no plane is copied twice. The 3 taps of one kh row
    accumulate in an fp16 partial flushed into fp32. POSITION_MAJOR runs all taps of one position at a time, keeping
    one partial live."""
    PLANES: gl.constexpr = (TH + 2) * (TW + 2)
    POSITIONS: gl.constexpr = TH * TW
    STEPS: gl.constexpr = 3 * (CIN // BLOCK_K)
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    mma_layout: gl.constexpr = _mma_layout(WARPS_M, WARPS_N)
    a_layout: gl.constexpr = _rows_layout(BLOCK_K, NUM_WARPS, 4)
    w_layout: gl.constexpr = _rows_layout(COUT, NUM_WARPS, 4)
    out_layout: gl.constexpr = _rows_layout(COUT, NUM_WARPS, 8)
    a_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)

    d_blocks = gl.cdiv(D, TD)
    sample = gl.program_id(2) // d_blocks
    x_ptr += sample.to(gl.int64) * H * W * D * CIN
    out_ptr += sample.to(gl.int64) * H * W * D * COUT
    h0 = gl.program_id(1) * TH
    w0 = gl.program_id(0) * TW
    d0 = gl.program_id(2) % d_blocks * TD
    d = d0 + gl.arange(0, TD, layout=gl.SliceLayout(1, a_layout))
    k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_layout))
    k_w = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, w_layout))
    n = gl.arange(0, COUT, layout=gl.SliceLayout(0, w_layout))

    a_smem = gl.allocate_shared_memory(gl.float16, [NUM_STAGES * PLANES, TD, BLOCK_K], _swizzled_layout(BLOCK_K))
    w_smem = gl.allocate_shared_memory(gl.float16, [NUM_STAGES * 9, BLOCK_K, COUT], _swizzled_layout(COUT))

    for step in gl.static_range(NUM_STAGES - 1):
        _issue_halo_loads(step, x_ptr, w_ptr, a_smem, w_smem, h0, w0, d, H, W, D, k_a, k_w, n, CIN, COUT, BLOCK_K, NUM_STAGES, TH, TW)

    accs = (gl.zeros([TD, COUT], gl.float32, mma_layout),) * POSITIONS
    for step in range(STEPS):
        _issue_halo_loads(
            step + NUM_STAGES - 1, x_ptr, w_ptr, a_smem, w_smem, h0, w0, d, H, W, D, k_a, k_w, n, CIN, COUT, BLOCK_K, NUM_STAGES, TH, TW
        )
        cp.wait_group(NUM_STAGES - 1)
        planes = step % NUM_STAGES * PLANES
        weights = step % NUM_STAGES * 9
        if POSITION_MAJOR:
            for position in gl.static_range(POSITIONS):
                for kh in gl.static_range(3):
                    part = gl.zeros([TD, COUT], gl.float16, mma_layout)
                    for kw in gl.static_range(3):
                        b = w_smem.index(weights + kh * 3 + kw).load(b_dot_layout)
                        a = a_smem.index(planes + (position // TW + kh) * (TW + 2) + position % TW + kw).load(a_dot_layout)
                        part = mma_v2(a, b, part)
                    accs = _with_item(accs, position, accs[position] + part.to(gl.float32))
        else:
            for kh in gl.static_range(3):
                parts = (gl.zeros([TD, COUT], gl.float16, mma_layout),) * POSITIONS
                for kw in gl.static_range(3):
                    b = w_smem.index(weights + kh * 3 + kw).load(b_dot_layout)
                    for position in gl.static_range(POSITIONS):
                        a = a_smem.index(planes + (position // TW + kh) * (TW + 2) + position % TW + kw).load(a_dot_layout)
                        parts = _with_item(parts, position, mma_v2(a, b, parts[position]))
                for position in gl.static_range(POSITIONS):
                    accs = _with_item(accs, position, accs[position] + parts[position].to(gl.float32))
    cp.wait_group(0)

    for position in gl.static_range(POSITIONS):
        _store_position(out_ptr, accs[position], h0 + position // TW, w0 + position % TW, d0, H, W, D, COUT, TD, out_layout)


@gluon.jit
def _issue_gemm_loads(
    step, x_ptr, w_ptr, a_smem, b_smem, row_offsets, h, w, d, H, W, D, k_a, k_b, n,
    CIN: gl.constexpr, COUT: gl.constexpr, BLOCK_K: gl.constexpr, NUM_STAGES: gl.constexpr,
):
    # Steps past the end reload a valid step into a buffer nobody reads, which keeps the loop free of tail cases.
    wrapped = step % (TAPS * (CIN // BLOCK_K))
    cin_offset = wrapped // TAPS * BLOCK_K
    tap = wrapped % TAPS
    kh = tap // 9 - 1
    kw = tap // 3 % 3 - 1
    kd = tap % 3 - 1
    inside = (h + kh >= 0) & (h + kh < H) & (w + kw >= 0) & (w + kw < W) & (d + kd >= 0) & (d + kd < D)
    a_rows = row_offsets + ((kh * W + kw) * D + kd) * CIN + cin_offset
    cp.async_copy_global_to_shared(a_smem.index(step % NUM_STAGES), x_ptr + a_rows[:, None] + k_a[None, :], mask=inside[:, None])
    b_rows = tap * CIN + cin_offset + k_b
    cp.async_copy_global_to_shared(b_smem.index(step % NUM_STAGES), w_ptr + b_rows[:, None] * COUT + n[None, :])
    cp.commit_group()


@gluon.jit
def _Conv_cl3d_implicit_gemm(
    x_ptr, w_ptr, out_ptr, H, W, D,
    CIN: gl.constexpr, COUT: gl.constexpr, BLOCK_K: gl.constexpr,
    TILE_H: gl.constexpr, TILE_W: gl.constexpr, TILE_D: gl.constexpr,
    NUM_STAGES: gl.constexpr, WARPS_M: gl.constexpr, WARPS_N: gl.constexpr, FP16_CHAIN: gl.constexpr, VEC: gl.constexpr,
):
    """Implicit GEMM: M is a TILE_H x TILE_W x TILE_D block of output voxels, N is COUT, K runs over 27 taps x CIN with
    rows gathered by cp.async into a ring of NUM_STAGES buffers. Products accumulate in an fp16 partial flushed into
    fp32 every FP16_CHAIN K steps. VEC: fp16 per cp.async."""
    BLOCK_M: gl.constexpr = TILE_H * TILE_W * TILE_D
    STEPS: gl.constexpr = TAPS * (CIN // BLOCK_K)
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    mma_layout: gl.constexpr = _mma_layout(WARPS_M, WARPS_N)
    a_layout: gl.constexpr = _rows_layout(BLOCK_K, NUM_WARPS, VEC)
    b_layout: gl.constexpr = _rows_layout(COUT, NUM_WARPS, VEC)
    out_layout: gl.constexpr = _rows_layout(COUT, NUM_WARPS, 8)
    a_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)

    d_blocks = gl.cdiv(D, TILE_D)
    sample = gl.program_id(2) // d_blocks
    x_ptr += sample.to(gl.int64) * H * W * D * CIN
    out_ptr += sample.to(gl.int64) * H * W * D * COUT
    h0 = gl.program_id(1) * TILE_H
    w0 = gl.program_id(0) * TILE_W
    d0 = gl.program_id(2) % d_blocks * TILE_D

    m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_layout))
    h = h0 + m // (TILE_W * TILE_D)
    w = w0 + m // TILE_D % TILE_W
    d = d0 + m % TILE_D
    row_offsets = ((h * W + w) * D + d) * CIN
    k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_layout))
    k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, b_layout))
    n = gl.arange(0, COUT, layout=gl.SliceLayout(0, b_layout))

    a_smem = gl.allocate_shared_memory(gl.float16, [NUM_STAGES, BLOCK_M, BLOCK_K], _swizzled_layout(BLOCK_K))
    b_smem = gl.allocate_shared_memory(gl.float16, [NUM_STAGES, BLOCK_K, COUT], _swizzled_layout(COUT))

    for step in gl.static_range(NUM_STAGES - 1):
        _issue_gemm_loads(step, x_ptr, w_ptr, a_smem, b_smem, row_offsets, h, w, d, H, W, D, k_a, k_b, n, CIN, COUT, BLOCK_K, NUM_STAGES)

    acc = gl.zeros([BLOCK_M, COUT], gl.float32, mma_layout)
    partial = gl.zeros([BLOCK_M, COUT], gl.float16, mma_layout)
    for step in range(STEPS):
        _issue_gemm_loads(
            step + NUM_STAGES - 1, x_ptr, w_ptr, a_smem, b_smem, row_offsets, h, w, d, H, W, D, k_a, k_b, n,
            CIN, COUT, BLOCK_K, NUM_STAGES,
        )
        cp.wait_group(NUM_STAGES - 1)
        a = a_smem.index(step % NUM_STAGES).load(a_dot_layout)
        b = b_smem.index(step % NUM_STAGES).load(b_dot_layout)
        partial = mma_v2(a, b, partial)
        if step % FP16_CHAIN == FP16_CHAIN - 1:
            acc += partial.to(gl.float32)
            partial = gl.zeros([BLOCK_M, COUT], gl.float16, mma_layout)
    cp.wait_group(0)
    acc += partial.to(gl.float32)

    mo = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, out_layout))
    ho = h0 + mo // (TILE_W * TILE_D)
    wo = w0 + mo // TILE_D % TILE_W
    do = d0 + mo % TILE_D
    no = gl.arange(0, COUT, layout=gl.SliceLayout(0, out_layout))
    out_rows = ((ho * W + wo) * D + do) * COUT
    inside = (ho < H) & (wo < W) & (do < D)
    gl.store(out_ptr + out_rows[:, None] + no[None, :], gl.convert_layout(acc.to(gl.float16), out_layout), mask=inside[:, None])
