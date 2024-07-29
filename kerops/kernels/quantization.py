import triton
from triton import language as tl


@triton.jit
def _QuantUint8Window_impl(
    input_ptr,
    output_ptr,
    numel,
    window,
    BLOCK_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)

    input_ptr += tid * BLOCK_SIZE
    output_ptr += tid * BLOCK_SIZE

    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < numel - tid * BLOCK_SIZE

    input = tl.load(input_ptr + offset, mask=mask).to(tl.float32)
    input = tl.minimum(tl.maximum(input, -window), window)  # clip
    input = (input + window) / (2 * window)  # normalize
    input *= 255
    input = input.to(tl.uint8)

    tl.store(output_ptr + offset, input, mask=mask)


@triton.jit
def _DequantUint8Window_impl(
    input_ptr,
    output_ptr,
    numel,
    window,
    BLOCK_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)

    input_ptr += tid * BLOCK_SIZE
    output_ptr += tid * BLOCK_SIZE

    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < numel - tid * BLOCK_SIZE

    input = tl.load(input_ptr + offset, mask=mask).to(tl.float32)
    input = input * (2 * window / 255) - window

    tl.store(output_ptr + offset, input, mask=mask)
