"""Measures the GPU's tensor-core and memory ceilings and the lower bound they give each Conv3d problem size."""
import re
import threading

import torch
import triton
from triton import language as tl
from triton.testing import do_bench

from kerops.ops.conv.conv import channel_pairs, tuning_size

MMA_BLOCK_M, MMA_BLOCK_N, MMA_BLOCK_K = 128, 128, 64
MMA_ITERATIONS = 4096


def bench_with_clock(fn) -> tuple[float, float]:
    """Median ms of `fn` and the median SM clock in MHz sampled while it runs."""
    clocks = []
    done = threading.Event()

    def sample_clock():
        while not done.wait(0.01):
            clocks.append(torch.cuda.clock_rate())

    sampler = threading.Thread(target=sample_clock)
    sampler.start()
    ms = do_bench(fn, warmup=50, rep=300, return_mode='median')
    done.set()
    sampler.join()
    return ms, sorted(clocks)[len(clocks) // 2]


@triton.jit
def _mma_loop(operand_ptr, out_ptr, iterations, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_DTYPE: tl.constexpr):
    a = tl.load(operand_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_K + tl.arange(0, BLOCK_K)[None, :])
    b = tl.load(operand_ptr + tl.arange(0, BLOCK_K)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
    acc = tl.zeros([BLOCK_M, BLOCK_N], ACC_DTYPE)
    for _ in range(iterations):
        acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE)
    offsets = tl.program_id(0) * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr + offsets, acc)


def mma_tflops(acc_dtype) -> tuple[float, float, str]:
    """Tensor-core TFLOPS of a register-resident dot loop, the SM clock, and the mma instruction it compiles to."""
    programs = 4 * torch.cuda.get_device_properties(0).multi_processor_count
    operand = torch.full([MMA_BLOCK_M * MMA_BLOCK_N], 1e-3, device='cuda', dtype=torch.float16)
    out = torch.empty(programs * MMA_BLOCK_M * MMA_BLOCK_N, device='cuda', dtype=torch.float32)

    def launch():
        return _mma_loop[(programs,)](operand, out, MMA_ITERATIONS, MMA_BLOCK_M, MMA_BLOCK_N, MMA_BLOCK_K, acc_dtype, num_warps=8)

    instruction = re.search(r'mma\.sync\.aligned\.\S+', launch().asm['ptx']).group(0)
    ms, mhz = bench_with_clock(launch)
    flop = 2 * programs * MMA_BLOCK_M * MMA_BLOCK_N * MMA_BLOCK_K * MMA_ITERATIONS
    return flop / ms / 1e9, mhz, instruction


def matmul_tflops() -> tuple[float, float]:
    size = 8192
    a = torch.randn(size, size, device='cuda', dtype=torch.float16)
    b = torch.randn(size, size, device='cuda', dtype=torch.float16)
    ms, mhz = bench_with_clock(lambda: a @ b)
    return 2 * size**3 / ms / 1e9, mhz


def copy_gb_per_s() -> float:
    source = torch.empty(2**28, device='cuda', dtype=torch.float16)
    target = torch.empty_like(source)
    ms, _ = bench_with_clock(lambda: target.copy_(source))
    return 2 * source.numel() * source.element_size() / ms / 1e6


def main():
    tflops_acc32, mhz_acc32, instruction_acc32 = mma_tflops(tl.float32)
    tflops_acc16, mhz_acc16, instruction_acc16 = mma_tflops(tl.float16)
    tflops_matmul, mhz_matmul = matmul_tflops()
    gb_per_s = copy_gb_per_s()

    print(f'mma fp32 acc   {tflops_acc32:6.1f} TFLOPS  {mhz_acc32:5.0f} MHz  {instruction_acc32}')
    print(f'mma fp16 acc   {tflops_acc16:6.1f} TFLOPS  {mhz_acc16:5.0f} MHz  {instruction_acc16}')
    print(f'cuBLAS matmul  {tflops_matmul:6.1f} TFLOPS  {mhz_matmul:5.0f} MHz  8192³ fp16')
    print(f'copy           {gb_per_s:6.0f} GB/s')
    print()
    print('lower bounds, ms: compute at fp32 / fp16 acc ceiling, memory (read input + weights, write output once)')
    for pair in channel_pairs():
        in_channels, out_channels = pair['in_channels'], pair['out_channels']
        size = tuning_size(in_channels, out_channels)
        flop = 2 * size**3 * 27 * in_channels * out_channels
        bytes_moved = 2 * (size**3 * (in_channels + out_channels) + 27 * in_channels * out_channels)
        print(
            f'{in_channels:>3}->{out_channels:<3} {size}³  fp32 acc {flop / tflops_acc32 / 1e9:6.3f}'
            f'  fp16 acc {flop / tflops_acc16 / 1e9:6.3f}  memory {bytes_moved / gb_per_s / 1e6:6.3f}'
        )


if __name__ == '__main__':
    main()
