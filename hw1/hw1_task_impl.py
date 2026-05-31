import statistics

import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        fn(*args)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return statistics.median(times)


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # Each `acc = acc * x + x` iteration does one multiply and one add = 2 FLOPs/element.
    total_flops = 2 * num_ops * num_elements

    if variant == "compiled":
        # Fused kernel: one read of x + one write of result at the kernel boundary.
        total_bytes = 2 * num_elements * bytes_per_element
    else:
        # Eager: each iteration launches a separate multiply and add kernel.
        # mul: read acc, read x, write tmp  -> 3 tensor accesses
        # add: read tmp, read x, write acc  -> 3 tensor accesses
        # = 6 element accesses per iteration.
        total_bytes = 6 * num_ops * num_elements * bytes_per_element

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# A1. These kernels are memory-bound: each fused kernel still reads x once and
# writes the result once, so the bytes transferred are the same regardless of
# num_ops. The runtime is set by HBM bandwidth, so it stays roughly flat. But
# the FLOPs counted in the numerator scale linearly with num_ops, so achieved
# FLOP/s = total_flops / time climbs proportionally even though we are just
# packing more arithmetic into the same memory transfer.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A2. A 1024x1024 FP32 matmul is only ~2.1 GFLOPs of work — far too small to
# saturate an H100's 132 SMs. It finishes in a fraction of a millisecond, so
# kernel launch overhead and tail effects from a low-occupancy tile schedule
# dominate. Also, the FP32 path does not use Tensor Cores, so the matmul has no
# special hardware advantage over a fused element-wise kernel; once AI is well
# past the ridge point, both are throughput-limited by the same FP32 CUDA-core
# pipeline, and the element-wise loop just happens to keep that pipeline full.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# A3. We have crossed the ridge point of the roofline and become compute-bound.
# For small num_ops the time is set by bandwidth (constant bytes, constant time).
# Once arithmetic intensity is high enough that the FP32 ALU pipeline cannot
# finish 2*num_ops FLOPs per element in the time it takes to stream the data,
# additional ops add real wall-clock time, so doubling num_ops roughly doubles
# the kernel duration.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# A4. Eager mode runs each `acc * x` and `+ x` as a separate kernel, materializing
# a fresh intermediate tensor for every iteration. The byte traffic grows linearly
# with num_ops (about 6 element accesses per iteration instead of 2 total), so
# arithmetic intensity stays pinned at a small constant (~1/12 FLOP/Byte) instead
# of rising with num_ops. On top of that, every iteration pays kernel launch
# overhead and re-reads x from HBM. The result is a vertical cluster of points
# stuck on the memory-bound part of the roofline, far below the compiled curve
# that fuses everything into a single kernel.
