# HW2: LLM Inference Optimization

## Goal

Write the fastest possible autoregressive generation loop for a tiny decoder-only transformer. You'll start from a slow baseline, identify what's wasting time, and apply a series of optimizations to reach a significant speedup.

## Setup

The model is a randomly-initialized 2-layer Llama built from a `LlamaConfig` (`hidden_size=2048`, `intermediate_size=6144`, 8 attention heads, 8 KV heads, `vocab_size=4096`). It runs on CUDA with synthetic random token IDs — no tokenizer, no pretrained weights — so the trace stays focused on the generation loop itself rather than I/O or model loading.

Two run lengths are used:

- **Profiling** runs `PROFILE_STEPS = 12` decode steps so the trace stays small enough to navigate in Perfetto.
- **Timing** runs `MAX_NEW_TOKENS = 128` decode steps from a `PROMPT_LEN = 1024` prompt, which is what the speedup numbers are measured against.

Each "step" is one forward pass through the model, so the slow baseline does 12 forward passes per profile and 128 per timed run.

## Speedup targets

Measured on an L40S against the V0 slow baseline (`slow_loop`, fp32, no KV cache):

- **Good:** > 3× speedup — bf16 + a couple of host-side fixes.
- **Great:** > 4× speedup — full stack, including `torch.compile`.

## File Layout

- `hw2_task.py`: **your only file to edit** — implement all three functions described below
- `utils.py`: provided helpers — `build_model`, `slow_loop`, `time_generation`, `get_input_ids` (do not modify)
- `results/`: output directory for Chrome trace files

## Run

From repository root:

```bash
python3 hw2/hw2_task.py
```

## Your Tasks

All three functions live in `hw2_task.py` and must be completed:

- `**profile(loop_fn, model, input_ids, trace_name)**` — wrap `loop_fn` with `torch.profiler`, print the summary table, and export a Chrome trace to `results/trace_name`.
- `**optimized_loop(model, input_ids, n_steps)**` — starts as a copy of `slow_loop`. Make it as fast as possible. Changes may span the loop body and the model loading in `generate_optimized()`.
- `**generate_optimized()**` — build the tiny Llama (consider dtype and other loading options too), then call `profile` and `time_generation` on `optimized_loop`.

## Constraints

- Stay within **PyTorch** and the libraries already pinned in `requirements.txt`. Anything built into PyTorch or `transformers` is fair game — figuring out *which* built-ins are worth reaching for is part of the exercise.
- **Do not** use dedicated inference engines such as vLLM, TensorRT-LLM, TGI, SGLang, llama.cpp, ExLlama, or similar.
- `utils.py` is provided and must not be modified — the slow baseline, model builder, and timing helpers must stay identical so your speedup numbers are comparable to the targets above.

## Background

### torch.profiler

`torch.profiler` records every PyTorch operator, CUDA kernel launch, and GPU kernel execution. See the [official docs](https://docs.pytorch.org/docs/2.11/profiler.html) for the full API. It produces two types of output:

1. **Summary table** — sorted by CPU or CUDA time, shows call counts and averages. Good for finding expensive operators.
2. **Chrome trace** — a JSON file you open at [ui.perfetto.dev](https://ui.perfetto.dev). Shows a timeline of all events on CPU threads and GPU streams.

### Reading Chrome Traces

When you open a trace in Perfetto you'll see several rows. The two most important are:

- **The CPU thread** (`python <pid>` → main thread) — a nested stack of colored bars, one per PyTorch operator. The outermost bars are high-level ops (e.g. `aten::linear`); inner bars are what they decompose into. The `aten::` prefix is PyTorch's C++ tensor library ("A Tensor Library") — every built-in op like `add`, `matmul`, or `item` lives in that namespace. At the very bottom of each stack you'll usually find a `cuda_runtime` event such as `cudaLaunchKernel` — the CPU handing off work to the GPU driver.
- **The GPU stream** (`stream N`) — where CUDA kernels actually execute on the hardware. A kernel bar here starts slightly after its `cudaLaunchKernel` on the CPU side.

The relationship between these rows tells you a lot:

```
cpu_op: aten::some_op
  └── cuda_runtime: cudaLaunchKernel   ← CPU hands off, then moves on immediately
                                              ↓  (async gap)
GPU stream: some_kernel                ← GPU executes it later
```

Because launches are asynchronous, a healthy trace has the CPU and GPU rows both densely filled and overlapping. When the CPU thread has long spans with **no `cudaLaunchKernel` at the bottom**, the CPU is doing real computation itself instead of delegating to the GPU — and the GPU stream goes quiet.

When investigating a slow trace, compare total GPU kernel time against the profile's wall-clock span: if kernel time ≪ wall the GPU is idle most of the time and you're dispatch bound (the gaps are either the CPU waiting on the GPU or the GPU waiting on the CPU to queue the next kernel), and if kernel time ≈ wall the GPU is saturated and you're compute bound (find which kernels dominate and ask whether that work is necessary and whether it's running on the fastest hardware path available).

**Trust the trace for structure, not for wall-clock.** `torch.profiler` pays a few microseconds of bookkeeping per aten op, which is negligible when ops are large but can dominate the CPU timeline when there are hundreds of tiny ops per step. Use the trace to find *what* to fix (kernel names and shapes, launch patterns, per-step sync points, redundant work); use the unprofiled `time_generation` numbers to confirm *how much* the fix actually helps.

**You don't have to lead with the trace.** Reading the baseline loop and asking "is anything here obviously wasteful?" is a perfectly valid approach for this homework — you can form a hypothesis from the code alone, apply the change, measure with `time_generation`, and use the trace to *understand* or *confirm* the speedup rather than to discover it. What matters is that each claim in your writeup is backed by either a trace observation or a timing measurement.

## Writeup

In the comment block at the bottom of `hw2_task.py`, briefly describe:

- What you changed and why
- The speedup each fix contributed
- Which fix had the biggest impact and why

