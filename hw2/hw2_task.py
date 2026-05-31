import torch
from torch.profiler import profile as torch_profile, ProfilerActivity
from transformers import DynamicCache

from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    device = input_ids.device
    token_buffer = torch.empty(n_steps, dtype=torch.long, device=device)

    with torch.inference_mode():
        past_key_values = DynamicCache()

        # Prefill: process the prompt once, cache K/V for every layer.
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_buffer[0] = next_token_id[0, 0]

        # Decode: feed only the new token each step, reuse the KV cache.
        for i in range(1, n_steps):
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            token_buffer[i] = next_token_id[0, 0]

    # Single CPU<->GPU sync at the end instead of one per step.
    return token_buffer.tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    # bf16 on L40S: matmuls run on tensor cores, halves memory bandwidth.
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix (measured cumulatively on L40S vs V0 fp32):
#
# 1. KV cache via DynamicCache + use_cache=True.
#    V0 re-runs the full prompt every step, so step t does O(prompt+t) work.
#    With the cache, prefill happens once and each decode step processes a
#    single token using cached K/V from prior steps. This is the dominant
#    structural fix — without it the loop is O(N^2) in total tokens.
#
# 2. bf16 model dtype.
#    fp32 matmuls don't hit the L40S tensor cores and move 2x the bytes.
#    Switching the build to torch.bfloat16 cuts memory bandwidth in half
#    and routes the GEMMs through tensor cores.
#
# 3. Drop the per-step .item() sync.
#    .item() forces a CPU<->GPU sync every iteration, which serializes the
#    launch queue and starves the GPU. The optimized loop writes each token
#    into a preallocated GPU buffer and calls .tolist() once at the end.
#
# 4. torch.inference_mode() around the loop.
#    Removes autograd bookkeeping (version counters, view tracking) that
#    .no_grad() still pays. Small but free win on a hot loop.
#
# 5. Avoid the growing torch.cat([generated_ids, next_token_id]) — with the
#    KV cache we only ever feed the latest token, so the concat disappears
#    entirely (no allocator churn, no copy of an N-token tensor each step).
#
# Biggest impact and why:
#
# The KV cache change is by far the biggest win. In the V0 trace each step's
# GPU stream gets longer than the last because attention re-scans the entire
# prefix; in the optimized trace every decode step is a fixed-cost slice
# (one-token QKV projection + cached attention + MLP). For PROMPT_LEN=1024
# and 128 new tokens, this collapses ~1024-1151 tokens of recomputation per
# step down to 1. bf16 is the second-biggest contributor — it's a constant
# factor (~2x) but stacks on top of the algorithmic fix.
