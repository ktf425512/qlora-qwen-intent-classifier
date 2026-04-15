"""
Measure inference latency and throughput for the merged model.

Loads ./merged_model/ and reports:
  - VRAM footprint (after load + peak during benchmarking)
  - Single-query latency distribution (p50 / p90 / p99 over 50 runs)
  - Batched throughput at batch sizes 1, 4, 16, 32
  - A back-of-envelope monthly cost comparison against an API provider

Output is written to outputs/benchmark.json for the README to reference.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("./merged_model")
OUT_JSON  = Path("./outputs/benchmark.json")
MAX_NEW_TOKENS = 16

SYSTEM_PROMPT = (
    "You are an intent classifier for customer-support messages. "
    "Respond with exactly one intent label from the allowed set. "
    "Output only the label, nothing else."
)

SAMPLE_MESSAGES = [
    "I need to cancel my subscription",
    "Can you tell me where my package is?",
    "How do I change my shipping address?",
    "What is your refund policy?",
    "I can't log into my account",
    "Give me the customer service phone number",
    "I want to leave some feedback about my recent experience",
    "When will my order arrive?",
]

# Assumptions for the cost comparison. Conservative defaults - adjust to
# your own hardware / electricity / utilization to re-run the math.
MONTHLY_VOLUME          = 10_000_000            # classifications per month
INPUT_TOKENS_PER_CALL   = 80                    # approx system+user tokens
OUTPUT_TOKENS_PER_CALL  = 5                     # short label output
GPU_PURCHASE_USD        = 1800                  # RTX 4090 retail
GPU_LIFETIME_HOURS      = 3 * 365 * 24          # 3-year amortization
GPU_POWER_KW            = 0.27                  # ~270 W under this workload
ELECTRICITY_USD_PER_KWH = 0.15
# GPT-4o-mini public pricing (check current pricing on openai.com/api/pricing)
API_INPUT_USD_PER_MTOK  = 0.15
API_OUTPUT_USD_PER_MTOK = 0.60


def build_prompt(tokenizer, message: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def measure_single_latency(model, tokenizer, messages, n_warmup=10, n_measure=50):
    prompts = [build_prompt(tokenizer, m) for m in messages]
    # warmup
    for i in range(n_warmup):
        enc = tokenizer(prompts[i % len(prompts)], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    # measure
    latencies_ms = []
    for i in range(n_measure):
        enc = tokenizer(prompts[i % len(prompts)], return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return latencies_ms


def measure_batch_throughput(model, tokenizer, messages, batch_size, n_batches=10):
    tokenizer.padding_side = "left"
    batch_msgs = [messages[i % len(messages)] for i in range(batch_size * n_batches)]
    prompts    = [build_prompt(tokenizer, m) for m in batch_msgs]

    # warmup (2 batches)
    for i in range(2):
        b = prompts[i * batch_size : (i + 1) * batch_size]
        enc = tokenizer(b, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(n_batches):
        b = prompts[i * batch_size : (i + 1) * batch_size]
        enc = tokenizer(b, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    n_total = batch_size * n_batches
    return n_total / elapsed_s, elapsed_s


def percentile(sorted_vals, pct):
    idx = max(0, min(len(sorted_vals) - 1, int(pct * len(sorted_vals))))
    return sorted_vals[idx]


def cost_comparison(best_throughput_per_sec: float) -> dict:
    """Monthly cost for MONTHLY_VOLUME classifications: self-hosted vs API."""
    # Self-hosted
    gpu_usd_per_hour   = GPU_PURCHASE_USD / GPU_LIFETIME_HOURS
    power_usd_per_hour = GPU_POWER_KW * ELECTRICITY_USD_PER_KWH
    total_usd_per_hour = gpu_usd_per_hour + power_usd_per_hour
    seconds_needed     = MONTHLY_VOLUME / best_throughput_per_sec
    self_hosted_usd    = (seconds_needed / 3600.0) * total_usd_per_hour

    # API (GPT-4o-mini public pricing)
    input_mtok         = (MONTHLY_VOLUME * INPUT_TOKENS_PER_CALL)  / 1e6
    output_mtok        = (MONTHLY_VOLUME * OUTPUT_TOKENS_PER_CALL) / 1e6
    api_usd            = input_mtok * API_INPUT_USD_PER_MTOK + output_mtok * API_OUTPUT_USD_PER_MTOK

    return {
        "monthly_volume":    MONTHLY_VOLUME,
        "input_tokens_call": INPUT_TOKENS_PER_CALL,
        "output_tokens_call": OUTPUT_TOKENS_PER_CALL,
        "self_hosted_usd_per_month": round(self_hosted_usd, 2),
        "self_hosted_breakdown": {
            "gpu_amortized_usd_per_hour":  round(gpu_usd_per_hour, 4),
            "electricity_usd_per_hour":    round(power_usd_per_hour, 4),
            "gpu_hours_needed_per_month":  round(seconds_needed / 3600.0, 2),
        },
        "api_usd_per_month":  round(api_usd, 2),
        "api_pricing_used": {
            "input_usd_per_mtok":  API_INPUT_USD_PER_MTOK,
            "output_usd_per_mtok": API_OUTPUT_USD_PER_MTOK,
        },
        "cost_ratio_api_over_self_hosted": round(api_usd / self_hosted_usd, 1) if self_hosted_usd > 0 else None,
    }


def main():
    print(f"Loading merged model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    vram_load_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  loaded. VRAM after load: {vram_load_gb:.2f} GB\n")

    # ─── Single-query latency ────────────────────────────────────────────
    print("Measuring single-query latency (10 warmup + 50 measured runs) ...")
    lat = measure_single_latency(model, tokenizer, SAMPLE_MESSAGES)
    lat_sorted = sorted(lat)
    lat_stats = {
        "mean":  statistics.mean(lat),
        "p50":   percentile(lat_sorted, 0.50),
        "p90":   percentile(lat_sorted, 0.90),
        "p99":   percentile(lat_sorted, 0.99),
        "min":   min(lat),
        "max":   max(lat),
    }
    print(f"  p50={lat_stats['p50']:.1f} ms   p90={lat_stats['p90']:.1f} ms   "
          f"p99={lat_stats['p99']:.1f} ms   mean={lat_stats['mean']:.1f} ms\n")

    # ─── Batched throughput ──────────────────────────────────────────────
    print("Measuring batched throughput ...")
    throughput_per_bs = {}
    for bs in [1, 4, 16, 32]:
        tp, total = measure_batch_throughput(model, tokenizer, SAMPLE_MESSAGES, bs, n_batches=10)
        throughput_per_bs[bs] = round(tp, 1)
        print(f"  batch={bs:>2}   {tp:6.1f} classifications/sec   ({total:.2f}s total)")
    best_tp = max(throughput_per_bs.values())

    vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak VRAM during benchmark: {vram_peak_gb:.2f} GB\n")

    # ─── Cost comparison ─────────────────────────────────────────────────
    costs = cost_comparison(best_tp)
    print(f"Cost comparison for {MONTHLY_VOLUME:,} classifications / month:")
    print(f"  self-hosted (amortized 4090 + electricity): ${costs['self_hosted_usd_per_month']:>7.2f} / month")
    print(f"  GPT-4o-mini API (at quoted pricing):        ${costs['api_usd_per_month']:>7.2f} / month")
    if costs["cost_ratio_api_over_self_hosted"]:
        print(f"  API / self-hosted ratio: {costs['cost_ratio_api_over_self_hosted']}x")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "vram_load_gb":            round(vram_load_gb, 2),
        "vram_peak_gb":            round(vram_peak_gb, 2),
        "latency_ms":              {k: round(v, 2) for k, v in lat_stats.items()},
        "throughput_per_bs":       throughput_per_bs,
        "best_throughput_per_sec": round(best_tp, 1),
        "cost_comparison":         costs,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
