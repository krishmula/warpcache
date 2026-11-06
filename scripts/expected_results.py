"""Expected decode latencies on the RTX 3090, from a roofline-plus-latency model.

These are predictions for the class slides, not measurements. Replace each one
with the benchmark number once it exists.

    uv run python scripts/expected_results.py            # table
    uv run python scripts/expected_results.py --json     # for the deck builder
"""

import json
import math
import sys

PEAK_BW = 936e9  # RTX 3090 GDDR6X, bytes/s
KERNEL_BW = 0.85 * PEAK_BW  # the Gate 0 smoke test reached 87% of peak DRAM throughput
SDPA_BW = 0.75 * PEAK_BW
SMS = 82
MAX_WARPS = SMS * 16  # 32-thread blocks hit the 16 resident blocks per SM limit first

HQ, HKV, D = 16, 8, 128
BYTES_PER_TOKEN = {
    "fp16": 2 * HKV * D * 2,
    "int8": 2 * HKV * D * 1 + 2 * HKV * 4,  # K and V rows plus one fp32 scale each
}

# Path A v1 is one warp per (request, query head) walking tokens one at a time.
# Each step waits on a K/V load, five shuffles, and two expf. These per-token
# costs are guesses in the 0.15 to 0.6 us range, not measurements.
T_TOK = {"fp16": 0.30e-6, "int8": 0.33e-6}

LAUNCH = 3e-6
MERGE = 25e-6  # about five eager PyTorch ops on the (m, l, u) workspace
SDPA_FIXED = 10e-6
CHUNKS = [2**i for i in range(1, 9)]

BATCHES = [1, 4, 16]
CONTEXTS = [1024, 4096, 16384]


def kernel_time(batch, ctx, fmt, chunks):
    warps = batch * HQ * chunks
    tokens_per_warp = math.ceil(ctx / chunks)
    waves = math.ceil(warps / MAX_WARPS)
    latency = waves * tokens_per_warp * T_TOK[fmt]
    memory = batch * ctx * BYTES_PER_TOKEN[fmt] / KERNEL_BW
    limiter = "latency" if latency > memory else "DRAM"
    return LAUNCH + max(latency, memory), limiter


def path_a(batch, ctx, fmt):
    t, limiter = kernel_time(batch, ctx, fmt, 1)
    return {"us": t * 1e6, "limiter": limiter}


def path_b(batch, ctx, fmt):
    best = None
    for c in CHUNKS:
        if ctx / c < 16:
            break
        t, limiter = kernel_time(batch, ctx, fmt, c)
        t += MERGE
        if best is None or t < best["us"] / 1e6:
            best = {"us": t * 1e6, "limiter": limiter, "chunks": c}
    return best


def sdpa(batch, ctx):
    t = SDPA_FIXED + batch * ctx * BYTES_PER_TOKEN["fp16"] / SDPA_BW
    return {"us": t * 1e6, "limiter": "DRAM"}


def dram_pct(batch, ctx, fmt, us):
    return 100 * batch * ctx * BYTES_PER_TOKEN[fmt] / (us * 1e-6) / PEAK_BW


def rows():
    out = []
    for batch in BATCHES:
        for ctx in CONTEXTS:
            row = {"batch": batch, "ctx": ctx, "sdpa_fp16": sdpa(batch, ctx)}
            for fmt in ("fp16", "int8"):
                for name, fn in (("a", path_a), ("b", path_b)):
                    r = fn(batch, ctx, fmt)
                    r["dram_pct"] = dram_pct(batch, ctx, fmt, r["us"])
                    row[f"{name}_{fmt}"] = r
            out.append(row)
    return out


def main():
    data = rows()
    if "--json" in sys.argv:
        print(json.dumps({"bytes_per_token": BYTES_PER_TOKEN, "rows": data}, indent=1))
        return
    cols = ["sdpa_fp16", "a_fp16", "a_int8", "b_fp16", "b_int8"]
    print(f"{'batch':>5} {'ctx':>6} " + " ".join(f"{c:>10}" for c in cols) + "  B chunks  B int8 limiter")
    for r in data:
        cells = " ".join(f"{r[c]['us']:>10.0f}" for c in cols)
        print(f"{r['batch']:>5} {r['ctx']:>6} {cells}  {r['b_fp16']['chunks']:>8}  {r['b_int8']['limiter']}")


if __name__ == "__main__":
    main()
