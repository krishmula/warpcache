"""Time Path A, Path B, SDPA and cache_write on the GPU with CUDA events.

Writes one JSON result line per case to results/bench/<host>-<time>.jsonl. Each
line carries the git commit and the path of the host manifest written at start.

    uv run python scripts/bench.py                               # full grid
    uv run python scripts/bench.py --kernel path_a --dtype fp16 --batch 1 --history 16384
    uv run python scripts/bench.py --compare A.jsonl B.jsonl     # medians within 3%?

Timing protocol, per case:
- Working set. Every case reads at least 24 MiB (4x the 3090's 6 MB L2). Small
  cases get several copies of their pages in one cache, each copy with its own
  block table, and the timed calls rotate through the copies. So a call never
  reads pages that the call before it left in L2.
- 10 warmup calls, then 50 timed calls. A CUDA event pair brackets each call.
  A GPU sleep runs first so the CPU queues all 50 calls before the GPU reaches
  them: the events measure GPU time, not Python launch time.
- wall_us is the host view: 50 back-to-back calls timed with the CPU clock.
  It is larger than the GPU time when the CPU cannot launch work fast enough,
  as with the PyTorch merge in Path B.
- Before timing, each decode case checks its output against SDPA on the same
  history, so a timing never comes from a call that did the wrong work.
"""

import argparse
import json
import math
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reference.quant import dequantize, quantize  # noqa: E402

HQ, HKV, D, PAGE = 16, 8, 128, 16
PEAK_GBPS = 936.0
MIN_WORKING_SET = 24 * 2**20
WARMUP, ITERS = 10, 50
SLEEP_CYCLES = 50_000_000  # about 30 ms at 3090 clocks, longer than queueing 50 calls
REPEAT_TOLERANCE = 0.03
MAX_REL_ERROR = 1e-2

KERNELS = ["path_a", "path_b", "sdpa", "cache_write"]
DTYPES = {"fp16": torch.float16, "int8": torch.int8}
BATCHES = [1, 4, 16]
HISTORIES = [4096, 16384]
CHUNKS = [2, 4, 8, 16, 32, 64, 128, 256]


def bytes_per_token(dtype: str) -> int:
    """Cache bytes one token occupies: K and V rows, plus one FP32 scale each for INT8."""
    if dtype == "fp16":
        return 2 * HKV * D * 2
    return 2 * HKV * D + 2 * HKV * 4


def case_key(line: dict) -> tuple:
    return (line["kernel"], line["dtype"], line["batch"], line["history"], line["chunks"])


def compare(a: list[dict], b: list[dict]) -> list[tuple[tuple, float]]:
    """Relative median difference |b - a| / a for each case in both runs."""
    b_by_key = {case_key(line): line for line in b}
    out = []
    for line in a:
        other = b_by_key.get(case_key(line))
        if other is not None:
            out.append((case_key(line), abs(other["median_us"] - line["median_us"]) / line["median_us"]))
    return out


def read_lines(path: str) -> list[dict]:
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


# Inputs


class Inputs:
    """One (batch, history) shape: an FP16 and an INT8 cache with `copies` page sets."""

    def __init__(self, batch: int, history: int, smallest_bytes: int):
        self.batch, self.history = batch, history
        pages = history // PAGE
        self.copies = max(1, math.ceil(MIN_WORKING_SET / smallest_bytes))
        num_pages = self.copies * batch * pages
        dev = "cuda"
        perm = torch.randperm(num_pages, device=dev).int()
        self.tables = list(perm.view(self.copies, batch, pages))
        self.seq_lens = torch.full((batch,), history, dtype=torch.int32, device=dev)
        self.q = [torch.randn(batch, HQ, D, device=dev).half() for _ in range(self.copies)]
        shape = (num_pages, PAGE, HKV, D)
        k16 = torch.randn(shape, device=dev).half()
        v16 = torch.randn(shape, device=dev).half()
        self.cache = {"fp16": (k16, v16, None, None), "int8": _quantize_cache(k16, v16)}

    def decode_args(self, dtype: str, copy: int) -> tuple:
        k, v, ks, vs = self.cache[dtype]
        return self.q[copy], k, v, ks, vs, self.tables[copy], self.seq_lens

    def gathered(self, dtype: str, copy: int) -> tuple[torch.Tensor, torch.Tensor]:
        """FP16 history laid out for SDPA: [batch, HKV, history, D]."""
        k, v, ks, vs = self.cache[dtype]
        table = self.tables[copy].long()

        def one(c, s):
            rows = c[table]  # [batch, pages, PAGE, HKV, D]
            if s is not None:
                rows = dequantize(rows, s[table]).half()
            return rows.reshape(self.batch, self.history, HKV, D).transpose(1, 2).contiguous()

        return one(k, ks), one(v, vs)


def _quantize_cache(k16, v16):
    out = []
    for x in (k16, v16):
        q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        s = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
        for i in range(0, x.shape[0], 4096):
            q[i : i + 4096], s[i : i + 4096] = quantize(x[i : i + 4096])
        out.append((q, s))
    (kq, ks), (vq, vs) = out
    return kq, vq, ks, vs


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q.unsqueeze(2), k, v, enable_gqa=True).squeeze(2)


# Timing


def time_calls(calls: list) -> dict:
    """GPU time per call with CUDA events, plus wall time per call, in microseconds."""
    n = len(calls)
    for i in range(WARMUP):
        calls[i % n]()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    torch.cuda._sleep(SLEEP_CYCLES)
    for i in range(ITERS):
        starts[i].record()
        calls[i % n]()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))

    t0 = time.perf_counter()
    for i in range(ITERS):
        calls[i % n]()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / ITERS * 1e6

    median = statistics.median(us)
    p10, p90 = us[ITERS // 10], us[ITERS - 1 - ITERS // 10]
    return {
        "median_us": median,
        "p10_us": p10,
        "p90_us": p90,
        "min_us": us[0],
        "max_us": us[-1],
        "spread_pct": 100 * (p90 - p10) / median,
        "wall_us": wall,
    }


def check_close(got: torch.Tensor, want: torch.Tensor, what: str) -> float:
    err = ((got.float() - want.float()).norm() / want.float().norm()).item()
    if not err < MAX_REL_ERROR:
        raise AssertionError(f"{what}: relative error {err:.2e} against SDPA")
    return err


# Cases


def decode_cases(inp: Inputs, args, decode):
    for dtype in args.dtype if {"path_a", "path_b"} & set(args.kernel) else []:
        want = sdpa(inp.q[0], *inp.gathered(dtype, 0))
        if "path_a" in args.kernel:
            yield _decode_case(inp, dtype, 1, want, decode)
        if "path_b" in args.kernel:
            for c in args.chunks:
                if inp.history // c >= PAGE:
                    yield _decode_case(inp, dtype, c, want, decode)
        del want
    if "sdpa" in args.kernel and "fp16" in args.dtype:
        kv = [inp.gathered("fp16", i) for i in range(inp.copies)]
        calls = [lambda i=i: sdpa(inp.q[i], *kv[i]) for i in range(inp.copies)]
        bytes_read = inp.batch * inp.history * bytes_per_token("fp16")
        yield {"kernel": "sdpa", "dtype": "fp16", "chunks": None, "bytes": bytes_read,
               "copies": inp.copies, **time_calls(calls)}
        del kv


def _decode_case(inp, dtype, chunks, want, decode):
    got = decode(*inp.decode_args(dtype, 0), num_chunks=chunks)
    err = check_close(got, want, f"{dtype} batch {inp.batch} history {inp.history} chunks {chunks}")
    calls = [lambda i=i: decode(*inp.decode_args(dtype, i), num_chunks=chunks) for i in range(inp.copies)]
    return {
        "kernel": "path_a" if chunks == 1 else "path_b",
        "dtype": dtype,
        "chunks": chunks,
        "bytes": inp.batch * inp.history * bytes_per_token(dtype),
        "copies": inp.copies,
        "rel_error_vs_sdpa": err,
        **time_calls(calls),
    }


def cache_write_cases(inp: Inputs, args, cache_write):
    """One new token per request, written to the last slot of its history."""
    batch, last = inp.batch, inp.history - 1
    for dtype in args.dtype:
        k, v, ks, vs = inp.cache[dtype]
        calls = []
        for i in range(inp.copies):
            slots = (inp.tables[i][:, last // PAGE].long() * PAGE + last % PAGE).contiguous()
            nk = torch.randn(batch, HKV, D, device="cuda").half()
            nv = torch.randn(batch, HKV, D, device="cuda").half()
            calls.append(lambda s=slots, nk=nk, nv=nv: cache_write(nk, nv, s, k, v, ks, vs))
        row = HKV * D * (2 if dtype == "fp16" else 1) + (HKV * 4 if dtype == "int8" else 0)
        yield {"kernel": "cache_write", "dtype": dtype, "chunks": None,
               "bytes": batch * 2 * (HKV * D * 2 + row), "copies": inp.copies, **time_calls(calls)}


# Main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kernel", nargs="+", choices=KERNELS, default=KERNELS)
    p.add_argument("--dtype", nargs="+", choices=list(DTYPES), default=list(DTYPES))
    p.add_argument("--batch", nargs="+", type=int, default=BATCHES)
    p.add_argument("--history", nargs="+", type=int, default=HISTORIES)
    p.add_argument("--chunks", nargs="+", type=int, default=CHUNKS, help="Path B chunk counts")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two result files")
    args = p.parse_args(argv)
    for h in args.history:
        if h % PAGE:
            p.error(f"history {h} is not a multiple of the page size {PAGE}")
    return args


def run_compare(a: str, b: str) -> int:
    worst = 0.0
    for key, diff in compare(read_lines(a), read_lines(b)):
        flag = "  OVER 3%" if diff > REPEAT_TOLERANCE else ""
        print(f"{' '.join(map(str, key)):<40} {100 * diff:6.2f}%{flag}")
        worst = max(worst, diff)
    print(f"worst median difference {100 * worst:.2f}%")
    return 0 if worst <= REPEAT_TOLERANCE else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.compare:
        return run_compare(*args.compare)
    if not torch.cuda.is_available():
        print("bench: needs an NVIDIA GPU, skipping")
        return 0

    from env_manifest import write_manifest

    from kernels.cache_write import cache_write
    from kernels.decode import decode

    manifest_path = write_manifest()
    manifest = json.loads(manifest_path.read_text())
    if manifest["git_dirty"]:
        print("bench: WARNING the worktree has uncommitted changes")
    out = ROOT / "results/bench" / f"{socket.gethostname()}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "git_commit": manifest["git_commit"],
        "git_dirty": manifest["git_dirty"],
        "manifest": str(manifest_path),
        "gpu": torch.cuda.get_device_name(),
        "warmup": WARMUP,
        "iters": ITERS,
    }

    over_peak = 0
    torch.manual_seed(0)
    with out.open("w") as f:
        for batch in args.batch:
            for history in args.history:
                smallest = batch * history * min(bytes_per_token(d) for d in args.dtype)
                inp = Inputs(batch, history, smallest)
                cases = []
                if set(args.kernel) & {"path_a", "path_b", "sdpa"}:
                    cases.append(decode_cases(inp, args, decode))
                if "cache_write" in args.kernel:
                    cases.append(cache_write_cases(inp, args, cache_write))
                for gen in cases:
                    for r in gen:
                        r = {"batch": batch, "history": history, **r, **common}
                        r["gbps"] = r["bytes"] / (r["median_us"] * 1e3)
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        over = r["gbps"] > PEAK_GBPS and r["kernel"] != "cache_write"
                        over_peak += over
                        print(
                            f"{r['kernel']:<11} {r['dtype']} b={batch:<2} h={history:<5} "
                            f"c={str(r['chunks']):<4} median {r['median_us']:9.1f} us "
                            f"spread {r['spread_pct']:5.1f}% wall {r['wall_us']:9.1f} us "
                            f"{r['gbps']:6.0f} GB/s{'  OVER PEAK' if over else ''}"
                        )
                del inp
                torch.cuda.empty_cache()
    print(f"wrote {out.relative_to(ROOT)}")
    if over_peak:
        print(f"bench: {over_peak} cases above {PEAK_GBPS:.0f} GB/s, the timing is wrong")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
