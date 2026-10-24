# Roadmap: class release

Replanned Saturday, October 17, 2026. Class completion is Thursday, November 5.
Submission is Monday, November 9. That leaves 19 days of work plus a 4-day buffer.

Phase 0 did not run on a GPU during the week of October 10, so every gate moved
one week. The step-by-step plan, with what to learn before each piece and the
check that closes it, is [docs/learning-roadmap.md](docs/learning-roadmap.md).
Its visual companion is [docs/learning-roadmap.html](docs/learning-roadmap.html).
This file keeps the project-level status, budget, findings, and risks.

| Gate | Old date | New date |
| --- | --- | --- |
| 0: smoke test passes on the final GPU | Tue Oct 13 | Mon Oct 19 |
| 1: Path A INT8 and FP16 pass every test | Sun Oct 18 | Sun Oct 25 |
| 2: Path B matches Path A, no NaNs | Sun Oct 25 | Sat Oct 31 |
| 3: sweep done, figures, paper draft | Sun Nov 1 | Tue Nov 3 |

vLLM integration moves to Part B after the class. FlashInfer is time-boxed to
2 hours in Phase 0, with PyTorch SDPA on a gathered FP16 history as the fallback
baseline.

## Status

Done:

- [x] Repository layout from the project plan
- [x] Scope spec with proposed defaults: [docs/scope.md](docs/scope.md)
- [x] PyTorch reference: quantization, paged cache, direct and split decode, merge ([reference/](reference/))
- [x] 26 reference tests passing on CPU: `uv run pytest`
- [x] GPU smoke test scripts, not yet run on a GPU: `scripts/gpu_smoke_test.sh`
- [x] Model recommended: Qwen3-1.7B (see Quantization finding)
- [x] Real K/V capture and INT8 error measurement on CPU: `scripts/kv_quant_error.py`

## Decide by Monday, October 19

These block everything else. Proposed defaults are in `docs/scope.md`.

| Decision | Recommendation | Why |
| --- | --- | --- |
| GPU access | Try a Vast.ai RTX 4090 first. Keep the host only if `scripts/gpu_smoke_test.sh` passes, including Nsight counters. Fall back to a Lambda A10. | Vast.ai hosts are Docker containers, and counter access depends on each host's driver setting. Lambda is a full VM where `sudo ncu` works. See GPU budget. |
| Dev workflow | VS Code Remote SSH into the GPU box. Write code on the Mac between sessions; only run the box to compile, test, and measure. | No CUDA on a Mac. Billing runs whenever the instance exists, even while you read or write. |
| Model | Qwen3-1.7B, pinned revision | 8 KV heads and head dim 128, the same KV head shape as Qwen3-8B and Llama-3-8B, so kernel results transfer. Ungated. Harder to quantize than Qwen2.5-1.5B, which makes it the more honest test. |
| External baseline | FlashInfer FP16 paged decode | Real paged decode kernel. Confirm it supports head dim 128 and page size 16 on your GPU in Phase 0. |
| Scope check | Email the instructor the revised title and abstract | The plan changed from "INT8 in vLLM" to a kernel study. Get that accepted now, not in November. |
| Weekly hours | Write down a real number | The schedule below assumes about 15 to 20 hours per week. |

## GPU budget

Estimate: **35 to 55 instance hours**, budget 60. The kernel study needs very
little memory: one layer's cache for Qwen3-1.7B is 4 KiB per token, so batch 32
at 16K context is 2 GiB. Any 24 GB card is plenty.

| Phase | Hours |
| --- | --- |
| 0: smoke test, pins, FlashInfer | 3 to 5 |
| 1: cache write, Path A, harness | 12 to 18 |
| 2: Path B, merge, profiling, selector | 12 to 18 |
| 3: final sweeps and reruns | 6 to 10 |
| 4: clean reproduction check | 2 to 3 |

Hours count instance time, not GPU busy time. Compiling and debugging take
most of it; the full sweep itself takes minutes.

| Option | Price checked Oct 11, 2026 | 60 hours | Notes |
| --- | --- | --- | --- |
| Vast.ai RTX 4090 | about $0.15 to $0.45/hr, set by each host | $10 to $27 | Containers. Nsight counters work only on some hosts. Host can vanish, so run the final sweep on one host in one sitting. 72 MB L2: benchmark working sets must be larger. |
| Lambda A10 24 GB | $1.29/hr | about $77 | Full VM, `sudo ncu` works. Cannot stop an instance, only terminate, which erases local disk. Attach a persistent filesystem at launch and script the setup. |

Either is far below the $200 to $300 in the original plan, which included vLLM
integration. Keep a log in `results/gpu_hours.md`.

## Quantization finding

Measured on CPU on October 11 with `scripts/kv_quant_error.py`: one 2,048-token
text (the project plan), calibration on the first half, error measured at 16
decode positions in the second half. Metric is relative L2 error of the
attention output, INT8 cache vs bf16 cache.

| Model | Per-token INT8, mean / worst | With static K smoothing, mean / worst |
| --- | --- | --- |
| Qwen2.5-1.5B | 2.7% / 40% | 1.5% / 10% |
| Qwen3-1.7B | 6.1% / 33% | 1.8% / 8% |

- V quantizes well (about 1% error). Almost all the error comes from K.
- K has outlier channels: the largest channel is up to 66x the median in early
  layers. Per-token scales are set by the outlier, so other channels lose precision.
- Static K smoothing fixes most of it: store K/s and multiply Q by s, where s is
  a fixed per-(layer, KV head, channel) factor from calibration. q·k is unchanged.
  Cost in the kernel: one extra multiply on Q per decode step and on K at cache
  write. Per-token scales and append behavior stay the same.
- This is QServe's SmoothAttention idea (Lin et al., 2024), applied after RoPE.
  Cite it; it is not a novel contribution.
- One text and one calibration split is a weak sample. Phase 0 reruns it on more text.

## Phases and gates

### Phase 0: Setup and data (Sat Oct 17 to Mon Oct 19)

- [ ] Start the GPU instance. Run `scripts/gpu_smoke_test.sh`. It must print `SMOKE TEST PASSED`.
- [ ] Pin working versions (driver, CUDA, PyTorch, FlashInfer) in `configs/env.md`. Commit the manifest from `results/env/`.
- [ ] Run `uv run pytest` on the GPU box too.
- [ ] Rerun `scripts/kv_quant_error.py` on 3 to 4 different long texts (8K tokens) with a calibration text separate from the evaluation text. Decide whether K smoothing goes into the class scope (see Quantization finding).
- [ ] Install FlashInfer and time one FP16 paged decode call with the scope shapes.
- [ ] Send the scope email.

**Done when:** smoke test passes on the GPU you will use for final numbers, captures exist, and `docs/scope.md` has no *proposed* items left.

### Phase 1: Cache write and Path A (Tue Oct 20 to Sun Oct 25)

- [ ] `csrc/` skeleton: one `.cu` file, pybind binding, loaded with `torch.utils.cpp_extension.load` (JIT) for fast iteration. Wrapper in `python/` checks the contract.
- [ ] `cache_write` kernel. Test against `PagedKVCache.write` on shuffled pages, partial tails, zeros, extreme values. Expect exact int8 match.
- [ ] Path A kernel: one block per (request, query head), online softmax, dequantize in registers. **Template it on the cache element type** so INT8 and FP16 share one kernel. That gives you the FP16 control for the factorial comparison at almost no cost.
- [ ] Port the reference tests to run the CUDA ops (`tests/test_cuda.py`, skipped without CUDA). Run `compute-sanitizer` on a small case.
- [ ] Benchmark harness in `benchmarks/`: CUDA events, warmup, repeats, working set larger than L2, one JSON line per measurement with shapes, config, and env manifest path. Build this now, not in Phase 3. Every later number goes through it.
- [ ] First numbers: Path A INT8 vs FP16 vs FlashInfer at a few points.

**Gate 1 (Sun Oct 25):** Path A INT8 passes all correctness tests. If not, cut to the single shape that works and spend Phase 2 finishing Path A.

### Phase 2: Split path, merge, profiling, selector (Mon Oct 26 to Sat Oct 31)

- [ ] Path B: grid over (request, query head, chunk). Each block writes (m, l, u) to a workspace. A merge kernel combines them. Handle empty chunks (m = -inf, l = 0) exactly as `reference/attention.py` does.
- [ ] Tests: split vs direct vs reference, including more chunks than tokens.
- [ ] Sweep chunk count (for example 2, 4, 8, 16) to see where splitting helps and where merge cost wins.
- [ ] Nsight Compute on 2 to 4 representative cases per path. Record achieved DRAM bandwidth as a percentage of peak, occupancy, and registers per thread. Decode attention should be memory bound, so bandwidth is the main story.
- [ ] Selector v1: a threshold rule from host-side metadata only (batch size, KV heads, SM count, max sequence length). No GPU-to-CPU copies per call.
- [ ] Start the paper: background, design, and thread mapping sections. None of these depend on results.

**Gate 2 (Sat Oct 31):** Path B matches Path A within tolerance and has no NaNs. If not, ship the honest fallback: reference, Path A, benchmarks, and a profile-backed explanation. Do not present broken dispatch as done.

**Stretch:** GQA head sharing in the kernel (one block per request and KV head) only starts if Gate 2 passes by Thu Oct 29. vLLM integration is Part B, after the class.

### Phase 3: Freeze and evaluate (Sun Nov 1 to Tue Nov 3)

- [ ] Feature freeze on Nov 1. Only bug fixes after this.
- [ ] Split workloads into a tuning set and a held-out set. Tune selector thresholds on the tuning set only.
- [ ] Full sweep: contexts 256, 1K, 4K, 8K, 16K; batch 1, 4, 16, 32; plus mixed-length batches. Compare INT8 A, INT8 B, FP16 A, FP16 B, selector, oracle, FlashInfer.
- [ ] `scripts/make_figures.py` reads raw JSON and produces every figure. No hand-edited plots.
- [ ] Quantization error table from captured model tensors.
- [ ] Complete paper draft by Tue Nov 3, including limitations and negative results.

### Phase 4: Finish (Wed Nov 4 to Thu Nov 5)

- [ ] Fresh GPU instance: follow README from scratch, rerun tests and one benchmark. Fix anything that breaks.
- [ ] Record the demo: tests passing, one benchmark run, one figure, one Nsight screenshot.
- [ ] Final paper pass.

### Buffer (Fri Nov 6 to Mon Nov 9)

Submission fixes only. No new implementation.

## Risks to watch

| Risk | Early signal | Response |
| --- | --- | --- |
| GPU or profiler access fails | Smoke test fails in Phase 0 | Switch provider immediately. Do not write kernels on a box you cannot profile. |
| K outlier channels make per-token INT8 lossy | Already observed (see Quantization finding) | Static K smoothing. If it does not hold on held-out text, report per-token error honestly and narrow claims. |
| INT8 is slower than FP16 | Phase 1 numbers | Valid result. Profile to explain (dequant cost, scale loads, too little parallelism). |
| Kernel debugging eats Phase 1 | Not correct by Thu Oct 22 | Simplify: fixed shapes as template constants, no vectorized loads yet. |
| GPU budget | Usage log | Keep a log in `results/gpu_hours.md`. Compile and debug on small cases. |

## Paper outline

1. Motivation: decode is memory bound and KV cache size limits serving.
2. Background: paged KV cache, GQA, INT8 symmetric quantization, online softmax.
3. Design: data contract, Path A and Path B thread mapping, merge, selector.
4. Methodology: hardware, versions, workloads, timing protocol, tuning vs held-out split.
5. Results: latency vs context and batch, crossover points, selector vs fixed choices and oracle, FlashInfer reference, storage bytes, quantization error.
6. Profiling: why each path wins where it wins.
7. Limitations and future work (Part B: serving integration, live conversion).
