# Roadmap: class release

Today is Saturday, October 10, 2026. Class completion is Thursday, November 5.
Submission is Monday, November 9. That leaves 26 days of work plus a 4-day buffer.

The original plan put setup in October 6 to 11. That week is mostly gone, so
Phase 0 is compressed to four days. The two hard gates (October 18 and
October 25) stay where they were.

## Status

Done:

- [x] Repository layout from the project plan
- [x] Scope spec with proposed defaults: [docs/scope.md](docs/scope.md)
- [x] PyTorch reference: quantization, paged cache, direct and split decode, merge ([reference/](reference/))
- [x] 26 reference tests passing on CPU: `uv run pytest`
- [x] GPU smoke test scripts, not yet run on a GPU: `scripts/gpu_smoke_test.sh`

## Decide this weekend

These block everything else. Proposed defaults are in `docs/scope.md`.

| Decision | Recommendation | Why |
| --- | --- | --- |
| GPU access | A full-VM provider (GCP, AWS, Lambda) or a school cluster node. One L4 or A100. | Container-only rentals often block Nsight Compute counters. Profiling is required for the paper, so verify it first. |
| Dev workflow | VS Code Remote SSH into the GPU box. Keep the repo on a persistent disk. Stop the instance after each session. | No CUDA on a Mac. Edit-compile-test must happen on the GPU host. |
| Model | Qwen2.5-1.5B, pinned revision | Head dim 128, only 2 KV heads, GQA ratio 6. Few heads means Path A launches few blocks at small batch, which makes the split path interesting to study. |
| External baseline | FlashInfer FP16 paged decode | Real paged decode kernel. Confirm it supports head dim 128 and page size 16 on your GPU in Phase 0. |
| Scope check | Email the instructor the revised title and abstract | The plan changed from "INT8 in vLLM" to a kernel study. Get that accepted now, not in November. |
| Weekly hours | Write down a real number | The schedule below assumes about 15 to 20 hours per week. |

## Phases and gates

### Phase 0: Setup and data (Sat Oct 10 to Tue Oct 13)

- [ ] Start the GPU instance. Run `scripts/gpu_smoke_test.sh`. It must print `SMOKE TEST PASSED`.
- [ ] Pin working versions (driver, CUDA, PyTorch, FlashInfer) in `configs/env.md`. Commit the manifest from `results/env/`.
- [ ] Run `uv run pytest` on the GPU box too.
- [ ] Capture model K/V: run Qwen2.5-1.5B with `use_cache=True` in Hugging Face Transformers on a few real prompts (short and ~8K tokens). `past_key_values` holds K after RoPE and V, which is exactly the cache-write boundary. Save a few layers as `.pt` files outside git (`captures/`).
- [ ] Run the quantization error test on captured tensors. Look for outlier channels in K. If error is large, fix the scheme now (see Risks).
- [ ] Install FlashInfer and time one FP16 paged decode call with the scope shapes.
- [ ] Send the scope email.

**Done when:** smoke test passes on the GPU you will use for final numbers, captures exist, and `docs/scope.md` has no *proposed* items left.

### Phase 1: Cache write and Path A (Wed Oct 14 to Sun Oct 18)

- [ ] `csrc/` skeleton: one `.cu` file, pybind binding, loaded with `torch.utils.cpp_extension.load` (JIT) for fast iteration. Wrapper in `python/` checks the contract.
- [ ] `cache_write` kernel. Test against `PagedKVCache.write` on shuffled pages, partial tails, zeros, extreme values. Expect exact int8 match.
- [ ] Path A kernel: one block per (request, query head), online softmax, dequantize in registers. **Template it on the cache element type** so INT8 and FP16 share one kernel. That gives you the FP16 control for the factorial comparison at almost no cost.
- [ ] Port the reference tests to run the CUDA ops (`tests/test_cuda.py`, skipped without CUDA). Run `compute-sanitizer` on a small case.
- [ ] Benchmark harness in `benchmarks/`: CUDA events, warmup, repeats, working set larger than L2, one JSON line per measurement with shapes, config, and env manifest path. Build this now, not in Phase 3. Every later number goes through it.
- [ ] First numbers: Path A INT8 vs FP16 vs FlashInfer at a few points.

**Gate 1 (Sun Oct 18):** Path A INT8 passes all correctness tests. If not, cut to the single shape that works, drop the vLLM stretch for good, and spend Phase 2 finishing Path A.

### Phase 2: Split path, merge, profiling, selector (Mon Oct 19 to Sun Oct 25)

- [ ] Path B: grid over (request, query head, chunk). Each block writes (m, l, u) to a workspace. A merge kernel combines them. Handle empty chunks (m = -inf, l = 0) exactly as `reference/attention.py` does.
- [ ] Tests: split vs direct vs reference, including more chunks than tokens.
- [ ] Sweep chunk count (for example 2, 4, 8, 16) to see where splitting helps and where merge cost wins.
- [ ] Nsight Compute on 2 to 4 representative cases per path. Record achieved DRAM bandwidth as a percentage of peak, occupancy, and registers per thread. Decode attention should be memory bound, so bandwidth is the main story.
- [ ] Selector v1: a threshold rule from host-side metadata only (batch size, KV heads, SM count, max sequence length). No GPU-to-CPU copies per call.
- [ ] Start the paper: background, design, and thread mapping sections. None of these depend on results.

**Gate 2 (Sun Oct 25):** Path B matches Path A within tolerance and has no NaNs. If not, ship the honest fallback: reference, Path A, benchmarks, and a profile-backed explanation. Do not present broken dispatch as done.

**Stretch:** vLLM integration only starts if Gate 2 passes by Thu Oct 22. Otherwise it moves to Part B after the class.

### Phase 3: Freeze and evaluate (Mon Oct 26 to Sun Nov 1)

- [ ] Feature freeze on Oct 26. Only bug fixes after this.
- [ ] Split workloads into a tuning set and a held-out set. Tune selector thresholds on the tuning set only.
- [ ] Full sweep: contexts 256, 1K, 4K, 8K, 16K; batch 1, 4, 16, 32; plus mixed-length batches. Compare INT8 A, INT8 B, FP16 A, FP16 B, selector, oracle, FlashInfer.
- [ ] `scripts/make_figures.py` reads raw JSON and produces every figure. No hand-edited plots.
- [ ] Quantization error table from captured model tensors.
- [ ] Complete paper draft by Sun Nov 1, including limitations and negative results.

### Phase 4: Finish (Mon Nov 2 to Thu Nov 5)

- [ ] Fresh GPU instance: follow README from scratch, rerun tests and one benchmark. Fix anything that breaks.
- [ ] Record the demo: tests passing, one benchmark run, one figure, one Nsight screenshot.
- [ ] Final paper pass.

### Buffer (Fri Nov 6 to Mon Nov 9)

Submission fixes only. No new implementation.

## Risks to watch

| Risk | Early signal | Response |
| --- | --- | --- |
| GPU or profiler access fails | Smoke test fails in Phase 0 | Switch provider immediately. Do not write kernels on a box you cannot profile. |
| K has outlier channels and per-token INT8 error is large | Captured-tensor error much worse than the synthetic test | Report it. Try per-channel K scales only if time allows; otherwise narrow claims. |
| INT8 is slower than FP16 | Phase 1 numbers | Valid result. Profile to explain (dequant cost, scale loads, too little parallelism). |
| Kernel debugging eats Phase 1 | Not correct by Thu Oct 16 | Simplify: fixed shapes as template constants, no vectorized loads yet. |
| GPU budget | Usage log | Keep a log in `results/gpu_hours.md`. Compile and debug on small cases. |

## Paper outline

1. Motivation: decode is memory bound and KV cache size limits serving.
2. Background: paged KV cache, GQA, INT8 symmetric quantization, online softmax.
3. Design: data contract, Path A and Path B thread mapping, merge, selector.
4. Methodology: hardware, versions, workloads, timing protocol, tuning vs held-out split.
5. Results: latency vs context and batch, crossover points, selector vs fixed choices and oracle, FlashInfer reference, storage bytes, quantization error.
6. Profiling: why each path wins where it wins.
7. Limitations and future work (Part B: serving integration, live conversion).
