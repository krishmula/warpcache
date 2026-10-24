# Learning roadmap: from zero CUDA to a shipped kernel study

Written Saturday, October 17, 2026. Class completion is Thursday, November 5.
Submission is Monday, November 9. That is 19 working days plus a 4-day buffer.

> **Superseded schedule.** On October 24 a 3-day plan in [ROADMAP.md](../ROADMAP.md)
> replaced the gates and phases below, and cut the `cache_write` kernel,
> FlashInfer, K smoothing, GQA head sharing, and mixed-length batches. The
> concept explanations and checks still apply. The GPU is an RTX 3090 (6 MB
> L2), not a 4090.

This doc is for one reader who has never written a CUDA kernel. It breaks every
milestone in [ROADMAP.md](../ROADMAP.md) into pieces of 2 to 6 hours. Each
piece names what to learn first, what to build, and the command that proves it
is done. Do the pieces in order. Do not start a piece until the previous check
is green.

The visual companion is [learning-roadmap.html](learning-roadmap.html). Open
it in a browser for the timeline, the GPU model diagrams, and the thread maps.

## What changed since October 10

The old plan put Gate 1 on October 18. Phase 0 has not run on a GPU yet, so the
gates move one week. The project keeps the same scope except for two cuts:

- vLLM integration moves to Part B after the class. The paper does not mention it as a stretch goal.
- FlashInfer is time-boxed to 2 hours. If it does not install and run in that
  time, the external baseline becomes PyTorch `scaled_dot_product_attention`
  on a gathered FP16 history. The paper says so.

New gates. Each gate has a fallback that still produces a complete submission.

| Gate | Date | Pass condition | Fallback if missed |
| --- | --- | --- | --- |
| 0 | Mon Oct 19 | Smoke test passes on the GPU you will use for final numbers | Switch provider the same day |
| 1 | Sun Oct 25 | Path A INT8 and FP16 pass every correctness test | Fix shapes to one config, finish Path A in Phase 2, skip Path B |
| 2 | Sat Oct 31 | Path B matches Path A within tolerance, no NaNs | Ship reference, Path A, benchmarks, profile-backed explanation |
| 3 | Tue Nov 3 | Full sweep done, every figure from `make_figures.py`, paper draft complete | Cut the sweep to the shapes that ran |

## Hours

The plan assumes about 25 hours a week for three weeks, about 70 hours total.
GPU instance time is about 40 of those hours. Learning time is front-loaded.

| Phase | Dates | Learn | Build | Of which GPU hours |
| --- | --- | --- | --- | --- |
| 0 | Sat Oct 17 to Mon Oct 19 | 8 | 4 | 4 |
| 1 | Tue Oct 20 to Sun Oct 25 | 6 | 18 | 14 |
| 2 | Mon Oct 26 to Sat Oct 31 | 4 | 16 | 12 |
| 3 | Sun Nov 1 to Tue Nov 3 | 1 | 10 | 6 |
| 4 | Wed Nov 4 to Thu Nov 5 | 0 | 5 | 3 |

If you only have 15 hours a week, cut in this order and say so in the paper:

1. GQA head sharing in the kernel (piece 2.7). The paper reports the per-query-head mapping.
2. FlashInfer. Use the SDPA baseline.
3. K smoothing. Report per-token INT8 error as measured.
4. Mixed-length batches in the final sweep.
5. Selector tuning on a held-out set. Ship the hand-set threshold and label it as such.

## The one idea behind the whole project

Decode attention reads the whole KV history once per step and does almost no
math on it. At 16K context, one layer of Qwen3-1.7B history is 64 MiB in FP16.
An RTX 4090 moves that in about 64 microseconds. The matching math is about
134 million floating point operations, which the same card finishes in about 2
microseconds. The kernel is memory bound by a factor of about 30. Every design
choice follows from that fact.

- INT8 halves the bytes, so it can halve the time, if dequantization is free enough.
- Path A gives one block to each (request, query head). At batch 1 that is 16 blocks on 128 streaming multiprocessors, so 88% of the chip is idle and the memory system is never saturated.
- Path B splits the history into chunks so there are enough blocks to fill the chip, at the cost of a merge step.
- The selector picks A or B from the batch shape before launch.

You do not need to understand everything about GPUs. You need to understand
enough to read bytes from global memory fast and keep enough blocks in flight.

## Concepts, in the order you need them

Each concept names the piece that first needs it. Read the resource, do the
exercise, then move on. Do not read ahead by more than one phase.

| Id | Concept | First needed | Resource | Exercise and check |
| --- | --- | --- | --- | --- |
| C1 | Execution model: kernel, grid, block, thread, warp of 32 lanes, streaming multiprocessor (SM) | 0.2 | CUDA C++ Programming Guide, "Programming Model". NVIDIA blog "An Even Easier Introduction to CUDA". | Write vector add with a grid-stride loop. Check against `torch.add` on 1 million elements. |
| C2 | Memory hierarchy: registers, shared memory, L1, L2, global memory. Coalesced loads: a warp reads one contiguous 128-byte line in one transaction. | 0.2 | Programming Guide, "Hardware Implementation" and "Maximize Memory Throughput". NVIDIA blog "How to Access Global Memory Efficiently". | Time a row sum with coalesced vs strided reads. The strided version should be at least 4x slower. |
| C3 | Launching a kernel from PyTorch: `load_inline`, pybind, `TORCH_CHECK`, `data_ptr`, the current stream, `c10::Half` and `__half` | 0.2 | PyTorch tutorial "Custom C++ and CUDA Extensions". `scripts/tiny_kernel.py` in this repo. | Extend `tiny_kernel.py` to take fp16 input and return fp16. Check with `torch.testing.assert_close`. |
| C4 | Warp shuffles and block reductions: `__shfl_xor_sync`, `__syncthreads`, shared memory as a scratchpad | 0.2 | NVIDIA blog "Using CUDA Warp-Level Primitives". PMPP 4th ed., chapter 10 (reduction). | Row max and row sum over 128 floats with one warp, 4 values per lane. Check against `torch.amax` and `torch.sum`. |
| C5 | Online softmax and the (m, l, u) merge | 0.3 | Milakov and Gimelshein 2018, "Online normalizer calculation for softmax". PyTorch blog "Flash-Decoding for long-context inference". `reference/attention.py`. | Rewrite `decode_attention` for one head as a Python loop over tokens that keeps only m, l, u. Check against the reference at lengths 1, 7, 300. |
| C6 | Paged cache indexing: slot, page, offset, block table, GQA head mapping | 1.2 | `docs/scope.md` tensor contract. `reference/paged.py`. | By hand, for page size 16: token 37 of a request whose block table starts [9, 2, 5] lives at page 5, offset 5. Query head 7 reads KV head 3. |
| C7 | Symmetric INT8 in CUDA: `__float2int_rn`, `char4` loads, per-vector scale broadcast | 1.2 | `reference/quant.py`. CUDA Math API for `__float2int_rn`. | Quantize one 128-vector with one warp. Match `reference.quant.quantize` exactly. |
| C8 | Templates on the cache element type | 1.4 | Any C++ template primer. vLLM `csrc/attention/attention_kernels.cuh` for a real example of dtype-templated paged attention. | Make the Path A kernel a template. FP16 and INT8 tests both pass from one source. |
| C9 | Debugging: `compute-sanitizer`, `printf` in a kernel, `CUDA_LAUNCH_BLOCKING=1`, tiny shapes first | 1.3 | Nsight Compute Sanitizer docs, "memcheck". | Run `compute-sanitizer` on the vector add. Zero errors. |
| C10 | Timing: CUDA events, warmup, repeats, a working set bigger than L2, variance | 1.6 | PyTorch docs for `torch.cuda.Event`. NVIDIA blog "How to Implement Performance Metrics in CUDA". | Time the same kernel 5 times. Standard deviation under 3% of the mean. |
| C11 | Profiling: Nsight Compute SpeedOfLight, achieved DRAM bandwidth, occupancy, registers per thread, memory bound vs latency bound | 2.4 | Nsight Compute "Kernel Profiling Guide". GPU MODE lecture 1 ("How to profile CUDA kernels in PyTorch"). | Profile Path A at batch 1, 16K. Record DRAM % of peak. Expect it low. Explain why in one sentence. |
| C12 | Filling the chip: blocks per SM, waves, tail effect | 2.1 | Programming Guide, "Maximize Utilization". The Flash-Decoding blog. | Compute blocks for Path A at batch 1, 4, 16, 32 and compare with 128 SMs. Predict which batch sizes need splitting. |
| C13 | Vectorized 16-byte loads with `int4` and `uint4` (optional) | 2.7 | NVIDIA blog "CUDA Pro Tip: Increase Performance with Vectorized Memory Access". | Replace per-lane 8-byte loads with 16-byte loads in Path A. Correctness tests still pass. Measure. |

Do not read FlashAttention or FlashInfer source for the kernel design. They
solve prefill, and their tiling will confuse the decode case. vLLM's paged
attention v1 and v2 are the closest real-world match for Path A and Path B.

## Phase 0: GPU box and CUDA basics (Sat Oct 17 to Mon Oct 19)

**Goal.** A GPU you can profile, three toy kernels you wrote yourself, and the
online softmax math in your head.

### 0.1 Get a GPU box and pass the smoke test (2 hours, GPU)

Learn first: nothing. Read `scripts/gpu_smoke_test.sh` once.

Build:

1. Start a Vast.ai RTX 4090 host with a PyTorch CUDA image. Pick a host that lists Nsight support or a bare-metal driver.
2. Clone the repo, run `uv sync`, run `scripts/gpu_smoke_test.sh`.
3. If `ERR_NVGPUCTRPERM` appears, terminate the host and try a Lambda A10 instead. Do not spend more than 2 hours total here.
4. Start `results/gpu_hours.md` with one line per session: date, provider, hours, cost.

Check: the script prints `SMOKE TEST PASSED`. Commit the manifest from `results/env/`.

### 0.2 Three toy kernels (4 hours, GPU, concepts C1 to C4)

Learn first: C1, C2, C3, C4. Spend 2 of the 4 hours reading.

Build, in a scratch file `scripts/toy_kernels.py` that uses `load_inline`:

1. Vector add with a grid-stride loop. Launch 256 threads per block.
2. Row sum of a [4096, 128] fp32 matrix with one warp per row, each lane holding 4 values, reduced with `__shfl_xor_sync`.
3. Row max of the same matrix. Same mapping.

Check: each kernel matches torch with `torch.testing.assert_close`. Run
`compute-sanitizer python scripts/toy_kernels.py` once and get zero errors.

Why this matters: the Path A inner loop is a row dot product plus a row max.
These three kernels are that loop with the softmax removed.

### 0.3 Online softmax by hand (2 hours, CPU, concept C5)

Learn first: C5.

Build: a Python function that loops over tokens one at a time and keeps only
m, l, and u. Then a second version that processes the history in chunks and
merges with `merge_states`. Put both in `scripts/online_softmax_by_hand.py`.

Check: both match `reference.attention.decode_attention` at lengths 1, 7, 300,
and the chunked version with more chunks than tokens. Write the update rule on
one line of paper. You will type it into the kernel in piece 1.3.

### 0.4 Pin the environment and send the scope email (1 hour)

Build:

1. Write driver, CUDA, PyTorch, and GPU model into `configs/env.md`.
2. Run `uv run pytest` on the GPU box.
3. Email the instructor the revised title and the abstract from `docs/scope.md`. Say vLLM integration is out of scope.

Check: you committed `configs/env.md` and sent the email.

### 0.5 Quantization rerun and FlashInfer (3 hours, time-boxed)

Build:

1. Rerun `scripts/kv_quant_error.py` on 3 long texts with a calibration text that differs from the evaluation text. 1 hour.
2. Decide K smoothing. Keep it if smoothed worst-case error stays under 10% on held-out text. The kernel cost is one multiply on Q.
3. `pip install flashinfer` and time one FP16 paged decode call at batch 4, context 4K, head dim 128, page size 16. 2 hours maximum. If it fails, switch to the SDPA baseline and note it in `docs/scope.md`.

Check: `docs/scope.md` has no item marked *proposed*.

**Gate 0 (Mon Oct 19).** Smoke test passed on the final GPU. Toy kernels pass.
Online softmax by hand matches the reference.

## Phase 1: Cache write and Path A (Tue Oct 20 to Sun Oct 25)

**Goal.** Two kernels, correct for INT8 and FP16, a benchmark harness, first numbers.

### 1.1 Extension skeleton with a no-op kernel (3 hours, GPU, concept C3)

Build:

1. `csrc/warpcache.cu` with one kernel that copies Q to the output.
2. `python/warpcache/__init__.py` that loads it with `torch.utils.cpp_extension.load` and checks the tensor contract from `docs/scope.md`: shapes, dtypes, device, contiguity. Raise on anything else.
3. `tests/test_cuda.py`, skipped without CUDA, with one test that the wrapper rejects a non-contiguous Q.

Check: `uv run pytest tests/test_cuda.py` passes on the GPU box. Rebuild takes under a minute. If it takes longer, pass `extra_cuda_cflags=["-O2"]` and build for one architecture only.

### 1.2 cache_write kernel (4 hours, GPU, concepts C6, C7)

Learn first: C6, C7.

Thread mapping: one warp per (token, KV head) vector of 128 values. Lane i
holds values 4i to 4i+3. Grid is N tokens times Hkv heads warps, 4 warps per
block.

Build:

1. Absmax via `__shfl_xor_sync`. Scale = absmax / 127, or 1.0 when absmax is 0.
2. Quantize with `__float2int_rn`, clamp to [-127, 127], store as `char4`.
3. Store the scale once per warp from lane 0.
4. Slot to (page, offset) with `slot / 16` and `slot % 16`.
5. FP16 cache: same mapping, store `__half` values, no scale.

Check: exact int8 match against `PagedKVCache.write` on shuffled pages, a
partial tail page, all-zero vectors, and values near fp16 max. Port
`test_write_*` cases from `tests/test_reference.py`.

### 1.3 Path A, FP16 first (6 hours, GPU, concepts C5, C9)

Learn first: C9. Reread C5.

Thread mapping, version 1. Keep it this simple until it is correct.

- One block of 128 threads (4 warps) per (request b, query head h). KV head g = h / (Hq / Hkv).
- Each lane holds 4 dims of q in registers, scaled by 1 / sqrt(128).
- Warp w handles tokens w, w + 4, w + 8, and so on. For token t: page = block_table[b][t / 16], offset = t % 16. Lane i loads K[page, offset, g, 4i to 4i+3]. That is 8 bytes per lane and 256 contiguous bytes per warp, so the load is coalesced.
- Partial dot over 4 values, then a 5-step shuffle reduce gives the score to every lane.
- Each warp keeps its own m and l in a register and its own u as 4 floats per lane. Apply the online update from piece 0.3, then load V the same way and accumulate.
- After the loop, write each warp's (m, l, u) to shared memory and have warp 0 merge the four states with the formula from `merge_states`. Write the output as fp16.

Build it in this order and test after each step:

1. Scores only. Kernel writes the raw score for token 0 to the output. Compare to Python.
2. Full online softmax with one warp active, three idle.
3. Four warps plus the shared-memory merge.

Check: matches `paged_decode_attention` on an FP16 cache at lengths 1, 7, 16,
17, 300, and a batch with lengths [1, 300, 17]. Tolerance `atol=1e-2,
rtol=1e-2` on fp16 output. Then `compute-sanitizer` on the length-17 case with
zero errors.

If it is not correct by Thursday, October 22: freeze head dim and page size as
template constants, drop the interleaved warp loop for a single warp, and get
that correct first.

### 1.4 Path A INT8 through the template (3 hours, GPU, concepts C7, C8)

Learn first: C8.

Build: template the kernel on the cache element type. For int8, lane i loads
one `char4`, converts to 4 floats, and multiplies by the per-(token, KV head)
scale that every lane loads from the same address. If K smoothing is in scope,
multiply q by the smoothing vector once at kernel start.

Check: the INT8 test compares against `paged_decode_attention` on the same
INT8 cache, so the only difference is accumulation order. Tolerance `atol=1e-2,
rtol=1e-2`. The FP16 tests still pass from the same source.

### 1.5 Port the correctness suite (2 hours, GPU)

Build: every test in `tests/test_reference.py` that touches the cache or
attention gets a CUDA twin in `tests/test_cuda.py`, parameterized over
`int8` and `float16`.

Check: `uv run pytest` on the GPU box is green, and the count of CUDA tests is
at least 12.

### 1.6 Benchmark harness (4 hours, GPU, concept C10)

Learn first: C10.

Build `benchmarks/bench.py`:

1. Allocate a cache whose bytes exceed 2x the L2 size. For a 4090 that is more than 144 MiB. Rotate through independent caches between repeats so no run hits a warm L2.
2. Time with `torch.cuda.Event`, 10 warmup calls, 50 timed calls, report the median and the interquartile range.
3. Write one JSON line per measurement: path, dtype, batch, context, num_splits, median microseconds, bytes read, achieved GB/s, git commit, and the env manifest path.
4. A `--quick` flag that runs three shapes in under a minute.

Check: run the same point twice. The two medians differ by under 3%. The
achieved GB/s for FP16 Path A at batch 32, context 4K is a plausible fraction
of 1008 GB/s. If it is above peak, the bytes count is wrong.

### 1.7 First numbers (2 hours, GPU)

Build: Path A INT8, Path A FP16, and the external baseline at batch 1 and 32,
context 1K and 16K. Put the table in `results/phase1.md`.

Check: the table exists with 12 numbers. Write one sentence per row that says
whether INT8 was faster and a guess at why. These guesses become profiling
questions in Phase 2.

**Gate 1 (Sun Oct 25).** All of `uv run pytest` passes on the GPU for INT8 and
FP16 Path A. The harness produces repeatable numbers.

## Phase 2: Path B, merge, profiling, selector (Mon Oct 26 to Sat Oct 31)

**Goal.** The split path, a profile that explains the numbers, and a selector.

### 2.1 Path B partial kernel (4 hours, GPU, concept C12)

Learn first: C12.

Build: reuse the Path A kernel body with a token range. The grid becomes
(B, Hq, num_splits). Block (b, h, c) handles tokens [c * chunk, min((c + 1) *
chunk, seq_len)). An empty range writes m = -inf, l = 0, u = 0, exactly as
`partial_state` does. Workspace tensors: m and l as [B, Hq, S] fp32, u as [B,
Hq, S, D] fp32.

The simplest way to share code is one `__device__` function that takes a token
range, called by both the Path A kernel and the Path B kernel.

Check: with num_splits = 1, the workspace m, l, u match `partial_state` on
the whole history. Path A output is unchanged.

### 2.2 Merge kernel (3 hours, GPU, concept C5)

Build: one block per (b, h), 128 threads, one per output dim. Thread d loops
over the S chunks, computes the global max, the weights exp(m_c - m_all), the
weighted l and u, and writes u / l as fp16.

Check: split output matches direct output and the reference at num_splits in
{2, 4, 8, 16}, including a 7-token history with 16 splits. No NaNs anywhere.
`compute-sanitizer` clean on one case.

### 2.3 Chunk sweep (2 hours, GPU)

Build: with the harness, sweep num_splits in {1, 2, 4, 8, 16, 32} at batch 1
and 4, context 4K and 16K, both dtypes.

Check: a table in `results/phase2_splits.md`. Expect splitting to help at batch
1 and to stop helping somewhere between batch 16 and 32. Record where.

### 2.4 Profile with Nsight Compute (4 hours, GPU, concept C11)

Learn first: C11. Spend 1.5 hours on the Kernel Profiling Guide first, or the
report will not make sense.

Build: `scripts/profile.sh` that runs `ncu --set full` on four cases: Path A
and Path B, each at batch 1, 16K and batch 32, 1K. Record DRAM throughput as a
percentage of peak, achieved occupancy, registers per thread, and the warp
stall reasons.

Check: a table in `results/profiles.md`. The Path A batch 1 case shows low DRAM
use and few active warps. The Path B batch 1 case shows higher DRAM
use. If it does not, the chunk count is too small or the merge
dominates. Explain each row in one sentence.

### 2.5 Selector v1 (2 hours, GPU)

Build: a rule in Python that reads only batch size, query heads, SM count, and
max sequence length. Direct blocks = B x Hq. If that is at least 2 x SM count,
pick A. Otherwise pick B with num_splits = ceil(2 x SMs / direct blocks),
capped so each chunk has at least 64 tokens. Also write an oracle in the
harness that runs both and keeps the faster one.

Check: on the chunk sweep shapes, the selector is never more than 10% slower
than the oracle. Record the number.

### 2.6 Paper sections that do not need results (3 hours, no GPU)

Build: background, design, and thread mapping sections of the paper. Draw the
Path A and Path B thread maps from the companion HTML page.

Check: a colleague who has not seen the code can explain Path B from the text.

### 2.7 Optional: share one KV head across its two query heads (3 hours, GPU, concept C13)

Only if Gate 2 passed by Thursday, October 29. One block per (request, KV
head) that holds both query heads' state and reads K and V once. This halves
bytes read and is the biggest easy speedup. Measure before and after.

**Gate 2 (Sat Oct 31).** Path B matches Path A within tolerance on every test.
The profile table explains the Phase 1 numbers.

## Phase 3: Freeze and evaluate (Sun Nov 1 to Tue Nov 3)

**Goal.** Every figure from raw data, paper draft complete.

### 3.1 Feature freeze and workload split (1 hour)

Build: tag the commit `freeze`. Write the tuning set and the held-out set into
`benchmarks/workloads.py`. Tuning: batch {1, 4, 16}, context {1K, 4K, 16K}.
Held out: batch {2, 8, 32}, context {256, 2K, 8K}, plus two mixed-length
batches.

### 3.2 Full sweep (4 hours, GPU, one sitting)

Build: run the harness over every configuration for INT8 A, INT8 B, FP16 A,
FP16 B, selector, oracle, and the external baseline. Run on one host in one
session. Save `results/sweep.jsonl`.

Check: the file has every expected row. A rerun of 5 random rows is within 5%.

### 3.3 Figures from raw data (3 hours, no GPU)

Build `scripts/make_figures.py` that reads `results/sweep.jsonl` and writes
every figure: latency vs context per batch, crossover points, selector vs
fixed vs oracle, achieved bandwidth, storage bytes, and the quantization error
table.

Check: delete `figures/`, run the script, every figure returns.

### 3.4 Paper draft (3 hours)

Build: methodology, results, profiling, limitations, negative results.

**Gate 3 (Tue Nov 3).** Draft complete with every figure.

## Phase 4: Finish (Wed Nov 4 to Thu Nov 5)

### 4.1 Clean reproduction (3 hours, GPU)

Start a fresh instance. Follow the README from a clean clone. Run the tests and
one benchmark. Fix what breaks and nothing else.

### 4.2 Demo and final pass (2 hours)

Record: tests passing, one benchmark, one figure, one Nsight screenshot. Final
read of the paper for the claims list in `docs/scope.md`.

## Buffer (Fri Nov 6 to Mon Nov 9)

Submission fixes only.

## Daily rhythm

- Write code on the Mac with the GPU box off. Start the box only to compile, test, and measure. Stop it when you go read.
- Every session starts with `uv run pytest` and ends with a commit and one line in `results/gpu_hours.md`.
- Test on shapes small enough to print: batch 2, length 17, page size 16. Only then scale up.
- When a kernel is wrong, shrink the shape until you can compare every number by hand. Then add `printf` for one thread.
- When a kernel is slow, profile before you guess.
