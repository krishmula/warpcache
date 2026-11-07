# Roadmap: class release

Replanned Saturday, October 24, 2026, after Gate 0 passed. Class completion is
Thursday, November 5. Submission is Monday, November 9.

The week-by-week gates from October 17 are replaced by the 3-day plan below.
[docs/learning-roadmap.md](docs/learning-roadmap.md) still explains each CUDA
concept and its check, but its schedule and scope are out of date. Where it
disagrees with this file, this file wins.

## Status

Done:

- [x] Scope and tensor contract: [docs/scope.md](docs/scope.md)
- [x] PyTorch reference: quantization, paged cache, direct and split decode, merge ([reference/](reference/))
- [x] 26 reference tests passing on CPU: `uv run pytest`
- [x] Model: Qwen3-1.7B (see Quantization finding)
- [x] Real K/V capture and INT8 error measurement on CPU: `scripts/kv_quant_error.py`
- [x] **Gate 0.** `scripts/gpu_smoke_test.sh` passed on a Vast.ai RTX 3090 VM, including Nsight Compute counters as root. Manifest: `results/env/ubuntu-20261024-191743.json`.
- [x] First profile: the smoke test's `scale_kernel` reached 87% of peak DRAM throughput and 18% of compute. It is memory bound, as decode attention will be.

## Decisions

| Decision | Choice | Why |
| --- | --- | --- |
| GPU | One RTX 3090 (24 GB, 82 SMs, 6 MB L2) on a Vast.ai Ubuntu 22.04 VM. All final numbers come from that card. | Full VM, so `ncu` reads counters as root. RunPod RTX 4090 pods are containers and the host blocks counters (`ERR_NVGPUCTRPERM`). Lambda said in October 2025 that Nsight Compute is not supported on its VMs. |
| Toolchain | Driver 580.95.05, CUDA 13.0 toolkit, torch 2.14.1+cu130, Python 3.12 | torch needs a CUDA toolkit with the same major version. PyTorch extensions need `ninja-build`. `.python-version` pins 3.12 because uv picked 3.15, which torch does not build for. |
| Dev workflow | Write code on the Mac. Run the VM only to compile, test, and measure. | No CUDA on a Mac. Billing runs whenever the instance exists. |
| Model | Qwen3-1.7B, pinned revision | 8 KV heads and head dim 128, the same KV head shape as Qwen3-8B and Llama-3-8B. |
| Baseline | PyTorch SDPA on a gathered FP16 history | FlashInfer is cut. |
| Cache write | A `cache_write` kernel on day 3, after Path B. `reference/quant.py` stays as the reference it must match exactly. | Restored November 1. The whole INT8 path then runs on the GPU, and the write cost per decode step can be measured. |
| Merge | PyTorch, with the math of `merge_states` | A CUDA merge kernel only if time allows. |
| Scope check | Email the instructor the revised title, abstract, and cut list | Day 1, step 1. |

## Cut, and the paper says so

- FlashInfer. PyTorch SDPA on a gathered FP16 history is the baseline.
- K smoothing, GQA head sharing, and mixed-length batches.
- The CUDA merge kernel, unless time allows. The merge runs in PyTorch.
- vLLM integration. It is Part B, after the class.

## Design

One kernel over a token range, with a grid of (request, query head, chunk),
templated on the cache element type (INT8 or FP16).

- With 1 chunk, it normalizes and writes the output. This is Path A.
- With more than 1 chunk, each block writes (m, l, u) to a workspace, and
  PyTorch merges them with the math of `merge_states`. This is Path B.

Why Path B matters: at batch 1, Path A launches 16 blocks on 82 SMs, so most of
the chip is idle at long context.

Every piece targets one op signature, with the reference as a stand-in until
the kernel exists:

`decode(q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, num_chunks)`

## 3-day plan

### Day 1. Math and skeleton, mostly on the Mac

- [ ] **Scope email** to the instructor. This is Day 1, step 1.
  - [ ] Draft the revised title from `docs/scope.md`, a short abstract, and the cut list from "Cut, and the paper says so".
  - [ ] Send it. Check: the instructor replies, or the sent mail is on record.
- [x] **Online softmax in Python**, about 2 hours.
  - [x] Stub and tests in `tests/test_online_softmax.py`.
  - [x] Online softmax over a range of tokens in `reference/online_softmax.py`, as a token loop and as chunks with a merge.
  - [x] Check: a hand-worked example, an empty range contributes nothing, a match with `reference.attention.decode_attention` at lengths 1, 7 and 300 and with more chunks than tokens, large scores stay finite, and token order does not change the output.
- [x] **Warm-up kernel**, about 1 hour. A warp row sum and row max over 128 floats with `__shfl_xor_sync`, in `kernels/warp_reduce.cu`.
  - [x] One warp per row, 4 values per lane, five shuffle rounds.
  - [x] Check: `tests/test_warp_reduce.py` matches `torch.sum` and `torch.amax` on random rows, every column counts, all-negative rows, and a max in the first and last column.
- [x] **Extension skeleton**, built with `load_inline` from a Python wrapper and compiled on the 3090.

### Day 2. Path A on the GPU

- [x] **FP16 decode kernel with 1 chunk.** 10 tests pass on the 3090, and `compute-sanitizer` reports 0 errors, October 31.
  - [x] Stub, host checks, and tests in `tests/test_decode.py`.
  - [x] One block of 32 threads per (request, query head). Each lane owns 4 of the 128 columns.
  - [x] Per token, find the physical page through the block table and the slot in the page, then the lane's 4 K and V numbers.
  - [x] Partial dot product, five `__shfl_xor_sync` rounds for the full score, divide by √128.
  - [x] Online softmax update of m, l and the lane's 4 u numbers. Write u / l as FP16.
  - [x] Check: one token returns its V row exactly, a match with the reference at lengths 2, 15, 16, 17, 300 and 2048, a batch with shuffled pages, and large scores stay finite.
- [x] **INT8 through the template.** All 20 decode tests pass on an RTX 4090, FP16 and INT8, November 3.
  - [x] The host path launches `decode_kernel<int8_t>` or `decode_kernel<__half>` and checks the scales.
  - [x] Template the kernel on the cache element type. Overloaded `flt_convert` turns either type into float.
  - [x] Load the K and V scale per (page, slot, KV head) inside `if constexpr`, so the FP16 copy never reads the null scale pointers.
- [x] **Port the tests and run `compute-sanitizer`.** 0 errors on the 4090, November 3.

**Cutoff.** If Path A INT8 does not pass every test by the end of day 2, day 3
finishes Path A and Path B is dropped.

### Day 3. Path B, numbers, writing

- [x] **Path B**, time-boxed to 4 hours. Each (request, query head) gets `num_chunks` blocks. Each block walks one chunk of the history and writes its partial state to the workspace. PyTorch merges the partial states.
  - [x] Step 1. Chunk ranges. The grid gets a third size, the chunk count. Each block works out its own chunk from `blockIdx.z` and `gridDim.z`: chunk size = history length / chunk count, rounded up, the same rule as `split_decode_attention`. The loop stops at the last token. Check: the 20 existing decode tests pass with 1 chunk. Passed November 4 on the 3090.
  - [x] Step 2. Workspace. With more than 1 chunk, the host function makes FP32 m and l `[B, HQ, C]` and u `[B, HQ, C, D]`. Each block writes its m, l and u to its own slot with no divide. `decode(...)` in C++ returns a list: the output for 1 chunk, or m, l and u. Check: `test_each_chunk_writes_its_partial_state` matches `partial_state` per chunk at 4 and 8 chunks, FP16 and INT8, and `compute-sanitizer` reports 0 errors. Passed November 4 on the 3090.
  - [x] Step 3. Merge. In `kernels/decode.py`, move the chunk dimension to the front, call `merge_states`, cast to FP16, and remove the `NotImplementedError`. Check: new tests at 2, 4 and 8 chunks, FP16 and INT8, match `paged_decode_attention`: `test_merged_chunks_match_reference`, plus `test_large_scores_stay_finite` at 4 chunks. Passed November 4 on the 3090.
  - [x] Step 4. More chunks than tokens. Empty chunks write m = -inf, l = 0, u = 0 and never divide. Check: `test_more_chunks_than_tokens` runs 5 tokens with 8 chunks through the full `decode(...)`, and `compute-sanitizer` reports 0 errors. No kernel change was needed. Passed November 4 on the 3090.
- [x] **The `cache_write` kernel**, time-boxed to 4 hours. One warp per (token, KV head) row, as in [docs/learning-roadmap.md](docs/learning-roadmap.md) section 1.2. If it misses the time box, ship with the PyTorch write and finish it in the November 6 to 9 buffer.
  - [x] Map each new token's slot to (page, slot in page) through the block table.
  - [x] Absmax of the 128 values with `__shfl_xor_sync`. Scale = absmax / 127, or 1.0 when absmax is 0. Lane 0 stores the scale.
  - [x] Quantize with `__float2int_rn`, clamp to [-127, 127], and store INT8. The FP16 cache stores the values as they are, with no scale.
  - [x] Check: an exact INT8 and scale match against `PagedKVCache.write` on shuffled pages, a partial last page, an all-zero row, and values near the FP16 maximum. The tests are in `tests/test_cache_write.py`, ported from `test_quant_zero_vector`, `test_quant_extreme_finite`, `test_gather_shuffled_pages_and_partial_tail` and `test_append_across_page_boundary_matches_bulk_write` in `tests/test_reference.py`. `compute-sanitizer` reports 0 errors.
- [x] **Timing** with CUDA events on the 3090 only. Ran November 7 with `scripts/bench.py`. Decode and SDPA results are in `results/bench/ubuntu-20261107-012514.jsonl`, and `cache_write` results are in `ubuntu-20261107-013340.jsonl`.
  - [x] Benchmark script. Working sets larger than 12 MiB so repeats do not read a warm L2. 10 warmup calls, 50 timed calls, median and spread. One result line per case, with the git commit and the host manifest.
  - [x] SDPA baseline on a gathered FP16 history.
  - [x] Run Path A, Path B and SDPA, FP16 and INT8, at batch size 1, 4 and 16 and history length 4K and 16K. Sweep the chunk count for Path B. Chunk counts 2 to 256, plus `path_b_kernel`, which times the Path B kernel without the merge.
  - [x] Time `cache_write` for one new token per request at the same batch sizes. Each event pair times 100 calls, because one call takes about 2 us and the event timer steps by about 1 us.
  - [x] Check: the same case run twice gives medians within 3%. Achieved bandwidth stays below the 3090's 936 GB/s peak. Result: all 210 decode, Path B kernel-only and SDPA cases repeat within 2.6%, and the fastest case reaches 868 GB/s. `cache_write` repeats within 1% in 9 of 12 cases. INT8 at batch size 16 and 4K differs by 2.2%, and FP16 at batch size 16 by 3.9% and 4.6% (76 and 91 ns of a median near 1.9 us). Accepted November 7 as a known exception. The paper reports `cache_write` as repeating within 5%.
- [x] **`ncu`** on 2 to 4 cases, for example Path A and Path B at batch size 1 and 16K history. Record DRAM throughput as a percentage of peak, achieved occupancy and registers per thread. Check: one sentence per case that explains the number. Done November 7 for Path A at batch size 1, Path B at batch size 1 (64 chunks) and 16 (128 chunks), and SDPA at batch size 16, all at 16K history. Reports are in `results/ncu/`.
- [x] **Selector rule** from the measurements, for example "use Path B when batch size times query heads is below the SM count". Label it as hand-set. Check: on the measured cases, it is never more than 10% slower than the best chunk count. Result: `select_chunks` in `kernels/decode.py`, hand-set, uses 128 chunks, capped at 16 tokens per chunk. It is within 3.8% of the best chunk count in all 24 cases, in both grid runs. Path A never won a measured case. The rule comes from the same data it is checked on.
- [x] **Figures and the paper.** The paper is `docs/paper.md`.
  - [x] Figure script that reads the benchmark result lines. `scripts/figures.py` writes PDF and PNG files to `results/figures/`.
  - [x] Figures: latency against history length and batch size for Path A, Path B and SDPA, INT8 against FP16, the chunk-count sweep, and the cost of `cache_write`. Also `merge_fix`, which shows the old merge's floor of about 400 us.
  - [x] Paper sections 1 to 4 from the outline below. They do not need results.
  - [x] Paper sections 5 and 6 from the measurements and the `ncu` cases. Section 7 from the cut list.

### After day 3, before November 5

- [ ] **Fresh GPU instance.**
  - [ ] Start a new host and run `scripts/gpu_smoke_test.sh`.
  - [ ] Follow the README from scratch, with no steps from memory.
  - [ ] Run `uv run pytest`, `compute-sanitizer` once, and one benchmark case.
  - [ ] Fix the README and anything else that breaks. Check: a second pass works with no changes.
- [ ] **Record the demo.**
  - [ ] Tests passing, one benchmark run, one figure, and one Nsight screenshot.
- [ ] **Final paper pass.**
  - [x] Every number in the paper matches a result line or an `ncu` report. Audited November 6 with a script against `results/bench`, `results/ncu/summary.md` and the manifest. Repeat this pass after any change to the paper.
  - [x] The selector is labelled hand-set. The cut list and SmoothAttention (Lin et al., 2024) are cited.

November 6 to 9 is buffer. Submission fixes only.

## GPU budget

The October 11 estimate was 35 to 55 instance hours for the full plan. The
3-day plan needs fewer. The kernel study needs little memory: one layer's cache
for Qwen3-1.7B is 4 KiB per token, so batch 16 at 16K context is 1 GiB.

Hours count instance time, not GPU busy time. Log every session in
[results/gpu_hours.md](results/gpu_hours.md).

The 3090 has a 6 MB L2. Benchmark working sets must be larger than 12 MiB so
repeats do not read a warm L2.

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
- One text and one calibration split is a weak sample. K smoothing is cut from
  the class scope, so the paper reports per-token INT8 error as measured and
  names smoothing as future work.

## Risks to watch

| Risk | Early signal | Response |
| --- | --- | --- |
| `cache_write` pushes day 3 past its timing and paper work | Not an exact match at the end of its 4 hours | Ship the decode results with the PyTorch write, and finish `cache_write` in the November 6 to 9 buffer. |
| Kernel debugging eats day 2 | Path A FP16 not correct by midday | Fixed shapes as template constants, no vectorized loads yet. The cutoff rule drops Path B. |
| INT8 is slower than FP16 | Day 3 numbers | Valid result. Profile to explain (dequant cost, scale loads, too little parallelism). |
| K outlier channels make per-token INT8 lossy | Already observed (see Quantization finding) | Report per-token error honestly and narrow claims. |
| The Vast.ai host disappears | Instance gone | Run the final timing on one host in one sitting. Rerun the smoke test on any new host first. |
| GPU budget | Usage log | Log sessions in `results/gpu_hours.md`. Compile and debug on small cases. |

## Paper outline

1. Motivation: decode is memory bound and KV cache size limits serving.
2. Background: paged KV cache, GQA, INT8 symmetric quantization, online softmax.
3. Design: data contract, the token-range kernel, Path A and Path B thread mapping, merge, selector rule.
4. Methodology: hardware, versions, workloads, timing protocol.
5. Results: latency vs context and batch, crossover points, hand-set selector vs fixed choices, SDPA baseline, storage bytes, quantization error.
6. Profiling: why each path wins where it wins.
7. Limitations and future work: the cut list above, and Part B (serving integration, live conversion).
