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
| Cache write | PyTorch, with `reference/quant.py` | The `cache_write` kernel is cut. |
| Merge | PyTorch, with the math of `merge_states` | A CUDA merge kernel only if time allows. |
| Scope check | Email the instructor the revised title, abstract, and cut list | Day 1, step 1. |

## Cut, and the paper says so

- The `cache_write` kernel. The cache is quantized in PyTorch with `reference/quant.py`.
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

- [ ] Send the scope email.
- [x] Online softmax in Python, as a token loop and as chunks with a merge. Match `reference.attention.decode_attention` at lengths 1, 7, and 300, and with more chunks than tokens. About 2 hours.
- [x] One warm-up kernel: a warp row sum and row max over 128 floats with `__shfl_xor_sync`. Match `torch.sum` and `torch.amax`. About 1 hour.
- [x] Extension skeleton with a no-op kernel, compiled on the 3090.

### Day 2. Path A on the GPU

- [x] The token-range kernel in FP16 with 1 chunk. Check it against the reference on tiny shapes first. 10 tests pass on the 3090, and `compute-sanitizer` reports 0 errors.
- [ ] INT8 through the template.
- [ ] Port the tests. Run `compute-sanitizer` once and get zero errors.

**Cutoff.** If Path A INT8 does not pass every test by the end of day 2, day 3
finishes Path A and Path B is dropped.

### Day 3. Path B, numbers, writing

- [ ] Path B, time-boxed to 4 hours. Turn on chunks above 1, merge in PyTorch, and test split vs direct vs reference, including more chunks than tokens.
- [ ] Time Path A, Path B, and SDPA with CUDA events, INT8 and FP16, at batch 1, 4, and 16, and context 4K and 16K.
- [ ] Run `ncu` on 2 to 4 cases. Record DRAM throughput as a percentage of peak.
- [ ] Write a selector rule from the measurements, for example "split when batch times query heads is below the SM count". Label it as hand-set.
- [ ] Make the figures and write the paper.

### Plumbing, built in parallel

- [ ] Test port for both paths
- [ ] Benchmark script
- [ ] SDPA baseline
- [ ] Figure script
- [ ] Scope email draft

### After day 3, before November 5

- [ ] Fresh GPU instance: follow README from scratch, rerun tests and one benchmark. Fix anything that breaks.
- [ ] Record the demo: tests passing, one benchmark run, one figure, one Nsight screenshot.
- [ ] Final paper pass.

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
