# Scope: class release (due November 5, 2026)

**Title:** Workload Aware CUDA Decode Attention over an INT8 Paged KV Cache

**Question:** How do INT8 storage and GPU work decomposition interact across
context lengths and batch sizes? Can a simple selector beat using one
execution configuration everywhere?

## Supported configuration

| Item | Value |
| --- | --- |
| GPU | One RTX 3090 (24 GB, 82 SMs, 6 MB L2) on a Vast.ai Ubuntu 22.04 VM. Driver 580.95.05, CUDA 13.0, torch 2.14.1+cu130. All final numbers come from that card. |
| Model | Qwen3-1.7B (28 layers, 16 query heads, 8 KV heads, GQA ratio 2, head dim 128). Pin the revision. Same KV head shape as Qwen3-8B and Llama-3-8B. |
| Head dim | 128 |
| Query / KV heads | 16 / 8 (from the model) |
| Page size | 16 tokens |
| Query tokens per request per layer | 1 (decode only) |
| Q and output dtype | fp16 in, fp32 accumulation, fp16 out |
| Cache formats | INT8 (primary), FP16 (control for the factorial comparison) |
| Scales | Symmetric, one fp32 scale per (token, KV head), separate for K and V |
| K smoothing | None. Cut from the class scope. See ROADMAP.md, "Quantization finding". |
| Rounding | Round half to even. Zero vectors get scale 1.0. Non-finite input is rejected. |
| Sequence length | At least 1 (the new token's K/V is written before attention) |

## Tensor contract

| Tensor | Shape | Dtype |
| --- | --- | --- |
| Q | [B, Hq, D] | fp16 |
| K cache, V cache | [P, page_size, Hkv, D] | int8 or fp16 |
| K scale, V scale | [P, page_size, Hkv] | fp32 (int8 only) |
| Block table | [B, max_pages] | int32 |
| Sequence lengths | [B] | int32 |
| Slot mapping (writes, PyTorch) | [N], slot = page * page_size + offset | int64 |
| Output | [B, Hq, D] | fp16 |

All tensors are contiguous. The CUDA wrapper checks shapes, dtypes, device, and
contiguity, and rejects anything else. `reference/paged.py` implements this
contract and is the correctness oracle.

## Operations

One CUDA op:

`decode(q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, num_chunks)`

- `num_chunks == 1` is Path A. One block per (request, query head) covers the
  whole history, normalizes, and writes the output.
- `num_chunks > 1` is Path B. One block per (request, query head, chunk) writes
  a partial state (m, l, u) to a workspace. PyTorch merges the states with the
  math of `merge_states` in `reference/attention.py`. An empty chunk writes
  m = -inf, l = 0, u = 0.
- `k_scale` and `v_scale` are ignored for an FP16 cache.

New tokens are written to the cache in PyTorch with `PagedKVCache.write`, which
quantizes with `reference/quant.py`. The selector is a hand-set rule written
from the measurements, and the paper labels it as hand-set.

## Out of scope

Prefill, multiple query tokens, prefix sharing, live format conversion,
multi-GPU, arbitrary models, and production packaging. Also cut, and the paper
says so: the `cache_write` kernel, FlashInfer (SDPA on a gathered FP16 history
is the baseline), K smoothing, GQA head sharing, mixed-length batches, and the
CUDA merge kernel unless time allows. vLLM integration is Part B, after the
class.

## Claims we can and cannot make

Can: kernel latency, storage bytes, numerical error, crossover points.
Cannot (without serving integration): throughput, concurrency, TTFT, or generation quality.
