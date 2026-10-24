# Scope: class release (due November 5, 2026)

**Title:** Workload Aware CUDA Decode Attention over an INT8 Paged KV Cache

**Question:** How do INT8 storage and GPU work decomposition interact across
context lengths, batch sizes, and unequal sequence lengths? Can a simple
selector beat using one execution configuration everywhere?

## Supported configuration

Values marked *proposed* are defaults until the first GPU session confirms them.

| Item | Value |
| --- | --- |
| GPU | *proposed:* one RTX 4090 on Vast.ai if a host passes the smoke test, otherwise one A10 on Lambda. All final numbers on one machine. |
| Model | Qwen3-1.7B (28 layers, 16 query heads, 8 KV heads, GQA ratio 2, head dim 128). Pin the revision. Same KV head shape as Qwen3-8B and Llama-3-8B. |
| Head dim | 128 |
| Query / KV heads | 16 / 8 (from the model) |
| Page size | 16 tokens |
| Query tokens per request per layer | 1 (decode only) |
| Q and output dtype | fp16 in, fp32 accumulation, fp16 out |
| Cache formats | INT8 (primary), FP16 (control for the factorial comparison) |
| Scales | Symmetric, one fp32 scale per (token, KV head), separate for K and V |
| K smoothing | *proposed:* static per-(layer, KV head, channel) factor s, calibrated offline. Store K/s, multiply Q by s in the kernel. See ROADMAP.md, "Quantization finding". |
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
| Slot mapping (writes) | [N], slot = page * page_size + offset | int64 |
| Output | [B, Hq, D] | fp16 |

All tensors are contiguous. The CUDA wrapper checks shapes, dtypes, device, and
contiguity, and rejects anything else. `reference/paged.py` implements this
contract and is the correctness oracle.

## Operations

1. `cache_write(k, v, slots, cache)`: quantize and store new tokens.
2. `decode_direct(q, cache, block_tables, seq_lens)`: Path A, one CUDA block per (request, query head).
3. `decode_split(q, cache, block_tables, seq_lens, num_splits)`: Path B, partial (m, l, u) per chunk, then a merge kernel.
4. `decode(q, ...)`: selector that picks A or B for the whole batch from host-side metadata.

## Out of scope

Prefill, multiple query tokens, prefix sharing, live format conversion,
multi-GPU, arbitrary models, and production packaging. vLLM integration is a
stretch goal and only starts if Gate 2 passes early (see ROADMAP.md).

## Claims we can and cannot make

Can: kernel latency, storage bytes, numerical error, crossover points.
Cannot (without serving integration): throughput, concurrency, TTFT, or generation quality.
