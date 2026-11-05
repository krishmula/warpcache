# warpcache

**CUDA decode attention over INT8 and FP16 paged KV caches, exploring quantization and GPU parallelism for LLM inference.**

warpcache is a focused CUDA kernel study: how do KV cache precision and the way
work is distributed across the GPU affect attention during token generation?
It pairs a readable PyTorch reference with a CUDA implementation, correctness
tests, and a documented experiment plan. Developed as a CS 259 class project.

[Quick start](#quick-start) · [CUDA usage](#cuda-usage) · [Design](#design) · [Status](#status) · [Documentation](#documentation)

## Why this project?

Each decode step reads a request's existing keys and values to produce attention
for one new query token. A smaller KV cache reduces storage, while distributing
a long history across more blocks can expose more GPU parallelism. Both choices
have costs: INT8 requires dequantization, and multiple chunks require a merge.

warpcache makes those tradeoffs explicit:

- **Two cache formats:** INT8 with separate per-token, per-KV-head FP32 scales
  for K and V, plus FP16 for comparison.
- **Two execution paths:** one block per query head and request, or multiple
  blocks working on chunks of the same history.
- **Online softmax:** FP32 accumulation without materializing the full attention
  score matrix in the CUDA kernel.
- **Reference-driven correctness:** tests for page boundaries, shuffled pages,
  large scores, partial states, and empty chunks.

## Status

**Research prototype.** Both CUDA paths are implemented. Performance measurements
and the workload selector are still pending; this repository does not yet claim
a speedup over PyTorch or a serving engine.

| Component | Current state |
| --- | --- |
| PyTorch reference | Quantization, paged cache writes, decode attention, and partial-state merge |
| CUDA Path A | FP16 and INT8 decode over the whole history |
| CUDA Path B | FP16 and INT8 decode over multiple chunks, with a PyTorch merge |
| GPU validation | Correctness and Compute Sanitizer runs recorded in the [roadmap](ROADMAP.md) |
| CUDA cache write | Planned; cache writes currently use the PyTorch reference |
| Benchmarks and profiling | Planned, with PyTorch SDPA on a gathered FP16 history as the baseline |
| Workload selector | Planned after measurements; chunk count is currently chosen by the caller |

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The repository pins
Python 3.12 for development. The reference implementation runs on CPU.

```sh
git clone https://github.com/krishmula/warpcache.git
cd warpcache
uv sync
uv run pytest
```

On a machine without an NVIDIA GPU, CUDA tests are skipped automatically.

### Try the reference on CPU

Run this from the repository root with `uv run python`, or save it as a Python
file in the repository root and run it with `uv run python <filename>.py`.

```python
import torch

from reference.paged import PagedKVCache, paged_decode_attention

torch.manual_seed(0)

# One request with 32 history tokens in two pages of 16 tokens each.
cache = PagedKVCache.allocate(
    num_pages=2, page_size=16, num_kv_heads=8,
    head_dim=128, dtype=torch.int8,
)
k = torch.randn(32, 8, 128)
v = torch.randn_like(k)
cache.write(k, v, slots=torch.arange(32))

q = torch.randn(1, 16, 128).half()
block_table = torch.tensor([[0, 1]], dtype=torch.int32)
seq_lens = torch.tensor([32], dtype=torch.int32)

output = paged_decode_attention(q, cache, block_table, seq_lens)
print(output.shape)  # torch.Size([1, 16, 128])
```

## CUDA usage

Requires an NVIDIA GPU, a CUDA-enabled PyTorch installation, a compatible CUDA
toolkit with `nvcc`, a C++ compiler, and Ninja. The toolkit must match PyTorch's
CUDA major version. Install Ninja on Ubuntu with `sudo apt-get install ninja-build`.
The extension compiles on the first call.

The recorded RTX 3090 environment uses Ubuntu 22.04, Python 3.12,
PyTorch 2.14.1+cu130, CUDA 13.0, and driver 580.95.05. See the
[environment manifest](results/env/ubuntu-20261104-233543.json) for details.

After running the CPU example, move its inputs to the GPU and choose a path:

```python
from kernels.decode import decode

args = (
    q.cuda(), cache.k.cuda(), cache.v.cuda(),
    cache.k_scale.cuda(), cache.v_scale.cuda(),
    block_table.cuda(), seq_lens.cuda(),
)

output_a = decode(*args, num_chunks=1)  # Path A
output_b = decode(*args, num_chunks=4)  # Path B
torch.testing.assert_close(output_a, output_b, rtol=1e-3, atol=1e-3)
```

Both paths return FP16 output of shape `[batch_size, 16, 128]`. For an FP16
cache, pass `None` for both scale arguments.

Run the CUDA correctness tests with:

```sh
uv run pytest tests/test_decode.py tests/test_warp_reduce.py
```

Before profiling on a new GPU host, run `bash scripts/gpu_smoke_test.sh`. It
records the environment, compiles a small CUDA extension, and checks Nsight
Compute counter access. This check also requires `ncu` on `PATH` and permission
to read GPU performance counters; the recorded VM required running `ncu` as root.

## Design

The CUDA kernel uses one block of 32 threads per `(request, query head, chunk)`.
Each lane handles four of the 128 head dimensions, with warp shuffles combining
the dot products. Keys and values are read through a block table from the paged
cache. INT8 values are dequantized as they are loaded.

| Path | Chunk count | Behavior |
| --- | --- | --- |
| **Path A** | 1 | Each block covers the whole history and writes the normalized output. |
| **Path B** | More than 1 | Each block writes a partial state `(m, l, u)` to an FP32 workspace; PyTorch merges the partial states. |

Here, `m` is the maximum attention score, `l` is the softmax denominator, and
`u` is the unnormalized weighted sum of values. An empty chunk contributes
`m = -inf`, `l = 0`, and `u = 0`.

Path B explores a specific limitation of Path A: at batch size 1 and 16 query
heads, Path A launches only 16 blocks on an RTX 3090 with 82 SMs. More chunks
create more blocks, at the cost of workspace traffic and the merge.

### Supported configuration

| Parameter | Value |
| --- | --- |
| Query tokens | One per request per decode call |
| Query heads / KV heads | 16 / 8, with grouped-query attention |
| Head dimension | 128 |
| Page size | 16 tokens |
| Query / output | FP16 |
| Accumulation / workspace | FP32 |
| Cache | INT8 or FP16 |
| INT8 scales | One FP32 scale per token and KV head, separately for K and V |
| Tensor layout | Contiguous; full shapes in the [tensor contract](docs/scope.md#tensor-contract) |

The shape matches Qwen3-1.7B. This is a standalone attention kernel study;
prefill, serving-engine integration, multi-GPU execution, and production
packaging are outside the current scope.

## Quantization error

Smaller storage does not guarantee acceptable numerical error. Initial CPU
experiments on real model K/V found that outlier channels in K can make
per-token INT8 quantization lossy. The current kernel does not implement K
smoothing. See the [quantization findings](ROADMAP.md#quantization-finding)
for the measured errors, experiment limits, and related work.

To reproduce the error measurement on your own text:

```sh
uv run --group capture python scripts/kv_quant_error.py \
  --model Qwen/Qwen3-1.7B --text path/to/your_text.txt
```

This runs on CPU and downloads the model. It measures attention-output error,
not generation quality.

## Documentation

| Path | Contents |
| --- | --- |
| [Scope and tensor contract](docs/scope.md) | Supported inputs, operations, and claim boundaries |
| [Roadmap](ROADMAP.md) | Implementation status, validation records, and experiment plan |
| [Learning roadmap](docs/learning-roadmap.md) | CUDA concepts and exercises; its original schedule is superseded by the roadmap |
| [kernels/](kernels/) | CUDA decode and warp-reduction kernels, with Python wrappers |
| [reference/](reference/) | PyTorch quantization, paged cache, attention, and merge |
| [tests/](tests/) | CPU reference and CUDA correctness tests |
| [scripts/](scripts/) | GPU environment checks and real-model quantization measurements |
| [results/](results/) | Environment manifests and GPU usage log |

## Feedback and contributions

Bug reports, numerical edge cases, and reproducibility feedback are welcome
through [GitHub issues](https://github.com/krishmula/warpcache/issues). For a GPU
report, include the commit, GPU model, PyTorch and CUDA versions, input shapes,
cache dtype, chunk count, and a minimal example.

## License

[MIT](LICENSE) © 2026 Krishna Mula.
