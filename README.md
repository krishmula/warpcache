# warpcache

CUDA decode attention over an INT8 paged KV cache, with a direct path, a
split-history path, and a workload selector. CS 259 class project.

- Scope and data contract: [docs/scope.md](docs/scope.md)
- Plan and status: [ROADMAP.md](ROADMAP.md)
- Step-by-step plan with what to learn first: [docs/learning-roadmap.md](docs/learning-roadmap.md)

## Setup

```sh
uv sync
uv run pytest            # reference tests, CPU only
```

To measure INT8 error on real model K/V (CPU, downloads the model):

```sh
uv run --group capture python scripts/kv_quant_error.py --model Qwen/Qwen3-1.7B --text some_long_text.txt
```

On a GPU host, run `scripts/gpu_smoke_test.sh` first. It records the
environment, compiles a tiny CUDA extension, and checks Nsight Compute access.

## Layout

| Path | Contents |
| --- | --- |
| `reference/` | PyTorch reference: quantization, paged cache, decode attention, merge |
| `tests/` | Correctness tests |
| `scripts/` | GPU smoke test and environment manifest |
| `docs/` | Scope, design, experiment protocol |
