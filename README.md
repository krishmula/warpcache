# warpcache

CUDA decode attention over an INT8 paged KV cache, with a direct path, a
split-history path, and a workload selector. CS 259 class project.

- Scope and data contract: [docs/scope.md](docs/scope.md)
- Plan and status: [ROADMAP.md](ROADMAP.md)

## Setup

```sh
uv sync
uv run pytest            # reference tests, CPU only
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
