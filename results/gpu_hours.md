# GPU hours

One row per instance session. Hours count instance time, not GPU busy time.
Fill hours and cost from the provider's billing page.

| Date | Provider | GPU | Hours | Cost (USD) | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-24 | RunPod | RTX 4090 (container) | TBD | TBD | Smoke test failed: `ERR_NVGPUCTRPERM`. Ruled out. Manifest `results/env/7bd7f93371ab-20261024-183904.json`. |
| 2026-10-24 | Vast.ai | RTX 3090 (VM) | TBD | TBD | Gate 0 passed. Manifest `results/env/ubuntu-20261024-191743.json`. |
| 2026-10-31 | Vast.ai | RTX 3090 (VM) | TBD | TBD | New host, smoke test passed. Decode v1 FP16: 10 tests pass, `compute-sanitizer` 0 errors. Manifest `results/env/ubuntu-20261031-201423.json`. |
| 2026-11-03 | Vast.ai | RTX 4090 (VM) | TBD | TBD | Correctness only. Smoke test passed, `ncu` counters work. Decode v1 FP16 and INT8: 20 tests pass, `compute-sanitizer` 0 errors. Manifest `results/env/ubuntu-20261103-023710.json`. |
| 2026-11-04 | Vast.ai | RTX 3090 (VM) | TBD | TBD | Smoke test passed, `ncu` counters work. Path B: 52 decode tests pass, `compute-sanitizer` 0 errors, full suite 97 pass. `git_dirty` was only the untracked CUDA keyring `.deb`. Manifest `results/env/ubuntu-20261104-233543.json`. |
| 2026-11-07 | Vast.ai | RTX 3090 (VM) | TBD | TBD | Smoke test passed, `ncu` counters work. `cache_write`: 14 tests pass with exact match, `compute-sanitizer` 0 errors, full suite 111 pass. Manifest `results/env/ubuntu-20261107-001822.json`. |
| 2026-11-07 | Vast.ai | RTX 3090 (VM) | TBD | TBD | New host for timing. Smoke test passed at `df60fb2`, `ncu` counters work, `git_dirty` false. Manifest `results/env/ubuntu-20261107-005603.json`. |
