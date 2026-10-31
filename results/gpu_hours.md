# GPU hours

One row per instance session. Hours count instance time, not GPU busy time.
Fill hours and cost from the provider's billing page.

| Date | Provider | GPU | Hours | Cost (USD) | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-24 | RunPod | RTX 4090 (container) | TBD | TBD | Smoke test failed: `ERR_NVGPUCTRPERM`. Ruled out. Manifest `results/env/7bd7f93371ab-20261024-183904.json`. |
| 2026-10-24 | Vast.ai | RTX 3090 (VM) | TBD | TBD | Gate 0 passed. Manifest `results/env/ubuntu-20261024-191743.json`. |
| 2026-10-31 | Vast.ai | RTX 3090 (VM) | TBD | TBD | New host, smoke test passed. Decode v1 FP16: 10 tests pass, `compute-sanitizer` 0 errors. Manifest `results/env/ubuntu-20261031-201423.json`. |
