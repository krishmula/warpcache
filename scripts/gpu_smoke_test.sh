#!/usr/bin/env bash
# First session on a GPU box: environment manifest, custom CUDA compile, Nsight counter access.
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi
nvcc --version
uv run python scripts/env_manifest.py
uv run python scripts/tiny_kernel.py

if ! command -v ncu >/dev/null; then
  echo "FAIL: ncu (Nsight Compute) not on PATH"; exit 1
fi
log=$(mktemp)
ncu --section SpeedOfLight uv run python scripts/tiny_kernel.py 2>&1 | tee "$log"
if grep -q ERR_NVGPUCTRPERM "$log"; then
  echo "FAIL: performance counters blocked. Try sudo ncu, or pick a provider that allows counter access."
  exit 1
fi
grep -q "Memory Throughput" "$log" || { echo "FAIL: ncu produced no metrics"; exit 1; }
echo "SMOKE TEST PASSED"
