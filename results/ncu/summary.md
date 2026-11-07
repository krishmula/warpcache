# Nsight Compute results, November 7, 2026

RTX 3090 on Vast.ai, manifest `results/env/ubuntu-20261107-005603.json`, commit
`825f57d`. Each report profiles one call from `scripts/ncu_case.py` with
`--set full`. `ncu` locks the SM clock at 1.39 GHz, so durations run about 15 to
28% longer than the boost-clock times from `scripts/bench.py`. The `.ncu-rep`
files are not in git (`*.ncu-rep` is ignored). The numbers below are copied from
`ncu --import <report> --page details --section SpeedOfLight --section Occupancy
--section LaunchStats`.

## Decode and SDPA kernels (FP16, 16K history)

| Case | Kernel | Grid | Block | Duration | DRAM % of peak | Achieved occupancy | Theoretical occupancy | Registers per thread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Path A, batch size 1 | `decode_kernel` | (16, 1, 1) | 32 | 10.32 ms | 0.74 | 2.08% | 33.33% | 34 |
| Path B, batch size 1, 64 chunks | `decode_kernel` | (16, 1, 64) | 32 | 180.38 us | 42.13 | 25.87% | 33.33% | 34 |
| Path B, batch size 16, 128 chunks | `decode_kernel` | (16, 16, 128) | 32 | 2.27 ms | 53.02 | 32.98% | 33.33% | 34 |
| SDPA, batch size 16 | `flash_fwd_splitkv_kernel` | (1, 5, 128) | 128 | 1.41 ms | 83.62 | 8.34% | 8.33% | 244 |
| SDPA, batch size 16 | `flash_fwd_splitkv_combine_kernel` | (64, 1, 1) | 128 | 7.10 us | 10.67 | 8.27% | 75% | 52 |

`decode_kernel` occupancy is limited by the 16 resident blocks per SM: a
32-thread block gives at most 16 of 48 warps. SDPA occupancy is limited by
81.92 KB of shared memory per block.

Path A runs 14,398,507 cycles for 16,384 tokens, about 879 cycles per token.

## Path B merge kernels

| Kernel | Batch size 1, 64 chunks | Batch size 16, 128 chunks |
| --- | --- | --- |
| `reduce_kernel` (max over chunks) | 5.31 us | 5.18 us |
| `elementwise_kernel` (subtract) | 3.62 us | 3.68 us |
| `vectorized_elementwise_kernel` (exp) | 2.88 us | 2.98 us |
| cuBLAS gemv `kernel` (l sum) | 3.68 us | 4.19 us |
| `gemvNSP_kernel` / `gemv2N_kernel` (u sum) | 4.58 us | 26.40 us, 78.94% DRAM |
| `elementwise_kernel` (divide) | 3.68 us | 3.84 us |
| `vectorized_elementwise_kernel` (to FP16) | 2.75 us | 2.88 us |
| Total | 26.50 us | 49.15 us |
