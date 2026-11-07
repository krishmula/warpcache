"""Run one benchmark case inside an NVTX range named "case", for Nsight Compute.

    ncu --nvtx --nvtx-include "case/" --set full -o results/ncu/NAME \
        uv run python scripts/ncu_case.py --kernel path_b --dtype fp16 --batch 1 --history 16384 --chunks 64

Two calls before the range compile the kernel and warm up, so ncu profiles
only the kernels of the third call: the decode kernel, plus the merge for
Path B, or the SDPA kernels.
"""

import argparse
import sys

import torch

from bench import Inputs, bytes_per_token, sdpa

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--kernel", choices=["path_a", "path_b", "sdpa"], required=True)
p.add_argument("--dtype", choices=["fp16", "int8"], default="fp16")
p.add_argument("--batch", type=int, required=True)
p.add_argument("--history", type=int, required=True)
p.add_argument("--chunks", type=int, default=1)
args = p.parse_args()
if not torch.cuda.is_available():
    sys.exit("ncu_case: needs an NVIDIA GPU")

from kernels.decode import decode  # noqa: E402

torch.manual_seed(0)
inp = Inputs(args.batch, args.history, args.batch * args.history * bytes_per_token(args.dtype))
if args.kernel == "sdpa":
    k, v = inp.gathered("fp16", 0)
    call = lambda: sdpa(inp.q[0], k, v)  # noqa: E731
else:
    chunks = 1 if args.kernel == "path_a" else args.chunks
    call = lambda: decode(*inp.decode_args(args.dtype, 0), num_chunks=chunks)  # noqa: E731

call()
call()
torch.cuda.synchronize()
with torch.cuda.nvtx.range("case"):
    call()
torch.cuda.synchronize()
print("ncu_case: done")
