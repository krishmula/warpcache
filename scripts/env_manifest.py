"""Record the environment that produced a set of results as JSON."""

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


manifest = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "git_commit": run(["git", "rev-parse", "HEAD"]),
    "git_dirty": bool(run(["git", "status", "--porcelain"])),
    "nvcc": run(["nvcc", "--version"]),
    "nvidia_smi": run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit",
         "--format=csv,noheader"]
    ),
}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    manifest["gpu"] = {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "memory_bytes": props.total_memory,
        "l2_bytes": getattr(props, "L2_cache_size", None),
    }

out = Path("results/env") / f"{manifest['host']}-{datetime.now():%Y%m%d-%H%M%S}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
print(f"wrote {out}")
