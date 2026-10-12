"""Measure per-token-head INT8 error on real K/V captured from a Hugging Face model.

Captures post-RoPE Q, K, and V at every layer through a custom attention function,
then compares decode attention over the full-precision cache against attention over
the INT8-reconstructed cache, with and without static K smoothing.

K smoothing divides K by a fixed per-(KV head, channel) factor s before quantizing
and multiplies Q by s, so q.k is unchanged (the idea behind QServe's SmoothAttention).
s is calibrated on the first half of the tokens; errors are measured on the second half.

uv run --group capture python scripts/kv_quant_error.py --model Qwen/Qwen2.5-1.5B --text FILE
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.attention import decode_attention  # noqa: E402
from reference.quant import dequantize, quantize  # noqa: E402

captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []


def capture_attention(module, query, key, value, attention_mask, **kwargs):
    # query [1, Hq, T, D], key/value [1, Hkv, T, D] after RoPE (and QK-norm if any).
    captured.append((query[0].transpose(0, 1), key[0].transpose(0, 1), value[0].transpose(0, 1)))
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--text", required=True, type=Path)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--queries", type=int, default=16, help="decode positions sampled per layer")
    p.add_argument("--save", type=Path, help="optional .pt file for the raw captures")
    args = p.parse_args()

    AttentionInterface.register("capture", capture_attention)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="capture"
    )
    ids = tok(args.text.read_text(), return_tensors="pt").input_ids[:, : args.tokens]
    with torch.no_grad():
        model(ids)
    if args.save:
        torch.save(captured, args.save)

    T = ids.shape[1]
    calib = T // 2
    positions = torch.linspace(calib, T - 1, args.queries).long().tolist()
    print(f"model={args.model} tokens={T} layers={len(captured)} calibration=tokens[0:{calib}]")
    print("attention output relative error, mean/max over sampled decode positions")
    print("layer  per-token        smoothed-K       K_channel_outlier_ratio")
    summary = {"per-token": [], "smoothed-K": []}
    for layer, (q, k, v) in enumerate(captured):
        q, k, v = q.float(), k.float(), v.float()
        group = q.shape[1] // k.shape[1]
        s = k[:calib].abs().amax(dim=0).clamp(min=1e-3).sqrt()
        s = s / s.mean(dim=-1, keepdim=True)
        vq = dequantize(*quantize(v))
        variants = {
            "per-token": (q, dequantize(*quantize(k))),
            "smoothed-K": (q * s.repeat_interleave(group, dim=0), dequantize(*quantize(k / s))),
        }
        row = []
        for name, (qv, kv) in variants.items():
            errs = [
                rel_err(decode_attention(qv[t], kv[: t + 1], vq[: t + 1]),
                        decode_attention(q[t], k[: t + 1], v[: t + 1]))
                for t in positions
            ]
            summary[name].append(errs)
            row.append(f"{sum(errs) / len(errs):.4f}/{max(errs):.4f}")
        channel_amax = k.abs().amax(dim=(0, 1))
        outlier = (channel_amax.max() / channel_amax.median()).item()
        print(f"{layer:5d}  {row[0]}    {row[1]}    {outlier:7.1f}")
    for name, rows in summary.items():
        flat = [e for r in rows for e in r]
        print(f"{name:10s}  mean {sum(flat) / len(flat):.4f}  worst {max(flat):.4f}")


if __name__ == "__main__":
    main()
