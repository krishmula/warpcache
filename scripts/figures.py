"""Draw the paper's figures from the benchmark result lines in results/bench.

    uv run python scripts/figures.py                      # results/bench -> results/figures
    uv run python scripts/figures.py --bench DIR --out DIR

Each case key keeps the line from the newest file, and file names sort by time.
Each figure is written as a PDF and a PNG.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MultipleLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench import BATCHES, CHUNKS, HISTORIES, case_key  # noqa: E402
from kernels.decode import select_chunks  # noqa: E402

# The only run with the merge from before the fix. Its path_b lines enter the
# case table under this kernel name.
OLD_MERGE_FILE = "ubuntu-20261107-011259.jsonl"
OLD_MERGE = "path_b_old_merge"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"

# Color follows the entity in every figure.
KERNEL_STYLE = {
    "path_a": dict(color=BLUE, marker="o", label="Path A"),
    "path_b": dict(color=ORANGE, marker="s", label="Path B"),
    "sdpa": dict(color=AQUA, marker="D", label="SDPA"),
}
HISTORY_COLOR = {4096: VIOLET, 16384: AQUA}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "figure.facecolor": SURFACE,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "pdf.fonttype": 42,
})


def history_label(history: int) -> str:
    return f"{history // 1024}K"


def load_cases(bench_dir: Path) -> dict[tuple, dict]:
    """Every result line, keyed by case key, the newest file winning each key."""
    cases = {}
    for path in sorted(Path(bench_dir).glob("*.jsonl")):
        for text in path.read_text().splitlines():
            if text.strip():
                line = json.loads(text)
                cases[case_key(line)] = line
    old = Path(bench_dir) / OLD_MERGE_FILE
    if old.exists():
        for text in old.read_text().splitlines():
            line = json.loads(text) if text.strip() else None
            if line and line["kernel"] == "path_b":
                cases[case_key({**line, "kernel": OLD_MERGE})] = line
    return cases


def median(cases, kernel, dtype, batch, history, chunks=None) -> float:
    return cases[(kernel, dtype, batch, history, chunks)]["median_us"]


def us_format(value, _pos=None) -> str:
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(us_format))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="minor", visible=False)


def chunk_x(ax, chunks):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(chunks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", visible=False)


def titles(fig, title, subtitle):
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, subtitle, ha="left", va="top", fontsize=9, color=INK_2)


def end_label(ax, x, y, text, dx=6):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK_2)


def latency_vs_history(cases):
    chunks = {h: select_chunks(h) for h in HISTORIES}
    fig, axes = plt.subplots(1, len(BATCHES), figsize=(9, 3.6), sharey=True)
    for ax, batch in zip(axes, BATCHES):
        series = {
            "path_a": [median(cases, "path_a", "fp16", batch, h, 1) for h in HISTORIES],
            "path_b": [median(cases, "path_b", "fp16", batch, h, chunks[h]) for h in HISTORIES],
            "sdpa": [median(cases, "sdpa", "fp16", batch, h) for h in HISTORIES],
        }
        for kernel, ys in series.items():
            style = KERNEL_STYLE[kernel]
            ax.plot(HISTORIES, ys, color=style["color"], marker=style["marker"],
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=style["label"])
            end_label(ax, HISTORIES[-1], ys[-1], f"{ys[-1]:,.0f}")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(HISTORIES))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: history_label(int(v))))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(HISTORIES[0] / 1.4, HISTORIES[-1] * 2.0)
        ax.grid(axis="x", visible=False)
        log_y(ax)
        ax.set_title(f"Batch size {batch}", loc="left", color=INK)
        ax.set_xlabel("History (tokens)")
    axes[0].set_ylabel("GPU median latency (us)")
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [f"Path B, {chunks[HISTORIES[0]]} chunks" if l == "Path B" else l for l in labels]
    fig.legend(handles, labels, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.99))
    titles(fig, "Decode latency against history, FP16",
           "Path B cuts Path A's latency at every batch size, but SDPA stays faster. "
           "Values at 16K are labeled.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def int8_vs_fp16(cases):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    groups = [(b, h) for b in BATCHES for h in HISTORIES]
    x = np.arange(len(groups))
    offsets = {"path_a": -0.12, "path_b": 0.12}
    ratios_all = []
    for kernel, dx in offsets.items():
        ratios = []
        for b, h in groups:
            c = 1 if kernel == "path_a" else select_chunks(h)
            ratios.append(median(cases, kernel, "int8", b, h, c) / median(cases, kernel, "fp16", b, h, c))
        ratios_all += ratios
        style = KERNEL_STYLE[kernel]
        label = style["label"] if kernel == "path_a" else f"Path B, {select_chunks(HISTORIES[0])} chunks"
        ax.vlines(x + dx, 1.0, ratios, color=style["color"], linewidth=2)
        ax.plot(x + dx, ratios, linestyle="none", color=style["color"], marker=style["marker"],
                markeredgecolor=SURFACE, markeredgewidth=1.5, markersize=7, label=label)
    ax.axhline(1.0, color=INK_2, linewidth=1)
    ax.annotate("1.0 = no gain", (-0.45, 1.0), xytext=(0, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, color=INK_2)
    span = max(0.06, max(abs(r - 1) for r in ratios_all) * 1.4)
    ax.set_ylim(1 - span, 1 + span)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
    ax.set_xticks(x, [f"{history_label(h)}" for _b, h in groups])
    for i in range(0, len(groups), len(HISTORIES)):
        ax.text(i + (len(HISTORIES) - 1) / 2, -0.16, f"Batch size {groups[i][0]}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("History", labelpad=22)
    ax.set_ylabel("INT8 time / FP16 time")
    ax.legend(loc="upper left", ncol=2)
    lo, hi = min(ratios_all), max(ratios_all)
    titles(fig, "INT8 against FP16 decode time",
           f"Every ratio lies between {lo:.2f} and {hi:.2f}. Halving the bytes read does not speed\n"
           "decode up, because the kernel is latency bound, not bandwidth bound.")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return fig


def chunk_sweep(cases):
    chunks = [1] + CHUNKS
    fig, axes = plt.subplots(1, len(BATCHES), figsize=(9.5, 3.9), sharey=True)
    for ax, batch in zip(axes, BATCHES):
        for h in HISTORIES:
            color = HISTORY_COLOR[h]
            full = [median(cases, "path_a", "fp16", batch, h, 1)]
            full += [median(cases, "path_b", "fp16", batch, h, c) for c in CHUNKS]
            kernel_only = [median(cases, "path_b_kernel", "fp16", batch, h, c) for c in CHUNKS]
            ax.plot(chunks, full, color=color, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            ax.plot(CHUNKS, kernel_only, color=color, linestyle=(0, (3, 2)), linewidth=1.5)
            pick = select_chunks(h)
            y = full[chunks.index(pick)]
            ax.plot([pick], [y], linestyle="none", marker="o", markersize=10, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=1.3)
            end_label(ax, chunks[-1], full[-1], history_label(h), dx=8)
        chunk_x(ax, chunks)
        ax.set_xlim(0.8, 256 * 1.9)
        log_y(ax)
        ax.set_title(f"Batch size {batch}", loc="left", color=INK)
        ax.set_xlabel("Chunks per request (1 = Path A)")
    axes[0].set_ylabel("GPU median latency (us)")
    handles = [
        Line2D([], [], color=HISTORY_COLOR[h], label=f"History {history_label(h)}") for h in HISTORIES
    ] + [
        Line2D([], [], color=MUTED, label="Path B with merge"),
        Line2D([], [], color=MUTED, linestyle=(0, (3, 2)), linewidth=1.5, label="Kernel alone, no merge"),
        Line2D([], [], linestyle="none", marker="o", markersize=9, markerfacecolor="none",
               markeredgecolor=INK, label=f"Selector's choice ({select_chunks(HISTORIES[0])})"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=5, bbox_to_anchor=(0.99, 0.88))
    titles(fig, "Path B latency against chunk count, FP16",
           "Latency falls as chunks are added until the GPU is full, then flattens. "
           "The gap between each solid line and its dashed line is the merge cost.")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return fig


def cache_write_cost(cases):
    history = HISTORIES[0]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(BATCHES))[::-1]
    rows = []
    for yi, batch in zip(y, BATCHES):
        w16 = median(cases, "cache_write", "fp16", batch, history)
        w8 = median(cases, "cache_write", "int8", batch, history)
        decode = min(median(cases, "path_b", "fp16", batch, history, c) for c in CHUNKS)
        rows.append((w16, w8, decode))
        ax.hlines(yi, w16, decode, color=GRID, linewidth=2, zorder=1)
        ax.plot([w16], [yi], "o", color=VIOLET, markersize=8, markeredgecolor=SURFACE, zorder=3)
        ax.plot([w8], [yi], "D", color=AQUA, markersize=7, markeredgecolor=SURFACE, zorder=3)
        ax.plot([decode], [yi], "s", color=ORANGE, markersize=8, markeredgecolor=SURFACE, zorder=3)
        ax.annotate(f"{w8:.1f} us", (w8, yi), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK_2)
        ax.annotate(f"{decode:,.0f} us  (write is {100 * w8 / decode:.1f}%)", (decode, yi),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1,)))
    ax.xaxis.set_major_formatter(FuncFormatter(us_format))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1, 1e4)
    ax.set_yticks(y, [f"Batch size {b}" for b in BATCHES])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("GPU median time per call (us, log scale)")
    handles = [
        Line2D([], [], linestyle="none", marker="o", color=VIOLET, markersize=7, label="cache_write, FP16"),
        Line2D([], [], linestyle="none", marker="D", color=AQUA, markersize=6, label="cache_write, INT8"),
        Line2D([], [], linestyle="none", marker="s", color=ORANGE, markersize=7,
               label="Fastest Path B decode, FP16"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    writes = [r[0] for r in rows] + [r[1] for r in rows]
    titles(fig, f"cache_write cost against decode, history {history_label(history)}",
           f"Writing one token takes {min(writes):.1f} to {max(writes):.1f} us, against "
           f"{min(r[2] for r in rows):,.0f} to {max(r[2] for r in rows):,.0f} us for the fastest decode.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def merge_fix(cases):
    batch, history = 1, HISTORIES[0]
    chunks = [1] + CHUNKS
    path_a = median(cases, "path_a", "fp16", batch, history, 1)
    old = [path_a] + [median(cases, OLD_MERGE, "fp16", batch, history, c) for c in CHUNKS]
    new = [path_a] + [median(cases, "path_b", "fp16", batch, history, c) for c in CHUNKS]
    kernel_only = [median(cases, "path_b_kernel", "fp16", batch, history, c) for c in CHUNKS]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(chunks, old, color=MUTED, marker="o", markersize=5, markeredgecolor=SURFACE,
            label="Path B, old merge (before the fix)")
    ax.plot(chunks, new, color=ORANGE, marker="s", markersize=5, markeredgecolor=SURFACE,
            label="Path B, new merge")
    ax.plot(CHUNKS, kernel_only, color=ORANGE, linestyle=(0, (3, 2)), linewidth=1.5,
            label="Kernel alone, no merge")
    end_label(ax, chunks[-1], old[-1], f"{old[-1]:,.0f}")
    end_label(ax, chunks[-1], new[-1], f"{new[-1]:,.0f}")
    end_label(ax, CHUNKS[-1], kernel_only[-1], f"{kernel_only[-1]:,.0f}", dx=6)
    floor = min(old[1:]) - min(kernel_only)
    i = CHUNKS.index(64)
    ax.annotate("", xy=(64, kernel_only[i]), xytext=(64, old[i + 1]),
                arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=0.8, shrinkA=2, shrinkB=2))
    ax.annotate(f"old merge adds about {floor:,.0f} us", (64, np.sqrt(kernel_only[i] * old[i + 1])),
                xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5,
                color=INK_2)
    chunk_x(ax, chunks)
    ax.set_xlim(0.8, 256 * 1.7)
    log_y(ax)
    ax.set_xlabel("Chunks per request (1 = Path A)")
    ax.set_ylabel("GPU median latency (us)")
    ax.legend(loc="lower left")
    titles(fig, f"The merge fix, batch size {batch}, history {history_label(history)}, FP16",
           f"Before the fix, the merge set a floor near {min(old[1:]):,.0f} us whatever the chunk count.")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


FIGURES = [
    ("latency_vs_history", latency_vs_history),
    ("int8_vs_fp16", int8_vs_fp16),
    ("chunk_sweep", chunk_sweep),
    ("cache_write_cost", cache_write_cost),
    ("merge_fix", merge_fix),
]


def main(argv=None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", type=Path, default=ROOT / "results" / "bench")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args(argv)
    cases = load_cases(args.bench)
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, draw in FIGURES:
        fig = draw(cases)
        for ext in ("pdf", "png"):
            path = args.out / f"{name}.{ext}"
            # No CreationDate, so rerunning on the same data gives the same PDF bytes.
            fig.savefig(path, dpi=200, metadata={"CreationDate": None} if ext == "pdf" else None)
            written.append(path)
        plt.close(fig)
    for path in written:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return written


if __name__ == "__main__":
    main()
