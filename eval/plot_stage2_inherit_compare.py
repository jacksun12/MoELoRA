import argparse
import ast
import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.plot_style import MOPRED_COLORS, MOPRED_PATTERNS, apply_global_plot_style, style_axes


apply_global_plot_style()


def _parse_ts(line):
    ts = line.split(" | ", 1)[0]
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")


def parse_stage2_summary(log_path):
    metrics = None
    stage2_start = None
    stage2_adapt_end = None
    stage2_end = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "[Stage-2] Running incremental re-clustering and overlap-driven update..." in line:
                stage2_start = _parse_ts(line)
            elif "[stage2][Router] centroid routing" in line:
                stage2_adapt_end = _parse_ts(line)
            elif "[Stage-2] {'text_generation_metrics':" in line:
                payload = line.split("[Stage-2] ", 1)[1]
                parsed = ast.literal_eval(payload)
                metrics = parsed["text_generation_metrics"]
                stage2_end = _parse_ts(line)
            elif "[Stage-2] text_generation_metrics=" in line:
                payload = line.split("text_generation_metrics=", 1)[1]
                metrics = ast.literal_eval(payload)
                stage2_end = _parse_ts(line)

    if metrics is None:
        raise ValueError(f"Failed to find Stage-2 metrics in {log_path}")

    adapt_seconds = None
    total_seconds = None
    if stage2_start and stage2_adapt_end:
        adapt_seconds = (stage2_adapt_end - stage2_start).total_seconds()
    if stage2_start and stage2_end:
        total_seconds = (stage2_end - stage2_start).total_seconds()

    return {
        "bleu1": float(metrics.get("bleu1", 0.0)),
        "rougeL_f1": float(metrics.get("rougeL_f1", 0.0)),
        "adapt_seconds": adapt_seconds,
        "total_stage2_seconds": total_seconds,
    }


def plot_compare(no_inherit, inherit, output_path, title):
    labels = ["No Inherit", "Inherit"]
    bleu_vals = [no_inherit["bleu1"], inherit["bleu1"]]
    rouge_vals = [no_inherit["rougeL_f1"], inherit["rougeL_f1"]]
    adapt_vals = [no_inherit["adapt_seconds"] or 0.0, inherit["adapt_seconds"] or 0.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="#f7f5f2")
    colors = MOPRED_COLORS[:2]
    payloads = [
        ("Stage-2 BLEU-1", bleu_vals, "score"),
        ("Stage-2 ROUGE-L F1", rouge_vals, "score"),
        ("Stage-2 Adaptation Time", adapt_vals, "seconds"),
    ]

    for ax, (subtitle, values, unit) in zip(axes, payloads):
        bars = ax.bar(labels, values, color=colors, edgecolor="#1a1a1a", linewidth=0.6)
        for idx, bar in enumerate(bars):
            bar.set_hatch(MOPRED_PATTERNS[idx % len(MOPRED_PATTERNS)])
        ax.set_title(subtitle, fontsize=16, fontweight="bold")
        style_axes(ax, grid_axis="y")
        ymax = max(values) * 1.18 if max(values) > 0 else 1.0
        ax.set_ylim(0, ymax)
        for bar, value in zip(bars, values):
            label = f"{value:.3f}" if unit == "score" else f"{value:.1f}s"
            ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.02, label, ha="center", va="bottom", fontsize=10)

    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_inherit_log", type=str, required=True)
    parser.add_argument("--inherit_log", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="eval/figures/stage2_inherit_compare.png")
    args = parser.parse_args()

    no_inherit = parse_stage2_summary(args.no_inherit_log)
    inherit = parse_stage2_summary(args.inherit_log)
    plot_compare(no_inherit, inherit, args.output_path, title="Stage-2 Inherit vs No-Inherit")
    print({"no_inherit": no_inherit, "inherit": inherit, "output_path": args.output_path})


if __name__ == "__main__":
    main()
