import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.plot_style import MOPRED_COLORS, MOPRED_PATTERNS, apply_global_plot_style, style_axes


apply_global_plot_style()


DISPLAY_NAMES = {
    "raw_base": "Base Model",
    "single_lora_r32": "LoRA r32",
    "mocle_4x8": "MoCLE 4x8",
    "hydralora_4x8": "HydraLoRA 4x8",
    "moe_lora_stage1": "MoE-LoRA Stage-1",
}
MARKERS = ["o", "s", "^", "D", "P", "X"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot quality vs system cost trade-off.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--metric", default="rougeL_f1", choices=["bleu1", "rougeL_f1"])
    parser.add_argument(
        "--x_axis",
        default="eval_peak_alloc_gb",
        choices=[
            "train_peak_alloc_gb",
            "train_peak_reserved_gb",
            "train_elapsed_sec",
            "eval_peak_alloc_gb",
            "eval_peak_reserved_gb",
            "eval_elapsed_sec",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results", [])
    if not rows:
        raise ValueError("No results found in input JSON.")

    fig, ax = plt.subplots(figsize=(9.2, 6.8), facecolor="#f7f5f2")
    ax.set_facecolor("#fcfbf8")

    for idx, row in enumerate(rows):
        x = row.get(args.x_axis)
        y = row.get(args.metric)
        if x is None or y is None:
            continue
        color = MOPRED_COLORS[idx % len(MOPRED_COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        ax.scatter([x], [y], s=220, color=color, edgecolors="#1a1a1a", linewidths=1.2, marker=marker, zorder=3)
        ax.text(
            x,
            y,
            f" {DISPLAY_NAMES.get(row['method'], row['method'])}",
            fontsize=12,
            fontweight="bold",
            color="#1f1f1f",
            va="center",
            ha="left",
        )

    x_labels = {
        "train_peak_alloc_gb": "Training Peak GPU Memory Allocated (GB)",
        "train_peak_reserved_gb": "Training Peak GPU Memory Reserved (GB)",
        "train_elapsed_sec": "Training Wall-Clock Time (s)",
        "eval_peak_alloc_gb": "Evaluation Peak GPU Memory Allocated (GB)",
        "eval_peak_reserved_gb": "Evaluation Peak GPU Memory Reserved (GB)",
        "eval_elapsed_sec": "Evaluation Wall-Clock Time (s)",
    }
    y_labels = {
        "bleu1": "BLEU-1",
        "rougeL_f1": "ROUGE-L F1",
    }
    ax.set_xlabel(x_labels[args.x_axis], fontsize=18, fontweight="bold")
    ax.set_ylabel(y_labels[args.metric], fontsize=18, fontweight="bold")
    style_axes(ax, grid_axis="both")

    title = f"Task Quality vs System Cost ({args.metric} vs {args.x_axis})"
    ax.set_title(title, fontsize=20, fontweight="bold", pad=16)

    output_path = args.output_path
    if not output_path:
        stem = os.path.splitext(os.path.basename(args.input_json))[0]
        output_path = os.path.join("eval", "figures", f"{stem}_{args.metric}_vs_{args.x_axis}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output_path": output_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
