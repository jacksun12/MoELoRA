import argparse
import ast
import os
import re
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.plot_style import MOPRED_COLORS, MOPRED_PATTERNS, apply_global_plot_style, style_axes


apply_global_plot_style()


STAGE2_TAGS = [
    "stage2_raw_base",
    "stage2_single_lora_r32",
    "stage2_raie",
    "Stage-2",
]


def _display_label(tag):
    return {
        "stage2_raw_base": "Base Model",
        "stage2_single_lora_r32": "LoRA r32",
        "stage2_raie": "RAIE",
        "Stage-2": "MoE-LoRA Stage-2",
    }.get(tag, tag)


def parse_stage2_metrics(log_path):
    baseline_pattern = re.compile(r"\[Baseline\]\[(?P<tag>[^\]]+)\]\s+(?P<payload>\{.*\})")
    stage_pattern = re.compile(r"\[(?P<stage>Stage-\d+)\]\s+(?P<payload>.*)")
    collected = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = baseline_pattern.search(line)
            if m:
                tag = m.group("tag")
                if tag not in {"stage2_raw_base", "stage2_single_lora_r32", "stage2_raie"}:
                    continue
                payload = ast.literal_eval(m.group("payload"))
                if "text_generation_metrics" in payload:
                    collected[tag] = payload["text_generation_metrics"]
                else:
                    collected[tag] = payload
                continue

            m = stage_pattern.search(line)
            if not m:
                continue
            if m.group("stage") != "Stage-2":
                continue
            payload_str = m.group("payload")
            if "text_generation_metrics=" in payload_str:
                raw = payload_str.split("text_generation_metrics=", 1)[1]
                collected["Stage-2"] = ast.literal_eval(raw)
            elif payload_str.startswith("{") and "text_generation_metrics" in payload_str:
                payload = ast.literal_eval(payload_str)
                collected["Stage-2"] = payload["text_generation_metrics"]

    return {tag: collected[tag] for tag in STAGE2_TAGS if tag in collected}


def plot_stage2_compare(metrics_by_tag, output_path, title=None):
    labels = list(metrics_by_tag.keys())
    if not labels:
        raise ValueError("No Stage-2 comparison metrics found to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="#f7f5f2")
    metric_keys = ["bleu1", "rougeL_f1"]

    for ax, metric_key in zip(axes, metric_keys):
        display_labels = [_display_label(tag) for tag in labels]
        values = [metrics_by_tag[tag].get(metric_key, 0.0) for tag in labels]
        bars = ax.bar(
            display_labels,
            values,
            width=0.92,
            color=MOPRED_COLORS[: len(labels)],
            edgecolor="#1a1a1a",
            linewidth=0.6,
        )
        for idx, bar in enumerate(bars):
            bar.set_hatch(MOPRED_PATTERNS[idx % len(MOPRED_PATTERNS)])
        ax.set_ylim(0.0, max(1.0, max(values) * 1.15))
        ax.set_title(metric_key, fontsize=18, fontweight="bold")
        ax.tick_params(axis="x", rotation=0)
        style_axes(ax, grid_axis="y")

    clean_title = os.path.splitext(title)[0] if title else "Stage-2 Baseline Comparison"
    fig.suptitle(clean_title, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_stage2_baseline_plot(log_path, output_path=""):
    if not output_path:
        stem = os.path.splitext(os.path.basename(log_path))[0]
        output_path = os.path.join("eval/figures", f"{stem}_stage2_compare.png")
    metrics_by_tag = parse_stage2_metrics(log_path)
    plot_stage2_compare(metrics_by_tag, output_path, title=os.path.basename(log_path))
    return {
        "log_path": log_path,
        "output_path": output_path,
        "labels": list(metrics_by_tag.keys()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()
    result = generate_stage2_baseline_plot(args.log_path, args.output_path)
    print(result)


if __name__ == "__main__":
    main()
