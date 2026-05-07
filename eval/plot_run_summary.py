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


TEXT_METRIC_KEYS = [
    "bleu1",
    "rougeL_f1",
]


def parse_log(path):
    baseline_metrics = {}
    stage_metrics = {}

    baseline_pattern = re.compile(r"\[Baseline\]\[(?P<tag>[^\]]+)\]\s+(?P<payload>\{.*\})")
    stage_pattern = re.compile(r"\[(?P<stage>Stage-\d+)\]\s+(?P<payload>.*)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = baseline_pattern.search(line)
            if m:
                payload = ast.literal_eval(m.group("payload"))
                if isinstance(payload, dict):
                    if "text_generation_metrics" in payload:
                        baseline_metrics[m.group("tag")] = payload["text_generation_metrics"]
                    else:
                        baseline_metrics[m.group("tag")] = payload
                continue

            m = stage_pattern.search(line)
            if not m:
                continue

            stage = m.group("stage")
            payload_str = m.group("payload")

            if "text_generation_metrics=" in payload_str:
                raw = payload_str.split("text_generation_metrics=", 1)[1]
                stage_metrics[stage] = ast.literal_eval(raw)
                continue

            if payload_str.startswith("{") and "text_generation_metrics" in payload_str:
                payload = ast.literal_eval(payload_str)
                stage_metrics[stage] = payload["text_generation_metrics"]

    return baseline_metrics, stage_metrics


def build_series(baseline_metrics, stage_metrics):
    labels = []
    metrics_by_label = {}

    for tag in ["raw_base", "single_lora_r32", "mocle_4x8", "hydralora_4x8"]:
        if tag in baseline_metrics:
            labels.append(tag)
            metrics_by_label[tag] = baseline_metrics[tag]

    for tag in ["Stage-1", "Stage-2"]:
        if tag in stage_metrics:
            labels.append(tag)
            metrics_by_label[tag] = stage_metrics[tag]

    return labels, metrics_by_label


def _display_label(tag):
    return {
        "raw_base": "Base Model",
        "single_lora_r32": "LoRA r32",
        "mocle_4x8": "MoCLE 4x8",
        "hydralora_4x8": "HydraLoRA 4x8",
        "Stage-1": "MoE-LoRA Stage-1",
        "Stage-2": "MoE-LoRA Stage-2",
    }.get(tag, tag)


def plot_text_metrics(labels, metrics_by_label, out_path, title=None):
    available_keys = [k for k in TEXT_METRIC_KEYS if any(k in metrics_by_label[label] for label in labels)]
    if not labels or not available_keys:
        raise ValueError("No text generation metrics found to plot.")

    fig, axes = plt.subplots(1, len(available_keys), figsize=(6 * len(available_keys), 5.5))
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.flatten()
        except Exception:
            axes = [axes]
    colors = MOPRED_COLORS

    for ax, metric_key in zip(axes, available_keys):
        display_labels = [_display_label(label) for label in labels]
        values = [metrics_by_label[label].get(metric_key, 0.0) for label in labels]
        bars = ax.bar(display_labels, values, width=0.92, color=colors[: len(labels)], edgecolor="#1a1a1a", linewidth=0.6)
        for idx, bar in enumerate(bars):
            bar.set_hatch(MOPRED_PATTERNS[idx % len(MOPRED_PATTERNS)])
        ax.set_title(metric_key, fontsize=18, fontweight="bold")
        ax.set_ylim(0.0, max(1.0, max(values) * 1.15))
        ax.tick_params(axis="x", rotation=0)
        style_axes(ax, grid_axis="y")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    clean_title = os.path.splitext(title)[0] if title else "Run Summary Metrics"
    fig.suptitle(clean_title, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_latest_log(log_dir):
    candidates = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.endswith(".log")
    ]
    if not candidates:
        raise FileNotFoundError(f"No log files found under {log_dir}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="eval/logs")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    log_path = args.log_path or resolve_latest_log(args.log_dir)
    output_path = args.output_path
    if not output_path:
        stem = os.path.splitext(os.path.basename(log_path))[0]
        output_path = os.path.join("eval/figures", f"{stem}_summary.png")

    baseline_metrics, stage_metrics = parse_log(log_path)
    labels, metrics_by_label = build_series(baseline_metrics, stage_metrics)
    plot_text_metrics(labels, metrics_by_label, output_path, title=os.path.basename(log_path))

    print(f"log_path={log_path}")
    print(f"output_path={output_path}")
    print(f"labels={labels}")


if __name__ == "__main__":
    main()
