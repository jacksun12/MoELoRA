import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.system.serving_sim import build_workload_profile, simulate_npu_hybrid_strategy, simulate_strategy
from src.utils.plot_style import MOPRED_COLORS, apply_global_plot_style, style_axes, style_legend


apply_global_plot_style()

DISPLAY_NAMES = {
    "all_cpu": "All CPU",
    "fifo_gpu": "FIFO GPU",
    "signature_gpu_cpu": "Signature GPU+CPU",
    "signature_length_gpu_cpu": "Sig+Length GPU+CPU",
    "npu_hybrid": "NPU Hybrid",
}

MARKERS = {
    "all_cpu": "o",
    "fifo_gpu": "s",
    "signature_gpu_cpu": "^",
    "signature_length_gpu_cpu": "D",
    "npu_hybrid": "X",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark synthetic Stage-3 serving strategies.")
    parser.add_argument("--workload", default="complex", choices=["simple", "complex", "fragmented", "bursty_hotspot"])
    parser.add_argument(
        "--strategies",
        default="all_cpu,fifo_gpu,signature_gpu_cpu,signature_length_gpu_cpu,npu_hybrid",
        help="Comma-separated strategies to benchmark.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def generate_serving_tradeoff_plot(output_json: str, output_path: str = ""):
    with open(output_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["results"]

    fig, ax = plt.subplots(figsize=(9.4, 6.8), facecolor="#f7f5f2")
    ax.set_facecolor("#fcfbf8")

    for idx, row in enumerate(rows):
        name = row["strategy"]
        ax.scatter(
            row["throughput_rps"],
            row["p95_latency_ms"],
            s=240,
            color=MOPRED_COLORS[idx % len(MOPRED_COLORS)],
            edgecolors="#1a1a1a",
            linewidths=1.2,
            marker=MARKERS.get(name, "o"),
            label=DISPLAY_NAMES.get(name, name),
            zorder=3,
        )

    ax.set_xlabel("Throughput (requests/s)", fontsize=18, fontweight="bold")
    ax.set_ylabel("P95 Latency (ms)", fontsize=18, fontweight="bold")
    style_axes(ax, grid_axis="both")
    style_legend(ax, loc="upper right", fontsize=13)

    if not output_path:
        stem = os.path.splitext(os.path.basename(output_json))[0]
        output_path = os.path.join("eval", "figures", f"{stem}_throughput_vs_p95.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_device_util_plot(output_json: str, output_path: str = ""):
    with open(output_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["results"]

    strategies = [DISPLAY_NAMES.get(row["strategy"], row["strategy"]) for row in rows]
    gpu_vals = [row["gpu_utilization"] for row in rows]
    cpu_vals = [row["cpu_utilization"] for row in rows]
    npu_vals = [row["npu_utilization"] for row in rows]

    fig, ax = plt.subplots(figsize=(10.5, 6.8), facecolor="#f7f5f2")
    ax.set_facecolor("#fcfbf8")
    x = list(range(len(strategies)))
    width = 0.24

    ax.bar([i - width for i in x], gpu_vals, width=width, color=MOPRED_COLORS[1], edgecolor="black", linewidth=1.2, hatch="/", label="GPU")
    ax.bar(x, cpu_vals, width=width, color=MOPRED_COLORS[0], edgecolor="black", linewidth=1.2, hatch="\\", label="CPU")
    ax.bar([i + width for i in x], npu_vals, width=width, color=MOPRED_COLORS[2], edgecolor="black", linewidth=1.2, hatch="x", label="NPU")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=0, ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Device Utilization", fontsize=18, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    style_axes(ax, grid_axis="y")
    style_legend(ax, loc="upper right", fontsize=13)

    if not output_path:
        stem = os.path.splitext(os.path.basename(output_json))[0]
        output_path = os.path.join("eval", "figures", f"{stem}_device_utilization.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    workload = build_workload_profile(args.workload, seed=args.seed)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]

    results = []
    for strategy in strategies:
        if strategy == "npu_hybrid":
            metrics = simulate_npu_hybrid_strategy(workload)
        else:
            metrics = simulate_strategy(workload, strategy=strategy)
        row = vars(metrics)
        row["workload"] = args.workload
        results.append(row)

    if not args.output_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_json = os.path.join("eval", "results", f"stage3_serving_{args.workload}_{ts}.json")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "workload": args.workload,
                "seed": args.seed,
                "strategies": strategies,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    latency_plot = generate_serving_tradeoff_plot(args.output_json)
    util_plot = generate_device_util_plot(args.output_json)
    print(json.dumps({"output_json": args.output_json, "latency_plot": latency_plot, "util_plot": util_plot, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
