import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .plot_style import MOPRED_COLORS, apply_global_plot_style, style_axes, style_legend


apply_global_plot_style()


@dataclass
class CurvePoint:
    stage: str
    track: str
    step: int
    loss: float


class TrainingCurveTracker:
    def __init__(self, out_dir: str = "eval/figures", run_name: str = "train"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.run_name = run_name
        self.points: List[CurvePoint] = []
        self.jsonl_path = os.path.join(self.out_dir, f"{run_name}_training_curves.jsonl")
        self.png_path = os.path.join(self.out_dir, f"{run_name}_training_curves.png")

    def log(self, stage: str, track: str, step: int, loss: float):
        loss = float(loss)
        if not np.isfinite(loss):
            return
        point = CurvePoint(stage=stage, track=track, step=int(step), loss=loss)
        self.points.append(point)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(point.__dict__) + "\n")

        self.render()

    def render(self):
        if not self.points:
            return

        grouped: Dict[str, List[CurvePoint]] = {}
        for p in self.points:
            key = f"{p.stage}:{p.track}"
            grouped.setdefault(key, []).append(p)

        plt.figure(figsize=(10, 6))
        ax1 = plt.subplot(1, 1, 1)

        for idx, (key, seq) in enumerate(grouped.items()):
            seq = sorted(seq, key=lambda x: x.step)
            xs = [p.step for p in seq if np.isfinite(p.loss)]
            ys_loss = [p.loss for p in seq if np.isfinite(p.loss)]
            if len(xs) == 0:
                continue
            ax1.plot(
                xs,
                ys_loss,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=MOPRED_COLORS[idx % len(MOPRED_COLORS)],
                label=key,
            )

        ax1.set_title("Training Loss Curves", fontsize=20, fontweight="bold", pad=16)
        ax1.set_xlabel("Global Training Step", fontsize=18, fontweight="bold")
        ax1.set_ylabel("Loss", fontsize=18, fontweight="bold")
        style_axes(ax1, grid_axis="both")
        style_legend(ax1, loc="upper right", fontsize=11)

        plt.tight_layout()
        plt.savefig(self.png_path, dpi=180)
        plt.close()
