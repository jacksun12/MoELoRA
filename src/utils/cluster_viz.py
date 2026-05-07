import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .plot_style import MOPRED_COLORS, MOPRED_PATTERNS, apply_global_plot_style, style_axes, style_legend

try:
    from umap import UMAP
except ImportError:  # pragma: no cover - optional dependency
    UMAP = None

apply_global_plot_style()


def _sample_points(features, labels, max_points=3000, random_state=42):
    n = len(features)
    if n <= max_points:
        return features, labels

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_points, replace=False)
    return features[idx], labels[idx]


def _sample_reference_and_current(
    reference_features,
    reference_labels,
    current_features,
    current_labels,
    max_points=3000,
    random_state=42,
):
    ref_n = len(reference_features)
    cur_n = len(current_features)
    total = ref_n + cur_n
    if total <= max_points:
        return reference_features, reference_labels, current_features, current_labels

    rng = np.random.default_rng(random_state)
    ref_budget = max(1, int(max_points * (ref_n / total)))
    cur_budget = max(1, max_points - ref_budget)
    ref_idx = rng.choice(ref_n, size=min(ref_n, ref_budget), replace=False)
    cur_idx = rng.choice(cur_n, size=min(cur_n, cur_budget), replace=False)
    return (
        reference_features[ref_idx],
        reference_labels[ref_idx],
        current_features[cur_idx],
        current_labels[cur_idx],
    )


def _palette(n):
    base = MOPRED_COLORS + ["#C2255C", "#2B8A3E", "#5F3DC4", "#1C7ED6", "#A61E4D"]
    if n <= len(base):
        return base[:n]
    cmap = plt.cm.get_cmap("tab20", n)
    return [cmap(i) for i in range(n)]


def _lighten_color(color, factor=0.65):
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    white = np.ones(3)
    return tuple(rgb + (white - rgb) * factor)


def _add_cov_ellipse(ax, points, color, n_std=1.8, alpha=0.10, linewidth=1.6, linestyle="--"):
    if len(points) < 3:
        return
    cov = np.cov(points[:, 0], points[:, 1])
    if cov.shape != (2, 2) or np.linalg.det(cov) <= 0:
        return
    mean = points.mean(axis=0)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if np.any(vals <= 0):
        return
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=_lighten_color(color, factor=0.82),
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(ell)


def _plot_2d(points_2d, labels, save_path, title, axis_labels=("dim-1", "dim-2"), subtitle=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 7.5), facecolor="#f7f5f2")
    ax.set_facecolor("#fcfbf8")
    unique_labels = np.unique(labels)
    colors = _palette(len(unique_labels))
    for color, lb in zip(colors, unique_labels):
        m = labels == lb
        cluster_points = points_2d[m]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=26,
            alpha=0.72,
            color=color,
            edgecolors="white",
            linewidths=0.35,
            label=f"Cluster {lb}",
        )
        centroid = cluster_points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=180,
            color=color,
            marker="X",
            edgecolors="#1f1f1f",
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            centroid[0],
            centroid[1],
            f" {lb}",
            fontsize=10,
            fontweight="bold",
            color="#1f1f1f",
            va="center",
            ha="left",
        )

    if title:
        ax.set_title(title, fontsize=24, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=16, color="#555555", ha="left")
    ax.set_xlabel(axis_labels[0], fontsize=18, fontweight="bold")
    ax.set_ylabel(axis_labels[1], fontsize=18, fontweight="bold")
    style_legend(ax, loc="upper right", ncol=1, fontsize=14)
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_2d(
    reference_points,
    reference_labels,
    current_points,
    current_labels,
    save_path,
    title,
    axis_labels=("dim-1", "dim-2"),
    subtitle=None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 8.4), facecolor="#f7f5f2")
    ax.set_facecolor("#fcfbf8")

    all_labels = np.unique(np.concatenate([reference_labels, current_labels]))
    color_map = {lb: color for lb, color in zip(all_labels, _palette(len(all_labels)))}

    for lb in np.unique(reference_labels):
        m = reference_labels == lb
        pts = reference_points[m]
        _add_cov_ellipse(ax, pts, color=color_map[lb])
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=14,
            alpha=0.18,
            color=_lighten_color(color_map[lb], factor=0.45),
            edgecolors="none",
            marker="o",
            label=f"Stage-1 Cluster {lb}",
        )
        centroid = pts.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=130,
            facecolors="none",
            edgecolors=color_map[lb],
            linewidths=1.4,
            marker="D",
            zorder=4,
        )

    for lb in np.unique(current_labels):
        m = current_labels == lb
        pts = current_points[m]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=38,
            alpha=0.84,
            color=color_map[lb],
            edgecolors="white",
            linewidths=0.42,
            marker="o",
            label=f"Stage-2 Cluster {lb}",
        )
        centroid = pts.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=190,
            color=color_map[lb],
            marker="X",
            edgecolors="#1f1f1f",
            linewidths=1.0,
            zorder=6,
        )
        ax.text(
            centroid[0],
            centroid[1],
            f" {lb}",
            fontsize=10.5,
            fontweight="bold",
            color="#1f1f1f",
            va="center",
            ha="left",
        )

    if title:
        ax.set_title(title, fontsize=24, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=16, color="#555555", ha="left")
    ax.set_xlabel(axis_labels[0], fontsize=18, fontweight="bold")
    ax.set_ylabel(axis_labels[1], fontsize=18, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        if l not in dedup:
            dedup[l] = h
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="upper right",
        fontsize=14,
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_linewidth(2.0)
        for text in legend.get_texts():
            text.set_fontweight("bold")
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _normalize_method_tokens(method):
    if method is None:
        return {"pca", "tsne"}

    method = str(method).lower().strip()
    aliases = {
        "both": {"pca", "tsne"},
        "strict": {"pca", "umap"},
        "strict_both": {"pca", "umap"},
        "all": {"pca", "tsne", "umap"},
        "overlay": {"pca", "umap"},
    }
    if method in aliases:
        return aliases[method]
    return {token.strip() for token in method.split(",") if token.strip()}


def visualize_clusters(
    features,
    labels,
    out_dir="eval/figures",
    prefix="stage1",
    method="both",
    max_points=3000,
    random_state=42,
    reference_features=None,
    reference_labels=None,
):
    os.makedirs(out_dir, exist_ok=True)
    overlay_mode = reference_features is not None and reference_labels is not None
    if overlay_mode:
        ref_x, ref_y, x, y = _sample_reference_and_current(
            np.asarray(reference_features),
            np.asarray(reference_labels),
            np.asarray(features),
            np.asarray(labels),
            max_points=max_points,
            random_state=random_state,
        )
    else:
        x, y = _sample_points(features, labels, max_points=max_points, random_state=random_state)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    methods = _normalize_method_tokens(method)

    if "pca" in methods:
        pca = PCA(n_components=2, random_state=random_state)
        if overlay_mode:
            pca.fit(ref_x)
            ref_pca_2d = pca.transform(ref_x)
            pca_2d = pca.transform(x)
        else:
            pca_2d = pca.fit_transform(x)
        pca_path = os.path.join(out_dir, f"{prefix}_pca_{ts}.png")
        variance = pca.explained_variance_ratio_
        if overlay_mode:
            _plot_overlay_2d(
                ref_pca_2d,
                ref_y,
                pca_2d,
                y,
                pca_path,
                title=f"{prefix.upper()} Drift Overlay via PCA",
                axis_labels=(f"PC-1 ({variance[0] * 100:.1f}%)", f"PC-2 ({variance[1] * 100:.1f}%)"),
                subtitle=f"Stage-1 anchors={len(ref_x)} | Stage-2 points={len(x)} | k={len(np.unique(y))}",
            )
        else:
            _plot_2d(
                pca_2d,
                y,
                pca_path,
                title=f"{prefix.upper()} Clusters via PCA",
                axis_labels=(f"PC-1 ({variance[0] * 100:.1f}%)", f"PC-2 ({variance[1] * 100:.1f}%)"),
                subtitle=f"n={len(x)} points | k={len(np.unique(y))} clusters",
            )
        saved.append(pca_path)

    if "umap" in methods:
        if UMAP is None:
            raise ImportError(
                "visualization_method requests UMAP, but umap-learn is not installed. "
                "Install `umap-learn` or switch to `pca`/`both`."
            )
        umap = UMAP(n_components=2, random_state=random_state, init="spectral")
        if overlay_mode:
            umap.fit(ref_x)
            ref_umap_2d = umap.transform(ref_x)
            umap_2d = umap.transform(x)
        else:
            umap_2d = umap.fit_transform(x)
        umap_path = os.path.join(out_dir, f"{prefix}_umap_{ts}.png")
        if overlay_mode:
            _plot_overlay_2d(
                ref_umap_2d,
                ref_y,
                umap_2d,
                y,
                umap_path,
                title="",
                axis_labels=("", ""),
                subtitle=None,
            )
        else:
            _plot_2d(
                umap_2d,
                y,
                umap_path,
                title="",
                axis_labels=("", ""),
                subtitle=None,
            )
        saved.append(umap_path)

    if "tsne" in methods:
        tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        if overlay_mode:
            combo_x = np.concatenate([ref_x, x], axis=0)
            combo_2d = tsne.fit_transform(combo_x)
            ref_tsne_2d = combo_2d[: len(ref_x)]
            tsne_2d = combo_2d[len(ref_x) :]
        else:
            tsne_2d = tsne.fit_transform(x)
        tsne_path = os.path.join(out_dir, f"{prefix}_tsne_{ts}.png")
        if overlay_mode:
            _plot_overlay_2d(
                ref_tsne_2d,
                ref_y,
                tsne_2d,
                y,
                tsne_path,
                title=f"{prefix.upper()} Joint Overlay via t-SNE (Non-Strict)",
                axis_labels=("t-SNE-1", "t-SNE-2"),
                subtitle=(
                    f"Stage-1 anchors={len(ref_x)} | Stage-2 points={len(x)} | "
                    f"k={len(np.unique(y))} | joint refit"
                ),
            )
        else:
            _plot_2d(
                tsne_2d,
                y,
                tsne_path,
                title=f"{prefix.upper()} Clusters via t-SNE",
                axis_labels=("t-SNE-1", "t-SNE-2"),
                subtitle=f"n={len(x)} points | k={len(np.unique(y))} clusters",
            )
        saved.append(tsne_path)

    return saved


def plot_cluster_style_mix(
    style_mix_summary,
    save_path,
    title,
    subtitle=None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.4, 6.8), facecolor="#f7f5f2")
    ax.set_facecolor("#fcfbf8")

    cluster_ids = sorted(style_mix_summary.keys())
    style_names = sorted(
        {
            part["style"]
            for cluster in style_mix_summary.values()
            for part in cluster.get("styles", [])
        }
    )
    style_colors = {style: color for style, color in zip(style_names, _palette(max(len(style_names), 1)))}

    bottoms = np.zeros(len(cluster_ids), dtype=float)
    x = np.arange(len(cluster_ids))

    for style in style_names:
        vals = []
        for cid in cluster_ids:
            parts = {p["style"]: p for p in style_mix_summary[cid].get("styles", [])}
            item = parts.get(style, {"ratio": 0.0, "count": 0})
            vals.append(float(item["ratio"]))
        vals = np.asarray(vals, dtype=float)
        bars = ax.bar(
            x,
            vals,
            bottom=bottoms,
            color=style_colors[style],
            edgecolor="#1a1a1a",
            linewidth=0.6,
            width=0.72,
            label=style,
        )
        hatch = MOPRED_PATTERNS[style_names.index(style) % len(MOPRED_PATTERNS)]
        for bar in bars:
            bar.set_hatch(hatch)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {cid}" for cid in cluster_ids], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)], fontsize=10)
    ax.set_ylabel("Style Composition Ratio", fontsize=11.5)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    style_legend(
        ax,
        title="Style Label",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=12,
        title_fontsize=13,
    )
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
