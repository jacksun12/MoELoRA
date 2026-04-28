import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _sample_points(features, labels, max_points=3000, random_state=42):
    n = len(features)
    if n <= max_points:
        return features, labels

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_points, replace=False)
    return features[idx], labels[idx]


def _plot_2d(points_2d, labels, save_path, title):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for lb in unique_labels:
        m = labels == lb
        plt.scatter(points_2d[m, 0], points_2d[m, 1], s=10, alpha=0.7, label=f"cluster {lb}")
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def visualize_clusters(
    features,
    labels,
    out_dir="eval/figures",
    prefix="stage1",
    method="both",
    max_points=3000,
    random_state=42,
):
    os.makedirs(out_dir, exist_ok=True)
    x, y = _sample_points(features, labels, max_points=max_points, random_state=random_state)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []

    if method in ("pca", "both"):
        pca = PCA(n_components=2, random_state=random_state)
        pca_2d = pca.fit_transform(x)
        pca_path = os.path.join(out_dir, f"{prefix}_pca_{ts}.png")
        _plot_2d(pca_2d, y, pca_path, title=f"{prefix.upper()} Clusters (PCA)")
        saved.append(pca_path)

    if method in ("tsne", "both"):
        tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        tsne_2d = tsne.fit_transform(x)
        tsne_path = os.path.join(out_dir, f"{prefix}_tsne_{ts}.png")
        _plot_2d(tsne_2d, y, tsne_path, title=f"{prefix.upper()} Clusters (t-SNE)")
        saved.append(tsne_path)

    return saved
