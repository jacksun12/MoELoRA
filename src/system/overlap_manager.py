import numpy as np


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), eps, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), eps, None)
    return a @ b.T


class ClusterOverlapManager:
    """
    Compare previous and new cluster layouts and choose update strategy.
    """

    def __init__(self, high_overlap_threshold: float = 0.75, mid_overlap_threshold: float = 0.5):
        self.high_overlap_threshold = high_overlap_threshold
        self.mid_overlap_threshold = mid_overlap_threshold

    def greedy_match(self, old_centroids: np.ndarray, new_centroids: np.ndarray):
        if old_centroids is None or len(old_centroids) == 0:
            return {}, [], 0.0

        sim = cosine_sim_matrix(new_centroids, old_centroids)
        pairs = []
        used_old = set()

        # Sort all candidate edges by similarity descending.
        edges = []
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                edges.append((float(sim[i, j]), i, j))
        edges.sort(reverse=True)

        mapping = {}
        for s, i, j in edges:
            if i in mapping or j in used_old:
                continue
            mapping[i] = j
            used_old.add(j)
            pairs.append((i, j, s))

        avg_overlap = float(np.mean([p[2] for p in pairs])) if pairs else 0.0
        return mapping, pairs, avg_overlap

    def choose_strategy(self, avg_overlap: float, optimize_for: str = "time"):
        if avg_overlap >= self.high_overlap_threshold:
            return "fast_rebuild"

        if avg_overlap >= self.mid_overlap_threshold:
            return "hybrid" if optimize_for == "performance" else "fast_rebuild"

        return "full_retrain" if optimize_for == "performance" else "hybrid"
