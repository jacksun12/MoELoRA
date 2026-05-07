import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


class SphericalKMeans:
    """轻量级球面 k-means（基于余弦相似度）实现。 / A lightweight spherical k-means (cosine-similarity) implementation."""

    def __init__(self, n_clusters: int, max_iter: int = 50, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None

    def _init_centroids(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(x.shape[0], size=self.n_clusters, replace=False)
        return x[idx].copy()

    def fit(self, features: np.ndarray):
        x = l2_normalize(features.astype(np.float32))
        centroids = l2_normalize(self._init_centroids(x))

        for _ in range(self.max_iter):
            sims = x @ centroids.T
            labels = np.argmax(sims, axis=1)

            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                members = x[labels == k]
                if len(members) == 0:
                    # 用随机样本重新初始化空簇。
                    # Reinitialize an empty cluster with a random point.
                    rand_idx = np.random.randint(0, x.shape[0])
                    new_centroids[k] = x[rand_idx]
                else:
                    new_centroids[k] = members.mean(axis=0)

            new_centroids = l2_normalize(new_centroids)
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = np.argmax(x @ centroids.T, axis=1)
        return self

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.labels_


class DBCHKSelector:
    """
    基于球面聚类标签上的 DB/CH 指标选择最优 k。
    Select the best k using DB/CH metrics computed on spherical-clustering labels.

    DB 越小越好，CH 越大越好。
    Lower DB and higher CH are preferred.
    """

    def __init__(self, k_min: int = 2, k_max: int = 8, random_state: int = 42):
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

    def _normalize(self, values: np.ndarray, reverse: bool = False) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        min_v, max_v = values.min(), values.max()
        if abs(max_v - min_v) < 1e-8:
            scores = np.ones_like(values)
        else:
            scores = (values - min_v) / (max_v - min_v)
        return 1.0 - scores if reverse else scores

    def select(self, features: np.ndarray):
        x = l2_normalize(features.astype(np.float32))
        n = x.shape[0]
        max_k = min(self.k_max, n - 1)
        min_k = min(self.k_min, max_k)
        if max_k < 2:
            return 1, np.zeros(n, dtype=np.int64), x.mean(axis=0, keepdims=True)

        candidates = []
        db_list = []
        ch_list = []

        for k in range(min_k, max_k + 1):
            model = SphericalKMeans(n_clusters=k, random_state=self.random_state)
            labels = model.fit_predict(x)
            unique = np.unique(labels)
            if len(unique) < 2:
                continue

            db = davies_bouldin_score(x, labels)
            ch = calinski_harabasz_score(x, labels)
            candidates.append((k, labels, model.centroids_))
            db_list.append(db)
            ch_list.append(ch)

        if not candidates:
            return 1, np.zeros(n, dtype=np.int64), x.mean(axis=0, keepdims=True)

        db_scores = self._normalize(np.array(db_list), reverse=True)
        ch_scores = self._normalize(np.array(ch_list), reverse=False)
        total_scores = 0.5 * db_scores + 0.5 * ch_scores
        best_idx = int(np.argmax(total_scores))

        best_k, best_labels, best_centroids = candidates[best_idx]
        return best_k, best_labels, best_centroids
