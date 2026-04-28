import numpy as np


class BudgetedRankAllocator:
    """Allocate integer ranks under a fixed budget using marginal utility curves."""

    def __init__(
        self,
        total_rank_budget: int,
        min_rank: int = 2,
        max_rank: int = 16,
        temperature: float = 1.5,
        min_extra_share: float = 0.10,
    ):
        self.total_rank_budget = total_rank_budget
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.temperature = max(float(temperature), 1e-6)
        self.min_extra_share = float(np.clip(min_extra_share, 0.0, 1.0))

    def allocate(self, marginal_gains, cluster_scores=None):
        gains = [np.asarray(x, dtype=np.float32) for x in marginal_gains]
        n = len(gains)
        if n == 0:
            return []

        if cluster_scores is None:
            cluster_scores = np.ones(n, dtype=np.float32)
        else:
            cluster_scores = np.asarray(cluster_scores, dtype=np.float32)
            if len(cluster_scores) != n:
                raise ValueError("cluster_scores must align with marginal_gains")

        max_total_capacity = n * self.max_rank
        if max_total_capacity < self.total_rank_budget:
            raise ValueError(
                f"Rank budget {self.total_rank_budget} exceeds total capacity {max_total_capacity} "
                f"under max_rank={self.max_rank} and n_clusters={n}."
            )

        base_budget = n * self.min_rank
        if base_budget > self.total_rank_budget:
            per = max(1, self.total_rank_budget // n)
            return [per] * n

        ranks = np.full(n, self.min_rank, dtype=int)
        remaining = int(self.total_rank_budget - base_budget)
        tie_break = np.linspace(0.0, 1e-6, num=n, dtype=np.float32)

        while remaining > 0:
            best_idx = -1
            best_gain = -np.inf
            for i in range(n):
                if ranks[i] >= self.max_rank:
                    continue
                extra_rank_idx = ranks[i] - self.min_rank
                if extra_rank_idx < gains[i].shape[0]:
                    gain = float(gains[i][extra_rank_idx])
                else:
                    gain = 0.0
                gain += float(cluster_scores[i]) * 1e-8 + float(tie_break[i])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx < 0:
                break
            ranks[best_idx] += 1
            remaining -= 1

        if int(ranks.sum()) != self.total_rank_budget:
            raise RuntimeError(
                f"Allocator failed to exhaust budget exactly: used={int(ranks.sum())}, "
                f"budget={self.total_rank_budget}"
            )
        return ranks.tolist()


def normalized_complexity(ppl_values, svd_values, ppl_weight: float = 0.6, svd_weight: float = 0.4):
    """
    Merge PPL and SVD complexity into one normalized score per cluster.
    Larger score => more difficult/private pattern => higher rank demand.
    """

    ppl = np.asarray(ppl_values, dtype=np.float32)
    svd = np.asarray(svd_values, dtype=np.float32)

    def norm(x):
        x = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(x)
        if not finite.any():
            return np.ones_like(x, dtype=np.float32)
        fill = float(np.nanmean(x[finite]))
        x = np.where(finite, x, fill).astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if abs(mx - mn) < 1e-8:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)

    ppl_n = norm(ppl)
    svd_n = norm(svd)
    score = ppl_weight * ppl_n + svd_weight * svd_n
    score = np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0)
    return score.tolist()
