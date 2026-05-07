import numpy as np


class BudgetedRankAllocator:
    """在固定预算下依据边际收益曲线分配整数 rank。 / Allocate integer ranks under a fixed budget using marginal utility curves."""

    def __init__(
        self,
        total_rank_budget: int,
        min_rank: int = 2,
        max_rank: int = 16,
    ):
        self.total_rank_budget = total_rank_budget
        self.min_rank = min_rank
        self.max_rank = max_rank

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
            ranks = np.zeros(n, dtype=int)
            for i in range(self.total_rank_budget):
                ranks[i % n] += 1
            return ranks.tolist()

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
