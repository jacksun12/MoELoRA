import math
import time
import torch
import numpy as np

# ==========================================
# 1. 模型质量评估 / Model Quality Metrics
# ==========================================

def calculate_ppl(loss):
    """计算困惑度，适用于语言建模任务。 / Compute perplexity for language-modeling tasks."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

def hit_rate_at_k(predictions, targets, k=10):
    """
    计算 HR@K（命中率）。
    Compute HR@K (Hit Rate).

    predictions: 模型预测得分或概率分布，形状为 (batch_size, num_classes)。
    predictions: model scores or probability distribution of shape (batch_size, num_classes).
    targets: 真实标签，形状为 (batch_size,)。
    targets: ground-truth labels of shape (batch_size,).
    """
    _, top_k_indices = torch.topk(predictions, k, dim=-1)
    targets = targets.unsqueeze(1)
    hits = (top_k_indices == targets).any(dim=1).float()
    return hits.mean().item()

def ndcg_at_k(predictions, targets, k=10):
    """
    计算 NDCG@K。
    Compute NDCG@K (Normalized Discounted Cumulative Gain).
    """
    _, top_k_indices = torch.topk(predictions, k, dim=-1)
    targets = targets.unsqueeze(1)
    
    # 找到 target 在 top-k 中的位置。
    # Find the target position inside the top-k list.
    hits = (top_k_indices == targets).nonzero(as_tuple=False)
    
    ndcg_sum = 0.0
    for hit in hits:
        # hit[1] 是 target 在 top-k 中的排名索引（0 到 k-1）。
        # hit[1] is the target rank index inside top-k (0 to k-1).
        rank = hit[1].item() + 1
        ndcg_sum += 1.0 / math.log2(rank + 1)
        
    return ndcg_sum / predictions.size(0)

# ==========================================
# 2. 系统效率分析 / System Efficiency Profiler
# ==========================================

class SystemProfiler:
    """
    用于精确测量系统延迟和显存占用的上下文管理器。
    Context manager for precise latency and memory measurements.

    对于系统论文，我们通常关心 P99 延迟和峰值显存。
    For systems papers, we usually care about P99 latency and peak memory.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.start_time = 0
        self.end_time = 0
        self.latencies = []

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        
        latency_ms = (self.end_time - self.start_time) * 1000
        self.latencies.append(latency_ms)

    def get_peak_memory_mb(self):
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def get_latency_stats(self):
        if not self.latencies:
            return {"avg": 0, "p99": 0}
        return {
            "avg_ms": np.mean(self.latencies),
            "p99_ms": np.percentile(self.latencies, 99)
        }
