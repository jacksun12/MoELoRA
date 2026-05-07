import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRouter(nn.Module):
    def __init__(self, hidden_size, initial_experts_count, top_k=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = initial_experts_count
        self.top_k = top_k
        self.register_buffer("centroids", torch.empty(0, hidden_size), persistent=False)

    def forward(self, sample_representations):
        """
        通过样本表示与簇中心的余弦相似度进行请求级路由。
        Route requests by cosine similarity between sample representations and cluster centroids.

        参数：
        Args:
            sample_representations:
                形状为 [batch, hidden] 的张量；若输入为二维，则为 [num_items, hidden]。
                Tensor of shape [batch, hidden] (or [num_items, hidden] for 2D inputs).

        返回：
        Returns:
            top_k_weights: 路由权重，形状为 [batch, top_k] / routing weights of shape [batch, top_k]
            top_k_indices: expert 索引，形状为 [batch, top_k] / expert indices of shape [batch, top_k]
        """
        if self.centroids.numel() == 0:
            raise RuntimeError("Centroid routing requires centroids to be set before forward.")
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k={self.top_k} exceeds num_experts={self.num_experts}")

        x = F.normalize(sample_representations, dim=-1)
        c = F.normalize(self.centroids, dim=-1)
        router_logits = x @ c.t()
        routing_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return top_k_weights, top_k_indices

    @torch.no_grad()
    def set_centroids(self, centroids):
        centroids = torch.as_tensor(centroids, dtype=self.centroids.dtype, device=self.centroids.device)
        if centroids.dim() != 2 or centroids.shape[1] != self.hidden_size:
            raise ValueError(
                f"Centroid shape mismatch: expected [k, {self.hidden_size}], got {tuple(centroids.shape)}"
            )
        self.centroids = centroids
        self.num_experts = centroids.shape[0]

    @torch.no_grad()
    def expand(self, num_new_experts=1):
        self.num_experts += num_new_experts

    @torch.no_grad()
    def rebuild(self, new_num_experts, inherit_map=None):
        """
        为新的簇布局重建路由器。
        Rebuild the router for a new cluster layout.

        inherit_map 表示新旧 expert 的继承关系：dict[new_idx] = old_idx。
        inherit_map describes new-to-old expert inheritance: dict[new_idx] = old_idx.
        """
        self.num_experts = new_num_experts
        if self.centroids.numel() > 0:
            new_centroids = torch.zeros(
                new_num_experts,
                self.hidden_size,
                device=self.centroids.device,
                dtype=self.centroids.dtype,
            )
            if inherit_map is not None:
                for new_idx, old_idx in inherit_map.items():
                    if 0 <= old_idx < self.centroids.shape[0] and 0 <= new_idx < new_num_experts:
                        new_centroids[new_idx] = self.centroids[old_idx]
            self.centroids = new_centroids
