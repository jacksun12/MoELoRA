import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRouter(nn.Module):
    def __init__(self, hidden_size, initial_experts_count, top_k=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = initial_experts_count
        self.top_k = top_k
        self.classifier = nn.Linear(hidden_size, self.num_experts, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.02)
        self.use_centroid_routing = False
        self.register_buffer("centroids", torch.empty(0, hidden_size), persistent=False)

    def forward(self, sample_representations):
        """
        Request-level routing.

        Args:
            sample_representations:
                Tensor of shape [batch, hidden] (or [num_items, hidden] for 2D inputs).

        Returns:
            top_k_weights: [batch, top_k]
            top_k_indices: [batch, top_k]
        """
        if self.use_centroid_routing:
            if self.centroids.numel() == 0:
                raise RuntimeError("Centroid routing enabled but centroids are not set.")
            x = F.normalize(sample_representations, dim=-1)
            c = F.normalize(self.centroids, dim=-1)
            router_logits = x @ c.t()
        else:
            router_logits = self.classifier(sample_representations)
        routing_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return top_k_weights, top_k_indices

    @torch.no_grad()
    def set_centroids(self, centroids):
        centroids = torch.as_tensor(centroids, dtype=self.classifier.weight.dtype, device=self.classifier.weight.device)
        if centroids.dim() != 2 or centroids.shape[1] != self.hidden_size:
            raise ValueError(
                f"Centroid shape mismatch: expected [k, {self.hidden_size}], got {tuple(centroids.shape)}"
            )
        self.centroids = centroids
        self.num_experts = centroids.shape[0]
        self.use_centroid_routing = True

    @torch.no_grad()
    def disable_centroid_routing(self):
        self.use_centroid_routing = False

    @torch.no_grad()
    def expand(self, num_new_experts=1):
        old_weight = self.classifier.weight.data
        old_dtype = self.classifier.weight.dtype
        old_device = self.classifier.weight.device
        self.num_experts += num_new_experts

        new_classifier = nn.Linear(self.hidden_size, self.num_experts, bias=False).to(
            device=old_device,
            dtype=old_dtype,
        )
        nn.init.normal_(new_classifier.weight, std=0.02)
        new_classifier.weight.data[: old_weight.size(0), :] = old_weight
        self.classifier = new_classifier

    @torch.no_grad()
    def rebuild(self, new_num_experts, inherit_map=None):
        """
        Rebuild router for a new cluster layout.
        inherit_map: dict[new_idx] = old_idx
        """
        old_weight = self.classifier.weight.data.clone()
        old_dtype = self.classifier.weight.dtype
        old_device = self.classifier.weight.device
        old_n = old_weight.size(0)

        new_classifier = nn.Linear(self.hidden_size, new_num_experts, bias=False).to(
            device=old_device,
            dtype=old_dtype,
        )
        nn.init.normal_(new_classifier.weight, std=0.02)

        if inherit_map is not None:
            for new_idx, old_idx in inherit_map.items():
                if 0 <= old_idx < old_n and 0 <= new_idx < new_num_experts:
                    new_classifier.weight.data[new_idx] = old_weight[old_idx]

        self.classifier = new_classifier
        self.num_experts = new_num_experts
        if self.centroids.numel() > 0:
            new_centroids = torch.zeros(
                new_num_experts,
                self.hidden_size,
                device=old_device,
                dtype=old_dtype,
            )
            if inherit_map is not None:
                for new_idx, old_idx in inherit_map.items():
                    if 0 <= old_idx < self.centroids.shape[0] and 0 <= new_idx < new_num_experts:
                        new_centroids[new_idx] = self.centroids[old_idx]
            self.centroids = new_centroids
