import torch
import torch.nn as nn

from .dynamic_router import DynamicRouter
from .evolving_expert import EvolvingLoRAExpert


class MoELoRALinear(nn.Module):
    def __init__(self, base_layer, initial_ranks=None, top_k=1, alpha=16, dropout=0.05):
        super().__init__()
        if initial_ranks is None:
            initial_ranks = [4, 4]

        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.router = DynamicRouter(in_features, len(initial_ranks), top_k)
        self.experts = nn.ModuleList(
            [EvolvingLoRAExpert(in_features, out_features, rank, alpha=alpha, dropout=dropout) for rank in initial_ranks]
        )

        self.force_expert_idx = None

    def set_force_expert(self, expert_idx):
        self.force_expert_idx = expert_idx

    def clear_force_expert(self):
        self.force_expert_idx = None

    @torch.no_grad()
    def set_router_centroids(self, centroids):
        self.router.set_centroids(centroids)

    def _build_request_representation(self, x):
        """
        Build one routing representation per request/sample.

        For sequence inputs [B, L, H], use mean pooling over the sequence dimension.
        For 2D inputs [N, H], route each row independently.
        """
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        raise ValueError(f"Unsupported input rank for request-level routing: shape={tuple(x.shape)}")

    def _broadcast_route_tensor(self, tensor, target_dim):
        out = tensor
        while out.dim() < target_dim:
            out = out.unsqueeze(1)
        return out

    def forward(self, x):
        base_output = self.base_layer(x)

        if self.force_expert_idx is not None:
            idx = int(self.force_expert_idx)
            return base_output + self.experts[idx](x)

        sample_repr = self._build_request_representation(x)
        routing_weights, selected_experts = self.router(sample_repr)
        moe_output = torch.zeros_like(base_output)

        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(x)
            for k in range(self.router.top_k):
                mask = (selected_experts[:, k] == expert_idx).to(x.dtype)
                weight = routing_weights[:, k]
                mask = self._broadcast_route_tensor(mask.unsqueeze(-1), expert_out.dim())
                weight = self._broadcast_route_tensor(weight.unsqueeze(-1), expert_out.dim())
                moe_output += expert_out * mask * weight

        return base_output + moe_output

    def add_expert(self, new_expert):
        self.experts.append(new_expert)
        self.router.expand(num_new_experts=1)

    @torch.no_grad()
    def rebuild_experts(self, new_ranks, inherit_map=None):
        """
        Rebuild experts by new rank layout.
        inherit_map: dict[new_idx] = old_idx
        """
        old_experts = list(self.experts)
        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features

        new_experts = []
        target_device = self.base_layer.weight.device
        target_dtype = self.base_layer.weight.dtype
        for new_idx, rank in enumerate(new_ranks):
            if inherit_map is not None and new_idx in inherit_map and inherit_map[new_idx] < len(old_experts):
                inherited = old_experts[inherit_map[new_idx]].resize_rank(rank)
                new_experts.append(inherited.to(device=target_device, dtype=target_dtype))
            else:
                fresh = EvolvingLoRAExpert(in_features, out_features, rank)
                new_experts.append(fresh.to(device=target_device, dtype=target_dtype))

        self.experts = nn.ModuleList(new_experts)
        self.router.rebuild(new_num_experts=len(new_ranks), inherit_map=inherit_map)
