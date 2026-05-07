import torch
import torch.nn as nn
import torch.nn.functional as F

from .evolving_expert import EvolvingLoRAExpert


class LearnedDynamicRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.classifier = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, sample_representations):
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k={self.top_k} exceeds num_experts={self.num_experts}")
        router_logits = self.classifier(sample_representations)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return top_k_weights, top_k_indices


class LearnedMoELoRALinear(nn.Module):
    def __init__(self, base_layer, initial_ranks=None, top_k=1, alpha=16, dropout=0.05):
        super().__init__()
        if initial_ranks is None:
            initial_ranks = [8, 8, 8, 8]

        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.router = LearnedDynamicRouter(in_features, len(initial_ranks), top_k=top_k)
        self.experts = nn.ModuleList(
            [EvolvingLoRAExpert(in_features, out_features, rank, alpha=alpha, dropout=dropout) for rank in initial_ranks]
        )

    def _build_request_representation(self, x):
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
