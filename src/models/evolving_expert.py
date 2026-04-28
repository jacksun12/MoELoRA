import math
import torch
import torch.nn as nn


class EvolvingLoRAExpert(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=16, dropout=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.rank == 0:
            return 0
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling

    @torch.no_grad()
    def delta_weight(self):
        return self.lora_A @ self.lora_B

    @torch.no_grad()
    def split_via_svd(self, rank_1, rank_2):
        curr_delta_w = self.delta_weight()
        U, S, Vh = torch.linalg.svd(curr_delta_w.float(), full_matrices=False)

        expert_1 = EvolvingLoRAExpert(self.in_features, self.out_features, rank_1, alpha=self.alpha)
        expert_2 = EvolvingLoRAExpert(self.in_features, self.out_features, rank_2, alpha=self.alpha)

        s_1 = torch.diag(torch.sqrt(S[:rank_1]))
        expert_1.lora_A.data = U[:, :rank_1] @ s_1
        expert_1.lora_B.data = s_1 @ Vh[:rank_1, :]

        s_2 = torch.diag(torch.sqrt(S[rank_1 : rank_1 + rank_2]))
        expert_2.lora_A.data = U[:, rank_1 : rank_1 + rank_2] @ s_2
        expert_2.lora_B.data = s_2 @ Vh[rank_1 : rank_1 + rank_2, :]
        return expert_1, expert_2

    @classmethod
    @torch.no_grad()
    def from_delta_weight(cls, delta_w, rank, alpha=16, dropout=0.05):
        in_features, out_features = delta_w.shape
        rank = max(1, int(rank))
        expert = cls(in_features, out_features, rank, alpha=alpha, dropout=dropout)

        U, S, Vh = torch.linalg.svd(delta_w.float(), full_matrices=False)
        take = min(rank, U.shape[1], Vh.shape[0], S.shape[0])

        s = torch.diag(torch.sqrt(S[:take]))
        expert.lora_A.data.zero_()
        expert.lora_B.data.zero_()
        expert.lora_A.data[:, :take] = U[:, :take] @ s
        expert.lora_B.data[:take, :] = s @ Vh[:take, :]
        return expert

    @torch.no_grad()
    def resize_rank(self, new_rank):
        return EvolvingLoRAExpert.from_delta_weight(
            self.delta_weight(),
            rank=new_rank,
            alpha=self.alpha,
            dropout=self.dropout.p,
        )
