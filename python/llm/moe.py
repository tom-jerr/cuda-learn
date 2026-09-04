import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w_gateup = nn.Linear(hidden_size, 2 * hidden_size)
        self.w_down = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor):
        gate_up = self.w_gateup(x)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        hidden = F.silu(gate) * up
        return self.w_down(hidden)


class MoE(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([Expert(hidden_size) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        x_flat = x.reshape(-1, d)

        topk_weight, topk_idx = torch.topk(
            F.softmax(self.gate(x_flat), dim=-1), self.top_k
        )
        topk_weight = topk_weight / torch.sum(topk_weight, dim=-1, keepdim=True)
        o = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            expert_mask = topk_idx == expert_id
            if not expert_mask.any():
                continue
            token_idx, k_idx = torch.where(expert_mask)
            tokens = x_flat[token_idx]
            y = expert(tokens) * topk_weight[token_idx, k_idx].unsqueeze(-1)  # [B*S, D]
            # 将 y 累加到输出中
            o.scatter_add_(0, index=token_idx.unsqueeze(-1).expand_as(y), src=y)
        return o.reshape(b, s, d)
