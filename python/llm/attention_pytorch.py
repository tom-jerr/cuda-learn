import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(self, num_q_heads, num_kv_heads, head_dim, hidden_size):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = self.hidden_size

        self.wq = nn.Linear(hidden_size, num_q_heads * head_dim)
        self.wk = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.wv = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.wo = nn.Linear(num_q_heads * head_dim, hidden_size)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        num_group = self.num_q_heads / self.num_kv_heads
        q = (
            q.reshape(b, s, self.num_q_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(b, self.num_kv_heads, num_group, s, self.head_dim)
        )
        k = k.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        s = q @ k.transpose(-2, -1)
        s = s / (self.head_dim**0.5)

        mask = torch.triu(torch.ones((s, s)), diagonal=1)
        s = s.mask_fill(mask, float("-inf"))

        p = F.softmax(s, dim=-1)
        o = p @ v
        o = o.reshape(b, self.num_q_heads, s, self.head_dim).transpose(1, 2)
        return o.reshape(b, s, d)


class MLA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
    ):
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.wq_a = nn.Linear(d_model, q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, num_heads * self.q_head_dim)
        self.wkv_a = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim)
        self.wkv_b = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
        )
        self.wo = nn.Linear(num_heads * v_head_dim, d_model)

    def forward(self, x):
        b, s, d = x.shape
        q_latent = self.wq_a(x)
        q_all = (
            self.wq_b(q_latent)
            .view(b, s, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
            .transpose(1, 2)
        )
        q_nope, q_rope = torch.split(q_all, dim=-1)

        kv_all = self.wkv_a(x)  # [B, S, kv_r + d_rope]
        c_kv, k_rope = torch.split(kv_all, dim=-1)
        # absorb
        w_uk, w_uv = torch.split(
            self.wkv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )

        q_absorbed = torch.einsum("bhsd,hdr->bhsr", q_nope, w_uk)
        score_nope = torch.einsum(
            "bhsr,btr->bhst",
            q_absorbed,
            c_kv,
        )
        # q_rope: [B, H, S, d_rope]
        # k_rope: [B, 1, T, d_rope]
        score_rope = torch.einsum("bhsr,bhtr->bhst", q_rope, k_rope)
        score = score_nope + score_rope
        p = torch.softmax(score, dim=-1)

        v = w_uv(c_kv).transpose(1, 2)  # [B,H,T,v_head_dim]

        # [B, H, S, T] × [B,H,T,v_head_dim]
        # -> [B, H, S, d_v]
        out = torch.einsum("bhst,bhtr->bhsr", p, v)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.v_head_dim)

        # [B, S, H * d_v] -> [B, S, hidden_dim]
        return self.wo(out)
