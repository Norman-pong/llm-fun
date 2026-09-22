"""T4 手写 Attention 与 GPT（rasbt ch3-4 / build-nanogpt 路线）。

Attention 一句话：每个 token 用自己当 Query 去「查询」所有位置的 Key，
相似度归一化成权重后加权聚合各处的 Value——信息按相关性流动。

GPT = 因果自注意力（只看过去）+ 残差 + LayerNorm + 前馈层的堆叠，
外加 token/position 嵌入与共享权重的输出头。全部按教学展开实现，
对齐基准是 torch 官方的 F.scaled_dot_product_attention(is_causal=True)。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """多头因果自注意力（GPT-2 形态：QKV 合并成一个 Linear）。

    forward 拆成 _qkv_heads / attend / _merge_heads_proj 三步，
    其中 attend 是纯手写数学（分头 → 缩放点积 → 因果 mask → softmax → 加权），
    测试用它直接对齐官方 SDPA 实现。
    """

    def __init__(self, emb_dim: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert emb_dim % n_head == 0, "emb_dim 必须能整除头数"
        self.n_head = n_head
        self.head_dim = emb_dim // n_head
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        # bool 下三角阵：mask[i,j]=True 表示位置 i 允许看位置 j
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def _qkv_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[B,T,C] -> 三个 [B,n_head,T,head_dim]。"""
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        split = lambda z: z.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        return split(q), split(k), split(v)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """手写的缩放点积注意力（因果）。输入输出均为 [B,n_head,T,head_dim]。"""
        att = q @ k.transpose(-2, -1) * self.head_dim**-0.5  # 相似度，除以 sqrt(d) 防止 softmax 饱和
        t = q.shape[-2]
        att = att.masked_fill(~self.mask[:t, :t], float("-inf"))  # 未来位置 → -inf → softmax 后为 0
        att = F.softmax(att, dim=-1)
        return att @ v

    def _merge_heads_proj(self, y: torch.Tensor) -> torch.Tensor:
        """[B,n_head,T,head_dim] -> [B,T,C]，过输出投影。"""
        b, _, t, _ = y.shape
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._qkv_heads(x)
        return self._merge_heads_proj(self.attend(q, k, v))


class FeedForward(nn.Module):
    """GPT-2 前馈层：Linear(C→4C) → GELU → Linear(4C→C)。"""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN 残差块（GPT-2）：x = x + attn(ln1(x)); x = x + ffn(ln2(x))。"""

    def __init__(self, emb_dim: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(emb_dim, n_head, block_size)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffn = FeedForward(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 256,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        # 权重共享（GPT-2）：输出投影与输入嵌入用同一矩阵，省 vocab*C 个参数且小模型更稳
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # GPT-2 的残差分支初始化：每个 block 有两条加性残差路径（attn/ffn），
        # 深层堆叠后方差线性增长，把两条路径的输出投影再缩小 sqrt(2*n_layer) 补偿
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.02 * (2 * n_layer) ** -0.5)
            torch.nn.init.normal_(block.ffn.net[2].weight, std=0.02 * (2 * n_layer) ** -0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 初始化：全部 normal(0, 0.02)。默认的 N(0,1) 会让 logits 方差
        随 emb_dim 放大，初始 loss 高达几十；0.02 起点让初始分布接近均匀。"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """LM 约定：位置 t 预测 t+1。传入 targets 必须是 idx 左移一位
        （x = idx[:, :-1], y = idx[:, 1:]），否则权重共享会出现
        「输出=当前输入嵌入」的恒等捷径，loss 虚假地接近 0。
        """
        _, t = idx.shape
        assert t <= self.block_size, f"序列长 {t} 超过 block_size {self.block_size}"
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """自回归采样：逐步把最后一个位置的分布采样结果拼回序列。"""
        self.eval()
        g = torch.Generator().manual_seed(seed)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]  # 超长时只保留最近的 block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # 低温→更贪心
            if top_k is not None:
                kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1).cpu()
            next_id = torch.multinomial(probs, 1, generator=g).to(idx.device)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
