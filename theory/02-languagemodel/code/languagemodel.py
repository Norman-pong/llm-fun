"""T2 语言建模：从计数 bigram 到 MLP（makemore 路线）。

语言模型的本质：P(下一个字符 | 前文)。两条实现路线——
1) BigramModel：直接数频次，context 只有 1 个字符（无参数学习的基线）；
2) MLPModel：Bengio 2003 风格——字符嵌入拼接 → 隐层 tanh → 全词表 softmax，
   context 扩展到 block 个字符，泛化能力来自嵌入空间的相似性。

数据约定：用 '.' 同时作起止符；数据集划分为 80/10/10（train/dev/test）。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SPECIAL = "."  # 起止符（也是词表 0 号）
BLOCK = 3  # MLP 的上下文长度（bigram 固定为 1）


# ---------- 数据 ----------


def load_names(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_vocab(words: list[str]) -> tuple[dict[str, int], list[str]]:
    chars = sorted({c for w in words for c in w})
    itos = [SPECIAL, *chars]  # '.' 固定在 0
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos


def build_dataset(
    words: list[str], stoi: dict[str, int], block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """每个词展开成 block 个样本：(...'.'+前缀, 下一个字符)。"""
    xs, ys = [], []
    for w in words:
        context = [0] * block
        for ch in [*w, SPECIAL]:
            xs.append(context)
            ys.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def split(words: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(words), generator=g).tolist()
    n = len(words)
    cut1, cut2 = int(n * 0.8), int(n * 0.9)
    return (
        [words[i] for i in idx[:cut1]],
        [words[i] for i in idx[cut1:cut2]],
        [words[i] for i in idx[cut2:]],
    )


# ---------- 基线：计数 bigram ----------


class BigramModel:
    """数频次的 bigram 语言模型（add-1 平滑）。context = 1 个字符。"""

    def __init__(self, vocab_size: int) -> None:
        # +1 平滑等价于先验计数 1，避免未出现过的二元组概率为 0
        self.counts = torch.ones(vocab_size, vocab_size)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        for prev, nxt in zip(X[:, -1].tolist(), y.tolist(), strict=True):
            self.counts[prev, nxt] += 1

    def probs(self) -> torch.Tensor:
        return self.counts / self.counts.sum(dim=1, keepdim=True)

    def nll(self, X: torch.Tensor, y: torch.Tensor) -> float:
        p = self.probs()
        return F.nll_loss(torch.log(p[X[:, -1]]), y).item()

    def sample(self, n: int, itos: list[str], seed: int = 0) -> list[str]:
        g = torch.Generator().manual_seed(seed)
        p = self.probs()
        out = []
        for _ in range(n):
            chars, prev = [], 0
            while True:
                nxt = torch.multinomial(p[prev], 1, generator=g).item()
                if nxt == 0:
                    break
                chars.append(itos[nxt])
                prev = nxt
            out.append("".join(chars))
        return out


# ---------- MLP 语言模型（Bengio 2003） ----------


class MLPModel(nn.Module):
    def __init__(self, vocab: int, block: int = BLOCK, emb_dim: int = 10, hidden: int = 128) -> None:
        super().__init__()
        self.block = block
        self.emb = nn.Embedding(vocab, emb_dim)
        self.fc1 = nn.Linear(block * emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab)
        self.last_hidden: torch.Tensor | None = None  # 诊断用：最近一次前向的隐层激活

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x).reshape(x.shape[0], -1)
        h = torch.tanh(self.fc1(e))
        self.last_hidden = h.detach()
        return self.fc2(h)  # logits；交叉熵内部含 softmax

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.forward(x), y)


def train_mlp(
    model: MLPModel,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    steps: int = 3000,
    batch_size: int = 128,
    lr: float = 0.1,
    device: str | None = None,
    seed: int = 42,
) -> list[float]:
    """小批 SGD（无 momentum，教学用最简形态）。返回每步 loss。"""
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        # CPU generator 采样索引后再搬去设备，避免 MPS 上 generator 设备不匹配
        idx = torch.randint(len(Xtr), (batch_size,), generator=g).to(device)
        loss = model.loss(Xtr[idx], ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def nll(model: MLPModel, X: torch.Tensor, y: torch.Tensor, device: str | None = None) -> float:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return F.cross_entropy(model(X.to(device)), y.to(device)).item()


@torch.no_grad()
def sample_names(
    model: MLPModel,
    itos: list[str],
    n: int,
    seed: int = 0,
    device: str | None = None,
    max_len: int = 20,
) -> list[str]:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        context = [0] * model.block
        chars = []
        for _step in range(max_len):  # 上限保护：模型未学好时 while True 可能停不下来
            logits = model(torch.tensor([context], device=device))
            probs = F.softmax(logits, dim=1).cpu()  # multinomial 走 CPU generator
            nxt = torch.multinomial(probs, 1, generator=g).item()
            if nxt == 0:
                break
            chars.append(itos[nxt])
            context = context[1:] + [nxt]
        out.append("".join(chars))
    return out


@torch.no_grad()
def diagnostics(model: MLPModel, X: torch.Tensor) -> dict[str, float]:
    """makemore 第 3/4 讲的激活/梯度诊断：隐层饱和度与各层 grad:weight 比。"""
    device = next(model.parameters()).device
    model(X[:512].to(device))
    h = model.last_hidden
    sat = (h.abs() < 0.01).float().mean().item()  # tanh 饱死比例，越低越健康
    stats = {"hidden_saturation": sat}
    for name, p in model.named_parameters():
        if p.grad is not None and p.dim() > 1:  # 只看权重矩阵
            stats[f"{name}_grad_to_weight_std_ratio"] = (p.grad.std() / p.std()).item()
    return stats
