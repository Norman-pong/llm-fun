"""T4 demo：小 GPT 在一段英文上过训练前后的生成对比 + 参数量速算。

    uv run python theory/04-gpt/code/demo_gpt.py
"""

from __future__ import annotations

import torch
from gpt import GPTModel, count_params

TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "the lazy dog sleeps while the quick fox runs. "
) * 40
BLOCK = 64


def build_ids(text: str) -> tuple[list[str], dict[str, int], torch.Tensor]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([[stoi[c] for c in text]])
    return chars, stoi, ids


def train(model: GPTModel, ids: torch.Tensor, steps: int = 300) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(steps):
        i = torch.randint(0, ids.shape[1] - BLOCK - 1, (16,))
        x = torch.stack([ids[0, j : j + BLOCK] for j in i])
        y = torch.stack([ids[0, j + 1 : j + BLOCK + 1] for j in i])
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def sample(model: GPTModel, chars: list[str], stoi: dict[str, int], prompt: str) -> str:
    idx = torch.tensor([[stoi[c] for c in prompt]])
    out = model.generate(idx, max_new_tokens=80, temperature=0.5, top_k=8, seed=1)
    return "".join(chars[i] for i in out[0].tolist())


def main() -> None:
    torch.manual_seed(0)
    chars, stoi, ids = build_ids(TEXT)
    print(f"语料 {len(TEXT)} 字符，词表 {len(chars)}")
    model = GPTModel(
        vocab_size=len(chars), emb_dim=96, n_head=4, n_layer=3, block_size=BLOCK
    )
    print(f"参数量: {count_params(model):,}（12·层数·d² + 词表·d 的近似可验算，见 notes.md）")

    print("\n[训练前] " + sample(model, chars, stoi, "the quick"))
    losses = train(model, ids)
    print(f"[训练 {len(losses)} 步] loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print("[训练后] " + sample(model, chars, stoi, "the quick"))


if __name__ == "__main__":
    main()
