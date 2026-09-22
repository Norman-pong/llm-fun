"""T5 预训练：在 tinyshakespeare 上训练 T4 的 GPT（MPS）。

相对 T4 demo 新增的工程件（rasbt 附录 D 路线）：
- LR 调度：线性 warmup + 余弦衰减到 10%（小数据集防过拟合的关键）
- 梯度裁剪（norm 1.0）：防训练尖峰
- 定期在验证集评估，训练结束记录完整曲线与超参到 runs/（已 ignore）

    uv run python theory/05-pretrain/code/train.py            # 完整训练
    uv run python theory/05-pretrain/code/train.py --steps 200 --smoke   # 冒烟（CI/测试用）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04-gpt" / "code"))

from gpt import GPTModel

DATA = Path(__file__).resolve().parent.parent / "input.txt"
RUNS = Path(__file__).resolve().parents[3] / "runs" / "05-pretrain"


def load_corpus(path: Path = DATA) -> tuple[list[str], dict[str, int], list[int]]:
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in text]
    return chars, stoi, ids


def split_ids(ids: list[int], frac: float = 0.9) -> tuple[list[int], list[int]]:
    cut = int(len(ids) * frac)
    return ids[:cut], ids[cut:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[j : j + block_size] for j in ix.tolist()])
    y = torch.stack([data[j + 1 : j + block_size + 1] for j in ix.tolist()])
    return x.to(device), y.to(device)


def lr_at(step: int, total: int, peak: float, warmup: int = 100) -> float:
    """warmup 线性升 → 余弦衰减到 peak 的 10%。"""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * min(progress, 1.0))))


@torch.no_grad()
def eval_loss(model: GPTModel, data: torch.Tensor, block_size: int, device: str) -> float:
    model.eval()
    g = torch.Generator().manual_seed(123)
    losses = []
    for _ in range(20):
        x, y = get_batch(data, block_size, 32, device, g)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(
    steps: int = 3000,
    block_size: int = 256,
    batch_size: int = 64,
    emb_dim: int = 192,
    n_head: int = 6,
    n_layer: int = 6,
    peak_lr: float = 1e-3,
    device: str | None = None,
    seed: int = 42,
    log_every: int = 500,
) -> dict:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    chars, stoi, ids = load_corpus()
    train_ids, val_ids = split_ids(ids)
    tr = torch.tensor(train_ids, dtype=torch.long)
    va = torch.tensor(val_ids, dtype=torch.long)
    print(f"语料 {len(ids)} 字符 | 词表 {len(chars)} | train/val {len(train_ids)}/{len(val_ids)} | 设备 {device}")

    model = GPTModel(
        vocab_size=len(chars), emb_dim=emb_dim, n_head=n_head,
        n_layer=n_layer, block_size=block_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_params:,}（{n_layer}L·{emb_dim}d·{n_head}H·ctx{block_size}）")

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(seed)
    history = {"train": [], "val": [], "lr": []}
    t0 = time.time()

    for step in range(steps):
        lr = lr_at(step, steps, peak_lr)
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        x, y = get_batch(tr, block_size, batch_size, device, gen)
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history["train"].append(loss.item())
        history["lr"].append(lr)
        if (step + 1) % log_every == 0 or step == steps - 1:
            vl = eval_loss(model, va, block_size, device)
            history["val"].append({"step": step + 1, "loss": vl})
            recent = history["train"][-log_every:]
            print(
                f"step {step + 1:5d} | train {sum(recent) / len(recent):.4f} | "
                f"val {vl:.4f} | lr {lr:.2e}"
            )

    wall = time.time() - t0
    final_val = history["val"][-1]["loss"]
    print(f"完成：{steps} 步 / {wall:.0f}s | 最终 val loss {final_val:.4f} "
          f"| ppl {math.exp(final_val):.1f}")

    out_dir = RUNS / f"s{steps}_d{emb_dim}_L{n_layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "history.json").write_text(
        json.dumps({"config": {
            "steps": steps, "block_size": block_size, "batch_size": batch_size,
            "emb_dim": emb_dim, "n_head": n_head, "n_layer": n_layer,
            "peak_lr": peak_lr, "seed": seed, "device": device,
        }, "n_params": n_params, **history}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"产物已存 {out_dir.relative_to(Path.cwd())}")

    torch.manual_seed(seed + 1)
    model.eval()
    prompt = torch.tensor([[stoi[c] for c in "\n"]], device=device)
    print("\n[采样 temperature=0.8]")
    print("".join(chars[i] for i in model.generate(prompt, 400, temperature=0.8, seed=seed).tolist()[0]))
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true", help="小模型冒烟：测试与 CI 用")
    args = ap.parse_args()
    if args.smoke:
        train(steps=args.steps, block_size=128, batch_size=32, emb_dim=64, n_head=4, n_layer=2, log_every=50)
    else:
        train(steps=args.steps)
