"""用 engine 训练小 MLP 拟合 XOR 数据：全批梯度下降，loss 应单调下降。

XOR（x1*x2 的符号）线性不可分，能直观体现「隐层 + 非线性」的价值。
直接运行本文件可看每个 epoch 的 loss 与最终准确率：
    uv run python theory/01-autograd/code/mlp_demo.py
"""

from __future__ import annotations

import random

from engine import MLP, Value


def make_xor_dataset(n: int = 24, seed: int = 7) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x1, x2 = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        xs.append([x1, x2])
        ys.append(1.0 if x1 * x2 > 0 else -1.0)
    return xs, ys


def mse_loss(model: MLP, xs: list[list[float]], ys: list[float]) -> Value:
    preds = [model([Value(a), Value(b)]) for a, b in xs]
    losses = [(p - y) ** 2 for p, y in zip(preds, ys, strict=True)]
    return sum(losses, start=Value(0.0)) * (1.0 / len(losses))


def accuracy(model: MLP, xs: list[list[float]], ys: list[float]) -> float:
    correct = sum(
        1
        for (a, b), y in zip(xs, ys, strict=True)
        if (model([Value(a), Value(b)]).data > 0) == (y > 0)
    )
    return correct / len(xs)


def train_mlp(
    epochs: int = 250, lr: float = 0.08, seed: int = 42
) -> tuple[list[list[float]], list[float], MLP, list[float]]:
    """全批梯度下降训练，返回 (xs, ys, model, 每 epoch 的 loss 列表)。"""
    random.seed(seed)  # 固定 MLP 参数初始化，保证实验可复现
    xs, ys = make_xor_dataset()
    model = MLP(2, [8, 1])
    losses: list[float] = []
    for _ in range(epochs):
        model.zero_grad()
        loss = mse_loss(model, xs, ys)
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        losses.append(loss.data)
    return xs, ys, model, losses


if __name__ == "__main__":
    xs, ys, model, losses = train_mlp()
    for epoch in range(0, len(losses), 10):
        print(f"epoch {epoch:3d}  loss {losses[epoch]:.6f}")
    print(f"epoch {len(losses) - 1:3d}  loss {losses[-1]:.6f}")
    print(f"最终准确率: {accuracy(model, xs, ys):.0%}")
