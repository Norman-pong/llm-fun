"""T2 DoD 验证（DoD 见 DESIGN.md 3.1 / 章 README）。

1) MLP 在 dev 上的 NLL 低于 bigram 基线（同数据同划分）
2) 从训练后的 MLP 采样出的「人名」合法（字符集内、无内部句点、长度合理）
3) 激活/梯度诊断可计算且数值有限
"""

from __future__ import annotations

import math
from pathlib import Path

from languagemodel import (
    BLOCK,
    BigramModel,
    MLPModel,
    build_dataset,
    build_vocab,
    diagnostics,
    load_names,
    nll,
    sample_names,
    split,
    train_mlp,
)

DATA = Path(__file__).resolve().parent.parent / "names.txt"


def _prepare():
    words = load_names(DATA)
    stoi, itos = build_vocab(words)
    tr, dev, _ = split(words)
    Xb, yb = build_dataset(tr, stoi, block=1)  # bigram 用 1 字符上下文
    Xtr, ytr = build_dataset(tr, stoi, BLOCK)
    Xdev, ydev = build_dataset(dev, stoi, BLOCK)
    return itos, stoi, (Xb, yb), (Xtr, ytr), (Xdev, ydev)


def _train_reference():
    """固定配置训练：既是测试基准也是 demo 复用入口。"""
    itos, stoi, (Xb, yb), (Xtr, ytr), (Xdev, ydev) = _prepare()
    model = MLPModel(vocab=len(itos))
    losses = train_mlp(model, Xtr, ytr, steps=6000, batch_size=128, lr=0.1)
    return model, losses, itos, stoi, (Xb, yb), (Xtr, ytr), (Xdev, ydev)


def test_bigram_baseline_sane():
    _, _, (Xb, yb), _, (Xdev, ydev) = _prepare()
    bigram = BigramModel(vocab_size=27)
    bigram.fit(Xb, yb)
    # 训练集 NLL 必须优于均匀分布（ln27≈3.30），且不过拟合到荒谬的低值
    train_nll = bigram.nll(Xb, yb)
    assert math.log(27) - 1.2 < train_nll < math.log(27), train_nll
    assert 2.2 < bigram.nll(Xdev[:, -1:], ydev) < 2.7  # dev NLL ≈ 2.45 量级


def test_mlp_beats_bigram_on_dev():
    model, losses, itos, _, (Xb, yb), _, (Xdev, ydev) = _train_reference()
    bigram = BigramModel(vocab_size=len(itos))
    bigram.fit(Xb, yb)
    bigram_dev = bigram.nll(Xdev[:, -1:], ydev)
    mlp_dev = nll(model, Xdev, ydev)
    # MLP 上下文更长 + 嵌入泛化，dev NLL 应明显优于 bigram（经验差 ≈ 0.15）
    assert mlp_dev < bigram_dev - 0.1, f"MLP {mlp_dev:.4f} vs bigram {bigram_dev:.4f}"
    assert losses[-1] < losses[0], "训练 loss 未下降"


def test_sampling_produces_valid_names():
    model, _, itos, *_ = _train_reference()
    names = sample_names(model, itos, n=50, seed=7)
    letters = set(itos[1:])  # 除 '.' 外全是合法字符
    assert len(names) == 50
    for name in names:
        assert set(name) <= letters, f"非法字符: {name!r}"
        assert 1 <= len(name) <= 20, f"长度异常: {name!r}"
    # 「可读」：绝大多数落在常见人名长度区间（欠训练模型会跑出超长串）
    readable = sum(1 for n in names if 2 <= len(n) <= 10)
    assert readable >= 0.85 * len(names)
    # 采样有多样性（不是同一个名字重复 50 次）
    assert len(set(names)) >= 20


def test_diagnostics_finite():
    model, _, _, _, _, (Xtr, _), _ = _train_reference()
    stats = diagnostics(model, Xtr)
    assert "hidden_saturation" in stats
    assert 0.0 <= stats["hidden_saturation"] <= 1.0
    for key, value in stats.items():
        assert math.isfinite(value), f"{key} 非有限值: {value}"
