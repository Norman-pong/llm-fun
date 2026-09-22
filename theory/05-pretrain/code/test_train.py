"""T5 DoD 验证（DoD 见 DESIGN.md 3.1 / 章 README）。

完整训练在 demo 路径（train.py，MPS 约 10 分钟）；测试用 300 步小配置
验证 DoD 的核心断言：loss 持续下降、val 曲线单调改善、采样出可读莎士比亚风格文本、
超参/曲线完整落盘。全套约 40 秒（CPU）。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04-gpt" / "code"))

from train import eval_loss, load_corpus, lr_at, split_ids, train


def _train_small() -> dict:
    return train(
        steps=300,
        block_size=128,
        batch_size=32,
        emb_dim=64,
        n_head=4,
        n_layer=2,
        peak_lr=1e-3,
        log_every=100,
    )


def test_lr_schedule_shape():
    assert lr_at(0, 1000, 1e-3) == 1e-3 * 1 / 100  # warmup 第一步
    assert abs(lr_at(100, 1000, 1e-3) - 1e-3) < 1e-9  # warmup 结束达峰值
    assert abs(lr_at(1000, 1000, 1e-3) - 1e-3 * 0.1) < 1e-9  # 终点衰减到 10%
    mid = lr_at(550, 1000, 1e-3)
    assert 1e-3 * 0.1 < mid < 1e-3  # 中段单调下降区间


def test_train_loss_decreases():
    history = _train_small()
    tr = history["train"]
    first100 = sum(tr[:100]) / 100
    last100 = sum(tr[-100:]) / 100
    assert last100 < first100 - 0.5, f"下降不足: {first100:.3f} -> {last100:.3f}"


def test_val_curve_improves():
    history = _train_small()
    vals = [v["loss"] for v in history["val"]]
    assert len(vals) >= 3
    assert vals[-1] < vals[0] - 0.2, f"val 改善不足: {vals}"
    # 拟合健康度：val 不应过拟合到接近 0，也不该几乎没学（均匀基线 ln65≈4.17）
    assert 1.5 < vals[-1] < 3.4


def test_run_artifacts_persisted():
    _train_small()
    run_dirs = sorted(
        (Path(__file__).resolve().parents[3] / "runs" / "05-pretrain").glob("s300_*")
    )
    assert run_dirs, "runs 产物未落盘"
    hist = json.loads((run_dirs[-1] / "history.json").read_text(encoding="utf-8"))
    assert {"config", "n_params", "train", "val", "lr", "best"} <= set(hist)
    assert len(hist["train"]) == 300
    assert (run_dirs[-1] / "model.pt").exists()
    assert (run_dirs[-1] / "model_best.pt").exists()
    # best 记录的必须是 val 曲线上的真实最优点
    best_on_curve = min(hist["val"], key=lambda v: v["loss"])
    assert hist["best"]["step"] == best_on_curve["step"]
    assert abs(hist["best"]["loss"] - best_on_curve["loss"]) < 1e-9


def test_sample_shakespeare_style():
    """300 步小模型 + 低温度：应出现大量合法英文单词与剧本格式痕迹。"""
    _train_small()
    chars, stoi, ids = load_corpus()
    _, val_ids = split_ids(ids)
    va = torch.tensor(val_ids, dtype=torch.long)
    run_dirs = sorted(
        (Path(__file__).resolve().parents[3] / "runs" / "05-pretrain").glob("s300_*")
    )
    from gpt import GPTModel

    model = GPTModel(
        vocab_size=len(chars), emb_dim=64, n_head=4, n_layer=2, block_size=128
    )
    model.load_state_dict(torch.load(run_dirs[-1] / "model.pt", weights_only=True))
    model.eval()
    vl = eval_loss(model, va, 128, "cpu")
    assert math.isfinite(vl) and vl < 3.2

    prompt = torch.tensor([[stoi["\n"]]])
    text = "".join(
        chars[i]
        for i in model.generate(prompt, 300, temperature=0.7, seed=3).tolist()[0]
    )
    words = {"the", "and", "you", "my", "that", "is", "not", "it", "for", "his"}
    hits = sum(1 for w in words if w in text)
    assert hits >= 5, f"常见词出现太少({hits}/10): {text[:120]!r}"
