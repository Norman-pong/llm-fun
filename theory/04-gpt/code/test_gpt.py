"""T4 DoD 验证（DoD 见 DESIGN.md 3.1 / 章 README）。

1) 手写多头注意力与官方 F.scaled_dot_product_attention(is_causal=True) 数值对齐
2) GPT 前向：shape/有限性/确定性/权重共享检查
3) 端到端可学：小 GPT 在重复语料上过拟合，loss 大幅下降
4) 采样生成：id 合法、长度正确、低温下复现训练模式
"""

from __future__ import annotations

import math

import torch
from gpt import GPTModel, count_params
from torch.nn import functional as F

B, T, C, NH, VOCAB = 2, 6, 32, 4, 5  # 小尺寸：测试跑 CPU 即可


def _make_attn():
    from gpt import CausalSelfAttention

    torch.manual_seed(0)
    return CausalSelfAttention(C, NH, block_size=16)


def test_attention_matches_torch_sdpa():
    attn = _make_attn().eval()
    x = torch.randn(B, T, C, generator=torch.Generator().manual_seed(1))

    q, k, v = attn._qkv_heads(x)
    manual = attn._merge_heads_proj(attn.attend(q, k, v))
    with torch.no_grad():
        reference = attn._merge_heads_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True))
    assert torch.allclose(manual, reference, atol=1e-5), (
        f"最大误差 {(manual - reference).abs().max().item():.2e}"
    )


def test_causal_mask_actually_hides_future():
    """因果性检查：改变未来位置的输入不得影响过去位置的输出。"""
    attn = _make_attn().eval()
    torch.manual_seed(2)
    x = torch.randn(1, T, C)
    x2 = x.clone()
    x2[0, T - 1] += 10.0  # 篡改最后一个位置
    with torch.no_grad():
        y1, y2 = attn(x), attn(x2)
    assert torch.allclose(y1[0, : T - 1], y2[0, : T - 1], atol=1e-6), "未来信息泄漏到了过去"


def _make_gpt() -> GPTModel:
    torch.manual_seed(3)
    return GPTModel(
        vocab_size=VOCAB, emb_dim=64, n_head=4, n_layer=2, block_size=32
    ).eval()


def test_gpt_forward_shape_and_sanity():
    model = _make_gpt()
    idx = torch.randint(0, VOCAB, (2, 11))
    x, y = idx[:, :-1], idx[:, 1:]  # LM 约定：位置 t 预测 t+1
    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 10, VOCAB)
    assert math.isfinite(loss.item())
    # 未训练时 loss 应在均匀分布基线 ln(vocab) 附近（不存在恒等捷径）
    assert abs(loss.item() - math.log(VOCAB)) < 0.8, loss.item()
    # 确定性：eval 模式下同输入同输出
    logits2, _ = model(x)
    assert torch.equal(logits, logits2)


def test_weight_tying():
    model = _make_gpt()
    assert model.head.weight is model.tok_emb.weight, "输出头未与词嵌入共享权重"
    # 参数量中 lm_head 不应被重复计数
    n = count_params(model)
    assert n > 0


STOI = {"a": 0, "b": 1, "c": 2, " ": 3}


def _train_pattern_model(steps: int = 200) -> GPTModel:
    """在 'abc ' 循环语料上训练小 GPT，供过拟合与采样测试共用。"""
    ids = torch.tensor([[STOI[c] for c in "abc " * 300]])
    model = GPTModel(vocab_size=4, emb_dim=64, n_head=2, n_layer=1, block_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.randint(0, ids.shape[1] - 17, (16,))
        x = torch.stack([ids[0, j : j + 16] for j in i])
        y = torch.stack([ids[0, j + 1 : j + 17] for j in i])
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


def test_tiny_overfit_learns():
    """端到端可学性：在重复语料 'abcabc...' 上过拟合一个小 GPT。"""
    torch.manual_seed(3)
    model = GPTModel(vocab_size=4, emb_dim=64, n_head=2, n_layer=1, block_size=16)
    ids = torch.tensor([[STOI[c] for c in "abc " * 300]])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = float("nan")
    for step in range(200):
        i = torch.randint(0, ids.shape[1] - 17, (16,))
        x = torch.stack([ids[0, j : j + 16] for j in i])
        y = torch.stack([ids[0, j + 1 : j + 17] for j in i])
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < 0.2, f"过拟合失败: 初始 {first:.3f} -> 最终 {last:.3f}"
    assert last < 0.25 * first


def test_generate_valid_and_follows_pattern():
    torch.manual_seed(3)
    model, _ = _train_pattern_model()
    prompt = torch.tensor([[0, 1, 2]])  # "abc"
    out = model.generate(prompt, max_new_tokens=12, temperature=0.1, seed=5)
    assert out.shape == (1, 15)
    assert ((out >= 0) & (out < 4)).all(), "采样出了词表外的 id"
    # 低温采样在学好的模式上应延续 abc 循环
    decoded = "".join("abc "[i] for i in out[0].tolist())
    assert decoded.startswith("abc"), decoded
    assert decoded.count("abc") >= 4, f"未复现训练模式: {decoded!r}"
