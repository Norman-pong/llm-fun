"""P2 DoD 验证（DoD 见 DESIGN.md 3.1 / 章 README）——检索与评测全本地，零 API key。

1) 端到端检索可跑：24 张知识卡片入库，任意问题返回 top-k（含相似度）
2) 评测集 ≥20 条，hit@1 / hit@3 有数字且达到阈值
3) 切块器行为正确（边界/重叠/短文本）
4) TF-IDF 语义基本正确（相关文档相似度 > 无关文档）
5) answer() 无 key 时优雅降级
"""

from __future__ import annotations

import numpy as np
from embedder import TfidfEmbedder, cosine, tokenize
from rag import EVAL_SET, RagIndex, answer, evaluate, load_cards, split_text


def _index() -> RagIndex:
    idx = RagIndex()
    idx.build(load_cards())
    return idx


# ---------- 切块 ----------


def test_split_text_overlap_and_bounds():
    text = "这是第一句话。" * 40  # 280 字符，超过 max_len=100
    chunks = split_text(text, max_len=100, overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) <= 100 + 20 for c in chunks)  # 单句不超限时块不超限太多
    # 相邻块有重叠（首尾内容衔接）
    assert any(chunks[i][-6:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_split_text_short_untouched():
    assert split_text("很短的文本。") == ["很短的文本。"]


# ---------- TF-IDF ----------


def test_tfidf_similarity_ranking():
    emb = TfidfEmbedder().fit(["attention computes softmax scores", "今天天气很好", "attention uses softmax"])
    v = emb.transform(["what is attention softmax", "attention softmax", "天气"])
    assert cosine(v[0], v[1]) > cosine(v[0], v[2])  # 相关 > 无关
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)  # L2 归一化


def test_tokenize_mixed():
    toks = tokenize("GPT 用 Attention 机制")
    assert "gpt" in toks and "attention" in toks
    # 中文无空格：抽取全部汉字串 "用机制" 后切 bigram（"用机"/"机制"），而非整词
    assert "机制" in toks and "用机" in toks


# ---------- 检索与评测（DoD 核心） ----------


def test_retrieve_returns_topk_with_similarity():
    idx = _index()
    results = idx.query("梯度裁剪是干什么的", top_k=3)
    assert len(results) == 3
    cids = [cid for cid, _, _ in results]
    assert "KB-02" in cids, f"期望 KB-02（含梯度裁剪）在 top-3: {cids}"
    sims = [sim for _, _, sim in results]
    assert sims == sorted(sims, reverse=True)


def test_eval_set_size_and_hitrate():
    assert len(EVAL_SET) >= 20  # DoD：≥20 条
    idx = _index()
    metrics = evaluate(idx)
    print(f"\nhit@1={metrics['hit@1']:.2%} hit@3={metrics['hit@3']:.2%} (n={metrics['n']})")
    assert metrics["hit@1"] >= 0.70, f"hit@1 过低: {metrics}"
    assert metrics["hit@3"] >= 0.90, f"hit@3 过低: {metrics}"


def test_answer_degrades_gracefully_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    idx = _index()
    out = answer(idx, "什么是 RAG", generate=True)
    assert out["retrieved"] and out["retrieved"][0]["cid"]
    assert out["answer"] is not None and "未生成" in out["answer"]  # 降级而非崩溃


def test_answer_retrieve_only_mode():
    idx = _index()
    out = answer(idx, "什么是 RAG", generate=False)
    assert out["answer"] is None
    assert out["retrieved"][0]["cid"] == "KB-16"


def test_cards_loaded():
    cards = load_cards()
    assert len(cards) == 24
    assert all(c.cid.startswith("KB-") and c.text for c in cards)
