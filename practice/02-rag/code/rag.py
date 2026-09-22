"""P2 RAG 核心：切块 → TF-IDF embedding → Chroma 向量检索 →（可选）LLM 生成。

零 API key、零服务依赖即可完成检索与评测（DoD）；
生成环节复用 P1 的 LLMClient，无 key 时优雅跳过。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "01-api" / "code"))

from embedder import TfidfEmbedder

CARDS = Path(__file__).resolve().parent / "knowledge_cards.md"
CARD_RE = re.compile(r"^## (KB-\d+)\n(.+?)(?=^## |\Z)", re.DOTALL | re.MULTILINE)


@dataclass
class Chunk:
    cid: str  # 卡片/块 id
    text: str


def load_cards(path: Path = CARDS) -> list[Chunk]:
    return [Chunk(cid=m.group(1), text=" ".join(m.group(2).split())) for m in CARD_RE.finditer(path.read_text(encoding="utf-8"))]


def split_text(text: str, max_len: int = 300, overlap: int = 60) -> list[str]:
    """滑窗切块：按句号/换行优先断句，块超长时滑窗切并保留重叠。"""
    sentences = [s for s in re.split(r"(?<=[。！？.!？])\s*|\n", text) if s.strip()]
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) > max_len and cur:
            chunks.append("".join(cur))
            # 重叠：保留尾部若干句，保证跨块上下文
            tail, tail_len = [], 0
            for t in reversed(cur):
                if tail_len + len(t) > overlap:
                    break
                tail.insert(0, t)
                tail_len += len(t)
            cur, cur_len = tail, tail_len
        cur.append(s)
        cur_len += len(s)
    if cur:
        chunks.append("".join(cur))
    return chunks


class RagIndex:
    """Chroma 嵌入式（PersistentClient）+ 自带 TF-IDF embedding。"""

    def __init__(self, persist_dir: str | None = None) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.Client()
        self.collection = self._client.get_or_create_collection(
            "kb-cards", metadata={"hnsw:space": "cosine"}  # 余弦距离（TF-IDF 已 L2 归一化）
        )
        self.embedder = TfidfEmbedder()

    def build(self, chunks: list[Chunk]) -> None:
        self._client.delete_collection("kb-cards")  # 幂等重建：评测/演示可重复跑
        self.collection = self._client.get_or_create_collection(
            "kb-cards", metadata={"hnsw:space": "cosine"}
        )
        self.embedder.fit([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.cid for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[e.tolist() for e in self.embedder.transform([c.text for c in chunks])],
            metadatas=[{"cid": c.cid} for c in chunks],
        )

    def query(self, question: str, top_k: int = 3) -> list[tuple[str, str, float]]:
        """返回 [(cid, text, similarity)]，相似度 = 1 - 余弦距离，按相似度降序。"""
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(question)], n_results=top_k
        )
        out = []
        for i in range(len(res["ids"][0])):
            cid = res["ids"][0][i]
            dist = res["distances"][0][i]
            out.append((cid, res["documents"][0][i], 1.0 - float(dist)))
        return out


# ---------- 评测集：24 问（DoD 要求 ≥20），含同义改写问法 ----------

EVAL_SET: list[tuple[str, str]] = [
    ("反向传播为什么 grad 要用加法累积？", "KB-01"),
    ("链式法则怎么在计算图上执行？", "KB-01"),
    ("学习率为什么要 warmup？", "KB-02"),
    ("余弦衰减和梯度裁剪分别解决什么问题？", "KB-02"),
    ("BPE 为什么用字节做初始词表？", "KB-03"),
    ("分词器怎么保证中文不会出现未知符号？", "KB-03"),
    ("Pre-LN 和 Post-LN 有什么区别？", "KB-04"),
    ("注意力里的 Q K V 分别是什么？", "KB-05"),
    ("点积为什么要除以根号 d？", "KB-05"),
    ("多头注意力每个头学什么？", "KB-06"),
    ("因果掩码是怎么挡住未来信息的？", "KB-07"),
    ("采样温度和 top-k 各控制什么？", "KB-08"),
    ("困惑度是什么意思？", "KB-09"),
    ("bigram 和 MLP 语言模型差在哪？", "KB-10"),
    ("训练 loss 降但验证 loss 升说明什么？", "KB-11"),
    ("GPT 参数量怎么估算？", "KB-12"),
    ("前馈层为什么扩 4 倍？", "KB-13"),
    ("什么是权重共享？", "KB-14"),
    ("为什么初始化用 0.02 的标准差？", "KB-15"),
    ("RAG 的三个步骤是什么？", "KB-16"),
    ("文本切块为什么要留重叠？", "KB-17"),
    ("TF-IDF 的词权重怎么算？", "KB-18"),
    ("hit-rate 指标怎么定义？", "KB-19"),
    ("API 429 错误要重试吗？401 呢？", "KB-20"),
]


def evaluate(index: RagIndex, eval_set: list[tuple[str, str]] | None = None) -> dict[str, float]:
    """hit@1 / hit@3：期望卡片是否出现在 top-k 检索结果。"""
    eval_set = eval_set or EVAL_SET
    hits1 = hits3 = 0
    for q, expected in eval_set:
        results = index.query(q, top_k=3)
        top_ids = [cid for cid, _, _ in results]
        if top_ids[0] == expected:
            hits1 += 1
        if expected in top_ids:
            hits3 += 1
    n = len(eval_set)
    return {"hit@1": hits1 / n, "hit@3": hits3 / n, "n": n}


# ---------- 生成环节（可选，需 API key） ----------

SYSTEM_PROMPT = (
    "你是知识库问答助手。只依据下面编号的资料回答问题，"
    "引用用到的资料编号（如 [KB-05]）；资料不足以回答时明确说不知道。\n\n资料：\n{context}"
)


def answer(index: RagIndex, question: str, top_k: int = 3, generate: bool = True) -> dict:
    """端到端：检索 →（有 key 且 generate 时）生成。"""
    results = index.query(question, top_k=top_k)
    payload = {
        "question": question,
        "retrieved": [{"cid": cid, "sim": round(sim, 4), "text": text[:120]} for cid, text, sim in results],
        "answer": None,
    }
    if not generate:
        return payload
    try:
        from client import LLMClient  # P1 资产

        llm = LLMClient()
    except Exception as e:  # noqa: BLE001 - 无 key 时优雅降级为纯检索
        payload["answer"] = f"[未生成：{e}] 已返回检索结果"
        return payload
    context = "\n".join(f"[{cid}] {text}" for cid, text, _ in results)
    payload["answer"] = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return payload
