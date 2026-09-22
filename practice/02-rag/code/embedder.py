"""P2 本地 embedding：从零实现的 TF-IDF（中英混合分词 + L2 归一化）。

不用 API、不下载模型——embedding 的本质就是「把文本变成可比较的向量」，
TF-IDF 是理解这件事的最小起点：词重要度 = 词频 × 稀有度。
P4 backlog 再升级到神经 embedding（如 bge）做对照。

中文分词用「CJK 字符 bigram + 英文/数字词」：无词典依赖，对短查询足够。
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """英文/数字切词（小写化）+ 汉字 bigram。"""
    tokens = [w.lower() for w in _WORD.findall(text)]
    cjk = "".join(_CJK.findall(text))
    tokens += [cjk[i : i + 2] for i in range(len(cjk) - 1)]
    return tokens


class TfidfEmbedder:
    """fit(corpus) 建 vocab 与 idf；transform(texts) 出 L2 归一化的稠密矩阵。"""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> TfidfEmbedder:
        df: Counter = Counter()
        for doc in corpus:
            df.update(set(tokenize(doc)))
        n = len(corpus)
        # 平滑 idf：未见词也有有限权重（查询侧新词不至于全 0）
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)]
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        assert self.idf is not None, "先 fit"
        mat = np.zeros((len(texts), len(self.vocab)))
        for r, text in enumerate(texts):
            for tok, tf in Counter(tokenize(text)).items():
                c = self.vocab.get(tok)
                if c is not None:
                    mat[r, c] = tf * self.idf[c]
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return mat / norm

    def embed_query(self, text: str) -> list[float]:
        return self.transform([text])[0].tolist()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
