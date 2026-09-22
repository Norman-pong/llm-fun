"""T3 分词器：BPE（Byte Pair Encoding）的从零实现（minBPE 路线）。

为什么需要分词器：模型看不懂字符串，只认整数 id。BPE 的做法——
1) 把文本当字节流（0-255 天然是初始词表，任何 UTF-8 文本都不会有未知符号）；
2) 反复找出最高频的相邻 id 对，合并成一个新 id，直到词表到目标大小。
训练结果 = 合并表 merges；编码 = 在字节序列上按「合并优先级」反复应用合并。

BasicTokenizer 处理原始字节；RegexTokenizer 先用 GPT-2 的正则把文本切成
「词块」（pre-tokenization），BPE 只在块内进行——防止模型把跨词边界的拼接
学成 token，也天然支持中文/emoji（多字节 UTF-8 落在 256 个初始字节内）。
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import regex as re


def get_stats(ids: list[int]) -> Counter:
    """统计相邻 id 对的出现次数。"""
    return Counter(pairwise(ids))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把 ids 中所有 pair 替换为 new_id（顺序扫描，不重叠回吃）。"""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BasicTokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}  # (id,id) -> 新 id；插入序即合并优先级
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for step in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            # 决策确定性：先比频次，同频取 pair 字典序（minbpe 按 first-seen，此处显式化）
            pair = max(stats, key=lambda p: (stats[p], p))
            new_id = 256 + step
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {step:3d}: {pair} -> {new_id} ({self.vocab[new_id]!r}) 剩余 {len(ids)} ids")

    def _apply_merges(self, ids: list[int]) -> list[int]:
        """编码的核心：每次贪心应用「最早学到的」可合并对，直到没有可合并的。"""
        while len(ids) >= 2:
            pair = min(
                (p for p in pairwise(ids) if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if pair is None:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        return self._apply_merges(list(text.encode("utf-8")))

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


# GPT-2 的 pre-tokenization 正则（minbpe 同款）：缩写、字母串、数字串、标点串、空白
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RegexTokenizer(BasicTokenizer):
    """GPT-2 风格：正则切块 + 块内 BPE。训练全程操作 id 列表，不回字符串。"""

    def __init__(self, pattern: str = GPT2_SPLIT_PATTERN) -> None:
        super().__init__()
        self.pattern = re.compile(pattern)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        # 每个词块维护一份 id 列表；合并只在块内发生，跨块边界永不相并
        chunks = [list(c.encode("utf-8")) for c in self.pattern.findall(text)]
        for step in range(vocab_size - 256):
            stats: Counter = Counter()
            for ids in chunks:
                stats.update(get_stats(ids))
            if not stats:
                break
            pair = max(stats, key=lambda p: (stats[p], p))
            new_id = 256 + step
            chunks = [merge(ids, pair, new_id) for ids in chunks]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {step:3d}: {pair} -> {new_id} ({self.vocab[new_id]!r})")

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self.pattern.findall(text):
            out.extend(self._apply_merges(list(chunk.encode("utf-8"))))
        return out
