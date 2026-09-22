"""T3 DoD 验证（DoD 见 DESIGN.md 3.1 / 章 README）。

1) Basic/Regex 两种分词器：编码→解码 roundtrip 无损（英文/中文/emoji/代码）
2) 合并行为与 minbpe 的经典确定性样例一致（README 的 aaabdaaabac 例子）
3) RegexTokenizer 遵守 GPT-2 切块边界，跨块不相并
4) BPE 确实压缩：训练语料上的 id 数显著少于字节数
"""

from __future__ import annotations

from pathlib import Path

from tokenizer import BasicTokenizer, RegexTokenizer

DATA = Path(__file__).resolve().parent.parent.parent / "02-languagemodel" / "names.txt"

ROUNDTRIP_TEXTS = [
    "hello world",
    "你好，世界！大模型理论学习",
    "emoji 🤖🔥 and tabs\t\tnewlines\n\n",
    "def train(model, lr=0.1): return model.fit(lr)  # code",
    "混合 mixed 文本 123 with numbers 456!",
    "",
    "a",
]

TOY = "aaabdaaabac"  # minbpe README 的经典样例


def _trained_basic() -> BasicTokenizer:
    tok = BasicTokenizer()
    tok.train(DATA.read_text(encoding="utf-8")[:200_000], vocab_size=512)
    return tok


def _trained_regex() -> RegexTokenizer:
    tok = RegexTokenizer()
    tok.train(DATA.read_text(encoding="utf-8")[:200_000], vocab_size=512)
    return tok


def test_toy_corpus_matches_expected_merges():
    """确定性样例（Sennrich BPE 论文经典语料 aaabdaaabac）的逐次合并可手算。

    字节序: [97,97,97,98,100,97,97,97,98,97,99]
    第 1 次合并: 最高频对 (97,97) 共 4 次 → 256，得到 [256,97,98,100,256,97,98,97,99]
    第 2 次合并: (256,97) 与 (97,98) 均 2 次，按本实现的确定性规则
    （同频取 pair 字典序大者，与 minbpe 的 first-seen 序在此例结果一致）选 (256,97) → 257
    编码结果: [257,98,100,257,98,97,99]
    """
    tok = BasicTokenizer()
    tok.train(TOY, vocab_size=258, verbose=False)
    assert tok.merges[(97, 97)] == 256
    assert tok.merges[(256, 97)] == 257
    assert tok.encode(TOY) == [257, 98, 100, 257, 98, 97, 99]
    assert tok.decode(tok.encode(TOY)) == TOY


def test_roundtrip_basic():
    tok = _trained_basic()
    for text in ROUNDTRIP_TEXTS:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"roundtrip 失败: {text!r}"
        assert all(0 <= i < 512 for i in ids)


def test_roundtrip_regex():
    tok = _trained_regex()
    for text in ROUNDTRIP_TEXTS:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"roundtrip 失败: {text!r}"
        assert all(0 <= i < 512 for i in ids)


def test_regex_respects_chunk_boundaries():
    """跨块不合并。GPT-2 正则把「空格+词」切成一块（` ?\\p{L}+`），
    所以 " ab" 可以整体成为 token，但块边界的字节对（如 b 与下一个块的 a）永不相并。
    """
    tok = RegexTokenizer()
    text = "ab ab ab ab"
    tok.train(text, vocab_size=259)
    assert (97, 98) in tok.merges  # 块内对 "ab"
    # 第一个 "ab" 无前导空格 → 单独成 256；后续 3 个 " ab"（含空格块内对 (32,256)）→ 257
    assert tok.encode(text) == [tok.merges[(97, 98)]] + [tok.merges[(32, 256)]] * 3
    # 关键性质：跨块字节对（前块尾 b=98 与后块首 a=97）绝不出现在 merges 中
    assert (98, 97) not in tok.merges
    assert tok.decode(tok.encode(text)) == text


def test_bpe_compresses_training_corpus():
    tok = _trained_basic()
    text = DATA.read_text(encoding="utf-8")[:200_000]
    raw_bytes = len(text.encode("utf-8"))
    token_count = len(tok.encode(text))
    assert token_count < 0.6 * raw_bytes, (
        f"压缩不足: {token_count} tokens vs {raw_bytes} bytes"
    )


def test_vocab_and_merge_consistency():
    tok = _trained_basic()
    assert len(tok.merges) == 512 - 256
    for (a, b), new_id in tok.merges.items():
        assert tok.vocab[new_id] == tok.vocab[a] + tok.vocab[b]
