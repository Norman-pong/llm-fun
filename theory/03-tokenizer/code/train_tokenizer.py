"""T3 demo：在 names.txt 上训练 BPE，看压缩率与学到的 token。

uv run python theory/03-tokenizer/code/train_tokenizer.py
"""

from __future__ import annotations

from pathlib import Path

from tokenizer import BasicTokenizer, RegexTokenizer

DATA = Path(__file__).resolve().parent.parent.parent / "02-languagemodel" / "names.txt"
VOCAB_SIZE = 512


def report(name: str, tok: BasicTokenizer | RegexTokenizer, samples: list[str]) -> None:
    text = DATA.read_text(encoding="utf-8")
    raw = len(text.encode("utf-8"))
    ids = tok.encode(text)
    print(f"\n[{name}] 词表 {VOCAB_SIZE}")
    print(f"  语料 {raw} 字节 -> {len(ids)} tokens（压缩率 {len(ids) / raw:.2%}）")
    for s in samples:
        enc = tok.encode(s)
        print(f"  {s!r} -> {enc}")
        assert tok.decode(enc) == s
    # 展示一部分学到的多字节 token
    multi = [(i, v) for i, v in sorted(tok.vocab.items()) if len(v) > 2][:15]
    print(
        "  学到的长 token 示例:",
        [v.decode("utf-8", errors="replace") for _, v in multi],
    )


def main() -> None:
    text = DATA.read_text(encoding="utf-8")
    basic = BasicTokenizer()
    basic.train(text, VOCAB_SIZE)
    report("BasicTokenizer（原始字节流）", basic, ["emma", "kaiden", "zhiling"])

    rx = RegexTokenizer()
    rx.train(text, VOCAB_SIZE)
    report("RegexTokenizer（GPT-2 切块）", rx, ["emma", "kaiden", "你好世界"])


if __name__ == "__main__":
    main()
