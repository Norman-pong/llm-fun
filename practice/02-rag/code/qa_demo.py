"""P2 demo：建库 → 评测 → 交互问答（无 key 时返回纯检索结果）。

    uv run python practice/02-rag/code/qa_demo.py            # 评测 + 示例问题
    uv run python practice/02-rag/code/qa_demo.py "什么是因果掩码"   # 指定问题
"""

from __future__ import annotations

import sys

from rag import RagIndex, answer, evaluate, load_cards


def main() -> None:
    idx = RagIndex()
    idx.build(load_cards())
    print(f"知识库就绪：{idx.collection.count()} 张卡片")

    m = evaluate(idx)
    print(f"评测（n={m['n']}）：hit@1 {m['hit@1']:.2%} | hit@3 {m['hit@3']:.2%}")

    question = sys.argv[1] if len(sys.argv) > 1 else "为什么点积要除以根号 d？"
    print(f"\n问题：{question}")
    out = answer(idx, question, generate=True)
    for r in out["retrieved"]:
        print(f"  [{r['cid']}] sim={r['sim']:.3f}  {r['text'][:60]}...")
    print(f"\n回答：{out['answer']}")


if __name__ == "__main__":
    main()
