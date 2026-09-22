"""P1 prompt 对照实验：同一问题 × 3 种 prompt 变体，输出结果对照表。

mock 测试见 test_client.py；真实运行（需要 .env 里的 API key）：

    uv run python practice/01-api/code/prompt_experiments.py "什么是注意力机制"
"""

from __future__ import annotations

import sys
import time

from client import LLMClient

# 变体设计：一次只改一个维度，才能把差异归因到该维度
VARIANTS = {
    "baseline": "回答问题：{q}",
    "concise": "用一句话回答问题，不超过 40 字：{q}",
    "structured": "分点回答问题，每点一行，最多 3 点：{q}",
}


def run_experiments(client: LLMClient, question: str) -> list[dict]:
    rows = []
    for name, template in VARIANTS.items():
        t0 = time.perf_counter()
        reply = client.chat([{"role": "user", "content": template.format(q=question)}])
        rows.append(
            {
                "variant": name,
                "prompt": template.format(q=question),
                "reply": reply,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            }
        )
    return rows


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "什么是注意力机制"
    try:
        client = LLMClient()
    except Exception as e:  # noqa: BLE001 - 给出可操作的指引而非栈回溯
        print(f"[跳过] {e}")
        sys.exit(1)
    rows = run_experiments(client, question)
    print(f"\n问题：{question}\n模型：{client.config.model}\n")
    for r in rows:
        print(f"--- [{r['variant']}] ({r['latency_ms']} ms) ---")
        print(f"prompt: {r['prompt']}")
        print(f"reply : {r['reply']}\n")


if __name__ == "__main__":
    main()
