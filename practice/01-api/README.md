# P1 API 与 Prompt 工程

## 学习目标

时间盒：2 周。封装一个生产可用的 OpenAI 兼容客户端（.env 配置/重试/结构化输出），
并建立 prompt 对照实验的方法论。

**DoD（pytest 验证，全部 mock 零网络）：**

- [x] OpenAI 兼容客户端 → `LLMClient`：.env 加载（缺失时清晰报错）、429/5xx/网络错误指数退避重试（4xx 不重试）、JSON mode + 修复重试的结构化输出
- [x] ≥3 组 prompt 对照实验 → baseline/concise/structured 三变体框架 + 结果表输出；mock 测试验证框架，真实运行待 `.env` key（脚本已就绪）

## 资源

1. **主线**：[llm-cookbook](https://github.com/datawhalechina/llm-cookbook) 必修一《ChatGPT Prompt Engineering for Developers》中文版——为什么读：吴恩达官方内容的中文对照，prompt 原则（清晰/结构化/迭代）与本章实验框架互补。
2. 辅助（按需）：[anthropics/courses](https://github.com/anthropics/courses) 的 prompt_evaluations——prompt 评测视角。

## 文件

| 文件 | 内容 |
|---|---|
| `code/client.py` | LLMConfig/LLMClient：配置加载、分类重试、chat_json 修复重试 |
| `code/prompt_experiments.py` | 3 变体对照实验：`uv run python practice/01-api/code/prompt_experiments.py "问题"` |
| `code/test_client.py` | DoD 验收（11 项，httpx.MockTransport） |

## 复盘（实现取舍）

- 不用 openai SDK、直接 httpx 封装：把 SDK 的魔法（鉴权/重试/解析）摊开写一遍是本章的学习目标；换兼容服务只需改 base_url。
- 重试分类（429/5xx/网络 → 退避；4xx → 立即失败）与「给模型一次 JSON 修复机会」是两个可迁移的工程模式。
- 真实 API 调用属学习者侧（.env 配 key 后运行实验脚本），代码正确性由 mock 测试全量覆盖——与 DESIGN.md 3.1 的 API 依赖备注一致。
