# P1 讲解笔记：API 客户端与 Prompt 工程

## 问题

- OpenAI SDK 替你做了哪些事？自己封装一遍能学到什么？
- 重试策略为什么分「可重试/不可重试」两类错误？
- 让模型稳定输出 JSON 有哪些坑？

## 1. 客户端封装的三个核心件（client.py）

### 配置：密钥只走环境变量

`load_config()` 用 python-dotenv 读 `.env`（OPENAI_API_KEY / OPENAI_BASE_URL /
OPENAI_MODEL），key 缺失时抛**带操作指引**的错误（复制 .env.example → 填 key），
不是裸 KeyError。`OPENAI_BASE_URL` 一个变量即可切换任何 OpenAI 兼容服务
（中转、本地 llama.cpp server、vLLM）——这就是「兼容协议」的价值。

### 重试：指数退避 + 错误分类

| 错误类型 | 例子 | 策略 | 原因 |
|---|---|---|---|
| 可重试 | 429 限流、5xx、网络超时 | 指数退避（0.5s→1s→2s），最多 3 次 | 瞬态故障，等一下就好 |
| 不可重试 | 401 key 错、400 参数错 | 立即失败 | 重试一万次也不会成功，反而烧配额/打日志 |

实现要点：退避用 `base * 2^attempt`；`max_retries` 是「额外重试次数」不是总次数；
每次重试前 sleep（测试里把 backoff_base 设 0 保持飞快）。

### 结构化输出：JSON mode + 修复重试

`chat_json()` 两层防御：
1. 请求带 `response_format: {"type": "json_object"}`——服务端会约束模型输出 JSON；
2. 本地 `json.loads` 解析，失败则把模型的错误输出拼回对话，追加一条
   「请只输出合法 JSON」再给一次机会。

第二层不能省：JSON mode 只是强烈倾向而非保证（截断、代码块包裹 ```json 都会出现）。
**「给模型一次自我修正机会」是 prompt 工程的通用模式**，比直接报错优雅得多。

## 2. Prompt 对照实验的方法论（prompt_experiments.py）

三种变体一次只改一个维度：

| 变体 | 模板 | 检验什么 |
|---|---|---|
| baseline | 回答问题：{q} | 无约束的默认行为 |
| concise | 用一句话回答，不超过 40 字 | 长度约束的服从性 |
| structured | 分点回答，最多 3 点 | 结构约束的服从性 |

记录 reply + latency_ms 输出对照表。mock 测试验证框架逻辑；
真实实验只需 `uv run python practice/01-api/code/prompt_experiments.py "问题"`
（需 `.env` 有 key）。

方法论核心：**控制变量**。如果同时改角色设定+格式+长度，效果好了你不知道
归因给谁。这也是后续 P2 评测（hit-rate 对比不同检索参数）的同一套思路。

## 3. 测试策略：httpx.MockTransport

所有 11 项测试零网络、零 key：注入 `httpx.MockTransport(handler)`，
handler 按调用次数返回预设响应（前两次 429、第三次 200），
精确断言重试次数、请求体里的 `response_format`、错误消息内容。

这是 API 客户端的标准测法——不 mock 会带来慢测试、依赖外部服务、
key 泄漏进 CI 三重问题。

## 4. 踩坑

- httpx 的 `Response(429, text=...)` 与 `Response(200, content=bytes)` 构造参数不同（text 接受 str，content 接受 bytes），混用会踩类型错。
- `json_mode` 下模型仍可能返回空串，客户端必须显式判空（`chat` 里已处理）。
- 测试要 monkeypatch 删掉真实环境里的 OPENAI_API_KEY，否则本机有 key 时
  「缺失报错」用例会假绿。

## 5. 与后续章节衔接

P2 的 RAG 生成环节直接复用本章 `LLMClient`（检索是本地的，生成走 API）；
P3 agent 的 tool-call 循环 = chat(messages) + 解析结构化输出 + 拼回 messages，
两个积木都来自本章。
