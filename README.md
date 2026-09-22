# llm-fun

个人大模型学习仓库：理论线从零实现（autograd → 语言模型 → GPT → 预训练），实践线做应用工程（API → RAG）。产出 = 笔记 + 可运行可测试的代码。

> 设计依据、章节 DoD、算力策略、参考项目全景见 **[DESIGN.md](DESIGN.md)**（后续迭代先读它）。

## 快速开始

```sh
uv sync --frozen                 # 安装环境（Python 3.12，uv 管理）
cp .env.example .env             # 填入 API key（实践线章节用，不进 git）
uv run pytest                    # 跑全部测试
uv run python theory/01-autograd/code/engine.py   # 跑某一章代码（示例）
```

## 学习进度（唯一进度源，完成即打勾）

### 理论线

- [x] [T1 autograd 与反向传播](theory/01-autograd/)
- [ ] T2 语言建模直觉（makemore）
- [ ] T3 分词器（BPE）
- [ ] T4 Attention 与 GPT
- [ ] T5 预训练（莎士比亚 mini-GPT）

### 实践线

- [ ] P1 API 与 Prompt
- [ ] P2 RAG（知识库问答 demo）

### 论文精读

- [ ] BPE（T3 时）
- [ ] Attention is All You Need（T4 时）
- [ ] GPT-2（T4/5 时）

> Backlog（T6-T8 现代架构/微调/毕业项目，P3-P5 Agent/微调实践/部署评测）见 DESIGN.md 第 3.2 节，触发条件到了才升级。
