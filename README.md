# llm-fun

个人大模型学习参考仓库 = **参考答案库 + 验收测试库**。理论线从零实现（autograd → 语言模型 → 分词器 → GPT → 预训练），实践线做应用工程（API → RAG）。

## 学习：一句话进入教练模式

装好环境（见下节）后，在本仓库目录打开任意兼容 AGENTS.md 的 agent（ZCode / Claude Code 等），说一句：

> 我是学习者，想学 t1

agent 会按 [TUTOR.md](TUTOR.md) 扮演教研员陪跑：**预习引导 → 陪你在仓库外自写 → 章内测试验收 → 对照复盘**，不泄参考答案、不动教材与参考实现、零 git 提交（可帮你维护笔记、跑演示小脚本）。agent 没自动读 AGENTS.md 时，把 TUTOR.md 内容贴给它即可。

## 准备环境（一次性）

```sh
# 1. 安装 uv（Python 3.12 由 .python-version 钉定）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆并同步依赖
git clone https://github.com/Norman-pong/llm-fun.git && cd llm-fun
uv sync --frozen
```

可选（仅 P1/P2 真实 API 调用需要）：`cp .env.example .env` 填 API key；不填则相关测试走本地/mock，不影响学习。

## 章节

| Tag | 章 | 状态 |
|---|---|---|
| `t1` | [T1 autograd 与反向传播](theory/01-autograd/) | ✅ |
| `t2` | [T2 语言建模直觉（bigram→MLP）](theory/02-languagemodel/) | ✅ |
| `t3` | [T3 分词器（BPE）](theory/03-tokenizer/) | ✅ |
| `t4` | [T4 Attention 与 GPT](theory/04-gpt/) | ✅ |
| `t5` | [T5 预训练（莎士比亚 mini-GPT）](theory/05-pretrain/) | ✅ |
| `p1` | [P1 API 与 Prompt](practice/01-api/) | ✅ |
| `p2` | [P2 RAG（知识库问答）](practice/02-rag/) | ✅ |

论文精读：[BPE](papers/bpe.md)（T3）、[Attention is All You Need](papers/attention-is-all-you-need.md)（T4）、[GPT-2](papers/gpt-2.md)（T4/5）。

> 每章完成打一个 tag，`git checkout t2` 可看该阶段完整快照。设计依据与 backlog（T6-T8、P3-P5）见 [DESIGN.md](DESIGN.md)。
