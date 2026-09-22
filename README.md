# llm-fun

个人大模型学习参考仓库：理论线从零实现（autograd → 语言模型 → GPT → 预训练），实践线做应用工程（API → RAG）。每章 = 参考实现 + DoD 验收测试 + 讲解笔记。

> 设计依据、章节 DoD、算力策略、参考项目全景见 **[DESIGN.md](DESIGN.md)**。

## 如何学习（tag 导航）

每章完成时打一个 tag，`git tag` 查看全部；`git checkout t2` 查看该阶段完整快照（看完 `git checkout main` 回来），或直接浏览章节目录。

推荐学法（看懂参考 ≠ 学会，动手才算）：

1. 看该章 README 的「资源」（主线视频/仓库）
2. **合上参考，自己从零写**一遍
3. 用章内 `test_*.py` 验收自己的实现（这就是「学懂了」的判据）
4. 对照参考实现查漏，读 notes.md 的推导与踩坑

## 章节状态（实现进度，完成即打 tag）

| Tag | 章 | 状态 |
|---|---|---|
| `t1` | [T1 autograd 与反向传播](theory/01-autograd/) | ✅ |
| `t2` | [T2 语言建模直觉（bigram→MLP）](theory/02-languagemodel/) | ✅ |
| `t3` | [T3 分词器（BPE）](theory/03-tokenizer/) | ✅ |
| `t4` | [T4 Attention 与 GPT](theory/04-gpt/) | ✅ |
| `t5` | [T5 预训练（莎士比亚 mini-GPT）](theory/05-pretrain/) | ✅ |
| `p1` | [P1 API 与 Prompt](practice/01-api/) | ✅ |
| `p2` | [P2 RAG（知识库问答）](practice/02-rag/) | ✅ |

论文精读（3 篇，已交付）：[BPE](papers/bpe.md)（T3）、[Attention is All You Need](papers/attention-is-all-you-need.md)（T4）、[GPT-2](papers/gpt-2.md)（T4/5）。

## 快速开始

```sh
uv sync --frozen                 # 安装环境（Python 3.12，uv 管理，已配阿里云镜像）
cp .env.example .env             # 填 API key（P1/P2 的真实调用用；不填则相关测试走本地/mock）
uv run pytest                    # 跑全部 DoD 验收测试
uv run python theory/01-autograd/code/mlp_demo.py   # 跑某一章 demo（各章 README 有清单）
```

> Backlog（T6-T8、P3-P5）见 DESIGN.md 3.2 节，触发条件到了才升级。
