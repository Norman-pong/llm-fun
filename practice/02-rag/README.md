# P2 RAG：知识库问答

## 学习目标

时间盒：4 周。端到端打通 RAG 管线：切块 → 本地 embedding → 向量检索 → 评测 →（可选）生成。

**DoD（pytest 验证，全本地零 key）：**

- [x] 端到端检索问答可跑 → Chroma 嵌入式（零服务）+ 从零 TF-IDF embedding（零 API key），24 张知识卡片入库，`qa_demo.py` 一键演示
- [x] ≥20 条评测集 hit-rate 有数字 → 24 问（含同义改写）：**hit@1 = 91.67%，hit@3 = 100%**
- [x] 回答生成走 P1 客户端，无 key 优雅跳过 → `answer()` 降级返回检索结果，测试覆盖

超时降级（砍 rerank 保闭环）：未触发。

## 资源

1. **主线**：[rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)（Part 1-9）——为什么读：LangChain 官方逐级搭建，与本章组件一一对应（切块→embedding→检索→生成）。
2. 辅助（按需）：[NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)——42 种进阶技巧卡片（rerank/混合检索等，backlog 方向）。

## 文件

| 文件 | 内容 |
|---|---|
| `code/embedder.py` | 从零 TF-IDF（中英混合分词 + L2 归一化） |
| `code/rag.py` | 切块器、Chroma 索引、24 问评测集、answer() 生成环节 |
| `code/knowledge_cards.md` | 检索语料：24 张 LLM 知识卡片（内容取自 T1-T5/P1 主题） |
| `code/test_rag.py` | DoD 验收（9 项） |
| `code/qa_demo.py` | demo：`uv run python practice/02-rag/code/qa_demo.py "问题"` |

## 复盘（实现取舍与实测）

- embedding 选从零 TF-IDF 而非调 API/下模型：学习目标是理解「文本→可比较向量」的本质，且全链路可测；P4 backlog 做神经 embedding 对照（hit@1 差距即改进空间，两处 miss 均为字面匹配局限）。
- 语料用本仓库学过的知识做卡片——学习闭环：RAG 检索的内容就是理论线的产出。
- 先评测后优化：hit-rate 与生成解耦，检索层可独立迭代（chunk 大小/top-k/ embedding 各是一个控制变量）。
