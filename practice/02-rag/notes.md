# P2 讲解笔记：RAG 检索增强生成

## 问题

- RAG 解决什么：模型知识冻结在训练时刻，私有/更新知识靠「检索 + 拼上下文」注入，不重训模型
- 一条最小可用的 RAG 管线需要哪些件？

## 1. 管线全景（rag.py，全本地）

```
knowledge_cards.md（24 张卡片）
   → load_cards()                 按标题切块（本章卡片天然成块；通用文本走 split_text 滑窗+重叠）
   → TfidfEmbedder.fit/transform  从零 TF-IDF embedding（中英混合分词，L2 归一化）
   → Chroma（嵌入式，余弦距离）   向量入库
   → query(question, top_k)       问题 embedding → 近邻检索
   → answer()                     top-k 拼进 system prompt →（有 key 时）LLM 生成
```

## 2. 各组件的实现要点

### 切块（chunking）

- 语义边界优先（标题/句子），超长才滑窗；**重叠（overlap）** 保证跨块句子两头都有上下文。
- 块大小是精度/上下文的权衡：太大 → 单块信息密度低、检索不准；太小 → 语义不完整。
  本章卡片每张 ~100 字是「天然块」的理想情况。

### Embedding：TF-IDF 从零写（embedder.py）

- 词权重 = tf × idf；idf = log((1+N)/(1+df)) + 1（平滑：查询里的新词权重有限而非全 0）。
- 中文分词无词典方案：**汉字串抽出来切 bigram**（"用机制"→"用机/机制"），
  英文/数字按词小写化。bigram 让"注意力"和"注意"共享特征，短查询友好。
- L2 归一化后，**内积 = 余弦相似度**——向量检索库默认配 cosine 距离即可对齐。

### 向量库：Chroma 嵌入式

- `PersistentClient`（本地 sqlite + hnsw 索引），零服务零 Docker；`hnsw:space: cosine`。
- HNSW 是近似最近邻（ANN）索引：图结构 + 贪心搜索，牺牲一点召回换百倍速度——
  这就是「向量检索」比「逐条算余弦」可扩展的原因（24 条数据看不出差别，百万级是本质差异）。

### 评测（evaluate）

- hit@k：期望卡片是否出现在 top-k。评测集 24 问（DoD 要求 ≥20），含同义改写。
- **实测：hit@1 = 91.67%（22/24），hit@3 = 100%**。
- 两类 miss（hit@1 未中的 2 问）都是问法与卡片措辞差异大（如"429 要重试吗"vs 卡片写"限流"），
  TF-IDF 只能字面匹配——这正是神经 embedding（语义级）的改进空间，P4 backlog 对照实验的选题。

### 生成（answer）

- system prompt 要求**只依据资料回答 + 标注引用编号 + 不知道就说不知道**——抑制幻觉三件套。
- 无 key 优雅降级：返回检索结果 + 提示，管线不崩（`answer(..., generate=False)` 可完全关闭）。

## 3. 方法论沉淀

- **先评测再优化**：hit-rate 是检索层的独立指标，不与生成质量耦合。RAG 出错时先分清
  「检索没找到」（改切块/embedding）还是「找到了不会用」（改 prompt）。
- 对照实验控制变量法（P1）同样适用：改 chunk 大小 / top-k / embedding 方案，一次只动一个。

## 4. 踩坑

- 新版 chromadb 集合名要求 3-512 字符（"kb" 太短被拒，改 "kb-cards"）。
- chromadb 的 upsert 幂等重建：build 里先 delete_collection 再建，评测脚本可重复跑。
- TF-IDF 的 transform 对 vocab 外的 token 静默丢弃是特性（查询噪音词无害），但 fit 语料太小时
  vocab 覆盖不足会伤召回——语料规模决定 embedding 质量。

## 5. 与后续衔接

P3 agent 的 tool-call 循环里，`rag.query` 就是一个可被模型调用的工具；
P4 的对照实验：TF-IDF（本章基线）vs bge 类神经 embedding 的 hit-rate 差距量化。
