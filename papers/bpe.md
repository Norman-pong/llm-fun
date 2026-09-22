# 论文精读：Neural Machine Translation of Rare Words with Subword Units（BPE）

> Sennrich, Haddow, Birch, ACL 2016。绑定章节：T3（theory/03-tokenizer）。

## 一句话贡献

把压缩算法界的 Byte Pair Encoding 引入 NLP，用「子词单元」同时解决机器翻译的罕见词（OOV）问题与词表膨胀问题——此后所有主流 LLM（GPT/Llama/Qwen）的 tokenizer 都是它的直系后代。

## 核心思路

1. 词表的两难：字符级无 OOV 但序列太长；词级序列短但开放词表必爆炸。
2. 本文方案：**在训练语料上统计符号对频率，反复合并最高频对**，得到一个「高频整词成 token、罕见词自动拆成子词」的自适应词表。
3. 关键性质：**编码确定性 + 可无损还原**——任何词要么在词表里，要么能拆成表内子词序列，永无 `<UNK>`。

## 值得注意的设计决策

- 原论文在**词级**运行 BPE（先分词，词内再做 BPE，词边界用 `</w>` 标记）；
  GPT-2 改为**字节级**（本章实现路线，minBPE 同款），彻底消除未知符号——中文/emoji 天然覆盖。
- 合并表的顺序就是优先级：编码时按「最早学到的对先合并」贪心执行（T3 的 `_apply_merges`）。
- 词表大小是超参：论文报告 10k-60k 区间，翻译质量对它不敏感（鲁棒性好），但推理成本线性受益。

## 与本仓库实现的对照

| 论文概念 | T3 实现 |
|---|---|
| 统计符号对频率 | `get_stats`（Counter） |
| 合并最高频对 | `train` 的 `max(stats, key=(count, pair))` |
| 词边界标记 `</w>` | GPT-2 正则 pre-tokenization（`RegexTokenizer` 切块即词边界） |
| 罕见词拆子词 | 字节级兜底：任何 UTF-8 字节序列都可表示 |

## 实验证据（论文）

英德/英法翻译：BPE 使 OOV 从 ~4-5% 降到 ~0%，BLEU 提升约 1；子词切分对罕见专有名词的音译尤其有效。

## 批判性视角

- BPE 偏向频率而非语义：`girlfriend` 可能拆成 `girl`+`friend`（幸运）也可能拆成无意义碎片（不幸）。
- 后继改进：Unigram LM（SentencePiece 的另一种算法，概率模型选词表）、WordPiece（BERT，选对似然提升最大的子词）——但字节级 BPE 仍是 LLM 时代的事实标准。
