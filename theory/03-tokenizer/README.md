# T3 分词器：BPE 从零实现

## 学习目标

时间盒：2 周。吃透 BPE：字节级初始词表、高频对合并、合并优先级编码、GPT-2 正则切块的作用。

**DoD（pytest 验证）：**

- [x] 编码→解码 roundtrip 无损 → Basic/Regex 双分词器在英文/中文/emoji/代码/空串 7 类文本上全部通过（词表 512）
- [x] 与 minbpe basic 行为一致 → 经典语料 `aaabdaaabac` 逐次合并结果与手算一致（含同频平票的确定性规则）；GPT-2 切块语义样例（`" ab"` 整体成 token、跨块对 (98,97) 永不合并）
- [x] BPE 有效压缩 → names.txt 上 228145 字节 → 133745 tokens（58.62%），Basic 版 49.75%

## 资源

1. **主线**：[minBPE](https://github.com/karpathy/minBPE) + 视频 [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)——为什么读：BPE 的所有核心决策（字节级、正则、特殊 token）在这条视频里都有第一性推导。
2. 辅助（按需）：[LLMs-from-scratch ch2 bonus: BPE from scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/05_bonus_bpe-from-scratch)——与本章互为对照。

## 文件

| 文件 | 内容 |
|---|---|
| `code/tokenizer.py` | BasicTokenizer / RegexTokenizer（GPT-2 正则，依赖 `regex` 包） |
| `code/test_tokenizer.py` | DoD 验收（6 项：toy 手算对齐、双 roundtrip、切块边界、压缩率、词表一致性） |
| `code/train_tokenizer.py` | demo：压缩率对比与 token 展示 |

## 复盘（实现取舍）

- 特殊 token（`<|endoftext|>` 等）未实现——预留到需要拼接训练语料的 T8；本章 DoD 不含它。
- 同频合并的平票规则显式化为「pair 字典序」，不依赖 dict 遍历序（minbpe 语义在此规则下可复现）。
- 实测中最有教学价值的对比：Basic 学出 `'an\n'` 跨行 token，Regex 学出 `'anna'` 词内 token——pre-tokenization 的必要性不用讲，看输出就懂（notes.md 第 3 节）。
