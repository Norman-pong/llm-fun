# T5 预训练：莎士比亚 mini-GPT

## 学习目标

时间盒：3 周。把 T4 的 GPT 在真实语料上完整训练一遍：LR 调度、梯度裁剪、验证评估、产物落盘——预训练的工程闭环。

**DoD（pytest 验证 + 完整训练实测）：**

- [x] 莎士比亚语料训练 loss 持续下降 → 3000 步完整训练（MPS 27.7 分钟）：train 2.21→0.96，val 最优 **1.5284**（step 1500，ppl 4.6，nanoGPT 参考区间内）；测试用 300 步小配置验证同断言（train 首末段差 >0.5，val 改善 >0.2）
- [x] 生成可读风格文本 → 完整训练采样具备说话人格式/正确拼写/古英语腔调（`KING RICHARD III: What is thy work?`，notes.md 全文）；测试断言常见词命中 ≥5/10
- [x] 记录训练曲线与超参 → `runs/05-pretrain/s3000_d192_L6/history.json`（config/train/val/lr 全量落盘，测试断言结构完整）

超时降级（缩语料/步数）：未触发（测试本身即 300 步降级版；完整训练为 demo 路径）。

## 资源

1. **主线**：[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) ch5 + 附录 D（训练循环增强）——为什么读：warmup/余弦/裁剪的每一步有实验对比。
2. 辅助：[nanoGPT](https://github.com/karpathy/nanoGPT) shakespeare_char 配置——10M 参数量级的标准参照（val ~1.47-1.6）。

## 文件

| 文件 | 内容 |
|---|---|
| `code/train.py` | 训练器：LR 调度/梯度裁剪/val 评估/产物落盘；`--smoke` 小配置 |
| `code/sample.py` | 加载 runs/ 里训好的模型交互：续写/温度对比/列出所有 run |
| `code/plot.py` | 把 runs/ 的 history.json 画成训练曲线 PNG（loss + LR 调度双面板） |
| `code/test_train.py` | DoD 验收（5 项，300 步小配置，CPU ~40s） |
| `input.txt` | tinyshakespeare 1,115,394 字符（来源 karpathy/char-rnn） |

运行完整训练（MPS 约 10 分钟）：

```sh
uv run python theory/05-pretrain/code/train.py            # 3000 步 10.7M 参数
uv run python theory/05-pretrain/code/train.py --steps 500 --smoke   # 快速体验
```

玩训好的模型（先跑过一次训练，产物在 `runs/`）：

```sh
uv run python theory/05-pretrain/code/sample.py --prompt "ROMEO:" --temperatures 0.5 0.8
```

## 复盘（实现取舍与实测）

- 模型与数据都复用前章资产（T4 的 GPTModel、字符级词表），本章增量全部在训练工程件——章节间的复用关系本身就是学习路径。
- 2,731,200 参数 = 6L×192d×6H×ctx256；3000 步 MPS 27.7 分钟（纯 CPU 同配置实测 9.7 小时，MPS 加速 ~15×）。val 最优 1.5284@1500 步、3000 步回升到 1.67——小语料过拟合的活教材，已写入 notes.md 第 3 节。
- 训练产物进 `runs/`（gitignore），仓库只留代码与曲线 JSON 的读取逻辑，保证 clone 后测试可独立复跑。
