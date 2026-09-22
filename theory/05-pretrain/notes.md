# T5 讲解笔记：预训练——把 T4 的 GPT 认真训一次

## 问题

- T4 的 300 步 toy 训练和「真正的预训练」差在哪？
- LR 调度、梯度裁剪这些工程件为什么必要？

## 1. 任务设定

语料 tinyshakespeare（1,115,394 字符，65 词表字符级），90/10 切 train/val。
模型：6 层 × 192 维 × 6 头 × ctx 256 = **2,731,200 参数**（速算：12·L·d² = 2.65M
+ 词嵌入 65·192 + 位置嵌入 256·192 ≈ 0.08M）——Mac MPS 半小时内可训完。
目标：val loss 进入 1.5-1.7（nanoGPT 同量级参照 1.47-1.6），生成文本具备
「剧本格式 + 拼写正确的英文词」。

## 2. 相对 T4 新增的三个工程件（rasbt 附录 D）

### LR 调度：warmup + 余弦衰减

```
lr(step) = peak × (step+1)/warmup        step < 100（线性热身）
         = peak × (0.1 + 0.45(1+cos(π·progress)))   之后（衰减到 10%）
```

- **为什么 warmup**：AdamW 的二阶动量方差估计在头几百步不可靠，初始大步长
  容易把参数踢进坏区域（T4 见过初始 loss=38 的教训——初始化敏感的模型对 lr 同样敏感）。
- **为什么余弦衰减**：小数据集（1M 字符 × 3000 步 ≈ 每个 token 被看 ~50 次）
  后期必须减小步长才能把 loss 磨下去；衰减到 10% 而非 0，保留微弱探索。

### 梯度裁剪（clip_grad_norm_, max=1.0）

个别 batch 的异常大梯度（数据里的罕见片段）会把参数打飞，loss 出现尖峰。
按全局范数缩放到 1.0 是保险丝——正常步不受影响（多数步范数 <1），
异常步被硬截断。这是几乎所有 LLM 训练脚本的标配。

### 定期 val 评估 + 产物落盘

每 500 步在验证集上评估（20 个 batch 平均），训练完把
`model.pt + history.json（超参/曲线）` 存进 `runs/05-pretrain/`（已 gitignore）。
**可复现性 = 固定 seed + 记录配置**，这是实验管理的最小闭环。

## 3. 训练配置与实测（MPS 实跑 1661s ≈ 27.7 分钟）

| 项 | 值 | 说明 |
|---|---|---|
| steps | 3000 | 每 token 平均被看过约 50 遍 |
| batch | 64 × 256 ctx | 每步 16K token |
| peak lr | 1e-3 | AdamW（β2=0.95，GPT-3 系配方）、wd=0.1 |
| 设备 | MPS | 27.7 分钟；同配置纯 CPU 约 **9.7 小时**（实测基准，MPS 加速 ~15×） |

实测曲线（`runs/05-pretrain/s3000_d192_L6/history.json`）：

```
step  500 | train 2.210 | val 1.819
step 1000 | train 1.467 | val 1.567
step 1500 | train 1.267 | val 1.528   ← val 最优点
step 2000 | train 1.142 | val 1.548   ← val 开始回升
step 3000 | train 0.960 | val 1.671   ← 典型过拟合
```

**最有价值的发现：过拟合曲线。** train loss 一路降到 0.96，val 却在 1500 步后
回升——1M 字符的小语料，3000 步意味着每个 token 被看约 50 遍，模型开始
「背诵」训练集。早停点即 1500 步（val 1.528，ppl 4.6），已在 nanoGPT 参考区间。
这解释了为什么真实预训练要万亿 token：不是数据多牛逼，而是**不让模型
在收敛前就把数据背完**（每个 token 只被看 1-2 遍，Chinchilla 配比）。

采样样本（temperature=0.8，模型权重为 3000 步版本）：

```
KING RICHARD III:
What is thy work?

DUKE OF AUMERLE:
Then comes this commonwealth make men my heart,
For I will entreat you then I'll lay the cousin.
```

说话人格式、拼写、古英语腔调全部成立——2.7M 参数、28 分钟训练即可达成。

### 设备选型结论（回应「能不能用 macOS 工具链」）

- **PyTorch MPS（Metal GPU）**：理论线全程默认启用（`torch.backends.mps.is_available()`），
  实测 15× 加速。代码与 CUDA 路径一致，学到的东西可直接迁移云上 GPU。
- **MLX**（Apple 原生框架）：同等任务通常再快 10-30%，但 API 是另一套——
  已在 DESIGN.md 第 6 节保留给 backlog P4（微调）使用，理论线不切换。

## 4. 为什么 val loss ≈ 1.53 值得停下脚思考

字符级 ppl 4.6 意味着：每个字符位置，模型把不确定性压缩到 5 选 1 以内。
一个 2.7M 参数、训了 28 分钟的模型已经「学会」了拼写、大小写、剧本格式、
说话人轮替。这就是 scaling 故事的起点：同样的配方，参数 ×1000、
数据 ×10000（每个 token 只看 1-2 遍而非 50 遍）、步数 ×1000，
就是 GPT-2 → GPT-3 → GPT-4 的路径（Chinchilla 定律决定参数/数据配比）。

## 5. 踩坑

- 下载 tinyshakespeare：nanoGPT 仓库 data/ 里只有 prepare.py 没有数据本体，
  原始出处是 char-rnn 的 `data/tinyshakespeare/input.txt`（README 里的源）。
- 跨章 import（T5 的 train.py 用 T4 的 gpt.py）：用
  `sys.path.insert(0, <04-gpt/code>)` 显式注入，测试与脚本同样处理——
  学习仓库没有 src/ 包，这是最轻的可读方案。
- MPS 上长时间训练注意内存压力：本配置峰值 ~1.5GB 统一内存，安全；
  更大模型前先查 Activity Monitor。

## 6. 理论线到此的完整拼图

T1 autograd（梯度从哪来）→ T2 语言建模（预测下一字符的本质）→
T3 分词器（文本变 id）→ T4 GPT 架构（注意力+残差堆叠）→ **T5 预训练**（工程闭环）。
从零到「自己训出一个会写莎士比亚的小模型」——这就是理论线的承诺范围，
后续 backlog（T6-T8）都建立在这块地基上。
