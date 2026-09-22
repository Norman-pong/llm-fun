# T4 Attention 与 GPT

## 学习目标

时间盒：4 周（理论线核心章）。手写多头因果注意力并数值对齐官方实现，组装完整 GPT 并端到端训练采样。

**DoD（pytest 验证）：**

- [x] 手写 multi-head attention 与 PyTorch 官方 `F.scaled_dot_product_attention(is_causal=True)` 数值对齐（atol=1e-5）
- [x] GPT 前向 shape/数值检查通过 → logits [B,T,V]、初始 loss≈ln(vocab)（<±0.8）、eval 确定性、权重共享生效
- [x] 能采样生成 → 低温 top-k 采样 id 合法且复现训练模式（'abc ' 循环语料，`abcabc...` 连续 ≥4 段）
- [x] 附加：因果性检查（篡改未来位置不改变过去输出）、过拟合测试（重复语料 loss 3.4→<0.2）

超时降级（超 6 周）：未触发。

## 资源

1. **主线**：[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) ch3-4——为什么读：注意力从单头逐步推导到多头，每步有数值验证，与本章测试互为对照。
2. 辅助：[build-nanogpt](https://github.com/karpathy/build-nanogpt) 讲座（3.5h 视频）——第一性视角讲 GPT-2 每个部件的「为什么」，尤其初始化与残差缩放。

## 文件

| 文件 | 内容 |
|---|---|
| `code/gpt.py` | CausalSelfAttention（手写 attend + SDPA 可对齐）、FeedForward、TransformerBlock（Pre-LN）、GPTModel（权重共享 + GPT-2 初始化）、generate（temperature/top-k） |
| `code/test_gpt.py` | DoD 验收（6 项） |
| `code/demo_gpt.py` | demo：训练前后采样对比 + 参数量验算 |

## 复盘（实现取舍与实测）

- 实测两个高价值坑并固化为回归测试：**恒等捷径**（权重共享 + target 未移位 → 未训练 loss≈0，正确约定是位置 t 预测 t+1）与**默认初始化初始 loss=38**（GPT-2 的 std=0.02 + 残差投影 1/√(2·layer) 缩放解决）。
- demo：344,544 参数小 GPT，3640 字符语料训 300 步 loss 3.42→0.04，采样从字符噪声变为流畅复现 `the quick brown fox...` 短语——「GPT 能 work」的最小证据。
- 参数量速算 12·L·d² + vocab·d 验算 GPT-2 124M ✓（notes.md 第 2 节）。
