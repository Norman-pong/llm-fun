# T1 autograd 与反向传播

## 学习目标

时间盒：2 周。吃透「反向传播 = 链式法则在计算图上的遍历」，不依赖 PyTorch 的 autograd 就能训练一个小网络。

**DoD（pytest 验证，达成才算完成）：**

- [ ] 手写 autograd 引擎与 `torch.autograd` 在 ≥10 个随机复合函数上梯度对齐（float32, atol=1e-5）
- [ ] 用手写引擎训练一个小 MLP，loss 单调下降

超时降级（超 3 周触发）：精读 micrograd 源码 + 写笔记，实现仅做到标量加/乘的反传。

## 资源

1. **主线**：[micrograd](https://github.com/karpathy/micrograd)（~100 行标量 autograd 引擎）+ 视频 [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)——为什么读：用最小代码量证明 backprop 本质，是建立直觉的最快路径。
2. 辅助（按需）：[d2l-zh](https://zh.d2l.ai/) 前几章——底座概念（张量/梯度下降）有缺口时才查，不通读。

## 复盘

（完成后填：学到什么、踩坑、还想深入的分支）
