# T1 autograd 与反向传播

## 学习目标

时间盒：2 周。吃透「反向传播 = 链式法则在计算图上的遍历」，不依赖 PyTorch 的 autograd 就能训练一个小网络。

**DoD（pytest 验证，达成才算完成）：**

- [x] 手写 autograd 引擎与 `torch.autograd` 在 ≥10 个随机复合函数上梯度对齐（float32, atol=1e-5）→ 实测 12 个复合函数 + 13 个单算子全过（`code/test_engine.py`）
- [x] 用手写引擎训练一个小 MLP，loss 单调下降 → XOR 分类，loss 1.133→0.164 单调降，准确率 96%（`code/mlp_demo.py`）

超时降级（超 3 周触发）：未触发。

## 资源

1. **主线**：[micrograd](https://github.com/karpathy/micrograd)（~100 行标量 autograd 引擎）+ 视频 [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)——为什么读：用最小代码量证明 backprop 本质，是建立直觉的最快路径。
2. 辅助（按需）：[d2l-zh](https://zh.d2l.ai/) 前几章——底座概念（张量/梯度下降）有缺口时才查，不通读。

## 文件

| 文件 | 内容 |
|---|---|
| `code/engine.py` | 手写 autograd 引擎（Value/Neuron/Layer/MLP，仅标准库） |
| `code/test_engine.py` | DoD 验证：梯度对齐 + MLP 单调收敛（26 项） |
| `code/mlp_demo.py` | XOR 训练 demo：`uv run python theory/01-autograd/code/mlp_demo.py` |
| `notes.md` | 推导、算子导数表、实验数据、踩坑 |

## 复盘

- **反向传播拆开就是三件事**：建图（DAG）、排序（拓扑序）、累加（链式法则）。`grad += ` 而非 `=` 是因为一个节点可被多条路径引用。
- 对齐测试抓到两个真实差异：torch 对未参与图的叶子梯度是 `None`（本引擎记 0）；float32 vs float64 的误差要靠控制表达式量级才能稳过 atol=1e-5。
- 全批 GD + 小 lr 才有严格单调的 loss 曲线，SGD 小批必有噪声——DoD 的「单调下降」选了全批是有意为之。
- 产出即验收：`uv run pytest theory/01-autograd/` 26 项全绿。
