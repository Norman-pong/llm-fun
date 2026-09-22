# T2 语言建模直觉：bigram → MLP

## 学习目标

时间盒：3 周。建立「语言模型 = P(下一个字符|前文)」的直觉：先数频次（bigram），
再用嵌入 + MLP（Bengio 2003）体感「泛化能力从哪来」。

**DoD（pytest 验证）：**

- [x] 人名数据集上 MLP dev NLL 低于 bigram 基线 → 实测 MLP **2.3087** vs bigram **2.4647**（差 0.156）
- [x] 采样出可读人名 → `'naz', 'aldi', 'deza', 'jarorsse'` 等名字感强的串，85%+ 落在 2~10 字符
- [x] 笔记含激活/梯度诊断 → `diagnostics()` 输出 tanh 饱和度与各层 grad:weight 比（notes.md 第 4 节）

超时降级（超 4.5 周触发）：未触发（agent 完成制下此机制转为学习者的自学节奏参考）。

## 资源

1. **主线**：[makemore](https://github.com/karpathy/makemore) + [nn-zero-to-hero](https://karpathy.ai/zero-to-hero.html) 第 2 讲（bigram）、第 3-4 讲（MLP 与诊断）——为什么读：同一任务螺旋升级，且第 4 讲的激活/梯度诊断方法是工程上真正常用的调试手段。
2. 辅助（按需）：[d2l-zh](https://zh.d2l.ai/) 语言模型章节。

## 文件

| 文件 | 内容 |
|---|---|
| `code/languagemodel.py` | 数据管线、BigramModel、MLPModel、训练/采样/诊断 |
| `code/test_languagemodel.py` | DoD 验收（4 项，含训练） |
| `code/train_mlp.py` | demo：基线对比 + 训练曲线 + 采样 + 诊断 |
| `names.txt` | makemore 人名数据集（32032 个，来源：karpathy/makemore） |

## 复盘（实现取舍）

- 用 torch 实现 MLP（T1 的手写引擎继续作为 autograd 原理的参考，语言模型章起切 torch——makemore 路线如此，且词表 softmax 的标量实现无教学增益）。
- 测试内训练 6000 步（MPS 约 8 秒）保证采样质量稳定过验收；训练不可复现的坑（MPS generator）已在 notes.md 第 5 节记录。
- 诊断做成 `diagnostics()` 函数而非 notebook 内联代码，让它可测试、可复用。
