"""T1 DoD 验证（DoD 定义见 DESIGN.md 3.1 / 章 README）。

1) 手写 autograd 与 torch.autograd 在 12 个随机复合函数上梯度对齐（float32, atol=1e-5）
2) 每个算子单独与 torch 对齐（含除法/指数/r 系操作）
3) 手写引擎训小 MLP，loss 单调下降且收敛

复合函数用「表达式 AST」生成：同一棵树分别交给手写引擎和 torch 求值，
保证两边算的是完全相同的函数，杜绝镜像误差。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from itertools import pairwise

import pytest
import torch
from engine import Value
from mlp_demo import train_mlp

# ---------- 单算子对齐 ----------

# 每条: (名字, 手写引擎实现, torch 实现, 叶子取值 a/b, 是否用到 b)
UNIT_CASES: list[tuple[str, Callable, Callable, float, float, bool]] = [
    ("add", lambda a, b: a + b, lambda ta, tb: ta + tb, 1.2, -0.7, True),
    ("mul", lambda a, b: a * b, lambda ta, tb: ta * tb, 1.2, -0.7, True),
    ("sub", lambda a, b: a - b, lambda ta, tb: ta - tb, 1.2, -0.7, True),
    ("neg", lambda a, b: -a, lambda ta, tb: -ta, 1.2, 0.5, False),
    ("div", lambda a, b: a / b, lambda ta, tb: ta / tb, 1.2, 0.5, True),
    ("rsub", lambda a, b: 2.0 - a, lambda ta, tb: 2.0 - ta, 1.2, 0.5, False),
    ("rdiv", lambda a, b: 2.0 / a, lambda ta, tb: 2.0 / ta, 1.2, 0.5, False),
    ("pow2", lambda a, b: a**2, lambda ta, tb: ta**2, 1.2, 0.5, False),
    ("pow_half", lambda a, b: a**0.5, lambda ta, tb: ta**0.5, 1.44, 0.5, False),
    ("relu_pos", lambda a, b: a.relu(), lambda ta, tb: ta.relu(), 1.2, 0.5, False),
    ("relu_neg", lambda a, b: b.relu(), lambda ta, tb: tb.relu(), 1.2, -0.7, True),
    ("tanh", lambda a, b: a.tanh(), lambda ta, tb: ta.tanh(), 1.2, 0.5, False),
    ("exp", lambda a, b: a.exp(), lambda ta, tb: ta.exp(), 0.7, 0.5, False),
]


@pytest.mark.parametrize(
    ("name", "eng_fn", "torch_fn", "a_val", "b_val", "uses_b"),
    UNIT_CASES,
    ids=[c[0] for c in UNIT_CASES],
)
def test_unit_op_grad_matches_torch(name, eng_fn, torch_fn, a_val, b_val, uses_b):
    a, b = Value(a_val), Value(b_val)
    out = eng_fn(a, b)
    out.backward()

    ta = torch.tensor(a_val, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(b_val, dtype=torch.float32, requires_grad=True)
    tout = torch_fn(ta, tb)
    tout.backward()

    # torch 对未参与输出图的叶子梯度保持 None（本引擎记 0.0），统一按 0 处理
    ta_grad = ta.grad.item() if ta.grad is not None else 0.0
    tb_grad = tb.grad.item() if tb.grad is not None else 0.0
    assert abs(a.grad - ta_grad) <= 1e-5, f"{name}: d/da {a.grad} vs {ta_grad}"
    if uses_b:
        assert abs(b.grad - tb_grad) <= 1e-5, f"{name}: d/db {b.grad} vs {tb_grad}"


# ---------- 随机复合函数对齐（DoD 核心：>=10 个） ----------

BINARY_OPS = ("add", "mul", "sub")
UNARY_OPS = ("neg", "tanh", "relu")


def build_composite(seed: int, n_ops: int = 8) -> tuple:
    """随机生成一棵表达式树。叶子是 3 个自由变量，之后随机叠加二元/一元算子。

    选值范围与算子组合刻意保持梯度有界（不引入 exp/pow 连锁），
    使 float32 的 torch 与 float64 的 Python 标量在 atol=1e-5 内可比。
    """
    rng = random.Random(seed)
    nodes: list[tuple] = [("var", i) for i in range(3)]
    for _ in range(n_ops):
        kind = rng.choice(BINARY_OPS + UNARY_OPS)
        if kind in BINARY_OPS:
            nodes.append((kind, rng.choice(nodes), rng.choice(nodes)))
        else:
            nodes.append((kind, rng.choice(nodes)))
    return nodes[-1]


def eval_engine(node: tuple, env: list[Value]) -> Value:
    kind = node[0]
    if kind == "var":
        return env[node[1]]
    if kind == "add":
        return eval_engine(node[1], env) + eval_engine(node[2], env)
    if kind == "mul":
        return eval_engine(node[1], env) * eval_engine(node[2], env)
    if kind == "sub":
        return eval_engine(node[1], env) - eval_engine(node[2], env)
    if kind == "neg":
        return -eval_engine(node[1], env)
    if kind == "tanh":
        return eval_engine(node[1], env).tanh()
    if kind == "relu":
        return eval_engine(node[1], env).relu()
    raise ValueError(f"未知算子: {kind}")


def eval_torch(node: tuple, env: list[torch.Tensor]) -> torch.Tensor:
    kind = node[0]
    if kind == "var":
        return env[node[1]]
    if kind == "add":
        return eval_torch(node[1], env) + eval_torch(node[2], env)
    if kind == "mul":
        return eval_torch(node[1], env) * eval_torch(node[2], env)
    if kind == "sub":
        return eval_torch(node[1], env) - eval_torch(node[2], env)
    if kind == "neg":
        return -eval_torch(node[1], env)
    if kind == "tanh":
        return eval_torch(node[1], env).tanh()
    if kind == "relu":
        return eval_torch(node[1], env).relu()
    raise ValueError(f"未知算子: {kind}")


@pytest.mark.parametrize("seed", range(12))
def test_composite_grad_matches_torch(seed: int):
    rng = random.Random(seed)
    leaf_vals = [rng.uniform(-1.5, 1.5) for _ in range(3)]
    ast = build_composite(seed)

    env_e = [Value(v) for v in leaf_vals]
    out_e = eval_engine(ast, env_e)
    out_e.backward()

    env_t = [torch.tensor(v, dtype=torch.float32, requires_grad=True) for v in leaf_vals]
    out_t = eval_torch(ast, env_t)
    out_t.backward()

    # 前向值先对齐：证明两边求值的是同一棵树
    assert abs(out_e.data - out_t.item()) <= 1e-5, (
        f"seed={seed} 前向不一致: {out_e.data} vs {out_t.item()}"
    )
    for i, (ve, vt) in enumerate(zip(env_e, env_t, strict=True)):
        vt_grad = vt.grad.item() if vt.grad is not None else 0.0
        assert abs(ve.grad - vt_grad) <= 1e-5, (
            f"seed={seed} var{i} 梯度不一致: {ve.grad} vs {vt_grad}"
        )


# ---------- MLP 训练：loss 单调下降（DoD 第 2 条） ----------


def test_mlp_loss_monotone_decreasing():
    _, _, _, losses = train_mlp()
    diffs = [later - prev for prev, later in pairwise(losses)]
    assert max(diffs) <= 1e-9, f"loss 出现上升: 最大步进 {max(diffs):.3e}"
    assert losses[-1] < 0.5 * losses[0], (
        f"收敛不足: 初始 {losses[0]:.4f} -> 最终 {losses[-1]:.4f}"
    )
    assert losses[-1] < 0.25, f"最终 loss {losses[-1]:.4f} 未达到拟合水平"
