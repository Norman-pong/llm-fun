"""T1 手写标量 autograd 引擎：反向传播 = 计算图上的链式法则遍历。

最小实现证明 autograd 不需要框架。每个算子只负责两件事：
前向时记住输入节点，并注册自己的局部导数如何分回给输入；
backward() 把全图拓扑排序后，从输出往输入逐节点累加梯度。
仅依赖标准库；对齐验证（与 torch.autograd）在 test_engine.py。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator


class Value:
    """可求导标量。data 是前向值；grad 是 d(最终输出)/d(self)，由 backward() 填充。"""

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = tuple(_children)
        self._op = _op  # 仅调试用：打印计算图时能看到每个节点是什么算子

    # ---------- 算子：前向构造子节点，并注册反向传播 ----------

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = d(a+b)/db = 1；用 += 因为同一节点可能被多条路径引用
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # 乘法法则：各拿对方的值
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | float) -> Value:
        return Value(other) + (-self)

    def __truediv__(self, other: Value | float) -> Value:
        return self * other**-1 if isinstance(other, Value) else self * Value(other) ** -1

    def __rtruediv__(self, other: Value | float) -> Value:
        return Value(other) * self**-1

    def __pow__(self, k: float) -> Value:
        # 只支持常数指数（变量指数需要 log 链式法则，本引擎不涉及）
        if not isinstance(k, (int, float)):
            raise TypeError("指数必须是常数 int/float")
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward() -> None:
            # d(x^k)/dx = k * x^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            # 分段常数：x>0 处导数 1，x<=0 处 0（x=0 取 0，与 PyTorch 一致）
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d tanh(x)/dx = 1 - tanh(x)^2
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            # exp 的导数是它自己
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # ---------- 反向传播 ----------

    def _topo_order(self) -> list[Value]:
        """迭代式 DFS 拓扑排序：保证调用某节点 _backward 前，它的输出梯度已算完。"""
        topo: list[Value] = []
        visited: set[int] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))
        return topo

    def backward(self) -> None:
        """从 self 出发反向传播。d(self)/d(self) = 1，其余按拓扑逆序累加。"""
        self.grad = 1.0
        for node in reversed(self._topo_order()):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


# ---------- 用 Value 搭一个极简神经网络（micrograd 风格） ----------


class Neuron:
    """单神经元：act = tanh(w·x + b)，可通过 act=None 关闭激活。"""

    def __init__(self, nin: int, act: str | None = "tanh") -> None:
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x: list[Value]) -> Value:
        act = sum(w * xi for w, xi in zip(self.w, x, strict=True)) + self.b
        if self.act == "tanh":
            return act.tanh()
        if self.act == "relu":
            return act.relu()
        return act

    def parameters(self) -> Iterator[Value]:
        yield from self.w
        yield self.b


class Layer:
    def __init__(self, nin: int, nout: int, act: str | None = "tanh") -> None:
        self.neurons = [Neuron(nin, act) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> Iterator[Value]:
        for n in self.neurons:
            yield from n.parameters()


class MLP:
    """多层感知机：隐层 tanh，输出层线性（回归用）。sizes 形如 [2, 8, 1]。"""

    def __init__(self, nin: int, sizes: list[int]) -> None:
        dims = [nin, *sizes]
        n_layers = len(sizes)
        self.layers = [
            Layer(dims[i], dims[i + 1], act="tanh" if i < n_layers - 1 else None)
            for i in range(n_layers)
        ]

    def __call__(self, x: list[Value]) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x[0]  # 本章只用到单输出

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0
