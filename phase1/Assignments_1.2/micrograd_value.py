"""Scalar micrograd engine: Value class + tiny MLP helpers."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


class Value:
    """A scalar value that records a computation graph for reverse-mode autodiff."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1.0)

    def __rtruediv__(self, other):
        return other * (self**-1.0)

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Value only supports int/float powers.")
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += power * (self.data ** (power - 1.0)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin: int, nonlin: str = "tanh"):
        scale = 1.0 / math.sqrt(nin)
        self.w = [Value(random.uniform(-scale, scale)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Sequence[Value | float]):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        if self.nonlin == "tanh":
            return act.tanh()
        if self.nonlin == "relu":
            return act.relu()
        if self.nonlin in ("linear", None):
            return act
        raise ValueError(f"Unknown activation: {self.nonlin}")

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int, nonlin: str = "tanh"):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: Sequence[int], nonlin: str = "tanh"):
        sizes = [nin] + list(nouts)
        self.layers = []
        for i in range(len(nouts)):
            act = nonlin if i != len(nouts) - 1 else "linear"
            self.layers.append(Layer(sizes[i], sizes[i + 1], act))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def mse_loss(preds: Iterable[Value], targets: Iterable[float]):
    losses = [(pred - target) ** 2 for pred, target in zip(preds, targets)]
    return sum(losses, Value(0.0)) / len(losses)


if __name__ == "__main__":
    x1 = Value(2.0, label="x1")
    x2 = Value(-3.0, label="x2")
    y = (x1 * x2 + x1.tanh() + x2.exp()) ** 2
    y.backward()
    print("y:", y)
    print("dy/dx1:", x1.grad)
    print("dy/dx2:", x2.grad)
