"""A small NumPy tensor autograd engine for MNIST-scale MLP training."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _as_array(data):
    return np.asarray(data, dtype=np.float32)


def _ensure_tensor(other):
    return other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)


def _unbroadcast(grad, target_shape):
    grad = np.asarray(grad, dtype=np.float32)
    if target_shape == ():
        return np.asarray(grad.sum(), dtype=np.float32)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


class Tensor:
    """A NumPy array that supports reverse-mode autodiff."""

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = _as_array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        if self.requires_grad:
            self.grad.fill(0.0)

    def __add__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            (self, other),
            "+",
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            (self, other),
            "*",
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            (self, other),
            "matmul",
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_ensure_tensor(other))

    def __rsub__(self, other):
        return _ensure_tensor(other) + (-self)

    def __truediv__(self, other):
        return self * (_ensure_tensor(other) ** -1.0)

    def __rtruediv__(self, other):
        return _ensure_tensor(other) * (self ** -1.0)

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Tensor only supports int/float powers.")
        out = Tensor(self.data**power, (self,), f"**{power}", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1.0)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0.0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            "sum",
            self.requires_grad,
        )

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            self.grad += np.ones_like(self.data, dtype=np.float32) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
            denom = math.prod(self.data.shape[a] for a in axes)
        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def backward(self, grad=None):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float32)
        self.grad = _as_array(grad)
        for node in reversed(topo):
            node._backward()


def cross_entropy(logits: Tensor, targets):
    """Mean softmax cross entropy with a direct backward formula."""

    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    batch = logits.data.shape[0]
    losses = -np.log(probs[np.arange(batch), targets] + 1e-12)
    out = Tensor(losses.mean(), (logits,), "cross_entropy", logits.requires_grad)

    def _backward():
        if logits.requires_grad:
            grad = probs.copy()
            grad[np.arange(batch), targets] -= 1.0
            grad /= batch
            logits.grad += grad * out.grad

    out._backward = _backward
    return out


def cross_entropy_np(logits, targets):
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    losses = -np.log(probs[np.arange(logits.shape[0]), targets] + 1e-12)
    return float(losses.mean())


def accuracy(logits, targets):
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == targets))


class MLPClassifier:
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, seed=42):
        rng = np.random.default_rng(seed)
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = math.sqrt(2.0 / fan_in)
            w = rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(Tensor(w, requires_grad=True))
            self.biases.append(Tensor(b, requires_grad=True))

    def __call__(self, x):
        h = x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i != len(self.weights) - 1:
                h = h.relu()
        return h

    def forward_np(self, x):
        h = np.asarray(x, dtype=np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w.data + b.data
            if i != len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        return h

    def parameters(self):
        return [p for pair in zip(self.weights, self.biases) for p in pair]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Adam:
    def __init__(self, params: Iterable[Tensor], lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
        self.v = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (p.grad**2)
            m_hat = self.m[i] / (1.0 - self.beta1**self.t)
            v_hat = self.v[i] / (1.0 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:
    def __init__(self, params: Iterable[Tensor], lr=0.01, momentum=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] + p.grad
            p.data -= self.lr * self.velocity[i]
