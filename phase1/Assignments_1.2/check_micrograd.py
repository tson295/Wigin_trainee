"""Quick checks for Assignment 1.2."""

from __future__ import annotations

import math

import numpy as np

from micrograd_value import Value
from tensorgrad_numpy import Adam, MLPClassifier, cross_entropy


def finite_diff_grad(fn, x, y, eps=1e-6):
    dx = (fn(x + eps, y) - fn(x - eps, y)) / (2.0 * eps)
    dy = (fn(x, y + eps) - fn(x, y - eps)) / (2.0 * eps)
    return dx, dy


def check_scalar_value():
    x = Value(2.0)
    y = Value(-3.0)
    z = (x * y + x.tanh() + y.exp()) ** 2
    z.backward()

    def fn(a, b):
        return (a * b + math.tanh(a) + math.exp(b)) ** 2

    dx_num, dy_num = finite_diff_grad(fn, 2.0, -3.0)
    print("scalar Value:")
    print("  forward:", z.data)
    print("  dx close:", abs(x.grad - dx_num) < 1e-5)
    print("  dy close:", abs(y.grad - dy_num) < 1e-5)


def check_tensor_training_step():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 16)).astype(np.float32)
    y = rng.integers(0, 3, size=64)
    model = MLPClassifier(16, [32], 3)
    optim = Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(20):
        model.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optim.step()
        losses.append(float(loss.data))

    print("tensorgrad MLP:")
    print("  first_loss:", f"{losses[0]:.4f}")
    print("  last_loss:", f"{losses[-1]:.4f}")
    print("  loss_decreased:", losses[-1] < losses[0])


if __name__ == "__main__":
    check_scalar_value()
    check_tensor_training_step()
