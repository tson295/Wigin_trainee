"""Verify NumPy Conv2D forward/backward against torch.nn.functional.conv2d."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from conv2d_numpy import Conv2D


def main():
    rng = np.random.default_rng(42)
    stride = 2
    padding = 1
    x_np = rng.normal(size=(2, 3, 7, 8)).astype(np.float64)
    w_np = rng.normal(size=(4, 3, 3, 3)).astype(np.float64)
    b_np = rng.normal(size=(4,)).astype(np.float64)

    conv = Conv2D(3, 4, kernel_size=3, stride=stride, padding=padding)
    conv.W = w_np.copy()
    conv.b = b_np.copy()

    out_np = conv.forward(x_np)
    dout_np = 2.0 * out_np / out_np.size
    dx_np, dW_np, db_np = conv.backward(dout_np)

    x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    w_t = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    out_t = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding)
    loss_t = (out_t**2).mean()
    loss_t.backward()

    checks = {
        "forward": np.allclose(out_np, out_t.detach().numpy(), atol=1e-9),
        "dX": np.allclose(dx_np, x_t.grad.detach().numpy(), atol=1e-9),
        "dW": np.allclose(dW_np, w_t.grad.detach().numpy(), atol=1e-9),
        "db": np.allclose(db_np, b_t.grad.detach().numpy(), atol=1e-9),
    }
    for name, passed in checks.items():
        print(f"{name}: {passed}")
    assert all(checks.values()), checks

    conv_no_bias = Conv2D(
        2, 3, kernel_size=(2, 3), stride=(2, 1), padding=(1, 2), bias=False
    )
    x_small = rng.normal(size=(1, 2, 5, 6)).astype(np.float64)
    out_small = conv_no_bias.forward(x_small)
    dout_small = rng.normal(size=out_small.shape).astype(np.float64)
    dx_small, dW_small, db_small = conv_no_bias.backward(dout_small)
    x_small_t = torch.tensor(x_small, dtype=torch.float64, requires_grad=True)
    w_small_t = torch.tensor(conv_no_bias.W, dtype=torch.float64, requires_grad=True)
    expected_small = F.conv2d(
        x_small_t,
        w_small_t,
        stride=(2, 1),
        padding=(1, 2),
    )
    expected_small.backward(torch.tensor(dout_small))
    assert db_small is None
    assert np.allclose(out_small, expected_small.detach().numpy(), atol=1e-9)
    assert np.allclose(dx_small, x_small_t.grad.numpy(), atol=1e-9)
    assert np.allclose(dW_small, w_small_t.grad.numpy(), atol=1e-9)
    print("asymmetric/no-bias: True")


if __name__ == "__main__":
    main()
