"""Pure NumPy Conv2D layer with forward and backward passes."""

from __future__ import annotations

import math

import numpy as np

def _pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Expected a scalar or a pair of values.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class Conv2D:
    """NCHW Conv2D: input (N, C_in, H, W), weight (C_out, C_in, KH, KW)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        seed=42,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if any(size <= 0 for size in self.kernel_size + self.stride):
            raise ValueError("kernel_size and stride must be positive.")
        if any(size < 0 for size in self.padding):
            raise ValueError("padding must be non-negative.")

        rng = np.random.default_rng(seed)
        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw
        scale = math.sqrt(2.0 / fan_in)
        self.W = rng.normal(
            0.0, scale, size=(out_channels, in_channels, kh, kw)
        ).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32) if bias else None

        self.x = None
        self.x_padded = None
        self.dW = None
        self.db = None

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError("Expected x with shape (N, C, H, W).")
        n, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}.")

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (ph, ph), (pw, pw)),
            mode="constant",
        )
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("Kernel/stride/padding produce an empty output.")

        out_dtype = np.result_type(x, self.W, self.b if self.b is not None else x)
        out = np.zeros((n, self.out_channels, out_h, out_w), dtype=out_dtype)

        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                window = x_padded[:, :, hs : hs + kh, ws : ws + kw]
                out[:, :, i, j] = np.tensordot(
                    window,
                    self.W,
                    axes=([1, 2, 3], [1, 2, 3]),
                )

        if self.b is not None:
            out += self.b.reshape(1, -1, 1, 1)

        self.x = x
        self.x_padded = x_padded
        return out

    def backward(self, dout):
        if self.x is None or self.x_padded is None:
            raise RuntimeError("Call forward before backward.")

        dout = np.asarray(dout)
        x = self.x
        x_padded = self.x_padded
        n, _, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        _, _, out_h, out_w = dout.shape
        expected_h = (h + 2 * ph - kh) // sh + 1
        expected_w = (w + 2 * pw - kw) // sw + 1
        expected_shape = (n, self.out_channels, expected_h, expected_w)
        if dout.shape != expected_shape:
            raise ValueError(
                f"Expected dout with shape {expected_shape}, got {dout.shape}."
            )

        dx_padded = np.zeros_like(x_padded, dtype=np.result_type(x, dout, self.W))
        dW = np.zeros_like(self.W, dtype=np.result_type(x, dout, self.W))
        db = None if self.b is None else dout.sum(axis=(0, 2, 3))

        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                window = x_padded[:, :, hs : hs + kh, ws : ws + kw]
                for oc in range(self.out_channels):
                    grad = dout[:, oc, i, j]
                    dW[oc] += np.sum(window * grad[:, None, None, None], axis=0)
                    dx_padded[:, :, hs : hs + kh, ws : ws + kw] += (
                        self.W[oc][None, :, :, :] * grad[:, None, None, None]
                    )

        if ph == 0 and pw == 0:
            dx = dx_padded
        else:
            dx = dx_padded[:, :, ph : ph + h, pw : pw + w]

        self.dW = dW
        self.db = db
        return dx, dW, db


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    conv = Conv2D(3, 4, kernel_size=3, stride=1, padding=1)
    x = rng.normal(size=(2, 3, 8, 8)).astype(np.float32)
    out = conv.forward(x)
    dout = rng.normal(size=out.shape).astype(np.float32)
    dx, dW, db = conv.backward(dout)
    print("out:", out.shape)
    print("dx:", dx.shape)
    print("dW:", dW.shape)
    print("db:", None if db is None else db.shape)
