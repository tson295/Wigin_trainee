"""A linear layer implemented with NumPy only.

Shapes:
    X: (batch_size, in_features)
    W: (in_features, out_features)
    b: (1, out_features) or (out_features,)
"""

import numpy as np


class linalg_numpy:
    """Compute ``Y = X @ W + b`` and its analytical gradients."""

    def __init__(self, W, b):
        self.W = np.asarray(W)
        self.b = np.asarray(b)
        self.X = None
        self.dX = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = np.asarray(X)
        return self.X @ self.W + self.b

    def backward(self, dY):
        if self.X is None:
            raise RuntimeError("Call forward() before backward().")

        dY = np.asarray(dY)
        self.dX = dY @ self.W.T
        self.dW = self.X.T @ dY
        self.db = np.sum(dY, axis=0)
        return self.dX, self.dW, self.db
