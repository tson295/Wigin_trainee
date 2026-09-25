from fbward_numpy import linalg_numpy

import torch
import torch.nn as nn
import numpy as np

N, d_in, d_out = 4, 3, 2
np.random.seed(42)

X_np = np.random.randn(N, d_in)
Y_true_np = np.random.randn(N, d_out)

W_init = np.random.randn(d_in, d_out)
b_init = np.random.randn(1, d_out)

linear_np = linalg_numpy(W_init, b_init)
Y_pred_np = linear_np.forward(X_np)
loss_np = np.mean((Y_pred_np - Y_true_np) ** 2)

dL_dY_np = 2 * (Y_pred_np - Y_true_np) / (N * d_out)
dX_np, dW_np, db_np = linear_np.backward(dL_dY_np)

X_torch = torch.tensor(X_np, dtype=torch.float32, requires_grad=True)
Y_true_torch = torch.tensor(Y_true_np, dtype=torch.float32)

linear_torch = nn.Linear(d_in, d_out, bias=True)
with torch.no_grad():
    linear_torch.weight.copy_(torch.tensor(W_init.T, dtype=torch.float32))
    linear_torch.bias.copy_(torch.tensor(b_init.squeeze(0), dtype=torch.float32))

Y_pred_torch = linear_torch(X_torch)
criterion = nn.MSELoss()
loss_torch = criterion(Y_pred_torch, Y_true_torch)
loss_torch.backward()

print(f"Loss (NumPy):   {loss_np:.6f}")
print(f"Loss (PyTorch): {loss_torch.item():.6f}")

print("\n--- Match Checks ---")

print("Forward Output match:",
      np.allclose(Y_pred_np, Y_pred_torch.numpy()))

print("dW match:",
      np.allclose(dW_np, linear_torch.weight.grad.numpy().T))

print("db match:",
      np.allclose(db_np, linear_torch.bias.grad.numpy()))

print("dX match:",
      np.allclose(dX_np, X_torch.grad.numpy()))
