"""Simple 3D loss landscape using Pandas, Seaborn, and Matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


N, d_in, d_out = 4, 3, 2
np.random.seed(42)
X = np.random.randn(N, d_in)
Y_true = np.random.randn(N, d_out)
W = np.random.randn(d_in, d_out)
b = np.random.randn(1, d_out)

w00_values = np.linspace(W[0, 0] - 3, W[0, 0] + 3, 50)
w10_values = np.linspace(W[1, 0] - 3, W[1, 0] + 3, 50)

rows = []
for w00 in w00_values:
    for w10 in w10_values:
        W_test = W.copy()
        W_test[0, 0] = w00
        W_test[1, 0] = w10

        Y_pred = X @ W_test + b
        loss = np.mean((Y_pred - Y_true) ** 2)
        rows.append([w00, w10, loss])

df = pd.DataFrame(rows, columns=["W[0,0]", "W[1,0]", "Loss"])
loss_table = df.pivot(index="W[1,0]", columns="W[0,0]", values="Loss")

W00, W10 = np.meshgrid(loss_table.columns, loss_table.index)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(W00, W10, loss_table.values, cmap="viridis")

ax.set_title("Linear Layer Loss Landscape")
ax.set_xlabel("W[0,0]")
ax.set_ylabel("W[1,0]")
ax.set_zlabel("MSE Loss")

output_path = Path(__file__).with_name("loss_landscape.png")
plt.tight_layout()
plt.savefig(output_path, dpi=150)

if plt.get_backend().lower() == "agg":
    plt.close(fig)
else:
    plt.show()

print(f"Saved: {output_path}")
