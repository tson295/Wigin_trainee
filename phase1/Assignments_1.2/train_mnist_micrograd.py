"""Train an MNIST MLP using the local NumPy tensor autograd engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorgrad_numpy import Adam, MLPClassifier, accuracy, cross_entropy, cross_entropy_np


def parse_hidden_dims(text: str):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def split_train_validation(x, y, val_size, rng):
    if val_size < 1:
        raise ValueError("--val-size must be at least 1")
    if len(x) < 2:
        raise ValueError("Need at least two samples to create train/validation splits")

    val_size = min(int(val_size), len(x) - 1)
    indices = rng.permutation(len(x))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def load_data(args):
    rng = np.random.default_rng(args.seed)
    from torchvision import datasets

    data_dir = Path(args.data_dir)
    train_set = datasets.MNIST(root=str(data_dir), train=True, download=True)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True)

    x_train_full = train_set.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
    y_train_full = train_set.targets.numpy().astype(np.int64)
    x_test = test_set.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)

    x_train_full = (x_train_full - 0.1307) / 0.3081
    x_test = (x_test - 0.1307) / 0.3081

    x_train, y_train, x_val, y_val = split_train_validation(
        x_train_full, y_train_full, args.val_size, rng
    )

    if args.limit_train:
        idx = rng.permutation(len(x_train))[: args.limit_train]
        x_train, y_train = x_train[idx], y_train[idx]
    if args.limit_test:
        idx = rng.permutation(len(x_test))[: args.limit_test]
        x_test, y_test = x_test[idx], y_test[idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


def iter_batches(x, y, batch_size, rng):
    indices = rng.permutation(len(x))
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def train_epoch(model, optimizer, x_train, y_train, batch_size, rng):
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for xb, yb in iter_batches(x_train, y_train, batch_size, rng):
        model.zero_grad()
        logits = model(xb)
        loss = cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.data) * len(xb)
        total_correct += int((np.argmax(logits.data, axis=1) == yb).sum())
        total_seen += len(xb)
    return total_loss / total_seen, total_correct / total_seen


def evaluate(model, x, y, batch_size):
    losses = []
    accs = []
    sizes = []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model.forward_np(xb)
        losses.append(cross_entropy_np(logits, yb))
        accs.append(accuracy(logits, yb))
        sizes.append(len(xb))
    sizes = np.asarray(sizes, dtype=np.float32)
    return float(np.average(losses, weights=sizes)), float(np.average(accs, weights=sizes))


def copy_model_state(model):
    return [param.data.copy() for param in model.parameters()]


def restore_model_state(model, state):
    for param, saved_data in zip(model.parameters(), state):
        param.data[...] = saved_data


def save_model_state(model, output_path):
    np.savez(
        output_path,
        **{f"param_{i}": param.data for i, param in enumerate(model.parameters())},
    )


def plot_history(history, output_path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--target-accuracy", type=float, default=0.97)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args)
    model = MLPClassifier(784, parse_hidden_dims(args.hidden_dims), 10, seed=args.seed)
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_acc = -np.inf
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, optimizer, x_train, y_train, args.batch_size, rng
        )
        val_loss, val_acc = evaluate(model, x_val, y_val, args.batch_size)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy_model_state(model)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    restore_model_state(model, best_state)
    test_loss, test_acc = evaluate(model, x_test, y_test, args.batch_size)

    metrics_path = out_dir / "mnist_metrics.json"
    model_path = out_dir / "mnist_best_model.npz"
    plot_path = out_dir / "mnist_learning_curves.png"
    metrics = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_model_state(model, model_path)
    plot_history(history, plot_path)

    status = "PASS" if test_acc >= args.target_accuracy else "NOT_YET"
    print(
        f"best_val_acc={best_val_acc:.4f} best_epoch={best_epoch} "
        f"test_acc={test_acc:.4f} status={status}"
    )
    print(f"saved_metrics={metrics_path}")
    print(f"saved_model={model_path}")
    print(f"saved_plot={plot_path}")


if __name__ == "__main__":
    main()
