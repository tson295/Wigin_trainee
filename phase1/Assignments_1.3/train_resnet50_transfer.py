"""Train ResNet50 from scratch on the notebook dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def ensure_kaggle_runtime() -> None:
    if not Path("/kaggle").is_dir() and not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return
    try:
        versions = {
            "torch": version("torch"),
            "torchvision": version("torchvision"),
            "Pillow": version("Pillow"),
        }
    except PackageNotFoundError:
        versions = {}
    ready = (
        versions.get("torch", "").startswith("2.5.1")
        and versions.get("torchvision", "").startswith("0.20.1")
        and versions.get("Pillow", "").startswith("11.3.0")
    )
    if ready:
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "Pillow"],
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--force-reinstall",
            "--no-cache-dir",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "Pillow==11.3.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
            "--extra-index-url",
            "https://pypi.org/simple",
        ]
    )


ensure_kaggle_runtime()

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability()
    architecture = f"sm_{major}{minor}"
    if architecture not in torch.cuda.get_arch_list():
        name = torch.cuda.get_device_name()
        raise RuntimeError(
            f"{name} ({architecture}) is not supported by this PyTorch build. "
            "Install torch 2.5.1 and torchvision 0.20.1 from the cu121 index "
            "before importing this file, or switch Kaggle to a T4 GPU."
        )
    return torch.device("cuda")


def make_transforms(image_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


def _find_dataset_dirs(root: Path) -> tuple[Path, Path] | None:
    candidates = [
        (root / "train", root / "val"),
        (root / "train", root / "test"),
        (root / "seg_train" / "seg_train", root / "seg_test" / "seg_test"),
    ]
    for train_dir, val_dir in candidates:
        if train_dir.is_dir() and val_dir.is_dir():
            return train_dir, val_dir
    return None


def discover_dataset_root() -> Path | None:
    search_roots = [Path("/kaggle/input"), Path("/content"), Path.cwd()]
    checked = set()
    for base in search_roots:
        base = base.expanduser().resolve()
        if not base.is_dir() or base in checked:
            continue
        checked.add(base)
        if _find_dataset_dirs(base) is not None:
            return base
        for train_dir in base.glob("**/seg_train/seg_train"):
            root = train_dir.parents[1]
            if _find_dataset_dirs(root) is not None:
                return root
    return None


def build_dataloaders(args):
    train_tf, eval_tf = make_transforms(args.image_size)
    if not args.data_root:
        raise SystemExit("Pass --data-root from the notebook dataset_path.")
    root = Path(args.data_root).expanduser().resolve()
    split_dirs = _find_dataset_dirs(root)
    if split_dirs is None:
        raise FileNotFoundError(f"Dataset splits not found under {root}.")
    train_dir, val_dir = split_dirs
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    class_names = train_ds.classes
    if val_ds.classes != class_names:
        raise ValueError(
            "Train/validation class folders differ: "
            f"{class_names} vs {val_ds.classes}."
        )
    if len(class_names) < args.min_classes:
        raise SystemExit(
            f"Dataset must contain at least {args.min_classes} classes, "
            f"found {len(class_names)}."
        )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, class_names


def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_all_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable


def configure_warmup(model: nn.Module) -> None:
    set_all_trainable(model, False)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def configure_finetune(model: nn.Module) -> None:
    set_all_trainable(model, False)
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def make_optimizer(model: nn.Module, stage: str, args):
    if stage == "warmup":
        parameter_groups = [{"params": model.fc.parameters(), "lr": args.head_lr}]
    elif stage == "finetune":
        parameter_groups = [
            {"params": model.layer4.parameters(), "lr": args.layer4_lr},
            {"params": model.fc.parameters(), "lr": args.head_lr},
        ]
    else:
        raise ValueError(f"Unknown optimization stage: {stage}")
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_seen += images.size(0)
    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.detach()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_seen += images.size(0)
    return total_loss / total_seen, total_correct / total_seen


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs: int,
    device,
    patience: int,
    stage: str,
    epoch_offset: int = 0,
):
    """Train one notebook-style stage and restore its best validation weights."""

    history = []
    best_val_acc = -float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    started = time.perf_counter()

    for local_epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch_offset + local_epoch,
            "stage": stage,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_weights)
    return model, history, time.perf_counter() - started


def run_scratch(model, train_loader, val_loader, args, device):
    criterion = nn.CrossEntropyLoss()
    history = []
    total_time = 0.0

    configure_warmup(model)
    optimizer = make_optimizer(model, "warmup", args)
    model, stage_history, elapsed = fit(
        model, train_loader, val_loader, criterion, optimizer,
        args.head_epochs, device, args.patience, "warmup",
    )
    history.extend(stage_history)
    total_time += elapsed

    configure_finetune(model)
    optimizer = make_optimizer(model, "finetune", args)
    model, stage_history, elapsed = fit(
        model, train_loader, val_loader, criterion, optimizer,
        args.finetune_epochs, device, args.patience, "finetune",
        epoch_offset=len(history),
    )
    history.extend(stage_history)
    total_time += elapsed

    best = max(history, key=lambda row: row["val_acc"])
    return {
        "history": history,
        "best_val_acc": best["val_acc"],
        "total_time_sec": total_time,
    }


def plot_curves(results, output_path: Path, warmup_epochs: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, payload in results.items():
        history = payload["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_acc"] for row in history], marker="o", label=f"{name} train")
        axes[0].plot(epochs, [row["val_acc"] for row in history], marker="o", label=f"{name} val")
        axes[1].plot(epochs, [row["train_loss"] for row in history], marker="o", label=f"{name} train")
        axes[1].plot(epochs, [row["val_loss"] for row in history], marker="o", label=f"{name} val")
    for axis, title, ylabel in (
        (axes[0], "Accuracy", "accuracy"),
        (axes[1], "Cross-entropy loss", "loss"),
    ):
        axis.axvline(warmup_epochs + 0.5, linestyle="--", color="black", alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-classes", type=int, default=5)
    parser.add_argument("--layer4-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    if argv is None and "ipykernel" in sys.modules:
        argv = list(sys.argv[1:])
        if "-f" in argv:
            index = argv.index("-f")
            del argv[index : index + 2]
    return parser.parse_known_args(argv)[0]


def notebook_dataset_path():
    value = globals().get("dataset_path")
    if value:
        return str(value)
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            value = shell.user_ns.get("dataset_path")
            if value:
                return str(value)
    except Exception:
        pass
    return ""


def main(data_root=None, out_dir=None):
    args = parse_args()
    if data_root is not None:
        args.data_root = str(data_root)
    if out_dir is not None:
        args.out_dir = str(out_dir)
    if not args.data_root:
        args.data_root = notebook_dataset_path()
    if not args.data_root:
        discovered_root = discover_dataset_root()
        if discovered_root is not None:
            args.data_root = str(discovered_root)
    device = get_device()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, class_names = build_dataloaders(args)
    num_classes = len(class_names)

    model = build_resnet50(num_classes).to(device)
    results = {
        "scratch": run_scratch(model, train_loader, val_loader, args, device)
    }

    payload = {
        name: {
            "accuracy": result["best_val_acc"],
            "training_time_sec": result["total_time_sec"],
            "history": result["history"],
        }
        for name, result in results.items()
    }
    metrics_path = out_dir / "resnet50_scratch_metrics.json"
    plot_path = out_dir / "resnet50_scratch_learning_curves.png"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_curves(results, plot_path, args.head_epochs)
    for name, result in results.items():
        print(
            f"{name}: accuracy={result['best_val_acc']:.4f} "
            f"training_time={result['total_time_sec']:.2f}s"
        )


if __name__ == "__main__":
    main()
