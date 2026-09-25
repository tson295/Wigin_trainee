"""Train the character-level GPT on Tiny Shakespeare."""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch

from part2_gpt import GPT, GPTConfig


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text(path: Path, download: bool = True) -> str:
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Shakespeare text at {path}. Pass --data-path or allow download."
        )
    text = path.read_text(encoding="utf-8")
    if len(text) < 100_000:
        raise ValueError("The Shakespeare corpus looks too small; expected about 1 MB.")
    return text


def encode_text(text: str):
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    return encoded, stoi, itos


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
):
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss))
        result[name] = float(np.mean(losses))
    model.train()
    return result


def learning_rate(step: int, args) -> float:
    if step < args.warmup_iters:
        return args.learning_rate * (step + 1) / max(1, args.warmup_iters)
    if step >= args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (step - args.warmup_iters) / max(
        1, args.lr_decay_iters - args.warmup_iters
    )
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coefficient * (args.learning_rate - args.min_lr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=5000)
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sample-tokens", type=int, default=500)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    text = load_text(args.data_path, download=not args.no_download)
    encoded, stoi, itos = encode_text(text)
    split = int((1.0 - args.val_fraction) * len(encoded))
    train_data, val_data = encoded[:split], encoded[split:]

    if min(train_data.numel(), val_data.numel()) <= args.block_size + 1:
        raise ValueError("Corpus split must be longer than block_size.")

    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} chars={len(text):,} vocab={len(stoi)} "
        f"parameters={model.num_parameters():,} amp={amp_enabled}"
    )
    best_val_loss = float("inf")
    history = []
    model.train()
    for step in range(args.max_iters):
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp_enabled,
        ):
            _, loss = model(x, y)
        assert loss is not None
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, args, device)
            row = {
                "step": step,
                "learning_rate": lr,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "val_perplexity": math.exp(min(20.0, losses["val"])),
            }
            history.append(row)
            print(
                f"step {step:05d} | lr {lr:.2e} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"val ppl {row['val_perplexity']:.3f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": vars(config),
                        "stoi": stoi,
                        "itos": itos,
                        "step": step,
                        "val_loss": best_val_loss,
                    },
                    args.out_dir / "char_gpt_best.pt",
                )

    (args.out_dir / "char_gpt_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    prompt_ids = torch.tensor(
        [[stoi[char] for char in args.prompt]], dtype=torch.long, device=device
    )
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    sample = "".join(itos[index] for index in generated)
    (args.out_dir / "char_gpt_sample.txt").write_text(sample, encoding="utf-8")
    print(f"saved={args.out_dir / 'char_gpt_best.pt'}")
    print(f"sample:\n{sample}")


if __name__ == "__main__":
    main()
