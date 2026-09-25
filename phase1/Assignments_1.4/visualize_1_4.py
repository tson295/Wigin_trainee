"""Build an interactive HTML report for Assignment 1.4.

Loads the saved character-GPT checkpoint (never retrains), runs forward passes to
pull out real attention maps, per-character loss and next-character predictions,
then does the same for the BPE tokenizer against tiktoken.

Usage:
    python visualize_1_4.py                      # uses outputs_validation_1000
    python visualize_1_4.py --ckpt-dir outputs_validation_500
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_kit import write as write_report  # noqa: E402

from part2_gpt import GPT, GPTConfig
from part2_train import encode_text, get_batch, load_text, set_seed
from part3_bpe import (
    FALLBACK_SPLIT_PATTERN,
    GPT2_SPLIT_PATTERN,
    GPT4_SPLIT_PATTERN,
    BytePairEncoding,
)

HERE = Path(__file__).resolve().parent
ATTENTION_TOKENS = 64
POSITION_TOKENS = 256


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def quantize(array: np.ndarray, high: float | None = None) -> tuple[str, float]:
    """Pack a non-negative float array into base64 uint8 plus its scale."""

    high = float(array.max()) if high is None else high
    high = high if high > 0 else 1.0
    scaled = np.clip(array / high * 255.0, 0, 255).astype(np.uint8)
    return base64.b64encode(scaled.tobytes()).decode("ascii"), high


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**payload["config"])
    model = GPT(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    itos = {int(key): value for key, value in payload["itos"].items()}
    return model, payload, payload["stoi"], itos


# --------------------------------------------------------------------------- #
# part 2 — character GPT
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_part2(args, device: torch.device) -> dict:
    ckpt_path = HERE / args.ckpt_dir / "char_gpt_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; nothing to visualize.")

    model, payload, stoi, itos = load_checkpoint(ckpt_path, device)
    config = model.config
    text = load_text(HERE / args.data_path, download=False)
    encoded, _, _ = encode_text(text)
    split = int(0.9 * len(encoded))
    train_data, val_data = encoded[:split], encoded[split:]

    # --- re-measure val loss from the restored weights (a real forward pass) ---
    set_seed(1337)
    losses = {"train": [], "val": []}
    for name, data in (("train", train_data), ("val", val_data)):
        for _ in range(args.eval_iters):
            x, y = get_batch(data, config.block_size, args.batch_size, device)
            _, loss = model(x, y)
            losses[name].append(float(loss))
    measured = {name: float(np.mean(values)) for name, values in losses.items()}

    # --- attention maps on a real Shakespeare passage ---
    passage = text[: ATTENTION_TOKENS]
    ids = torch.tensor(
        [[stoi[char] for char in passage]], dtype=torch.long, device=device
    )
    logits, _ = model(ids)

    maps, head_stats = [], []
    rows = torch.arange(ATTENTION_TOKENS, device=device)
    distance = (rows[:, None] - rows[None, :]).float()
    valid = rows >= 8  # skip warm-up positions where causality forces the answer
    for layer_index, block in enumerate(model.blocks):
        weights = block.attn.last_attention_weights
        assert weights is not None, "forward did not populate last_attention_weights"
        weights = weights[0].float().cpu()
        for head_index in range(weights.size(0)):
            attention = weights[head_index]
            packed, _ = quantize(attention.numpy(), high=1.0)
            maps.append(
                {"layer": layer_index, "head": head_index, "data": packed}
            )
            probs = attention[valid.cpu()]
            entropy = float(
                (-(probs.clamp_min(1e-9).log() * probs).sum(-1)).mean()
            )
            span = float((probs * distance.cpu()[valid.cpu()]).sum(-1).mean())
            head_stats.append(
                {
                    "layer": layer_index,
                    "head": head_index,
                    "entropy": entropy,
                    "span": span,
                    "self": float(attention.diagonal()[valid.cpu()].mean()),
                    "sink": float(attention[valid.cpu()][:, 0].mean()),
                    "previous": float(
                        attention.diagonal(-1)[valid.cpu()[1:]].mean()
                    ),
                }
            )

    # --- per-character loss over a longer window ---
    window = 256
    span_text = text[1000 : 1000 + window + 1]
    span_ids = torch.tensor(
        [[stoi[char] for char in span_text]], dtype=torch.long, device=device
    )
    span_logits, _ = model(span_ids[:, :-1])
    per_char = F.cross_entropy(
        span_logits[0].float(), span_ids[0, 1:], reduction="none"
    ).cpu()
    char_loss = [
        {"char": span_text[index + 1], "loss": float(per_char[index])}
        for index in range(window)
    ]

    # --- next-character distribution at the end of the attention passage ---
    probabilities = F.softmax(logits[0, -1].float(), dim=-1)
    top = torch.topk(probabilities, 10)
    predictions = [
        {"char": itos[int(index)], "p": float(value)}
        for value, index in zip(top.values, top.indices)
    ]

    # --- positional embedding structure ---
    position = model.position_embedding.weight[:POSITION_TOKENS].float()
    position = position / position.norm(dim=-1, keepdim=True)
    similarity = (position @ position.T).cpu().numpy()
    pos_packed, pos_high = quantize(
        np.clip(similarity, 0.0, None), high=float(similarity.max())
    )

    # --- generation ---
    prompt = "ROMEO:"
    prompt_ids = torch.tensor(
        [[stoi[char] for char in prompt]], dtype=torch.long, device=device
    )
    set_seed(7)
    samples = {}
    for label, kwargs in (
        ("greedy", {"do_sample": False}),
        ("t=0.8, top-k 50", {"temperature": 0.8, "top_k": 50}),
        ("t=1.2, top-k 50", {"temperature": 1.2, "top_k": 50}),
    ):
        out = model.generate(prompt_ids, max_new_tokens=420, **kwargs)[0].tolist()
        samples[label] = "".join(itos[index] for index in out)

    histories = {}
    for directory in sorted(HERE.glob("outputs_*")):
        history_file = directory / "char_gpt_history.json"
        if history_file.exists():
            histories[directory.name] = json.loads(
                history_file.read_text(encoding="utf-8")
            )

    return {
        "checkpoint": str(Path(args.ckpt_dir) / "char_gpt_best.pt"),
        "config": payload["config"],
        "step": payload["step"],
        "saved_val_loss": payload["val_loss"],
        "measured": measured,
        "measured_val_ppl": math.exp(measured["val"]),
        "parameters": model.num_parameters(),
        "vocab": "".join(itos[index] for index in sorted(itos)),
        "histories": histories,
        "passage": passage,
        "attention": maps,
        "head_stats": head_stats,
        "char_loss": char_loss,
        "predictions": predictions,
        "position_similarity": {
            "data": pos_packed,
            "high": pos_high,
            "n": POSITION_TOKENS,
        },
        "samples": samples,
        "device": str(device),
    }


# --------------------------------------------------------------------------- #
# part 3 — BPE tokenizer
# --------------------------------------------------------------------------- #
def show_bytes(raw: bytes) -> str:
    """Render token bytes for display.

    A BPE token often ends mid-way through a multi-byte character, so it has no
    valid decoding on its own. ``backslashreplace`` shows those bytes as ``\\xe2``
    instead of collapsing them to U+FFFD, which loses which byte it actually was.
    """

    return raw.decode("utf-8", errors="backslashreplace")


def token_bytes(tokenizer, token_id: int) -> bytes:
    """The real bytes a single token stands for.

    ``vocab`` for a tokenizer built from tiktoken lives in cl100k's permuted byte
    space, and ``decode`` only undoes that permutation on the joined output. Reading
    ``vocab[id]`` directly would render every token as a substitution cipher, so undo
    it here too — except for special tokens, whose entries were never permuted.
    """

    raw = tokenizer.vocab[token_id]
    if (
        tokenizer.inverse_byte_shuffle is not None
        and token_id not in set(tokenizer.special_tokens.values())
    ):
        raw = bytes(tokenizer.inverse_byte_shuffle[byte] for byte in raw)
    return raw


def segment(tokenizer, text: str, allowed_special="none_raise") -> list[dict]:
    ids = tokenizer.encode(text, allowed_special=allowed_special)
    return [
        {"id": token_id, "text": show_bytes(token_bytes(tokenizer, token_id))}
        for token_id in ids
    ]


def collect_part3(args) -> dict:
    text = load_text(HERE / args.data_path, download=False)
    corpus = text[: args.bpe_chars]
    holdout = text[args.bpe_chars : args.bpe_chars + 20_000]

    # --- compression as vocabulary grows ---
    curve = []
    merge_log = []
    for vocab_size in (256, 320, 384, 512, 768, 1024):
        tokenizer = BytePairEncoding()
        if vocab_size > 256:
            tokenizer.train(corpus, vocab_size)
        ids = tokenizer.encode(holdout)
        curve.append(
            {
                "vocab_size": vocab_size,
                "tokens": len(ids),
                "bytes_per_token": len(holdout.encode("utf-8")) / len(ids),
            }
        )
        if vocab_size == 1024:
            for (left, right), new_id in list(tokenizer.merges.items())[:40]:
                merge_log.append(
                    {
                        "rank": new_id - 256 + 1,
                        "left": show_bytes(tokenizer.vocab[left]),
                        "right": show_bytes(tokenizer.vocab[right]),
                        "result": show_bytes(tokenizer.vocab[new_id]),
                        "id": new_id,
                    }
                )
            shakespeare_tokenizer = tokenizer

    # --- the three split patterns on the same probe strings ---
    probes = [
        "Hello WORLD'S 1234567 dog.",
        "IT'S ok\nline2",
        "Xin chào Việt Nam 42",
    ]
    pattern_rows = []
    for label, pattern in (
        ("GPT-2", GPT2_SPLIT_PATTERN),
        ("GPT-4", GPT4_SPLIT_PATTERN),
        ("fallback (re)", FALLBACK_SPLIT_PATTERN),
    ):
        probe_tokenizer = BytePairEncoding(pattern=pattern)
        pattern_rows.append(
            {
                "label": label,
                "splits": [probe_tokenizer._split(probe) for probe in probes],
            }
        )

    result = {
        "corpus_chars": len(corpus),
        "holdout_chars": len(holdout),
        "curve": curve,
        "merges": merge_log,
        "probes": probes,
        "patterns": pattern_rows,
        "own_segmentation": segment(
            shakespeare_tokenizer, "First Citizen:\nBefore we proceed any further"
        ),
        "tiktoken": None,
    }

    # --- verification against tiktoken ---
    try:
        import tiktoken
    except ImportError:
        return result

    ours = BytePairEncoding.from_tiktoken("cl100k_base")
    reference = tiktoken.get_encoding("cl100k_base")
    samples = [
        "Hello, world!",
        "Xin chào — attention và BPE.",
        "\n\nTo be, or not to be;\n",
        "emoji: 🐍 café naïve 12345",
        "def forward(self, x): return self.mlp(x)",
    ]
    checks = []
    for sample in samples:
        ours_ids = ours.encode(sample)
        reference_ids = reference.encode(sample)
        checks.append(
            {
                "text": sample,
                "ours": ours_ids,
                "reference": reference_ids,
                "match": ours_ids == reference_ids,
                "pieces": [
                    show_bytes(reference.decode_single_token_bytes(token_id))
                    for token_id in reference_ids
                ],
            }
        )

    big = text[:200_000]
    big_ours = ours.encode(big)
    big_reference = reference.encode(big)

    special = "hello <|endoftext|> world"
    special_ours = ours.encode(special, allowed_special="all")
    special_reference = reference.encode(special, allowed_special={"<|endoftext|>"})

    result["tiktoken"] = {
        "version": tiktoken.__version__,
        "encoding": "cl100k_base",
        "merges_recovered": len(ours.merges),
        "checks": checks,
        "bulk": {
            "chars": len(big),
            "tokens": len(big_reference),
            "match": big_ours == big_reference,
        },
        "special": {
            "text": special,
            "ours": special_ours,
            "reference": special_reference,
            "match": special_ours == special_reference,
        },
        "cl100k_segmentation": segment(ours, "First Citizen:\nBefore we proceed any further"),
    }
    return result


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="outputs_validation_1000")
    parser.add_argument("--data-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--out", default="outputs_viz/report.html")
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bpe-chars", type=int, default=120_000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-part3", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"device={device}")

    print("part 2: loading checkpoint and running forward passes ...")
    part2 = collect_part2(args, device)
    print(
        f"  measured val loss {part2['measured']['val']:.4f} "
        f"(ppl {part2['measured_val_ppl']:.3f})"
    )

    part3 = {}
    if not args.skip_part3:
        print("part 3: training probe tokenizers and comparing to tiktoken ...")
        part3 = collect_part3(args)
        if part3.get("tiktoken"):
            print(f"  bulk match vs tiktoken: {part3['tiktoken']['bulk']['match']}")

    out_path, artifact_path = write_report(
        HERE / args.out,
        HERE / "report_template.html",
        {"part2": part2, "part3": part3},
    )
    print(f"wrote {out_path} ({out_path.stat().st_size/1024:.0f} KB) - open this one")
    print(f"     {artifact_path.name} - body-only copy for publishing")


if __name__ == "__main__":
    main()
