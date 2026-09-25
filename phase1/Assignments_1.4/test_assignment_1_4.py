"""Fast correctness checks for Assignment 1.4."""

from __future__ import annotations

import argparse

import torch

from part1_attention import MultiHeadAttention, ScaledDotProductAttention
from part1_transformer import TransformerBlock
from part2_gpt import GPT, GPTConfig
from part3_bpe import BytePairEncoding


def check_attention() -> None:
    torch.manual_seed(7)
    attention = ScaledDotProductAttention()
    q = torch.randn(2, 3, 5)
    k = torch.randn(2, 4, 5)
    v = torch.randn(2, 4, 6)
    output, weights = attention(q, k, v)
    expected_weights = torch.softmax(q @ k.transpose(-2, -1) / 5**0.5, dim=-1)
    expected_output = expected_weights @ v
    assert output.shape == (2, 3, 6)
    assert torch.allclose(weights, expected_weights, atol=1e-6)
    assert torch.allclose(output, expected_output, atol=1e-6)

    mha = MultiHeadAttention(d_model=12, n_heads=3).eval()
    x = torch.randn(2, 5, 12)
    causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    first = mha(x, mask=causal)
    changed = x.clone()
    changed[:, 1:] += 100.0
    second = mha(changed, mask=causal)
    assert torch.allclose(first[:, :1], second[:, :1], atol=1e-5)
    assert first.shape == x.shape
    print("attention: PASS")


def check_transformer_and_gpt() -> None:
    torch.manual_seed(11)
    block = TransformerBlock(d_model=16, n_heads=4, dropout=0.0)
    x = torch.randn(2, 8, 16, requires_grad=True)
    y = block(x, is_causal=True)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    model = GPT(
        GPTConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=4,
            n_embd=32,
            dropout=0.0,
        )
    )
    idx = torch.randint(0, 32, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, 32)
    assert loss is not None and torch.isfinite(loss)
    generated = model.generate(idx[:, :2], max_new_tokens=4, do_sample=False)
    assert generated.shape == (3, 6)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    print("transformer/gpt: PASS")


def check_bpe(require_tiktoken: bool) -> None:
    text = "To be, or not to be — that is the question.\nTo be!"
    tokenizer = BytePairEncoding()
    tokenizer.train(text * 20, vocab_size=300)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text

    try:
        import tiktoken
    except ImportError:
        if require_tiktoken:
            raise
        print("bpe: PASS (round-trip; tiktoken not installed, comparison skipped)")
        return

    ours = BytePairEncoding.from_tiktoken("cl100k_base")
    reference = tiktoken.get_encoding("cl100k_base")
    samples = [
        "Hello, world!",
        "Xin chào — attention và BPE.",
        "\n\nTo be, or not to be;\n",
        "emoji: 🐍 café naïve 12345",
    ]
    for sample in samples:
        ours_ids = ours.encode(sample)
        reference_ids = reference.encode(sample)
        assert ours_ids == reference_ids, (sample, ours_ids, reference_ids)
        assert ours.decode(ours_ids) == reference.decode(reference_ids)
    special_sample = "hello <|endoftext|> world"
    ours_special = ours.encode(special_sample, allowed_special="all")
    reference_special = reference.encode(
        special_sample, allowed_special={"<|endoftext|>"}
    )
    assert ours_special == reference_special
    print("bpe: PASS (round-trip + exact cl100k_base tiktoken comparison)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-tiktoken", action="store_true")
    args = parser.parse_args()
    check_attention()
    check_transformer_and_gpt()
    check_bpe(args.require_tiktoken)
    print("all checks: PASS")


if __name__ == "__main__":
    main()
