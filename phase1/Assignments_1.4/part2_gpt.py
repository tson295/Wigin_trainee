"""A small decoder-only GPT assembled from the local attention primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from part1_transformer import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384 #d_model
    dropout: float = 0.2
    bias: bool = True


class GPT(nn.Module):
    """Character-level GPT with tied input/output embeddings."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if config.block_size <= 0:
            raise ValueError("block_size must be positive.")

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.n_embd,
                    n_heads=config.n_head,
                    expansion=4,
                    dropout=config.dropout,
                    bias=config.bias,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence).")
        batch, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size "
                f"{self.config.block_size}."
            )
        if idx.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise TypeError("idx must contain integer token IDs.")

        positions = torch.arange(sequence_length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        x = self.drop(x)
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=idx.device,
            )
        )
        for block in self.blocks:
            x = block(x, mask=causal_mask, is_causal=False)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx.")
            loss = F.cross_entropy(
                logits.reshape(batch * sequence_length, -1),
                targets.reshape(batch * sequence_length),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressively append tokens to ``idx``."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = idx[:, -self.config.block_size :]
                logits, _ = self(context)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    if top_k <= 0:
                        raise ValueError("top_k must be positive.")
                    top_k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, top_k)
                    logits[logits < values[:, [-1]]] = float("-inf")
                probabilities = F.softmax(logits, dim=-1)
                if do_sample:
                    next_token = torch.multinomial(probabilities, num_samples=1)
                else:
                    next_token = probabilities.argmax(dim=-1, keepdim=True)
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            self.train(was_training)
        return idx

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.position_embedding.weight.numel()
        return total
