"""Transformer block built from the attention primitives in ``attention.py``."""

from __future__ import annotations

import torch
from torch import nn

from part1_attention import MultiHeadAttention

"""Feed forward network"""
class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=bias),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(
            d_model=d_model,
            expansion=expansion,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Encoder block forward."""
        x = x + self.attn(self.ln_1(x), mask=mask, is_causal=is_causal)
        x = x + self.mlp(self.ln_2(x))
        return x
