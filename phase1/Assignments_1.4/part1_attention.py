from __future__ import annotations

import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[:-1] != key.shape[:-1] and query.ndim == key.ndim:
            if query.shape[:-2] != key.shape[:-2]:
                raise ValueError("query and key batch/head dimensions differ.")
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("key and value sequence lengths must match.")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("query and key feature dimensions must match.")

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(query.size(-1))

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask.to(dtype=scores.dtype)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, value)
        return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.last_attention_weights: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x = x.view(batch, length, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, d_model).")
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {x.size(-1)}.")
        if context is None:
            context = x
        if context.ndim != 3 or context.size(0) != x.size(0):
            raise ValueError("context must have shape (batch, sequence, d_model).")
        if context.size(-1) != self.d_model:
            raise ValueError("context feature dimension must equal d_model.")

        if is_causal:
            if context is not x:
                raise ValueError("is_causal=True is only valid for self-attention.")
            query_length = x.size(1)
            causal = torch.tril(
                torch.ones(
                    query_length,
                    query_length,
                    dtype=torch.bool,
                    device=x.device,
                )
            )
            mask = causal if mask is None else (mask & causal)

        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(context))
        value = self._split_heads(self.v_proj(context))
        attended, weights = self.attention(query, key, value, mask=mask)
        self.last_attention_weights = weights
        output = self.out_proj(self._merge_heads(attended))
        return self.resid_dropout(output)
