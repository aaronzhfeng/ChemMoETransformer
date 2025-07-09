"""
models/transformer_layers.py
----------------------------

Baseline Transformer encoder & decoder blocks.

The only deviation from PyTorch's built‑in nn.TransformerEncoderLayer is
that **the feed‑forward sub‑layer is constructed via a factory** so we can
swap in MoE later (batch 3).  Until MoE is implemented we default to the
standard PositionwiseFeedForward block.

Notation
--------
Input shapes strictly follow PyTorch convention: (S, B, D)
  * S - sequence length
  * B - batch size
  * D - d_model
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Position‑wise Feed‑Forward (dense version)
# ---------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Placeholder for MoE FFN – will be imported in Batch 3
try:
    from models.moe_layer import MoEFFN  # noqa: F401
except ModuleNotFoundError:
    MoEFFN = None  # type: ignore


def _ffn_factory(
    use_moe: bool, d_model: int, d_ff: int, num_experts: int, dropout: float
) -> nn.Module:
    """Return Dense FFN or MoE FFN depending on flag."""
    if use_moe:
        if MoEFFN is None:
            raise ImportError("MoEFFN not yet implemented – wait for Batch 3.")
        return MoEFFN(d_model, d_ff, n_experts=num_experts, dropout=dropout)
    return PositionwiseFeedForward(d_model, d_ff, dropout)


# ---------------------------------------------------------------------
# Encoder & Decoder Layers
# ---------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """One encoder block: self‑attention → FFN (+ norm & residual)."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        use_moe: bool = False,
        num_experts: int = 0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = _ffn_factory(use_moe, d_model, d_ff, num_experts, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        # Self‑attention expects (L, B, D) input
        attn_out, _ = self.self_attn(
            src, src, src, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        ffn_out = self.ffn(src)
        src = src + self.dropout2(ffn_out)
        src = self.norm2(src)

        return src


class DecoderLayer(nn.Module):
    """One decoder block: masked self‑attn → x‑attn → FFN."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        use_moe: bool = False,
        num_experts: int = 0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=False
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=False
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = _ffn_factory(use_moe, d_model, d_ff, num_experts, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        # Masked self‑attention
        self_attn_out, _ = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(self_attn_out)
        tgt = self.norm1(tgt)

        # Encoder‑decoder (cross) attention
        cross_out, _ = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=None,
        )
        tgt = tgt + self.dropout2(cross_out)
        tgt = self.norm2(tgt)

        # FFN
        ffn_out = self.ffn(tgt)
        tgt = tgt + self.dropout3(ffn_out)
        tgt = self.norm3(tgt)

        return tgt


# ---------------------------------------------------------------------
# Helper mask generator
# ---------------------------------------------------------------------
def subsequent_mask(size: int) -> torch.Tensor:
    """Square subsequent mask (L, L) with -inf above diag."""
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    mask = mask.float().masked_fill(mask, float("-inf"))
    return mask
