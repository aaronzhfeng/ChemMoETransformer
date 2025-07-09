"""
models/embedding.py
-------------------

Token, positional, and (optional) segment‑aware positional encodings.

Why separate file?
  * Keeps embedding logic decoupled from Transformer blocks
  * Allows easy swap between sinusoidal, learned, or compound‑aware encodings
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Standard token embedding + √d_model scale factor."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, B) int64 token ids
        return self.embed(x) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """Classic Transformer sinusoidal PE (Vaswani et al., 2017).

    Args
    ----
    d_model : int
        Embedding dimension.
    dropout : float
        Dropout applied after adding positions.
    max_len : int
        Pre‑compute PE up to this length (can be > longest seq).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre‑compute table of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (max_len, d_model) -> (1, max_len, d_model) so it can be broadcast in add_
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (L, B, d_model).

        Returns
        -------
        Tensor of same shape with positional encodings added then dropout.
        """
        # x is (L, B, d_model); need to match (L, 1, d_model)
        L = x.size(0)
        x = x + self.pe[:, :L].transpose(0, 1)
        return self.dropout(x)


class CompoundPositionalEncoding(nn.Module):
    """
    Segment‑aware positional encoding inspired by ChemTransformer.

    Instead of unique position index per *token*, we keep one learned
    embedding per *compound segment* (e.g. all tokens belonging to same
    reactant share position id 0, next reactant id 1, etc.).  You pass in
    a tensor of segment ids shaped (L, B) alongside tokens.

    When segment ids are None we fall back to the sinusoidal encoding.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_segments: int = 512,
        fallback: Optional[SinusoidalPositionalEncoding] = None,
    ):
        super().__init__()
        self.seg_embed = nn.Embedding(max_segments, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fallback = fallback

    def forward(
        self, x: torch.Tensor, segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            (L, B, d_model)
        segment_ids : Tensor | None
            (L, B) ints. 0‑based compound id per token.

        Returns
        -------
        Tensor (L, B, d_model)
        """
        if segment_ids is None:
            if self.fallback is None:
                return x
            return self.fallback(x)

        x = x + self.seg_embed(segment_ids)
        return self.dropout(x)
