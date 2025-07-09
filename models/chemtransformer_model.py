"""
models/chemtransformer_model.py
--------------------------------

High‑level wrapper around:

    * Token + positional embeddings
    * N × EncoderLayer
    * N × DecoderLayer
    * Final linear generator (d_model → vocab)

Handles:
    * Building MoE vs dense layers per `moe_layers` list (1‑based index)
    * Creating source pad masks + subsequent decoder mask
    * Aggregating `aux_loss` from every MoEFFN layer
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.embedding import TokenEmbedding, SinusoidalPositionalEncoding
from models.transformer_layers import (
    EncoderLayer,
    DecoderLayer,
    subsequent_mask,
)
from utils.dataset_utils import PAD_ID, BOS_ID, EOS_ID


class ChemMoETransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        moe_layers: List[int],
        num_experts: int,
        padding_idx: int = PAD_ID,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        # Embedding + PE
        self.tok_embed = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout)

        # Encoder
        enc_layers: List[nn.Module] = []
        for i in range(1, n_layer + 1):  # 1‑based for readability
            use_moe = i in moe_layers
            enc_layers.append(
                EncoderLayer(
                    d_model,
                    n_head,
                    d_ff,
                    dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                )
            )
        self.encoder = nn.ModuleList(enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        for i in range(1, n_layer + 1):
            use_moe = i in moe_layers
            dec_layers.append(
                DecoderLayer(
                    d_model,
                    n_head,
                    d_ff,
                    dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                )
            )
        self.decoder = nn.ModuleList(dec_layers)

        # Final generator
        self.generator = nn.Linear(d_model, vocab_size, bias=False)

        # buffer for caching tgt masks
        self.register_buffer("_tgt_mask_cache", torch.empty(0), persistent=False)

    # -----------------------------------------------------------------
    #  Helper: create / reuse subsequent mask
    # -----------------------------------------------------------------
    def _get_subsequent_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        if self._tgt_mask_cache.size(0) >= tgt_len:
            return self._tgt_mask_cache[:tgt_len, :tgt_len]
        mask = subsequent_mask(tgt_len).to(device)
        self._tgt_mask_cache = mask
        return mask

    # -----------------------------------------------------------------
    #  Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        src_tokens: torch.Tensor,       # (L_src, B)
        decoder_in: torch.Tensor,       # (L_tgt, B) shifted‑right <BOS> + product‑prefix
        src_pad_mask: torch.Tensor,     # (B, L_src) True at PAD positions
        tgt_pad_mask: torch.Tensor,     # (B, L_tgt)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits   : (L_tgt, B, vocab_size)
        aux_loss : scalar tensor (sum of MoE load‑balancing losses)
        """
        device = src_tokens.device

        # ---------- Encoder ----------
        src_emb = self.pos_enc(self.tok_embed(src_tokens))
        enc_out = src_emb
        for layer in self.encoder:
            enc_out = layer(enc_out, src_key_padding_mask=src_pad_mask)

        # ---------- Decoder ----------
        tgt_mask = self._get_subsequent_mask(decoder_in.size(0), device)
        tgt_emb = self.pos_enc(self.tok_embed(decoder_in))
        dec_out = tgt_emb
        for layer in self.decoder:
            dec_out = layer(
                dec_out,
                enc_out,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )

        logits = self.generator(dec_out)

        # ---------- Aggregate MoE auxiliary loss ----------
        aux_loss = torch.tensor(0.0, device=device)
        for layer in list(self.encoder) + list(self.decoder):
            if hasattr(layer.ffn, "aux_loss"):
                aux_loss = aux_loss + layer.ffn.aux_loss
        return logits, aux_loss
