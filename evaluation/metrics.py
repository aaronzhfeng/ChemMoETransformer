"""
evaluation/metrics.py
---------------------

Standalone helpers shared by the evaluate script and (optionally)
future notebooks.

Conventions
-----------
* logits shape  (L, B, V)
* tokens shape  (L, B)
* special.ids imported from utils.dataset_utils
"""
from __future__ import annotations

import torch
from utils.dataset_utils import PAD_ID, EOS_ID


def token_accuracy(logits: torch.Tensor, tgt_tokens: torch.Tensor) -> float:
    """Fraction of non‑PAD tokens predicted correctly."""
    pred = logits.argmax(dim=-1)                     # (L, B)
    mask = tgt_tokens.ne(PAD_ID)
    correct = (pred.eq(tgt_tokens) & mask).sum()
    return (correct / mask.sum()).item()


def sequence_accuracy(logits: torch.Tensor, tgt_tokens: torch.Tensor) -> float:
    """Exact match on full product sequence (Top‑1)."""
    pred = logits.argmax(dim=-1)                     # (L, B)
    L, B = tgt_tokens.shape
    correct = 0
    for b in range(B):
        ref = tgt_tokens[:, b]
        end = (ref == EOS_ID).nonzero(as_tuple=False)
        if len(end):
            ref_len = end[0].item() + 1
        else:
            ref_len = (ref != PAD_ID).sum().item()
        if ref_len == 0:
            continue
        if pred[:ref_len, b].eq(ref[:ref_len]).all():
            correct += 1
    return correct / B
