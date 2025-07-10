"""
models/moe_layer.py
-------------------

Homogeneous Mixture‑of‑Experts feed‑forward network with *expert‑choice*
routing (Zhou et al., 2022).  Designed as a **drop‑in replacement** for the
dense Position‑wise FFN inside Transformer blocks.

Key design points
-----------------
*  All experts share the same architecture (two‑layer FFN).
*  Router is a single linear layer → softmax over experts.
*  *Expert‑choice* routing = each expert claims up to `capacity`
   tokens; tokens that would overflow are sent to a fallback expert 0.
*  Two auxiliary load‑balancing losses are returned:
     (i)  Importance loss   – encourages uniform router probabilities
     (ii) Capacity  loss    – penalises overflow to keep load balanced
   They are stored on `self.aux_loss` so the training loop can add
   them to the main Cross‑Entropy loss (weight to be set in trainer).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Expert sub‑network (dense FFN)
# ---------------------------------------------------------------------
class _DenseExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(F.relu(self.lin1(x))))


# ---------------------------------------------------------------------
# MoE‑FFN with Expert‑Choice Routing
# ---------------------------------------------------------------------
class MoEFFN(nn.Module):
    """
    Parameters
    ----------
    d_model : int
        Model hidden size.
    d_ff : int
        Hidden size inside each expert FFN.
    n_experts : int
        Number of homogeneous expert networks.
    dropout : float
        Dropout used in experts.
    capacity_factor : float
        Multiplier controlling tokens/expert capacity.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        # Router : (d_model) → logits over experts
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # FFN experts
        self.experts = nn.ModuleList(
            [_DenseExpert(d_model, d_ff, dropout=dropout) for _ in range(n_experts)]
        )

        # slot to store aux loss for trainer
        self.aux_loss = torch.tensor(0.0)

    # ----------------------- helpers ---------------------------------
    @staticmethod
    def _importance_loss(router_probs: torch.Tensor) -> torch.Tensor:
        """Encourage router to give equal mass to each expert."""
        # router_probs: (T, n_experts)
        expert_prob = router_probs.mean(dim=0)                  # (n_experts,)
        return (expert_prob * expert_prob).sum() * router_probs.size(0)

    @staticmethod
    def _capacity_loss(
        assignments: torch.Tensor,  # (T,) int64 expert id or -1 for overflow
        n_experts: int,
        capacity: int,
    ) -> torch.Tensor:
        """Penalise experts that overflow their capacity."""
        counts = torch.bincount(
            torch.clamp(assignments, min=0), minlength=n_experts
        ).float()  # (n_experts,)
        overflow = torch.clamp(counts - capacity, min=0.0)
        return overflow.sum()

    # ------------------------ core forward ---------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (L, B, d_model).

        Returns
        -------
        Same shape tensor.  self.aux_loss stores load‑balancing loss.
        """
        L, B, D = x.shape
        T = L * B                          # number of tokens

        # Flatten tokens so routing is token‑wise
        tokens = x.reshape(T, D)           # (T, D)

        # ---------------- Router softmax -----------------------------
        logits = self.router(tokens)                       # (T, n_experts)
        router_probs = F.softmax(logits, dim=1)

        # Importance loss (Eq. 1 in GShard/GLaM); scale by T
        imp_loss = self._importance_loss(router_probs)

        # ---------------- Expert‑choice assignment --------------------
        # Sort tokens per expert by descending probability so each expert
        # can *choose* its favourite tokens up to `capacity`.
        capacity = int(math.ceil(T / self.n_experts * self.capacity_factor))

        # argmax expert for tie‑break ranking
        top1_prob, top1_idx = router_probs.max(dim=1)      # (T,)
        # For each expert build (prob, token_id) list
        expert_lists: List[List[Tuple[float, int]]] = [[] for _ in range(self.n_experts)]
        for tok_id, (e, p) in enumerate(zip(top1_idx.tolist(), top1_prob.tolist())):
            expert_lists[e].append((p, tok_id))

        # Choose up to `capacity` highest‑prob tokens for each expert
        assignments = torch.full((T,), -1, dtype=torch.long, device=x.device)
        for e, lst in enumerate(expert_lists):
            lst.sort(key=lambda z: z[0], reverse=True)
            chosen = [tok_id for _, tok_id in lst[:capacity]]
            idx = torch.tensor(chosen, dtype=torch.long, device=x.device)
            assignments[idx] = e

        cap_loss = self._capacity_loss(assignments, self.n_experts, capacity)

        # ---------------- Dispatch to experts ------------------------
        expert_inputs = [[] for _ in range(self.n_experts)]
        for tok_id, e in enumerate(assignments.tolist()):
            if e >= 0:  # -1 means overflow (we route to expert 0)
                expert_inputs[e].append(tok_id)
            else:
                expert_inputs[0].append(tok_id)

        # Gather -> run expert -> scatter, with dtype alignment under AMP
        out_tokens = torch.empty_like(tokens)             # (T, D)
        for e, tok_ids in enumerate(expert_inputs):
            if not tok_ids:
                continue
            idx = torch.tensor(tok_ids, device=x.device)
            expert_out = self.experts[e](tokens[idx])
            # Cast to match out_tokens’ dtype (e.g. FP16 under autocast)
            expert_out = expert_out.to(out_tokens.dtype)
            out_tokens[idx] = expert_out

        # Reshape back to (L, B, D)
        out = out_tokens.reshape(L, B, D)

        # Store auxiliary loss for trainer (importance + capacity, normalised)
        self.aux_loss = (imp_loss + cap_loss) / T
        return out
