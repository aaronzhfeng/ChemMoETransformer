"""
training/trainer.py
-------------------

Handles one full training run:
  * epoch loop over train DataLoader
  * validation at end of each epoch
  * checkpointing + logging

Assumes:
  * model returns  (logits, aux_loss)
  * build_dataloader returns dict‑style batches
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


class Trainer:
    def __init__(self, model, cfg, dataloaders, vocab_size: int, log_path: Path):
        self.model = model
        self.cfg = cfg
        self.train_dl = dataloaders["train"]
        self.val_dl = dataloaders["val"]
        self.device = torch.device(cfg.training.device)
        self.pad_id = cfg.special.PAD_ID if hasattr(cfg, "special") else 0
        self.seq_eos_id = cfg.special.EOS_ID if hasattr(cfg, "special") else 3

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.max_lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.max_epoch,
            eta_min=cfg.training.min_lr,
        )
        self.grad_clip = cfg.training.grad_clip
        self.aux_w = getattr(cfg.training, "aux_loss_weight", 0.02)

        self.scaler = GradScaler()  # AMP
        self.vocab_size = vocab_size
        self.ckpt_dir = Path(cfg.output.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self._init_log()

    # ------------------------------------------------------------------
    def _init_log(self):
        with open(self.log_path, "w") as f:
            f.write("epoch,split,loss,tok_acc,seq_acc,aux_loss\n")

    def _log(self, epoch, split, loss, tok_acc, seq_acc, aux):
        line = f"{epoch},{split},{loss:.4f},{tok_acc:.4f},{seq_acc:.4f},{aux:.4f}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------
    @staticmethod
    def _accuracy(logits, tgt_tokens, pad_id, eos_id):
        # logits: (L, B, V), tgt_tokens: (L, B)
        pred = logits.argmax(dim=-1)  # (L, B)
        mask = tgt_tokens.ne(pad_id)
        tok_correct = pred.eq(tgt_tokens) & mask
        tok_acc = tok_correct.sum().item() / mask.sum().item()

        # seq accuracy: correct only if all positions correct up to EOS (or PAD)
        # Flatten batch dimension
        L, B = tgt_tokens.shape
        seq_correct = 0
        for b in range(B):
            # from 0 .. first PAD or EOS
            ref = tgt_tokens[:, b]
            until = (ref == eos_id).nonzero(as_tuple=False)
            if len(until) > 0:
                ref_len = until[0].item() + 1
            else:
                ref_len = (ref != pad_id).sum().item()
            if ref_len == 0:
                continue
            if pred[:ref_len, b].eq(ref[:ref_len]).all():
                seq_correct += 1
        seq_acc = seq_correct / B
        return tok_acc, seq_acc

    # ------------------------------------------------------------------
    def _run_epoch(self, epoch, train: bool = True):
        dl = self.train_dl if train else self.val_dl
        self.model.train(train)

        total_loss = total_aux = 0.0
        total_tok, correct_tok = 0, 0
        total_seq, correct_seq = 0, 0

        for batch in dl:
            src = batch["src_tokens"].to(self.device)
            dec_in = batch["decoder_in"].to(self.device)
            tgt = batch["tgt_tokens"].to(self.device)
            src_mask = batch["src_pad_mask"].to(self.device)
            tgt_mask = batch["tgt_pad_mask"].to(self.device)

            with autocast():
                logits, aux = self.model(src, dec_in, src_mask, tgt_mask)
                L, B, V = logits.shape
                loss = self.criterion(logits.view(L * B, V), tgt.view(L * B))
                loss = loss + self.aux_w * aux

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            tok_acc, seq_acc = self._accuracy(logits.detach(), tgt, self.pad_id, self.seq_eos_id)

            total_loss += loss.item()
            total_aux += aux.item()
            total_tok += 1
            correct_tok += tok_acc
            total_seq += 1
            correct_seq += seq_acc

        avg_loss = total_loss / total_tok
        avg_aux = total_aux / total_tok
        avg_tok_acc = correct_tok / total_tok
        avg_seq_acc = correct_seq / total_seq
        split = "train" if train else "val"
        self._log(epoch, split, avg_loss, avg_tok_acc, avg_seq_acc, avg_aux)
        return avg_loss

    # ------------------------------------------------------------------
    def train(self):
        best_val = math.inf
        for epoch in range(1, self.cfg.training.max_epoch + 1):
            self._run_epoch(epoch, train=True)
            with torch.no_grad():
                val_loss = self._run_epoch(epoch, train=False)

            self.scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                self._save_ckpt(epoch, best=True)
            if epoch % self.cfg.output.ckpt_interval == 0:
                self._save_ckpt(epoch)

    # ------------------------------------------------------------------
    def _save_ckpt(self, epoch, best=False):
        tag = "best" if best else f"epoch{epoch}"
        ckpt_path = self.ckpt_dir / f"model_{tag}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            ckpt_path,
        )
