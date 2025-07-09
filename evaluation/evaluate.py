#!/usr/bin/env python
"""
evaluation/evaluate.py
----------------------

Example
-------
$ python evaluation/evaluate.py \
      --config config/example_moe_config.yaml  \
      --checkpoint checkpoints/model_best.pt   \
      --split test
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch

from utils.config_parser import load_config
from utils.dataset_utils import build_dataloader, _load_vocab
from models import build_model
from evaluation.metrics import token_accuracy, sequence_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config used at train time")
    p.add_argument("--checkpoint", required=True, help="*.pt file from training")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=None, help="override batch size")
    return p.parse_args()


@torch.no_grad()
def run_eval(
    model,
    dl,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_tok_acc = total_seq_acc = 0.0
    total_aux = 0.0
    n_batches = 0

    for batch in dl:
        src = batch["src_tokens"].to(device)
        dec_in = batch["decoder_in"].to(device)
        tgt = batch["tgt_tokens"].to(device)
        src_mask = batch["src_pad_mask"].to(device)
        tgt_mask = batch["tgt_pad_mask"].to(device)

        logits, aux = model(src, dec_in, src_mask, tgt_mask)

        total_tok_acc += token_accuracy(logits, tgt)
        total_seq_acc += sequence_accuracy(logits, tgt)
        total_aux += aux.item()
        n_batches += 1

    return {
        "token_acc": total_tok_acc / n_batches,
        "seq_acc": total_seq_acc / n_batches,
        "aux_loss": total_aux / n_batches,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.training.device)

    # ----- vocab & data loader -----
    vocab = _load_vocab(cfg.data.vocab_file)
    vocab_size = len(vocab)

    batch_size = args.batch_size or cfg.data.batch_size
    dl = build_dataloader(
        root=cfg.data.data_root,
        split=args.split,
        batch_size=batch_size,
        vocab_file=cfg.data.vocab_file,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    # ----- model -----
    model = build_model(cfg, vocab_size).to(device)

    # ----- load checkpoint -----
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)

    # ----- evaluate -----
    metrics = run_eval(model, dl, device)
    print(
        f"[{args.split}] Token‑Acc={metrics['token_acc']:.4f} "
        f"Seq‑Acc={metrics['seq_acc']:.4f} "
        f"Aux‑Loss={metrics['aux_loss']:.4f}"
    )

    # ----- save csv next to checkpoint -----
    out_csv = Path(args.checkpoint).with_name("eval_results.csv")
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["checkpoint", "split", "token_acc", "seq_acc", "aux_loss"])
        writer.writerow(
            [
                Path(args.checkpoint).name,
                args.split,
                f"{metrics['token_acc']:.4f}",
                f"{metrics['seq_acc']:.4f}",
                f"{metrics['aux_loss']:.4f}",
            ]
        )
    print(f"Results appended to {out_csv}")


if __name__ == "__main__":
    main()
