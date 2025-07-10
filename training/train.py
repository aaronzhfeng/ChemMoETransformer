#!/usr/bin/env python
"""
training/train.py
-----------------

Usage
-----
$ python training/train.py config/example_moe_config.yaml --debug
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import sys
sys.path.append('.')
from utils.config_parser import load_config
from utils.dataset_utils import build_dataloader, _load_vocab
from models import build_model
from training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to YAML config")
    p.add_argument("--debug", action="store_true", help="limit dataset to 1k samples")
    return p.parse_args()


def subset_loader(dl, n_samples=1000):
    from torch.utils.data import Subset

    ds = dl.dataset
    idx = list(range(min(n_samples, len(ds))))
    new_ds = Subset(ds, idx)
    return torch.utils.data.DataLoader(
        new_ds,
        batch_size=dl.batch_size,
        shuffle=False,
        num_workers=dl.num_workers,
        collate_fn=dl.collate_fn,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    # ----- vocab & data loaders -----
    vocab = _load_vocab(cfg.data.vocab_file)
    vocab_size = len(vocab)

    dl_train = build_dataloader(
        root=cfg.data.data_root,
        split="train",
        batch_size=cfg.data.batch_size,
        vocab_file=cfg.data.vocab_file,
        num_workers=cfg.data.num_workers,
    )
    dl_val = build_dataloader(
        root=cfg.data.data_root,
        split="val",
        batch_size=cfg.data.batch_size,
        vocab_file=cfg.data.vocab_file,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    if args.debug:
        dl_train = subset_loader(dl_train, 1000)
        dl_val = subset_loader(dl_val, 1000)

    # ----- model -----
    model = build_model(cfg, vocab_size).to(cfg.training.device)

    # ----- trainer -----
    log_path = Path(cfg.output.log_file)
    trainer = Trainer(
        model,
        cfg,
        dataloaders={"train": dl_train, "val": dl_val},
        vocab_size=vocab_size,
        log_path=log_path,
    )
    trainer.train()


if __name__ == "__main__":
    main()
