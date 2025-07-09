import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.dataset_utils import (
    build_dataloader,
    _load_vocab,
    PAD_ID,
    BOS_ID,
    EOS_ID,
)


def _make_toy_corpus(root: Path):
    # 4 special + 3 real tokens
    vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "C", "O", ">"]
    (root / "vocab_smiles.txt").write_text("\n".join(vocab))

    src_lines = ["C C", "C O"]
    tgt_lines = ["> C", "> O"]
    for split in ["train", "val", "test"]:
        (root / f"src-{split}.txt").write_text("\n".join(src_lines))
        (root / f"tgt-{split}.txt").write_text("\n".join(tgt_lines))


def test_loader_smiles_txt():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_toy_corpus(root)

        dl = build_dataloader(
            root=root,
            split="train",
            batch_size=2,
            vocab_file=root / "vocab_smiles.txt",
            num_workers=0,
        )
        batch = next(iter(dl))
        assert batch["src_tokens"].shape[1] == 2  # B=2
        # Check BOS token inserted in decoder_in[0]
        assert (batch["decoder_in"][0] == BOS_ID).all()
        # No PAD tokens in tiny example
        assert batch["src_pad_mask"].sum() == 0


def test_vocab_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_toy_corpus(root)
        vocab = _load_vocab(root / "vocab_smiles.txt")
        assert vocab["C"] == 4
        assert vocab["<PAD>"] == PAD_ID
        assert vocab["<EOS>"] == EOS_ID
