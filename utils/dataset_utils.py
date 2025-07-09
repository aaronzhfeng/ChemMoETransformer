"""
utils/dataset_utils.py  –  v2  (TXT token lists → ID tensors)

Supports two disk layouts:

A.  *.npz  (already‑binarised)                        – unchanged
B.  src‑*.txt / tgt‑*.txt  +  vocab.{pkl|txt}         – NEW

If *.npz is present we still prefer it because it’s faster, but when only
the tokenised TXT files are found we:

1. load a vocabulary file once (dict: token → int id)
2. map each token to its id (unknown → _UNK)
3. append the _EOS id
4. return integer tensors ready for the Transformer.

Token id conventions follow Graph2SMILES  
(_PAD = 0, _UNK = 1, _SOS = 2, _EOS = 3) :contentReference[oaicite:1]{index=1}
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# -------------------------------------------------------------------------
#  Special symbols – keep in ONE place to avoid magic numbers everywhere
# -------------------------------------------------------------------------
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2   # <SOS>  (we use it as <BOS>)
EOS_ID = 3


# -------------------------------------------------------------------------
#  Helper: vocab loader (handles .pkl or plain‑text)
# -------------------------------------------------------------------------
def _load_vocab(vocab_file: Union[str, Path]) -> dict[str, int]:
    vocab_path = Path(vocab_file)
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    if vocab_path.suffix == ".pkl":
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:  # txt
        vocab = {}
        with open(vocab_path, "r") as f:
            for i, line in enumerate(f):
                token = line.strip().split("\t")[0]
                vocab[token] = i
    return vocab


# -------------------------------------------------------------------------
#  Core dataclass
# -------------------------------------------------------------------------
@dataclass
class ReactionExample:
    src: Sequence[int]  # list[int]  (already ends with EOS_ID)
    tgt: Sequence[int]


# -------------------------------------------------------------------------
#  Dataset
# -------------------------------------------------------------------------
class ReactionDataset(Dataset):
    def __init__(self, root: Union[str, Path], split: str, vocab_file: str | Path):
        """
        Parameters
        ----------
        root        path holding either *.npz OR src‑/tgt‑*.txt files
        split       "train" | "val" | "test"
        vocab_file  vocabulary mapping tokens → ids
        """
        self.root = Path(root)
        self.split = split
        self.vocab = _load_vocab(vocab_file)

        # --- prefer pre‑binarised .npz if available ----------------------
        npz_file = self.root / f"{split}.npz"
        if npz_file.exists():
            data = np.load(npz_file)
            self.src = [row.tolist() for row in data["src_token_ids"]]
            self.tgt = [row.tolist() for row in data["tgt_token_ids"]]
            return

        # --- otherwise fall back to tokenised TXT -----------------------
        src_txt = self.root / f"src-{split}.txt"
        tgt_txt = self.root / f"tgt-{split}.txt"
        if not (src_txt.exists() and tgt_txt.exists()):
            raise FileNotFoundError(
                f"Could not find .npz or src/tgt txt files for split='{split}' inside {root}"
            )

        self.src, self.tgt = self._load_txt_pair(src_txt, tgt_txt)

    # ---------------- internal helpers ---------------------------------
    def _line_to_ids(self, line: str) -> List[int]:
        ids = []
        for tok in line.strip().split():
            ids.append(self.vocab.get(tok, UNK_ID))
        ids.append(EOS_ID)
        return ids

    def _load_txt_pair(self, src_file: Path, tgt_file: Path) -> Tuple[List, List]:
        with open(src_file, "r") as f:
            src_lines = f.readlines()
        with open(tgt_file, "r") as f:
            tgt_lines = f.readlines()

        assert len(src_lines) == len(tgt_lines), "src/tgt line count mismatch"

        src_ids, tgt_ids = [], []
        for s, t in zip(src_lines, tgt_lines):
            src_ids.append(self._line_to_ids(s))
            tgt_ids.append(self._line_to_ids(t))
        return src_ids, tgt_ids

    # ---------------- Dataset API --------------------------------------
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return ReactionExample(self.src[idx], self.tgt[idx])


# -------------------------------------------------------------------------
#  Collate & DataLoader
# -------------------------------------------------------------------------
def _pad(seqs: List[Sequence[int]], pad_id: int = PAD_ID) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    batch = batch.t().contiguous()                   # (L, B)
    key_pad = batch.eq(pad_id).t()                   # (B, L)
    return batch, key_pad


def reaction_collate_fn(batch: List[ReactionExample]):
    src, tgt = [b.src for b in batch], [b.tgt for b in batch]
    src_tok, src_mask = _pad(src)
    tgt_tok, tgt_mask = _pad(tgt)

    # shift‑right for decoder input
    dec_in = torch.full_like(tgt_tok, PAD_ID)
    dec_in[1:] = tgt_tok[:-1]
    dec_in[0] = BOS_ID

    return {
        "src_tokens": src_tok,
        "tgt_tokens": tgt_tok,
        "decoder_in": dec_in,
        "src_pad_mask": src_mask,
        "tgt_pad_mask": tgt_mask,
    }


def build_dataloader(
    root: str | Path,
    split: str,
    batch_size: int,
    vocab_file: str | Path,
    num_workers: int = 2,
    shuffle: bool | None = None,
):
    dataset = ReactionDataset(root, split, vocab_file)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=reaction_collate_fn,
    )
