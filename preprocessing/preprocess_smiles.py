#!/usr/bin/env python
"""
preprocessing/preprocess_smiles.py
----------------------------------

CLI that:

1.  Reads a raw reaction dataset where each line is either
        <reactant_smiles(s)>> <product_smiles>         (USPTO format)
    or two columns separated by TAB / ',':
        <reactants> <TAB> <products>

2.  Tokenises SMILES (or SELFIES) strings into whitespace‑separated tokens.

3.  Splits into train/val/test (80/10/10 by default).

4.  Writes  src-{split}.txt  /  tgt-{split}.txt       (tokens)
            vocab_{repr}.txt                         (token → id order)

Tokens follow the same regex as Graph2SMILES so our loader’s vocabulary
matches theirs.

Example
-------
$ python preprocessing/preprocess_smiles.py \
        --input uspto_480k.csv \
        --output_dir preprocessed/USPTO_480k_smiles \
        --representation smiles
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

try:
    import selfies  # Optional, only needed for SELFIES
except ImportError:
    selfies = None

# --------------------------------------------------------------------------
#  SMILES tokeniser (same regex as Graph2SMILES)
# --------------------------------------------------------------------------
_REGEX = re.compile(
    r"""
Cl?|Br?|Si?|Se?|Li?|Na?|Mg?|Ca?|Al?|Sn?|                   # two‑letter elements
\[[^\]]+\]|                                                # atoms in brackets
\d+|                                                      # ring numbers
\%\d{2}|                                                  # ring numbers >= 10
\#|\(|\)|\.|=|\/|\\|\-|\+|\:|\@|\?|>|\*|\$|\!|\||\:|\%|~|  # special chars
[a-zA-Z]                                                  # single‑letter atoms
""",
    re.X,
)


def tokenize_smiles(s: str) -> List[str]:
    return _REGEX.findall(s)


# --------------------------------------------------------------------------
def to_selfies_tokens(s: str) -> List[str]:
    if selfies is None:
        raise ImportError("`pip install selfies` to use SELFIES representation.")
    sf = selfies.encoder(s)
    return list(selfies.split_selfies(sf))


# --------------------------------------------------------------------------
def parse_reaction_line(
    line: str, sep: str = ">>"
) -> Tuple[str, str]:  # returns (src, tgt)
    if sep in line:
        src, tgt = line.split(sep)
    else:  # try comma or tab
        parts = re.split(r"[\t,]", line)
        if len(parts) != 2:
            raise ValueError(f"Cannot parse line: {line[:100]}")
        src, tgt = parts
    return src.strip(), tgt.strip()


# --------------------------------------------------------------------------
def write_split(
    examples: List[Tuple[str, str]],
    out_dir: Path,
    representation: str,
    split: str,
    tokenizer,
):
    src_path = out_dir / f"src-{split}.txt"
    tgt_path = out_dir / f"tgt-{split}.txt"

    with open(src_path, "w") as fs, open(tgt_path, "w") as ft:
        for src_raw, tgt_raw in examples:
            src_tok = tokenizer(src_raw)
            tgt_tok = tokenizer(tgt_raw)
            fs.write(" ".join(src_tok) + "\n")
            ft.write(" ".join(tgt_tok) + "\n")


def build_vocab(examples: List[Tuple[str, str]], tokenizer, min_freq=1) -> List[str]:
    counter = Counter()
    for s, t in examples:
        counter.update(tokenizer(s))
        counter.update(tokenizer(t))
    # special tokens first
    vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    vocab.extend([tok for tok, freq in counter.items() if freq >= min_freq])
    return vocab


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="raw dataset file (txt/csv)")
    p.add_argument(
        "--output_dir",
        required=True,
        help="where to write src/tgt splits + vocab file",
    )
    p.add_argument(
        "--representation",
        choices=["smiles", "selfies"],
        default="smiles",
        help="tokenisation scheme",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose tokenizer
    tokenizer = tokenize_smiles if args.representation == "smiles" else to_selfies_tokens

    # ---------------- 1. Load raw lines ----------------------------------
    raw_examples: List[Tuple[str, str]] = []
    with open(args.input, "r") as f:
        # auto‑detect CSV dialect
        sample = f.readline()
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.reader(f, dialect)
        for row in reader:
            line = ",".join(row)
            try:
                src, tgt = parse_reaction_line(line)
            except ValueError:
                continue
            raw_examples.append((src, tgt))

    # Shuffle
    random.shuffle(raw_examples)

    # ---------------- 2. Split ------------------------------------------
    N = len(raw_examples)
    n_train = int(N * args.train_ratio)
    n_val = int(N * args.val_ratio)
    train = raw_examples[:n_train]
    val = raw_examples[n_train : n_train + n_val]
    test = raw_examples[n_train + n_val :]

    # ---------------- 3. Vocabulary -------------------------------------
    vocab = build_vocab(train, tokenizer)
    vocab_file = out_dir / f"vocab_{args.representation}.txt"
    with open(vocab_file, "w") as vf:
        for tok in vocab:
            vf.write(tok + "\n")

    # ---------------- 4. Write splits -----------------------------------
    write_split(train, out_dir, args.representation, "train", tokenizer)
    write_split(val, out_dir, args.representation, "val", tokenizer)
    write_split(test, out_dir, args.representation, "test", tokenizer)

    print(
        f"Done!  {len(train)} train / {len(val)} val / {len(test)} test "
        f"→ tokens saved under {out_dir}"
    )
    print(f"Vocab size: {len(vocab)}  →  {vocab_file}")


if __name__ == "__main__":
    main()
