"""
preprocessing/data_split.py
---------------------------

Utility helpers in case you want programmatic control from a notebook.

Most workflows can just call the CLI `preprocess_smiles.py` directly.
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple


def split_dataset(
    data: Sequence[Tuple[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    random.seed(seed)
    data = list(data)
    random.shuffle(data)
    N = len(data)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test
