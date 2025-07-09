
## Repository Structure

```plaintext
ChemMoETransformer/
│
├── config/
│   ├── default_config.yaml
│   └── example_moe_config.yaml          # MoE in layers 2‑4‑6
│
├── models/
│   ├── __init__.py                      # exposes build_model, MoE FFN
│   ├── embedding.py                     # token & positional encodings
│   ├── transformer_layers.py            # encoder / decoder blocks
│   ├── moe_layer.py                     # MoE‑FFN (expert‑choice routing)
│   └── chemtransformer_model.py         # full encoder‑decoder wrapper
│
├── preprocessing/
│   ├── preprocess_smiles.py             # CLI: raw reactions → tokenised *.txt
│   └── data_split.py                    # helper splitting utilities
│
├── utils/
│   ├── __init__.py
│   ├── config_parser.py                 # YAML → namespace loader
│   └── dataset_utils.py                 # TXT/NPZ loader + DataLoader
│
├── training/
│   ├── train.py                         # CLI for training (debug flag)
│   └── trainer.py                       # epoch loop, logging, checkpoints
│
├── evaluation/
│   ├── metrics.py                       # token / sequence accuracy
│   └── evaluate.py                      # CLI: load ckpt → report metrics
│
├── tests/                               # pytest unit tests
│   ├── test_dataset_utils.py
│   ├── test_moe_layer.py
│   └── test_forward_pass.py
│
├── examples/
│   └── QuickStart.ipynb                 # run training in notebook
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # lint + tests on every push
│
├── pyproject.toml                       # packaging & dev dependencies
└── README.md                            # (updated – see next section)

```

Below is an **updated repository schema** (after Batches 0 – 9) followed by a **comprehensive `README.md`** that documents every component we have implemented so far.

---

## Repository tree

```text
ChemMoETransformer/
│
├── config/
│   ├── default_config.yaml
│   └── example_moe_config.yaml          # MoE in layers 2‑4‑6
│
├── models/
│   ├── __init__.py                      # exposes build_model, MoE FFN
│   ├── embedding.py                     # token & positional encodings
│   ├── transformer_layers.py            # encoder / decoder blocks
│   ├── moe_layer.py                     # MoE‑FFN (expert‑choice routing)
│   └── chemtransformer_model.py         # full encoder‑decoder wrapper
│
├── preprocessing/
│   ├── preprocess_smiles.py             # CLI: raw reactions → tokenised *.txt
│   └── data_split.py                    # helper splitting utilities
│
├── utils/
│   ├── __init__.py
│   ├── config_parser.py                 # YAML → namespace loader
│   └── dataset_utils.py                 # TXT/NPZ loader + DataLoader
│
├── training/
│   ├── train.py                         # CLI for training (debug flag)
│   └── trainer.py                       # epoch loop, logging, checkpoints
│
├── evaluation/
│   ├── metrics.py                       # token / sequence accuracy
│   └── evaluate.py                      # CLI: load ckpt → report metrics
│
├── tests/                               # pytest unit tests
│   ├── test_dataset_utils.py
│   ├── test_moe_layer.py
│   └── test_forward_pass.py
│
├── examples/
│   └── QuickStart.ipynb                 # run training in notebook
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # lint + tests on every push
│
├── pyproject.toml                       # packaging & dev dependencies
└── README.md                            # (updated – see next section)
```

---

## `README.md`

````markdown
# ChemMoETransformer

*A Mixture‑of‑Experts (MoE) upgrade of the classic sequence‑to‑sequence
“Molecular Transformer” for forward reaction prediction.*

<div align="center">
<img src="https://img.shields.io/badge/python-3.9%2B-blue">
<img src="https://github.com/your‑org/ChemMoETransformer/actions/workflows/ci.yml/badge.svg" alt="CI">
</div>

---

## ✨ Key features

| Feature | Details |
|---------|---------|
| **Encoder–decoder Transformer** | 6‑layer baseline identical to Schwaller *et al.* (2019) |
| **Homogeneous MoE blocks** | Feed‑forward sub‑layers optionally replaced with `MoEFFN` (top‑k = expert‑choice routing) |
| **Config‑driven capacity scaling** | `moe_layers: [2,4,6]`, `num_experts: 4` etc. in YAML |
| **Flexible data ingestion** | Accepts either (i) tokenised `src-*.txt` / `tgt-*.txt` or (ii) pre‑binarised `*.npz` |
| **Quick‑debug mode** | `python training/train.py ... --debug` trains on 1 000 samples |
| **End‑to‑end CLI** | Preprocess → Train → Evaluate with three commands |
| **Packaging & CI** | Editable install (`pip install -e .`), pytest unit tests, Ruff lint, GitHub workflow |

---

## 1  Installation

```bash
git clone https://github.com/your‑org/ChemMoETransformer.git
cd ChemMoETransformer
pip install -e .[dev]        # core deps + pytest/ruff for dev
````

GPU training requires **PyTorch ≥ 2.1** with CUDA (install from
[pytorch.org](https://pytorch.org)).

---

## 2  Data preparation

```bash
# Example: preprocess raw USPTO‑480k lines  "reactants>>product"
python preprocessing/preprocess_smiles.py \
       --input uspto_480k_raw.txt \
       --output_dir preprocessed/USPTO_480k_smiles \
       --representation smiles
```

Output folder layout:

```
preprocessed/USPTO_480k_smiles/
   ├─ src-train.txt   tgt-train.txt
   ├─ src-val.txt     tgt-val.txt
   ├─ src-test.txt    tgt-test.txt
   └─ vocab_smiles.txt         # token → id (PAD, UNK, BOS, EOS first)
```

Update `config/example_moe_config.yaml`:

```yaml
data:
  data_root: preprocessed/USPTO_480k_smiles
  vocab_file: preprocessed/USPTO_480k_smiles/vocab_smiles.txt
```

---

## 3  Training

### Quick smoke‑test (1 000 samples on GPU)

```bash
python training/train.py config/example_moe_config.yaml --debug
```

* Logs per‑epoch metrics to `train_moe.log`.
* Saves checkpoints to `checkpoints/` every `ckpt_interval` epochs.

### Full training

Remove the `--debug` flag to train on the complete dataset.

---

## 4  Evaluation

```bash
python evaluation/evaluate.py \
       --config config/example_moe_config.yaml \
       --checkpoint checkpoints/model_best.pt \
       --split test
```

Example output:

```
[test] Token‑Acc=0.8973 Seq‑Acc=0.6158 Aux‑Loss=0.0039
Results appended to checkpoints/eval_results.csv
```

---

## 5  Configuration reference (`config/*.yaml`)

| Key                        | Description                                          | Example   |
| -------------------------- | ---------------------------------------------------- | --------- |
| `model.moe_layers`         | 1‑based indices of encoder/decoder layers to use MoE | `[2,4,6]` |
| `model.num_experts`        | Experts per MoE layer                                | `4`       |
| `training.aux_loss_weight` | λ for load‑balancing loss                            | `0.02`    |
| `data.batch_size`          | tokens per batch                                     | `64`      |
| `training.grad_clip`       | gradient norm clipping                               | `0.5`     |
| *…*                        | *see YAML files*                                     |           |

---

## 6  MoE architecture details

* **Experts:** two‑layer FFN (`d_model → d_ff → d_model`) identical to
  dense baseline.
* **Routing:** *Expert‑choice* (Zhou 2022) – each expert claims up to a
  fixed capacity of tokens; overflow routed to expert 0.
  Balancing losses (`importance`, `capacity`) are added to training
  objective (`λ ≈ 0.02`).
* **Parallelism:** The current implementation runs experts on a single
  GPU; upgrading to tensor/ expert parallelism (e.g., with
  `torch.distributed` or `fairscale`) only requires swapping the
  per‑expert `ModuleList` with sharded modules.

---

## 7  Development workflow

| Step            | Command                                             |
| --------------- | --------------------------------------------------- |
| **Tests**       | `pytest`                                            |
| **Lint**        | `ruff .`                                            |
| **CI**          | GitHub Actions runs lint + tests on Python 3.9‑3.11 |
| **Build wheel** | `python -m build`                                   |

---

