Below is a **road‑map for writing the code base in small, reviewable batches**.
For each batch I list the exact files that will be delivered in a single assistant‑response, the purpose of the batch, and any external dependency it unblocks.  This lets you (i) inspect or run unit tests on a minimal slice, (ii) request tweaks, then (iii) move to the next batch when satisfied.

| Batch                             | Files produced in that response                                                                                                                                                                 | Purpose / what you can already do afterwards                                                   | Dependencies satisfied            |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------- |
| **0 — Skeleton**                  | `README.md` (refined), empty folder structure via `mkdir -p` statement in the answer                                                                                                            | Finalises project metadata; provides commands to create directory tree locally                 | none                              |
| **1 — Config & Utility core**     | `config/default_config.yaml`  <br> `config/example_moe_config.yaml`  <br> `utils/config_parser.py`  <br> `utils/dataset_utils.py` (stub with dataclass definitions & batching helpers)          | You can parse YAML into a python `Cfg` object and load a toy dataset; unit‑test config parsing | batch 0                           |
| **2 — Baseline model primitives** | `models/embedding.py` (token, positional, segment encodings)  <br> `models/transformer_layers.py` *(baseline FFN, attention, & wrapper classes; MoE hooks left TODO)*                           | Lets you instantiate the vanilla ChemTransformer block and run a forward pass on fake data     | batch 1                           |
| **3 — MoE engine**                | `models/moe_layer.py`  (experts, router, gating, importance + capacity losses)  <br> `models/__init__.py` (exposes `MoEFFN`)                                                                    | Compiles & unit‑tests the MoE FFN on random tensors; you can measure load‑balancing loss       | batches 1‑2                       |
| **4 — Transformer integration**   | Updated `models/transformer_layers.py` (adds `use_moe` logic, swaps FFN→MoEFFN when layer id in cfg)  <br> `models/chemtransformer_model.py` (full encoder‑decoder, `build_model(cfg)` factory) | You can build **ChemMoETransformer** end‑to‑end and compare param counts vs baseline           | batches 1‑3                       |
| **5 — Training loop**             | `training/train.py`  <br> `training/trainer.py` (epoch loop, gradient clip, LR scheduler, checkpointing)                                                                                        | Allows full‑dataset training run on one GPU; logs token‑ & sequence‑accuracy                   | batches 1‑4                       |
| **6 — Evaluation & metrics**      | `evaluation/metrics.py` (token / seq accuracy, load‑balance stats)  <br> `evaluation/evaluate.py`                                                                                               | Evaluate checkpoints; inspect expert utilisation histograms                                    | batch 5                           |
| **7 — Pre‑processing scripts**    | `preprocessing/preprocess_smiles.py`  <br> `preprocessing/data_split.py`                                                                                                                        | Converts raw USPTO CSV to tokenised `.npz`; builds vocab; writes splits                        | none (can be run before training) |
| **8 — Tests & examples**          | `tests/test_moe_layer.py`  <br> `tests/test_forward_pass.py`  <br> Jupyter notebook `examples/QuickStart.ipynb`                                                                                 | Quick sanity and reproducibility check                                                         | batches 1‑4                       |
| **9 — Packaging & CI (optional)** | `setup.py` or `pyproject.toml`  <br> GitHub Actions workflow YAML                                                                                                                               | PyPI install & automatic lint / test on push                                                   | all previous                      |

### Ordering rationale

1. **Early config & utils** give you the plumbing to unit‑test later modules in isolation.
2. Writing the **baseline blocks first** (batch 2) ensures we reproduce ChemTransformer behaviour before we touch MoE.
3. The **MoE layer** (batch 3) is self‑contained and can be benchmarked immediately on dummy tensors; its interface then stays stable for the rest of the project.
4. Batches **4–6** progressively wire the system together and enable training, then inference.
5. **Pre‑processing (batch 7)** is independent; you can run it in parallel to coding batches 3‑4.
6. Tests and CI come last so that they cover the final API.

### Citations to design guidance

* The need for load‑balancing and the expert‑choice router is drawn from slide 13 (*“Mixture of Experts … Two branches”*) and slide 14 (*“each specialist claims a fixed share … keeps all specialists busy”*) .
* Homogeneous MoE swaps dense FFNs for 8‑expert blocks, cutting compute by \~40 % on protein language modelling (slide 15) .

---

**Next action:**
If this batch plan looks good, let me know and I will deliver **Batch 0 (skeleton & refined README)** in the next message.
