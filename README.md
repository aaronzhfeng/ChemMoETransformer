
## Repository Structure

```plaintext
ChemMoETransformer/
├── config/
│   ├── default_config.yaml
│   └── example_moe_config.yaml          # example config (uses MoE in layers 2,4,6)
├── models/
│   ├── __init__.py
│   ├── chemtransformer_model.py         # baseline ChemTransformer model definition
│   ├── moe_layer.py                     # MoE layer implementation (experts & gating logic)
│   └── transformer_layers.py            # Transformer blocks (modified to plug in MoE layers)
├── training/
│   ├── train.py                        # script to train the model
│   └── trainer.py                      # training loop and epoch routines
├── evaluation/
│   ├── evaluate.py                     # script to evaluate a trained model
│   └── metrics.py                      # evaluation metrics (accuracy, etc.)
├── preprocessing/
│   ├── preprocess_smiles.py            # script to preprocess SMILES data into model inputs
│   └── data_split.py                   # helpers for dataset splitting and tokenization
├── utils/
│   ├── __init__.py
│   ├── dataset_utils.py                # dataset loading, batching utilities
│   └── config_parser.py                # utility to parse config files
└── README.md
```

# ChemMoETransformer

**ChemMoETransformer** is a sequence-to-sequence Transformer model for chemical reaction prediction that extends the baseline **ChemTransformer** model. It introduces Mixture-of-Experts (MoE) layers into selected Transformer blocks to enhance model capacity and accuracy. The goal is to improve the prediction of reaction outcomes (product molecules given reactants) by allowing multiple expert sub-networks to specialize in different reaction types, all while keeping computation efficient. MoE enables scaling up the model’s capacity (many expert networks) without a proportional increase in computation cost, which helps the model capture diverse chemical transformation patterns better than a single-expert baseline.

## Model Architecture

The architecture is based on a standard Transformer encoder-decoder (as used in ChemTransformer) with **6 layers** each for encoder and decoder by default. We modify this architecture by replacing the feed-forward network (FFN) sublayers in some Transformer blocks with MoE layers. All other components (multi-head attention, embeddings, etc.) remain the same as the baseline for a fair comparison. Key architecture features and modifications include:

* **Mixture-of-Experts Feed-Forward Layers:** In specified Transformer layers, the single FFN is replaced with an MoE block that contains multiple parallel expert FFNs and a router (gating network). During a forward pass, the gating mechanism assigns each token’s hidden representation to one or more expert networks for processing, instead of a single FFN. This allows the model to effectively have a much larger representational capacity, as different experts can learn different aspects of the reaction data, while the computational cost grows sub-linearly with the number of experts.

* **Homogeneous Expert Networks:** All experts in an MoE layer share the same architecture and dimensionality (they are **homogeneous** experts). Each expert is essentially a feed-forward subnetwork with the same size and shape as the original Transformer FFN. This makes the MoE a drop-in replacement – each expert could have been an FFN in a standard Transformer. Homogeneous experts simplify integration and ensure that any expert could handle any token; they differ only in the learned weights.

* **Expert-Choice Routing:** We use the *expert-choice* routing algorithm (Zhou *et al.*, 2022) for the MoE layers. In traditional MoE, a softmax gate might send most tokens to a few “popular” experts, causing imbalance. In expert-choice routing, **each expert is allocated a fixed share of the incoming tokens** instead of letting the gate purely decide. In practice, this means each expert will get roughly an equal portion of tokens to process (keeping all experts busy), which avoids load imbalances and training instability. This simple routing rule leads to better utilization of all experts and significantly faster training – roughly **2× faster** in practice – while matching the accuracy of dense (non-MoE) models on benchmark tasks. By keeping the workload balanced, we address the load-balancing challenge identified in scaling up MoE models.

* **Configurable MoE Placement:** Not every Transformer layer has to use MoE. We provide flexibility to enable MoE in specific layers via configuration. For example, one can choose to apply MoE layers to the 2nd, 4th, and 6th layers of the encoder and decoder (and use standard FFNs in the other layers). This selective use of MoE allows controlling computational overhead and experimenting with different patterns of MoE insertion. By default, our example configuration uses MoE in layers **2, 4, 6** of a 6-layer Transformer (see the configuration section below).

These modifications are designed as a **drop-in enhancement** over the baseline ChemTransformer. Importantly, integrating MoE layers does *not* require any additional chemical knowledge features or reaction-specific labels – the gating is entirely learned from data, and experts specialize automatically. The rest of the model workflow (tokenization of SMILES, training objective, etc.) remains unchanged from the baseline.

## Configuration

Configuration files (in the `config/` folder) use YAML format to specify model parameters, dataset paths, and training settings. They include fields to configure the MoE integration. For instance, you can specify which layers use MoE and how many expert networks to use. Below is an example snippet from a configuration file (`config/example_moe_config.yaml`):

```yaml
model:
  n_layer: 6             # number of layers in encoder (and decoder)
  d_model: 512           # model dimensionality
  n_head: 8              # number of attention heads
  d_ff: 2048             # feed-forward network dimension
  dropout: 0.1
  moe_layers: [2, 4, 6]  # use MoE in 2nd, 4th, 6th Transformer layers
  num_experts: 4         # number of experts in each MoE layer
data:
  data_root: data/USPTO_480k/   # path to dataset and splits
  batch_size: 64
training:
  max_epoch: 50
  max_lr: 1e-3
  min_lr: 1e-5
output:
  ckpt_dir: checkpoints/
  ckpt_interval: 5
```

In this example, the model is a 6-layer Transformer where layers 2, 4, and 6 are MoE layers (each of those layers will have 4 expert FFNs). You can adjust `moe_layers` to use different layers or set it empty (`[]`) to disable MoE entirely. The rest of the config specifies training hyperparameters (learning rate schedule, number of epochs, batch size, etc.), and dataset paths. The training script will read this config file to set up the model and training procedure.

## Data Preprocessing

Chemical reactions are represented as textual strings (SMILES notation) for reactants and products. Before training, you should preprocess these reaction SMILES into a tokenized format suitable for the model:

* **SMILES Tokenization:** The provided script `preprocessing/preprocess_smiles.py` takes raw reaction data (e.g. a CSV or TXT file of reactions) and converts each reaction into a sequence of tokens. Typically, the input format expects a separation of reactant and product SMILES (for example, `"reactants>>product"` strings). The script will split reactant and product SMILES, tokenize them (e.g., breaking into atomic tokens, digits, special symbols), and build a vocabulary of tokens.

* **Dataset Splits:** The preprocessing will also create training, validation, and test splits from the data (if not already provided). For instance, it can output files or serialized datasets like `train.npz`, `val.npz`, etc., along with a token dictionary (`dictionary.pkl`) mapping tokens to indices. These outputs are stored in a `data/` directory (as specified by `data_root` in the config).

Before training, ensure that the configuration’s `data_root` points to the directory containing the preprocessed dataset files. After running the preprocessing step, you should have files like `train.npz`, `val.npz`, `test.npz` (or similar) and a vocabulary file in the `data_root` folder.

## Training

Once the data is prepared and the configuration is set, you can train the ChemMoETransformer model. Training uses a standard sequence-to-sequence approach with cross-entropy loss on the product SMILES tokens. Here are the steps to train the model:

1. **Prepare Data:** Make sure you have run the preprocessing step above and updated the config file to point to the processed dataset. (By default, the config expects a path like `data/USPTO_480k/` with the necessary files.)

2. **Configure MoE Settings:** Adjust the configuration YAML to set the desired MoE layers and number of experts (as shown in the example config). You can also tweak other hyperparameters (learning rate, epochs, etc.) in this file.

3. **Run Training Script:** Execute the training script with the config file as an argument. For example:

   ```bash
   python training/train.py config/example_moe_config.yaml
   ```

   This will start training the model using the parameters defined in the config. The training script will load the dataset, initialize the Transformer model with MoE layers as specified, and begin the training loop. During training, it logs progress including training loss and accuracy metrics on both training and validation sets. Model checkpoints are saved to the directory defined in the config (`output.ckpt_dir`, e.g. a `checkpoints/` folder) at regular intervals (e.g. every 5 epochs).

**Training Metrics:** The training process reports two primary accuracy metrics: **token-level accuracy** and **sequence-level accuracy**. Token-level accuracy measures how often the model correctly predicts each individual token in the product SMILES. Sequence-level (reaction) accuracy is stricter – it measures how often the entire predicted product sequence exactly matches the ground truth product (this is the “Top-1 full sequence accuracy”). For instance, on a typical dataset the baseline ChemTransformer achieved \~98% token accuracy but only \~58% full sequence accuracy, indicating that getting every token right in a long SMILES is much harder. By introducing MoE layers, we aim to improve the full sequence accuracy by capturing more complex reaction patterns without sacrificing token-level performance. During training, you can monitor these metrics per epoch in the logs (they are written to a log file, e.g., `log.txt`, as well as printed to console).

## Evaluation

After training, you can evaluate the final model on a held-out test set to measure its performance on unseen reactions. Use the `evaluation/evaluate.py` script for this purpose. For example:

```bash
python evaluation/evaluate.py --model checkpoints/best_model.pt --test data/USPTO_480k/test.npz
```

In the above, `--model` points to a saved model checkpoint (in this case, the best model saved during training) and `--test` points to the tokenized test dataset. The evaluation script will load the trained model and the test data, then generate predicted product SMILES for each reaction in the test set. It computes the same metrics as during training: token-level accuracy and full sequence accuracy (Top-1 accuracy). It may also compute Top-3 or Top-5 accuracy if configured to generate multiple candidate outputs per reactant (this is optional and would require the model to output multiple guesses). The results will be displayed or saved, allowing you to compare the model’s performance to the baseline or other models.

**Expected Results:** We anticipate that ChemMoETransformer will outperform the baseline on sequence-level accuracy of product prediction. For example, if the baseline achieved \~60% exact match on a large dataset, the MoE-enhanced model might improve this (hypothetically to 65% or more) by leveraging expert specializations. Additionally, because expert-choice routing keeps all experts utilized, the model training and inference remain efficient. We also monitor the load balancing during evaluation – ideally each expert in an MoE layer handles a similar number of tokens, confirming the effectiveness of the routing mechanism.

## Supported Datasets

ChemMoETransformer supports reaction prediction datasets in standard SMILES format. You can train and evaluate the model on commonly used reaction datasets, for example:

* **USPTO-480k:** A large collection of \~480,000 patent reactions extracted from the United States Patent Office database. This dataset provides reactions in SMILES format (usually as reactant>>product strings). It is a popular benchmark for reaction outcome prediction. Our config files include an example setup for USPTO-480k (with data path `data/USPTO_480k/`).

* **Open Reaction Database (ORD):** An open-access database of chemical reactions curated from literature. It contains a wide variety of reactions (with reagents and conditions in many cases). The model can be trained on a processed version of ORD by converting it into the appropriate format. (Ensure to preprocess the SMILES and splits similarly to USPTO before training.)

* **USPTO-50k (optional):** A smaller benchmark dataset of 50k reactions (a subset of USPTO) often used for quick evaluation. While our focus is on larger data, the code can equally run on USPTO-50k for rapid prototyping or comparisons with literature.

Before training on any new dataset, update the configuration file’s `data_root` path and ensure the preprocessing script is compatible with the input file format of that dataset. The model does not depend on any dataset-specific features – as long as you can provide reaction SMILES pairs, ChemMoETransformer can learn from them.

## Citation

If you use **ChemMoETransformer** in your research or find our work useful, please consider citing the following:

* **Molecular Transformer (Schwaller *et al.*, 2019)** – The baseline sequence-to-sequence Transformer model for reaction prediction that our project builds upon. (ACS Central Science 5(9):1572–1583, 2019.)

* **Zhou *et al.*, 2022 (Expert-Choice MoE Routing)** – Introduced the expert-choice routing algorithm for MoE, which we adopt to improve load balancing.

* **Sun *et al.*, 2024 (Protein MoE Model)** – Demonstrated that swapping dense layers with MoE layers in a protein language model achieved the same accuracy as a much larger model with \~40% less compute, highlighting MoE efficacy on sequence data relevant to our work.

*We will update this section with our own ChemMoETransformer publication details once available.*
