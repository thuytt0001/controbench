# ControBench

Official code repository for **ControBench: An Interaction-Aware Benchmark for Controversial Discourse Analysis on Social Networks** (IEEE ICDM 2026).

ControBench is a benchmark for controversial discourse analysis built from Reddit discussions on three topics: Trump, abortion, and religion. It represents conversations as a heterogeneous graph with user and post nodes and three edge types (publish, comment-on-post, comment-on-user), where user-comment-user edges carry dual semantic features encoding both a reply and the parent comment it responds to. This repository contains the full data-construction pipeline and the evaluation code for all model families reported in the paper (GNNs, PLMs, LLMs, and baselines).

---

## Repository Structure

```
.
├── data/                              # Raw and processed Reddit data
├── models/                            # Model implementations / adaptations
├── Reddit_BERT.py                     # Step 1: generate BERT embeddings
├── emb_integrate.py                   # Step 2: build embedded graph data
├── split_dataset.py                   # Step 3: train/val/test split
├── tune.py                            # GNN hyperparameter search
├── eval_gnn_multiseed.py             # GNN evaluation (multi-seed + 200-user)
├── eval_plm_multiseed.py             # PLM evaluation (multi-seed + 200-user)
├── llm_train.py                       # LLM evaluation
├── label_propagation.py              # Label Propagation baseline
├── cola.py                            # COLA multi-agent baseline
├── text_classification_experiments.py # Shared PLM training utilities
├── datasets.py                        # Dataset loading utilities
├── run_experiments.py                 # Convenience experiment runner
└── requirements.txt                   # Python dependencies
```

The `models/` folder contains the implementations and adaptations of the different published model architectures evaluated in the benchmark (e.g., RGCN, HAN, HinSAGE, Hetero2Net, H²G-Former, H₂GCN, ACMGNN, GCN+GFS).

---

## Installation

```bash
pip install -r requirements.txt
```

For the LLM experiments, set your API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

---

## Data Construction Pipeline

Run these three steps in order to build the benchmark from the raw Reddit data.

### Step 1 — Generate embeddings

Run `Reddit_BERT.py` to generate BERT embeddings for the raw data (post titles, post content, and comment text).

```bash
python Reddit_BERT.py
```

### Step 2 — Build the embedded graph

Run `emb_integrate.py` to create the embedded version of the graph data. This attaches the embeddings from Step 1 to the nodes and edges and writes the result into the `embedded_data/` folder.

```bash
python emb_integrate.py
```

### Step 3 — Split into train / validation / test

Run `split_dataset.py` to split the data into stratified train/validation/test sets (60/20/20).

```bash
python split_dataset.py
```

---

## Reproducing Experiments

After completing the data pipeline, run the file for the model family you want to reproduce.

### Graph Neural Networks

Run the hyperparameter search **first**, then the multi-seed evaluation.

```bash
# 1. hyperparameter search
python tune.py --all_models --all_datasets

# 2. multi-seed + same-200-user evaluation (loads best configs from tune.py)
python eval_gnn_multiseed.py --all_models --all_datasets --no_search
```

### Pre-trained Language Models

```bash
python eval_plm_multiseed.py --all_models --all_datasets
```

### Large Language Models

```bash
python llm_train.py
```

### Baselines

```bash
# Label Propagation
python label_propagation.py

# COLA (multi-agent LLM baseline)
python cola.py
```

---

## Notes

- **Evaluation protocol.** GNNs use a transductive protocol on a single merged graph; PLMs use standard supervised train/val/test; LLMs are evaluated on a stratified 200-user sample per dataset (seed = 42) with deterministic decoding. GNNs and PLMs are additionally evaluated on the same 200-user sample to enable direct cross-category comparison.
- **No label leakage.** All model families classify each user from content only. No ground-truth stance label of any neighboring user is exposed to any model at inference time.
- **Reproducibility.** GNN and PLM results are averaged over three seeds (42, 123, 456). LLMs use temperature = 0.0.

---

## Citation

If you use ControBench in your research, please cite:

```bibtex
@inproceedings{controbench2026,
  title     = {ControBench: An Interaction-Aware Benchmark for Controversial Discourse Analysis on Social Networks},
  author    = {Ta, Thanh Thuy and Zhu, Jiaqi and Liu, Xuan and Lin, Shang and Rabbany, Reihaneh and Rabusseau, Guillaume and Chen, Lihui and Zheng, Yilun and Luan, Sitao},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```
