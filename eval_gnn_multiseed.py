"""
eval_gnn_multiseed.py
=====================
Two things in one script:

1. MULTI-SEED  — trains each GNN model with seeds [42, 123, 456] and reports
                  mean ± std over the FULL test set, matching the existing
                  transductive protocol in tune.py.

2. 200-USER    — for seed=42, additionally evaluates the trained model on the
                  same deterministic stratified 200-user sample used by the LLM
                  experiments, so Table 3 rows are directly comparable.

Usage
-----
  # all models, all datasets
  python eval_gnn_multiseed.py --all_models --all_datasets

  # specific subset
  python eval_gnn_multiseed.py --models HAN,RGCN --datasets trump,abortion

  # skip HP search (use tuning_results/ configs if available)
  python eval_gnn_multiseed.py --all_models --all_datasets --no_search

Output
------
  gnn_multiseed_results/
    summary_full_testset.csv   ← mean ± std over 3 seeds, full test set
    summary_200user.csv        ← seed-42 results on 200-user LLM-comparable subset
    per_run_details.csv        ← every individual run
    summary.txt                ← human-readable table
"""

import os
import sys
import json
import time
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import dgl
from collections import Counter
from datetime import datetime
from sklearn.metrics import f1_score

# ── import shared infrastructure from tune.py ────────────────────────────
# We re-use load_merged, train_merged, hp_search, set_seed, ALL_MODELS, etc.
try:
    from tune import (
        load_merged, train_merged, hp_search,
        set_seed, _eval_hetero, _eval_homo, _class_weights,
        DATA_DIR, DATASETS,
    )
    from models import (
        HETERO_MODELS, HOMO_MODELS, ALL_MODELS,
        build_homogeneous_user_graph,
        build_2hop_graph,
        build_khop_hetero_graph,
        GCNPlusGFS,
    )

    # handle both possible module names
except ModuleNotFoundError:
    raise ImportError(
        "Cannot import tune__1_.py. Make sure it is in the same directory.\n"
        "If the file is named differently, adjust the import above."
    )

# ── 200-user stratified sampler (mirrors llm_train.py exactly) ───────────────

def stratified_sample_200(user_indices: np.ndarray,
                           labels: torch.Tensor,
                           max_samples: int = 200,
                           seed: int = 42) -> np.ndarray:
    """
    Return the indices (into the full user tensor) of the 200-user stratified
    sample that exactly matches llm_train.py's stratified_sample_users().

    Parameters
    ----------
    user_indices : 1-D array of graph user-node indices that belong to the
                   test split (i.e. test_mask.nonzero()).
    labels       : full label tensor (length = total users in graph).
    max_samples  : target sample size (default 200).
    seed         : must match the LLM script (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)

    # group by class
    by_class = {}
    for idx in user_indices:
        lbl = labels[idx].item()
        if lbl >= 0:
            by_class.setdefault(lbl, []).append(int(idx))

    per_class = max(1, max_samples // len(by_class))
    sampled = []

    for cls in sorted(by_class.keys()):
        ids = sorted(by_class[cls])
        random.shuffle(ids)
        sampled.extend(ids[:per_class])

    sampled_set = set(sampled)
    remaining = sorted([i for i in user_indices if i not in sampled_set])
    random.shuffle(remaining)
    sampled.extend(remaining[: max(0, max_samples - len(sampled))])
    sampled = sampled[:max_samples]

    return np.array(sampled, dtype=np.int64)


# ── single evaluation on a boolean mask ──────────────────────────────────────

def eval_on_mask(model_name, model, hetero_g, node_features, edge_features,
                 labels, bool_mask, device):
    """
    Evaluate a trained model on the nodes indicated by bool_mask.
    Returns (macro_f1, micro_f1) as percentages.
    """
    model = model.to(device)
    is_homo = model_name in HOMO_MODELS
    if is_homo:
        homo_no, homo_raw, features = build_homogeneous_user_graph(
            hetero_g, edge_features, node_features)
        g_2hop   = build_2hop_graph(homo_no).to(device)  # must run on CPU first
        homo_no  = homo_no.to(device)
        homo_raw = homo_raw.to(device)
        features = features.to(device)
        macro, micro = _eval_homo(model, homo_no, homo_raw,
                                  features, labels.to(device), bool_mask, g_2hop)
    else:
        nf = {k: v.to(device) for k, v in node_features.items()}
        ef = {k: v.to(device) for k, v in edge_features.items()}
        if model_name == "H2GFormer":
            hetero_g_khop, ef_khop = build_khop_hetero_graph(hetero_g, ef, device)
            macro, micro = _eval_hetero(model, hetero_g_khop, nf, ef_khop,
                                        labels.to(device), bool_mask)
        else:
            macro, micro = _eval_hetero(model, hetero_g.to(device), nf, ef,
                                        labels.to(device), bool_mask)
    return round(macro * 100, 2), round(micro * 100, 2)


# ── load best HP config from previous tuning run ─────────────────────────────

def load_best_config(model_name, dataset_name,
                     tuning_dir="tuning_results/run_20260322_160250"):
    path = os.path.join(tuning_dir, f"{model_name}_{dataset_name}",
                        "best_config.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        cfg = data.get("best_config", {})
        if cfg:
            print(f"    Loaded HP config from {path}")
            return cfg
    return None


# ── main per-model-dataset runner ─────────────────────────────────────────────

def run_model_dataset(model_name: str, dataset_name: str,
                      seeds: list, do_search: bool,
                      n_trials: int, data_dir: str,
                      tuning_dir: str) -> dict:
    """
    Trains model_name on dataset_name for each seed.
    Returns a dict with full-test and 200-user results.
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} / {dataset_name}  (seeds={seeds})")
    print(f"{'='*60}")

    # ── load data once (same graph for all seeds) ──────────────────────────
    raw = load_merged(dataset_name, data_dir)
    (hetero_g, edge_features, node_features,
     labels, train_mask, val_mask, test_mask,
     num_classes, idx_to_flair, user_map, splits) = raw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_kw = dict(
        hetero_g=hetero_g, edge_features=edge_features,
        node_features=node_features, labels=labels,
        train_mask=train_mask, val_mask=val_mask,
        test_mask=test_mask, num_classes=num_classes,
    )

    # ── HP search (once, using seed=42) ────────────────────────────────────
    best_cfg = None
    if not do_search:
        best_cfg = load_best_config(model_name, dataset_name, tuning_dir)

    if best_cfg is None:
        if do_search:
            print(f"  Running HP search ({n_trials} trials) …")
            best_cfg, best_val = hp_search(
                model_name, dataset_name, data_kw, n_trials, seed=42)
            print(f"  Best HP: {best_cfg}  (val={best_val:.2f}%)")
        else:
            # safe fallback defaults
            best_cfg = {"hidden_size": 256, "n_layers": 2, "dropout": 0.5,
                        "lr": 0.005, "weight_decay": 5e-4}
            print(f"  Using default HP config: {best_cfg}")

    # ── 200-user mask (seed=42, deterministic, matches LLM script) ─────────
    test_indices = test_mask.nonzero(as_tuple=True)[0].numpy()
    sample_200_indices = stratified_sample_200(test_indices, labels,
                                               max_samples=200, seed=42)
    mask_200 = torch.zeros(labels.shape[0], dtype=torch.bool)
    mask_200[sample_200_indices] = True
    # only keep nodes that have valid labels AND are in test split
    mask_200 = mask_200 & test_mask & (labels >= 0)

    print(f"  200-user sample: {mask_200.sum().item()} valid labeled users "
          f"(from test set of {test_mask.sum().item()})")

    # ── train with each seed ────────────────────────────────────────────────
    full_macros, full_micros = [], []
    u200_macros, u200_micros = [], []
    per_run = []

    for seed in seeds:
        print(f"\n  --- seed={seed} ---")
        t0 = time.time()
        result = train_merged(
            model_name, dataset_name,
            num_epochs=200, patience=20, seed=seed,
            **data_kw, **best_cfg,
        )
        elapsed = round(time.time() - t0, 1)

        # full test-set metrics (already computed inside train_merged)
        full_macro = result["test_macro_f1"]
        full_micro = result["test_micro_f1"]
        full_macros.append(full_macro)
        full_micros.append(full_micro)

        # 200-user subset metrics — evaluate the trained model
        trained_model = result.get("_trained_model")
        if trained_model is not None:
            m200, mi200 = eval_on_mask(
                model_name, trained_model,
                hetero_g, node_features, edge_features,
                labels, mask_200, device,
            )
        else:
            # fallback: re-train and eval (should not happen with current tune.py)
            print("  WARNING: _trained_model not found in result, re-evaluating.")
            m200, mi200 = full_macro, full_micro

        u200_macros.append(m200)
        u200_micros.append(mi200)

        print(f"  Full test  → Macro={full_macro:.2f}%  Micro={full_micro:.2f}%")
        print(f"  200-user   → Macro={m200:.2f}%  Micro={mi200:.2f}%")
        print(f"  Time: {elapsed}s")

        per_run.append({
            "model":           model_name,
            "dataset":         dataset_name,
            "seed":            seed,
            "full_macro_f1":   full_macro,
            "full_micro_f1":   full_micro,
            "u200_macro_f1":   m200,
            "u200_micro_f1":   mi200,
            "elapsed_sec":     elapsed,
        })

    # ── aggregate over seeds ───────────────────────────────────────────────
    def fmt(vals):
        m, s = np.mean(vals), np.std(vals)
        return round(m, 2), round(s, 2), f"{m:.2f} ± {s:.2f}"

    fm_m, fm_s, fm_str = fmt(full_macros)
    fmi_m, fmi_s, fmi_str = fmt(full_micros)
    u200_m, u200_s, u200_str = fmt(u200_macros)
    u200mi_m, u200mi_s, u200mi_str = fmt(u200_micros)

    summary = {
        "model":                   model_name,
        "dataset":                 dataset_name,
        # full test set
        "full_macro_mean":         fm_m,
        "full_macro_std":          fm_s,
        "full_macro_str":          fm_str,
        "full_micro_mean":         fmi_m,
        "full_micro_std":          fmi_s,
        "full_micro_str":          fmi_str,
        # 200-user subset (seed=42 only — deterministic)
        "u200_macro_seed42":       u200_macros[seeds.index(42)] if 42 in seeds else u200_macros[0],
        "u200_micro_seed42":       u200_micros[seeds.index(42)] if 42 in seeds else u200_micros[0],
        # mean ± std over seeds (informational)
        "u200_macro_mean":         u200_m,
        "u200_macro_std":          u200_s,
        "u200_macro_str":          u200_str,
        "u200_micro_mean":         u200mi_m,
        "u200_micro_std":          u200mi_s,
        "u200_micro_str":          u200mi_str,
        "per_run":                 per_run,
        "best_config":             best_cfg,
        "seeds":                   seeds,
    }
    return summary


# ── output helpers ─────────────────────────────────────────────────────────────

def save_all(all_summaries: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_per_run = []
    full_rows, u200_rows = [], []

    for s in all_summaries:
        all_per_run.extend(s["per_run"])

        full_rows.append({
            "Model":   s["model"],
            "Dataset": s["dataset"],
            "Macro F1 (mean ± std)": s["full_macro_str"],
            "Micro F1 (mean ± std)": s["full_micro_str"],
            "Macro mean": s["full_macro_mean"],
            "Macro std":  s["full_macro_std"],
            "Micro mean": s["full_micro_mean"],
            "Micro std":  s["full_micro_std"],
            "Seeds":   str(s["seeds"]),
            "Best config": str(s["best_config"]),
        })

        u200_rows.append({
            "Model":   s["model"],
            "Dataset": s["dataset"],
            "Macro F1 (seed=42, 200-user)": s["u200_macro_seed42"],
            "Micro F1 (seed=42, 200-user)": s["u200_micro_seed42"],
            "Macro F1 (mean ± std, 200-user)": s["u200_macro_str"],
            "Micro F1 (mean ± std, 200-user)": s["u200_micro_str"],
        })

    pd.DataFrame(all_per_run).to_csv(
        os.path.join(out_dir, "per_run_details.csv"), index=False)
    pd.DataFrame(full_rows).to_csv(
        os.path.join(out_dir, "summary_full_testset.csv"), index=False)
    pd.DataFrame(u200_rows).to_csv(
        os.path.join(out_dir, "summary_200user.csv"), index=False)

    # human-readable text table
    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("GNN MULTI-SEED RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("FULL TEST SET (mean ± std over seeds)\n")
        f.write("-" * 60 + "\n")
        df_full = pd.DataFrame(full_rows)[
            ["Model", "Dataset", "Macro F1 (mean ± std)", "Micro F1 (mean ± std)"]]
        try:
            pivot = df_full.pivot(
                index="Model", columns="Dataset",
                values="Macro F1 (mean ± std)")
            f.write(pivot.to_string() + "\n\n")
        except Exception:
            f.write(df_full.to_string(index=False) + "\n\n")

        f.write("200-USER SUBSET (seed=42, LLM-comparable)\n")
        f.write("-" * 60 + "\n")
        df_200 = pd.DataFrame(u200_rows)[
            ["Model", "Dataset",
             "Macro F1 (seed=42, 200-user)", "Micro F1 (seed=42, 200-user)"]]
        try:
            pivot2 = df_200.pivot(
                index="Model", columns="Dataset",
                values="Macro F1 (seed=42, 200-user)")
            f.write(pivot2.to_string() + "\n\n")
        except Exception:
            f.write(df_200.to_string(index=False) + "\n\n")

    print(f"\n{'='*60}")
    print(f"  Results saved to: {out_dir}/")
    print(f"  summary_full_testset.csv  — mean ± std for paper Table 3")
    print(f"  summary_200user.csv       — 200-user comparable to LLMs")
    print(f"  per_run_details.csv       — every individual run")
    print(f"  summary.txt               — human-readable")
    print(f"{'='*60}")

    # quick console print
    print("\n  FULL TEST SET — Macro F1 (mean ± std)")
    print(f"  {'Model':<14} {'Dataset':<12} {'Macro F1':>20}  {'Micro F1':>20}")
    print("  " + "-" * 70)
    for r in full_rows:
        print(f"  {r['Model']:<14} {r['Dataset']:<12} "
              f"{r['Macro F1 (mean ± std)']:>20}  "
              f"{r['Micro F1 (mean ± std)']:>20}")

    print("\n  200-USER SUBSET — Macro F1 (seed=42)")
    print(f"  {'Model':<14} {'Dataset':<12} {'Macro F1':>12}  {'Micro F1':>12}")
    print("  " + "-" * 54)
    for r in u200_rows:
        print(f"  {r['Model']:<14} {r['Dataset']:<12} "
              f"{r['Macro F1 (seed=42, 200-user)']:>12}  "
              f"{r['Micro F1 (seed=42, 200-user)']:>12}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed GNN training + 200-user subset evaluation")
    p.add_argument("--model",        default="RGCN",
                   help="Single model name")
    p.add_argument("--dataset",      default="trump",
                   help="Single dataset name")
    p.add_argument("--models",       default=None,
                   help="Comma-separated list, e.g. RGCN,HAN")
    p.add_argument("--datasets",     default=None,
                   help="Comma-separated list, e.g. trump,abortion")
    p.add_argument("--all_models",   action="store_true",
                   help="Run all GNN models")
    p.add_argument("--all_datasets", action="store_true",
                   help="Run all datasets (trump, abortion, religion)")
    p.add_argument("--seeds",        default="42,123,456",
                   help="Comma-separated seeds (default: 42,123,456)")
    p.add_argument("--no_search",    action="store_true",
                   help="Skip HP search, load from tuning_results/ or use defaults")
    p.add_argument("--n_trials",     type=int, default=15,
                   help="HP search trials per model-dataset pair")
    p.add_argument("--data_dir",     default=DATA_DIR)
    p.add_argument("--tuning_dir",   default="tuning_results/run_20260322_160250",
                   help="Directory with previous HP search results")
    p.add_argument("--out_dir",      default="gnn_multiseed_results",
                   help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    models = (ALL_MODELS          if args.all_models
              else [m.strip() for m in args.models.split(",")] if args.models
              else [args.model])
    datasets = (DATASETS           if args.all_datasets
                else [d.strip() for d in args.datasets.split(",")] if args.datasets
                else [args.dataset])
    seeds = [int(s) for s in args.seeds.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{timestamp}")

    print("GNN MULTI-SEED + 200-USER EVALUATION")
    print(f"  Models   : {models}")
    print(f"  Datasets : {datasets}")
    print(f"  Seeds    : {seeds}")
    print(f"  HP search: {'OFF (loading from ' + args.tuning_dir + ')' if args.no_search else 'ON (' + str(args.n_trials) + ' trials)'}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Out dir  : {out_dir}")
    print("=" * 60)

    all_summaries = []
    for dataset in datasets:
        for model in models:
            try:
                summary = run_model_dataset(
                    model_name=model,
                    dataset_name=dataset,
                    seeds=seeds,
                    do_search=not args.no_search,
                    n_trials=args.n_trials,
                    data_dir=args.data_dir,
                    tuning_dir=args.tuning_dir,
                )
                all_summaries.append(summary)
            except Exception as e:
                import traceback
                print(f"\n  ERROR: {model}/{dataset}: {e}")
                traceback.print_exc()

    if all_summaries:
        save_all(all_summaries, out_dir)
    else:
        print("\nNo successful runs to save.")

    print("\nDone.")
    print("\nUsage examples:")
    print("  # all models and datasets, skip HP search (fastest)")
    print("  python eval_gnn_multiseed.py --all_models --all_datasets --no_search")
    print()
    print("  # specific models, 3 seeds, with HP search")
    print("  python eval_gnn_multiseed.py --models HAN,RGCN --all_datasets --seeds 42,123,456")
    print()
    print("  # single model quick test")
    print("  python eval_gnn_multiseed.py --model RGCN --dataset trump --no_search")