"""
eval_plm_multiseed.py
=====================
Two things in one script:

1. MULTI-SEED  — trains each PLM model with seeds [42, 123, 456] and reports
                  mean ± std over the FULL test set, matching the existing
                  text_classification_experiments.py protocol.

2. 200-USER    — for each seed, additionally evaluates the trained model on
                  the same deterministic stratified 200-user sample used by
                  the LLM experiments, so Table 3 rows are directly comparable.

Models evaluated: BERT, RoBERTa, SimCSE, Sentence-BERT

Usage
-----
  # all models, all datasets
  python eval_plm_multiseed.py --all_datasets

  # specific subset
  python eval_plm_multiseed.py --models BERT,RoBERTa --datasets trump,abortion

  # skip HP tuning (use defaults)
  python eval_plm_multiseed.py --all_datasets --no_search

Output
------
  plm_multiseed_results/
    summary_full_testset.csv   ← mean ± std over 3 seeds, full test set
    summary_200user.csv        ← mean ± std on 200-user LLM-comparable subset
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
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

# ── optional heavy deps ────────────────────────────────────────────────────────
TORCH_AVAILABLE         = False
TRANSFORMERS_AVAILABLE  = False
SIMCSE_AVAILABLE        = False
SBERT_AVAILABLE         = False

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except Exception:
    pass

try:
    from transformers import (
        BertTokenizer, BertForSequenceClassification,
        RobertaTokenizer, RobertaForSequenceClassification,
        TrainingArguments, Trainer,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception:
    pass

try:
    from simcse import SimCSE
    SIMCSE_AVAILABLE = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    pass

# ── import data loading from the PLM experiment script ────────────────────────
try:
    from text_classification_experiments import (
        SplitDatasetProcessor, CustomDataset,
    )
except ModuleNotFoundError:
    raise ImportError(
        "Cannot import text_classification_experiments.py. "
        "Make sure it is in the same directory."
    )

DATASETS    = ["trump", "abortion", "religion"]
ALL_MODELS  = ["BERT", "RoBERTa", "SimCSE", "Sentence-BERT"]
DATA_DIR    = "split_datasets_enriched_2"


# ── deterministic seed helper ─────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ── 200-user stratified sampler (mirrors llm_train.py exactly) ───────────────

def stratified_sample_200(test_texts: list,
                           test_labels: list,
                           max_samples: int = 200,
                           seed: int = 42):
    """
    Return indices (into test_texts / test_labels) of the 200-user sample
    that matches llm_train.py's stratified_sample_users().

    Returns (sampled_indices, sampled_texts, sampled_labels).
    """
    random.seed(seed)
    np.random.seed(seed)

    by_class: dict = {}
    for i, lbl in enumerate(test_labels):
        by_class.setdefault(lbl, []).append(i)

    per_class = max(1, max_samples // len(by_class))
    sampled_idx = []

    for cls in sorted(by_class.keys()):
        ids = sorted(by_class[cls])
        random.shuffle(ids)
        sampled_idx.extend(ids[:per_class])

    sampled_set = set(sampled_idx)
    remaining   = sorted([i for i in range(len(test_labels))
                          if i not in sampled_set])
    random.shuffle(remaining)
    sampled_idx.extend(remaining[: max(0, max_samples - len(sampled_idx))])
    sampled_idx = sampled_idx[:max_samples]

    sampled_texts  = [test_texts[i]  for i in sampled_idx]
    sampled_labels = [test_labels[i] for i in sampled_idx]
    return sampled_idx, sampled_texts, sampled_labels


# ── HP search config ──────────────────────────────────────────────────────────

BERT_ROBERTA_CONFIGS = [
    (2e-5, 16, 5, 0.01, 256),
    (1e-5, 16, 3, 0.01, 256),
    (3e-5, 32, 8, 0.10, 512),
    (2e-5, 32, 5, 0.10, 256),
]

SIMCSE_CONFIGS = [
    {"C": 1.0,  "solver": "lbfgs",     "max_iter": 1000, "class_weight": "balanced"},
    {"C": 0.1,  "solver": "saga",      "max_iter": 1000, "class_weight": "balanced"},
    {"C": 10.0, "solver": "liblinear", "max_iter": 1000, "class_weight": "balanced"},
    {"C": 1.0,  "solver": "lbfgs",     "max_iter": 2000, "class_weight": None},
    {"C": 0.01, "solver": "lbfgs",     "max_iter": 1000, "class_weight": "balanced"},
]

SBERT_MODELS   = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-MiniLM-L12-v2"]
SBERT_CLF_CONFIGS = SIMCSE_CONFIGS


# ── individual model trainers ─────────────────────────────────────────────────

def _metrics(y_true, y_pred):
    return (round(f1_score(y_true, y_pred, average="macro",  zero_division=0) * 100, 2),
            round(f1_score(y_true, y_pred, average="micro",  zero_division=0) * 100, 2))


def train_bert_roberta(model_name: str,
                       X_train, X_val, X_test, X_200,
                       y_train, y_val, y_test, y_200,
                       seed: int, do_search: bool,
                       experiment_dir: str):
    """Train BERT or RoBERTa, return (full_macro, full_micro, u200_macro, u200_micro)."""
    if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
        print(f"  ⚠  {model_name} skipped — torch/transformers not available")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    hf_name  = "bert-base-uncased" if model_name == "BERT" else "roberta-base"
    tok_cls  = BertTokenizer      if model_name == "BERT" else RobertaTokenizer
    model_cls = BertForSequenceClassification if model_name == "BERT" \
                else RobertaForSequenceClassification

    tokenizer = tok_cls.from_pretrained(hf_name)

    # ── optional HP search ─────────────────────────────────────────────────
    if do_search:
        best_score, best_params = 0.0, None
        for lr, bs, epochs, wd, max_len in BERT_ROBERTA_CONFIGS:
            try:
                unique_labels = sorted(set(y_train))
                l2i = {l: i for i, l in enumerate(unique_labels)}
                i2l = {i: l for l, i in l2i.items()}
                tr_ds = CustomDataset(X_train, y_train, tokenizer, max_len)
                va_ds = CustomDataset(X_val,   y_val,   tokenizer, max_len)
                mdl   = model_cls.from_pretrained(
                    hf_name, num_labels=len(unique_labels),
                    problem_type="single_label_classification",
                    label2id=l2i, id2label=i2l).to(device)
                targs = TrainingArguments(
                    output_dir=os.path.join(experiment_dir, "hp_search"),
                    num_train_epochs=epochs,
                    per_device_train_batch_size=bs,
                    per_device_eval_batch_size=bs,
                    warmup_steps=100, weight_decay=wd, learning_rate=lr,
                    logging_steps=9999, evaluation_strategy="epoch",
                    save_strategy="no", report_to=None, seed=seed,
                    fp16=device.type=="cuda", dataloader_num_workers=0,
                )
                def compute_metrics(ep):
                    preds, lbls = ep
                    return {"f1_macro": f1_score(lbls, np.argmax(preds, 1),
                                                 average="macro", zero_division=0)}
                trainer = Trainer(model=mdl, args=targs,
                                  train_dataset=tr_ds, eval_dataset=va_ds,
                                  compute_metrics=compute_metrics)
                trainer.train()
                val_f1 = trainer.evaluate().get("eval_f1_macro", 0)
                if val_f1 > best_score:
                    best_score, best_params = val_f1, (lr, bs, epochs, wd, max_len)
                del mdl, trainer, tr_ds, va_ds
                if device.type == "cuda": torch.cuda.empty_cache()
            except Exception as e:
                print(f"    HP trial failed: {e}")
        if best_params is None:
            best_params = (2e-5, 16, 5, 0.01, 256)
    else:
        best_params = (2e-5, 16, 5, 0.01, 256)

    lr, bs, epochs, wd, max_len = best_params

    # ── final training on train+val ────────────────────────────────────────
    set_seed(seed)
    X_trval = X_train + X_val
    y_trval = y_train + y_val
    unique_labels = sorted(set(y_trval))
    l2i = {l: i for i, l in enumerate(unique_labels)}
    i2l = {i: l for l, i in l2i.items()}

    tr_ds   = CustomDataset(X_trval, y_trval, tokenizer, max_len)
    te_ds   = CustomDataset(X_test,  y_test,  tokenizer, max_len)
    u200_ds = CustomDataset(X_200,   y_200,   tokenizer, max_len)

    final_mdl = model_cls.from_pretrained(
        hf_name, num_labels=len(unique_labels),
        problem_type="single_label_classification",
        label2id=l2i, id2label=i2l).to(device)

    targs = TrainingArguments(
        output_dir=os.path.join(experiment_dir, "final"),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        warmup_steps=100, weight_decay=wd, learning_rate=lr,
        logging_steps=9999, evaluation_strategy="no",
        save_strategy="no", report_to=None, seed=seed,
        fp16=device.type=="cuda", dataloader_num_workers=0,
    )
    trainer = Trainer(model=final_mdl, args=targs, train_dataset=tr_ds)
    trainer.train()

    y_test_num = [l2i[l] for l in y_test]
    y_200_num  = [l2i.get(l, 0) for l in y_200]

    te_preds  = np.argmax(trainer.predict(te_ds).predictions, 1)
    u200_preds = np.argmax(trainer.predict(u200_ds).predictions, 1)

    full_macro, full_micro = _metrics(y_test_num, te_preds)
    u200_macro, u200_micro = _metrics(y_200_num,  u200_preds)

    del final_mdl, trainer, tr_ds, te_ds, u200_ds
    if device.type == "cuda": torch.cuda.empty_cache()
    return full_macro, full_micro, u200_macro, u200_micro


def _encode_simcse(X_train, X_val, X_test, X_200):
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    def enc(X):
        emb = model.encode(X)
        return emb.cpu().numpy() if hasattr(emb, "cpu") else emb
    return enc(X_train), enc(X_val), enc(X_test), enc(X_200)


def train_simcse(X_train, X_val, X_test, X_200,
                 y_train, y_val, y_test, y_200,
                 seed: int, do_search: bool):
    if not SIMCSE_AVAILABLE:
        print("  ⚠  SimCSE skipped — pip install simcse")
        return None
    set_seed(seed)
    try:
        Xtr_e, Xv_e, Xte_e, X200_e = _encode_simcse(X_train, X_val, X_test, X_200)
    except Exception as e:
        print(f"  SimCSE encoding failed: {e}")
        return None

    if do_search:
        best_score, best_cfg = 0.0, None
        for cfg in SIMCSE_CONFIGS:
            clf = LogisticRegression(random_state=seed, **cfg)
            clf.fit(Xtr_e, y_train)
            val_f1 = f1_score(y_val, clf.predict(Xv_e),
                              average="macro", zero_division=0)
            if val_f1 > best_score:
                best_score, best_cfg = val_f1, cfg
        if best_cfg is None:
            best_cfg = SIMCSE_CONFIGS[0]
    else:
        best_cfg = SIMCSE_CONFIGS[0]

    X_trval_e = np.vstack([Xtr_e, Xv_e])
    y_trval   = y_train + y_val
    final_clf = LogisticRegression(random_state=seed, **best_cfg)
    final_clf.fit(X_trval_e, y_trval)

    full_macro, full_micro = _metrics(y_test, final_clf.predict(Xte_e))
    u200_macro, u200_micro = _metrics(y_200,  final_clf.predict(X200_e))
    return full_macro, full_micro, u200_macro, u200_micro


def train_sbert(X_train, X_val, X_test, X_200,
                y_train, y_val, y_test, y_200,
                seed: int, do_search: bool):
    if not SBERT_AVAILABLE:
        print("  ⚠  Sentence-BERT skipped — pip install sentence-transformers")
        return None
    set_seed(seed)

    if do_search:
        best_score, best_model_name, best_cfg = 0.0, None, None
        best_Xte_e, best_X200_e = None, None
        for mn in SBERT_MODELS:
            try:
                sbert = SentenceTransformer(mn)
                Xtr_e = sbert.encode(X_train, show_progress_bar=False)
                Xv_e  = sbert.encode(X_val,   show_progress_bar=False)
                for cfg in SBERT_CLF_CONFIGS:
                    clf   = LogisticRegression(random_state=seed, **cfg)
                    clf.fit(Xtr_e, y_train)
                    val_f1 = f1_score(y_val, clf.predict(Xv_e),
                                      average="macro", zero_division=0)
                    if val_f1 > best_score:
                        best_score     = val_f1
                        best_model_name = mn
                        best_cfg        = cfg
                        best_Xte_e  = sbert.encode(X_test, show_progress_bar=False)
                        best_X200_e = sbert.encode(X_200,  show_progress_bar=False)
            except Exception as e:
                print(f"  SBERT {mn} failed: {e}")
        if best_model_name is None:
            return None
    else:
        sbert = SentenceTransformer(SBERT_MODELS[0])
        Xtr_e = sbert.encode(X_train, show_progress_bar=False)
        Xv_e  = sbert.encode(X_val,   show_progress_bar=False)
        best_Xte_e  = sbert.encode(X_test, show_progress_bar=False)
        best_X200_e = sbert.encode(X_200,  show_progress_bar=False)
        best_model_name = SBERT_MODELS[0]
        best_cfg        = SBERT_CLF_CONFIGS[0]

    sbert_final = SentenceTransformer(best_model_name)
    Xtr_e_f = sbert_final.encode(X_train + X_val, show_progress_bar=False)
    final_clf = LogisticRegression(random_state=seed, **best_cfg)
    final_clf.fit(Xtr_e_f, y_train + y_val)

    full_macro, full_micro = _metrics(y_test, final_clf.predict(best_Xte_e))
    u200_macro, u200_micro = _metrics(y_200,  final_clf.predict(best_X200_e))
    return full_macro, full_micro, u200_macro, u200_micro


# ── per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(dataset_name: str, models: list, seeds: list,
                do_search: bool, data_dir: str, out_dir: str) -> list:
    """
    Returns a list of summary dicts, one per (model, dataset) pair.
    """
    print(f"\n{'='*60}")
    print(f"  PLM EXPERIMENTS: {dataset_name.upper()}  (seeds={seeds})")
    print(f"{'='*60}")

    processor = SplitDatasetProcessor(dataset_name, data_dir)
    train_u, val_u, test_u, labels = processor.load_split_data()

    X_train, y_train, _ = processor.create_text_features(train_u)
    X_val,   y_val,   _ = processor.create_text_features(val_u)
    X_test,  y_test,  _ = processor.create_text_features(test_u)

    print(f"  Split sizes: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

    # ── 200-user subset (deterministic, seed=42 — same every model/seed) ──
    _, X_200, y_200 = stratified_sample_200(X_test, y_test, max_samples=200, seed=42)
    print(f"  200-user sample: {len(X_200)} users, "
          f"classes: {dict(Counter(y_200))}")

    summaries = []
    for model_name in models:
        print(f"\n  ── {model_name} ──")
        full_macros, full_micros = [], []
        u200_macros, u200_micros = [], []
        per_run = []

        for seed in seeds:
            print(f"    seed={seed} …", flush=True)
            set_seed(seed)
            exp_dir = os.path.join(out_dir, f"{model_name}_{dataset_name}_seed{seed}")
            os.makedirs(exp_dir, exist_ok=True)
            t0 = time.time()

            try:
                if model_name in ("BERT", "RoBERTa"):
                    res = train_bert_roberta(
                        model_name,
                        X_train, X_val, X_test, X_200,
                        y_train, y_val, y_test, y_200,
                        seed=seed, do_search=do_search,
                        experiment_dir=exp_dir,
                    )
                elif model_name == "SimCSE":
                    res = train_simcse(
                        X_train, X_val, X_test, X_200,
                        y_train, y_val, y_test, y_200,
                        seed=seed, do_search=do_search,
                    )
                elif model_name == "Sentence-BERT":
                    res = train_sbert(
                        X_train, X_val, X_test, X_200,
                        y_train, y_val, y_test, y_200,
                        seed=seed, do_search=do_search,
                    )
                else:
                    print(f"    Unknown model: {model_name}")
                    res = None
            except Exception as e:
                import traceback
                print(f"    ERROR seed={seed}: {e}")
                traceback.print_exc()
                res = None

            elapsed = round(time.time() - t0, 1)
            if res is not None:
                full_macro, full_micro, u200_macro, u200_micro = res
                full_macros.append(full_macro)
                full_micros.append(full_micro)
                u200_macros.append(u200_macro)
                u200_micros.append(u200_micro)
                print(f"    Full test → Macro={full_macro:.2f}%  Micro={full_micro:.2f}%")
                print(f"    200-user  → Macro={u200_macro:.2f}%  Micro={u200_micro:.2f}%")
                per_run.append({
                    "model":         model_name,
                    "dataset":       dataset_name,
                    "seed":          seed,
                    "full_macro_f1": full_macro,
                    "full_micro_f1": full_micro,
                    "u200_macro_f1": u200_macro,
                    "u200_micro_f1": u200_micro,
                    "elapsed_sec":   elapsed,
                })

        if not full_macros:
            print(f"  ⚠  {model_name}/{dataset_name}: all seeds failed, skipping.")
            continue

        def fmt(vals):
            m, s = np.mean(vals), np.std(vals)
            return round(m, 2), round(s, 2), f"{m:.2f} ± {s:.2f}"

        fm_m, fm_s, fm_str   = fmt(full_macros)
        fmi_m, fmi_s, fmi_str = fmt(full_micros)
        u200_m, u200_s, u200_str = fmt(u200_macros)
        u200mi_m, u200mi_s, u200mi_str = fmt(u200_micros)

        summaries.append({
            "model":   model_name,
            "dataset": dataset_name,
            # full test set
            "full_macro_mean":  fm_m,
            "full_macro_std":   fm_s,
            "full_macro_str":   fm_str,
            "full_micro_mean":  fmi_m,
            "full_micro_std":   fmi_s,
            "full_micro_str":   fmi_str,
            # 200-user — use seed=42 value as the "canonical" comparable number
            "u200_macro_seed42": u200_macros[seeds.index(42)] if 42 in seeds else u200_macros[0],
            "u200_micro_seed42": u200_micros[seeds.index(42)] if 42 in seeds else u200_micros[0],
            "u200_macro_mean":  u200_m,
            "u200_macro_std":   u200_s,
            "u200_macro_str":   u200_str,
            "u200_micro_mean":  u200mi_m,
            "u200_micro_std":   u200mi_s,
            "u200_micro_str":   u200mi_str,
            "per_run":          per_run,
            "seeds":            seeds,
        })
        print(f"\n  {model_name}/{dataset_name} summary:")
        print(f"    Full test : {fm_str}  /  {fmi_str}")
        print(f"    200-user  : {u200_str}  /  {u200mi_str}")

    return summaries


# ── output helpers ─────────────────────────────────────────────────────────────

def save_all(all_summaries: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_per_run, full_rows, u200_rows = [], [], []

    for s in all_summaries:
        all_per_run.extend(s["per_run"])
        full_rows.append({
            "Model":   s["model"],
            "Dataset": s["dataset"],
            "Macro F1 (mean ± std)": s["full_macro_str"],
            "Micro F1 (mean ± std)": s["full_micro_str"],
            "Macro mean": s["full_macro_mean"],
            "Macro std":  s["full_macro_std"],
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

    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("PLM MULTI-SEED RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("FULL TEST SET (mean ± std over seeds)\n")
        f.write("-" * 60 + "\n")
        df_full = pd.DataFrame(full_rows)
        try:
            pivot = df_full.pivot(
                index="Model", columns="Dataset",
                values="Macro F1 (mean ± std)")
            f.write(pivot.to_string() + "\n\n")
        except Exception:
            f.write(df_full.to_string(index=False) + "\n\n")

        f.write("200-USER SUBSET (seed=42, LLM-comparable)\n")
        f.write("-" * 60 + "\n")
        df_200 = pd.DataFrame(u200_rows)
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

    print("\n  FULL TEST SET — Macro F1 (mean ± std)")
    print(f"  {'Model':<16} {'Dataset':<12} {'Macro F1':>20}  {'Micro F1':>20}")
    print("  " + "-" * 72)
    for r in full_rows:
        print(f"  {r['Model']:<16} {r['Dataset']:<12} "
              f"{r['Macro F1 (mean ± std)']:>20}  "
              f"{r['Micro F1 (mean ± std)']:>20}")

    print("\n  200-USER SUBSET — Macro F1 (seed=42)")
    print(f"  {'Model':<16} {'Dataset':<12} {'Macro F1':>12}  {'Micro F1':>12}")
    print("  " + "-" * 56)
    for r in u200_rows:
        print(f"  {r['Model']:<16} {r['Dataset']:<12} "
              f"{r['Macro F1 (seed=42, 200-user)']:>12}  "
              f"{r['Micro F1 (seed=42, 200-user)']:>12}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed PLM training + 200-user subset evaluation")
    p.add_argument("--model",        default="BERT")
    p.add_argument("--dataset",      default="trump")
    p.add_argument("--models",       default=None,
                   help="Comma-separated, e.g. BERT,RoBERTa")
    p.add_argument("--datasets",     default=None,
                   help="Comma-separated, e.g. trump,abortion")
    p.add_argument("--all_models",   action="store_true")
    p.add_argument("--all_datasets", action="store_true")
    p.add_argument("--seeds",        default="42,123,456")
    p.add_argument("--no_search",    action="store_true",
                   help="Skip HP search, use default configs")
    p.add_argument("--data_dir",     default=DATA_DIR)
    p.add_argument("--out_dir",      default="plm_multiseed_results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    models   = (ALL_MODELS if args.all_models
                else [m.strip() for m in args.models.split(",")] if args.models
                else [args.model])
    datasets = (DATASETS   if args.all_datasets
                else [d.strip() for d in args.datasets.split(",")] if args.datasets
                else [args.dataset])
    seeds    = [int(s) for s in args.seeds.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("PLM MULTI-SEED + 200-USER EVALUATION")
    print(f"  Models   : {models}")
    print(f"  Datasets : {datasets}")
    print(f"  Seeds    : {seeds}")
    print(f"  HP search: {'OFF (defaults)' if args.no_search else 'ON'}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Torch      : {TORCH_AVAILABLE}")
    print(f"  Transformers: {TRANSFORMERS_AVAILABLE}")
    print(f"  SimCSE     : {SIMCSE_AVAILABLE}")
    print(f"  SBERT      : {SBERT_AVAILABLE}")
    print("=" * 60)

    all_summaries = []
    for dataset in datasets:
        try:
            summaries = run_dataset(
                dataset_name=dataset,
                models=models,
                seeds=seeds,
                do_search=not args.no_search,
                data_dir=args.data_dir,
                out_dir=out_dir,
            )
            all_summaries.extend(summaries)
        except Exception as e:
            import traceback
            print(f"\n  ERROR dataset={dataset}: {e}")
            traceback.print_exc()

    if all_summaries:
        save_all(all_summaries, out_dir)
    else:
        print("\nNo successful runs to save.")

    print("\nDone.")
    print("\nUsage examples:")
    print("  # fastest: all models, all datasets, no HP search")
    print("  python eval_plm_multiseed.py --all_models --all_datasets --no_search")
    print()
    print("  # full run with HP tuning")
    print("  python eval_plm_multiseed.py --all_models --all_datasets --seeds 42,123,456")
    print()
    print("  # single model quick test")
    print("  python eval_plm_multiseed.py --model BERT --dataset trump --no_search")