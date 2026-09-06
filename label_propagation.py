"""
Usage
-----
    python label_propagation.py                        # all 5 datasets
    python label_propagation.py --dataset trump        # single dataset
    python label_propagation.py --alpha 0.85           # skip tuning
    python label_propagation.py --max_iter 200 --tol 1e-8
    python label_propagation.py --data_dir split_datasets
"""

import os
import json
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from datasets import set_deterministic_behavior

# ── helpers ───────────────────────────────────────────────────────────────────

def build_adjacency(src, dst, num_nodes):
    """
    Build an undirected, self-looped, row-normalised sparse adjacency.
    src / dst are integer arrays of edge endpoints.
    """
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)

    A = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    A = A + sp.eye(num_nodes, format="csr", dtype=np.float32)   # self-loops

    row_sums = np.asarray(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_norm = sp.diags(1.0 / row_sums).dot(A)
    return A_norm


def label_propagation(A_norm, Y, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Iterative LP:  F = alpha * A_norm * F  +  (1 - alpha) * Y
    Returns the soft-label matrix F of shape (N, C).
    """
    F = Y.copy().astype(np.float32)
    delta = float("inf")
    for i in range(max_iter):
        F_new = alpha * A_norm.dot(F) + (1.0 - alpha) * Y
        delta = float(np.abs(F_new - F).max())
        F = F_new
        if delta < tol:
            print(f"    Converged after {i+1} iterations (delta={delta:.2e})")
            return F
    print(f"    Did not converge in {max_iter} iterations (final delta={delta:.2e})")
    return F


def build_seed_matrix(labels, seed_mask, num_classes):
    """One-hot Y matrix; rows for unlabelled/test nodes are all-zero."""
    N = len(labels)
    Y = np.zeros((N, num_classes), dtype=np.float32)
    for i in np.where(seed_mask)[0]:
        if labels[i] >= 0:
            Y[i, labels[i]] = 1.0
    return Y


def evaluate(F, labels, mask):
    """Return (macro_f1, micro_f1) for nodes in mask with valid labels."""
    valid = mask & (labels >= 0)
    if valid.sum() == 0:
        return 0.0, 0.0
    preds  = np.argmax(F[valid], axis=1)
    truths = labels[valid]
    macro  = f1_score(truths, preds, average="macro",  zero_division=0)
    micro  = f1_score(truths, preds, average="micro",  zero_division=0)
    return macro, micro


# ── JSON loading (no DGL required) ────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_ucu_edges(data, user_map):
    """Return (src_array, dst_array) for user_comment_user edges."""
    src_list, dst_list = [], []
    for edge in data["edges"]:
        if edge["type"] == "user_comment_user":
            s, d = edge["source"], edge["target"]
            if s in user_map and d in user_map:
                src_list.append(user_map[s])
                dst_list.append(user_map[d])
    return (np.array(src_list, dtype=np.int32),
            np.array(dst_list, dtype=np.int32))


# ── per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(dataset_name, data_dir="split_datasets_enriched_2",
                alpha=None, alpha_grid=None,
                max_iter=100, tol=1e-6):

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_name.upper()}")
    print(f"{'='*60}")

    train_data = load_json(os.path.join(data_dir, dataset_name, "train.json"))
    val_data   = load_json(os.path.join(data_dir, dataset_name, "validation.json"))
    test_data  = load_json(os.path.join(data_dir, dataset_name, "test.json"))

    # ── unified user map (train ∪ val ∪ test) ──
    all_user_ids, seen = [], set()
    for data in (train_data, val_data, test_data):
        for node in data["nodes"]:
            if node["type"] == "user" and node["id"] not in seen:
                all_user_ids.append(node["id"])
                seen.add(node["id"])

    user_map  = {uid: i for i, uid in enumerate(all_user_ids)}
    num_users = len(user_map)

    # ── flair -> idx (sorted for reproducibility) ──
    flairs = sorted(set(
        n["label"]
        for data in (train_data, val_data, test_data)
        for n in data["nodes"]
        if n["type"] == "user" and "label" in n
    ))
    flair_to_idx = {f: i for i, f in enumerate(flairs)}
    idx_to_flair = {i: f for f, i in flair_to_idx.items()}
    num_classes  = len(flair_to_idx)

    # ── labels and split masks ──
    labels     = np.full(num_users, -1, dtype=np.int64)
    train_mask = np.zeros(num_users, dtype=bool)
    val_mask   = np.zeros(num_users, dtype=bool)
    test_mask  = np.zeros(num_users, dtype=bool)

    for node in train_data["nodes"]:
        if node["type"] == "user" and node["id"] in user_map:
            idx = user_map[node["id"]]
            train_mask[idx] = True
            if node.get("label") in flair_to_idx:
                labels[idx] = flair_to_idx[node["label"]]

    for node in val_data["nodes"]:
        if node["type"] == "user" and node["id"] in user_map:
            idx = user_map[node["id"]]
            val_mask[idx] = True
            if node.get("label") in flair_to_idx:
                labels[idx] = flair_to_idx[node["label"]]

    for node in test_data["nodes"]:
        if node["type"] == "user" and node["id"] in user_map:
            idx = user_map[node["id"]]
            test_mask[idx] = True
            if node.get("label") in flair_to_idx:
                labels[idx] = flair_to_idx[node["label"]]

    print(f"  Classes     : {num_classes}")
    print(f"  Train users : {int(train_mask.sum())}  |  "
          f"Val users : {int(val_mask.sum())}  |  "
          f"Test users : {int(test_mask.sum())}")

    # ── combined adjacency (train + val + test ucu edges) ──
    src_parts, dst_parts = [], []
    for data in (train_data, val_data, test_data):
        s, d = extract_ucu_edges(data, user_map)
        src_parts.append(s)
        dst_parts.append(d)
    src_all = np.concatenate(src_parts)
    dst_all = np.concatenate(dst_parts)
    print(f"  UCU edges   : {len(src_all)} directed")

    A_norm = build_adjacency(src_all, dst_all, num_users)

    # ── alpha tuning using the real validation set ──
    if alpha is not None:
        best_alpha = alpha
        print(f"  Using fixed alpha = {best_alpha}")
    else:
        if alpha_grid is None:
            alpha_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        print(f"  Tuning alpha over {alpha_grid} using validation set …")

        Y_train = build_seed_matrix(labels, train_mask, num_classes)

        best_alpha     = alpha_grid[0]
        best_val_macro = -1.0

        for a in alpha_grid:
            F = label_propagation(A_norm, Y_train, alpha=a,
                                  max_iter=max_iter, tol=tol)
            macro, _ = evaluate(F, labels, val_mask)
            print(f"    alpha={a:.2f}  ->  val macro F1 = {macro:.4f}")
            if macro > best_val_macro:
                best_val_macro = macro
                best_alpha     = a

        print(f"  Best alpha = {best_alpha}  "
              f"(val macro F1 = {best_val_macro:.4f})")

    # ── final run: seed train+val nodes, predict test nodes ──
    print(f"  Final propagation with alpha={best_alpha} …")
    seed_mask = train_mask | val_mask
    Y_final = build_seed_matrix(labels, seed_mask, num_classes)
    F_final = label_propagation(A_norm, Y_final, alpha=best_alpha,
                                max_iter=max_iter, tol=tol)

    test_macro,  test_micro  = evaluate(F_final, labels, test_mask)
    train_macro, train_micro = evaluate(F_final, labels, train_mask)

    print(f"\n  Results")
    print(f"  Train  Macro F1 : {train_macro*100:.2f}%   "
          f"Micro F1 : {train_micro*100:.2f}%")
    print(f"  Test   Macro F1 : {test_macro*100:.2f}%   "
          f"Micro F1 : {test_micro*100:.2f}%")

    return {
        "dataset"        : dataset_name,
        "alpha"          : best_alpha,
        "num_users"      : num_users,
        "num_classes"    : num_classes,
        "train_macro_f1" : round(train_macro * 100, 2),
        "train_micro_f1" : round(train_micro * 100, 2),
        "test_macro_f1"  : round(test_macro  * 100, 2),
        "test_micro_f1"  : round(test_micro  * 100, 2),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Label Propagation on ControBench")
    parser.add_argument("--dataset",  type=str,   default=None,
                        help="Single dataset (default: all five)")
    parser.add_argument("--data_dir", type=str,   default="split_datasets_enriched_2")
    parser.add_argument("--alpha",    type=float, default=None,
                        help="Fixed alpha; if omitted, tuned automatically")
    parser.add_argument("--max_iter", type=int,   default=100)
    parser.add_argument("--tol",      type=float, default=1e-6)
    parser.add_argument("--output",   type=str,   default="lp_results.json")
    args = parser.parse_args()

    set_deterministic_behavior(42)

    all_datasets    = ["trump", "abortion", "religion"]
    datasets_to_run = [args.dataset] if args.dataset else all_datasets
    alpha_grid      = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    all_results = []
    for ds in datasets_to_run:
        result = run_dataset(
            dataset_name = ds,
            data_dir     = args.data_dir,
            alpha        = args.alpha,
            alpha_grid   = alpha_grid,
            max_iter     = args.max_iter,
            tol          = args.tol,
        )
        all_results.append(result)

    # ── summary table ──
    print("\n\n" + "="*68)
    print("  LABEL PROPAGATION  --  Test Set Results")
    print("="*68)
    print(f"{'Dataset':<14} {'Alpha':>6}  {'Macro F1':>10}  {'Micro F1':>10}")
    print("-"*68)
    for r in all_results:
        print(f"{r['dataset']:<14} {r['alpha']:>6.2f}  "
              f"{r['test_macro_f1']:>9.2f}%  {r['test_micro_f1']:>9.2f}%")
    print("="*68)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()