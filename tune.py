"""
Unified hyperparameter tuning for all ControBench GNN models.

Protocol (standard semi-supervised / transductive node classification):
  - One merged graph containing all train/val/test users, posts, and edges.
  - Same fixed user split (60/20/20) as the PLM/LLM experiments.
  - Loss computed on train users only; val used for HP tuning and early stopping;
    test evaluated once at the end.
  - All 9 graph models use this setup for the main paper table.

Optional inductive appendix for HinSAGE:
  python tune.py --model HinSAGE --dataset trump --inductive

Usage
-----
  python tune.py --model RGCN --dataset trump
  python tune.py --dataset abortion --all_models
  python tune.py --all_models --all_datasets
  python tune.py --models RGCN,HAN --datasets trump,abortion
  python tune.py --model HAN --dataset lgbtq --no_search
  python tune.py --data_dir /path/to/splits --all_models --all_datasets
"""

import os
import sys
import json
import time
import random
import itertools
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

import dgl
from sklearn.metrics import f1_score



from models import (
    HETERO_MODELS, HOMO_MODELS, ALL_MODELS,
    build_homogeneous_user_graph,
    build_2hop_graph,
    build_khop_hetero_graph,
    GCNPlusGFS,
)

DATASETS = ["trump", "abortion","religion"]
DATA_DIR = "split_datasets_enriched_2"

# ── HP grids ──────────────────────────────────────────────────────────────────

BASE_GRID = {
    "hidden_size":  [128, 256],
    "n_layers":     [2, 3],
    "dropout":      [0.3, 0.5],
    "lr":           [0.001, 0.005],
    "weight_decay": [1e-4, 5e-4],
}

MODEL_EXTRA_GRID = {
    "H2GFormer":  {"num_heads":    [4, 8]},
    "HAN":        {"num_heads":    [4, 8]},
    "GCNPlusGFS": {"split_ratio":  [0.3, 0.5, 0.7]},
}


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    dgl.seed(seed)
    os.environ.update({"PYTHONHASHSEED": str(seed),
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    torch.set_num_threads(1)


# ── data loading ──────────────────────────────────────────────────────────────

def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _post_feat(node):
    emb, temb = node.get("embedding"), node.get("title_embedding")
    if emb and temb:
        return torch.tensor(emb + temb, dtype=torch.float)
    if emb:
        return torch.tensor(emb, dtype=torch.float)
    if temb:
        return torch.tensor(temb, dtype=torch.float)
    return torch.zeros(768)


def _build_dgl(user_map, post_map, edges_raw):
    """Build DGL heterogeneous graph from raw edge list."""
    pub, com, ucu = [], [], []
    com_feats, ucu_feats, ucu_reply = [], [], []

    for e in edges_raw:
        src, dst, et = e["source"], e["target"], e["type"]
        if et == "user_publish_post" and src in user_map and dst in post_map:
            pub.append((user_map[src], post_map[dst]))
        elif et == "user_comment_post" and src in user_map and dst in post_map:
            com.append((user_map[src], post_map[dst]))
            com_feats.append(torch.tensor(e["embedding"], dtype=torch.float)
                             if "embedding" in e else torch.zeros(768))
        elif et == "user_comment_user" and src in user_map and dst in user_map:
            ucu.append((user_map[src], user_map[dst]))
            ucu_feats.append(torch.tensor(e["embedding"], dtype=torch.float)
                             if "embedding" in e else torch.zeros(768))
            ucu_reply.append(torch.tensor(e["reply_embedding"], dtype=torch.float)
                             if "reply_embedding" in e else torch.zeros(768))

    num_u, num_p = len(user_map), len(post_map)
    g = dgl.heterograph(
        {("user", "publish",           "post"): pub  or [(0, 0)],
         ("user", "comment",           "post"): com  or [(0, 0)],
         ("user", "user_comment_user", "user"): ucu  or [(0, 0)]},
        num_nodes_dict={"user": num_u, "post": num_p}
    )
    # remove the placeholder edges we added to satisfy DGL's non-empty requirement
    for etype, lst in [("publish", pub), ("comment", com), ("user_comment_user", ucu)]:
        if not lst:
            g = dgl.remove_edges(g, torch.tensor([0]), etype=etype)

    if com_feats and g.num_edges("comment") > 0:
        g.edges["comment"].data["feat"] = torch.stack(com_feats)
    if ucu_feats and g.num_edges("user_comment_user") > 0:
        g.edges["user_comment_user"].data["feat"]       = torch.stack(ucu_feats)
        g.edges["user_comment_user"].data["reply_feat"] = torch.stack(ucu_reply)

    return g


def load_separate(dataset_name, data_dir=DATA_DIR):
    """
    Inductive loading — three independent DGL graphs, one per split.
    Matches datasets.py / train__2_.py used by the original tune_hyperparams.py.
    """
    paths = {s: os.path.join(data_dir, dataset_name,
                              "validation.json" if s == "val" else f"{s}.json")
             for s in ("train", "val", "test")}
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run split_dataset.py first.")

    splits = {s: _read_json(p) for s, p in paths.items()}

    # flair vocab from train only
    train_user_nodes = [n for n in splits["train"]["nodes"] if n["type"] == "user"]
    flairs = sorted(set(n["label"] for n in train_user_nodes if "label" in n))
    flair_to_idx = {f: i for i, f in enumerate(flairs)}
    idx_to_flair = {i: f for f, i in flair_to_idx.items()}
    num_classes  = len(flair_to_idx)

    graphs, label_tensors = {}, {}
    for s, data in splits.items():
        user_nodes = {n["id"]: n for n in data["nodes"] if n["type"] == "user"}
        post_nodes = {n["id"]: n for n in data["nodes"] if n["type"] == "post"}
        user_map   = {uid: i for i, uid in enumerate(user_nodes)}
        post_map   = {pid: i for i, pid in enumerate(post_nodes)}

        g = _build_dgl(user_map, post_map, data["edges"])

        # post features
        post_feat_list = [_post_feat(pn) for pn in post_nodes.values()]
        if post_feat_list:
            max_d = max(f.shape[0] for f in post_feat_list)
            post_feat_list = [torch.cat([f, torch.zeros(max_d - f.shape[0])])
                              if f.shape[0] < max_d else f for f in post_feat_list]
            g.nodes["post"].data["feat"] = torch.stack(post_feat_list)
        else:
            g.nodes["post"].data["feat"] = torch.zeros(len(post_map), 768)
        g.nodes["user"].data["feat"] = torch.zeros(len(user_map), 768)

        # labels
        labels = torch.full((len(user_map),), -1, dtype=torch.long)
        for uid, n in user_nodes.items():
            if n.get("label") in flair_to_idx:
                labels[user_map[uid]] = flair_to_idx[n["label"]]

        graphs[s]       = g
        label_tensors[s] = labels

    print(f"  {dataset_name} (separate): "
          f"train={graphs['train'].num_nodes('user')} "
          f"val={graphs['val'].num_nodes('user')} "
          f"test={graphs['test'].num_nodes('user')} classes={num_classes}")

    return (graphs["train"], graphs["val"], graphs["test"],
            label_tensors["train"], label_tensors["val"], label_tensors["test"],
            num_classes, idx_to_flair)


def load_merged(dataset_name, data_dir=DATA_DIR):
    """
    Transductive loading — all splits merged into one graph with masks.
    Matches tune_homo.py / tune_hetero__1_.py.
    """
    paths = {s: os.path.join(data_dir, dataset_name,
                              "validation.json" if s == "val" else f"{s}.json")
             for s in ("train", "val", "test")}
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run split_dataset.py first.")

    splits = {s: _read_json(p) for s, p in paths.items()}

    # label vocab: train only, same threshold as split_dataset.py
    train_users  = [n for n in splits["train"]["nodes"] if n["type"] == "user"]
    label_counts = Counter(n["label"] for n in train_users if "label" in n)
    total_train  = len(train_users)
    min_pct      = splits["train"].get("min_class_percentage",
                                        0.05 if dataset_name == "trump" else 0.01)
    valid_labels = sorted(l for l, c in label_counts.items()
                          if c / total_train >= min_pct)
    flair_to_idx = {f: i for i, f in enumerate(valid_labels)}
    idx_to_flair = {i: f for f, i in flair_to_idx.items()}
    num_classes  = len(flair_to_idx)

    # collect unique users/posts across all splits
    all_user_ids, seen_u = [], set()
    for s in ("train", "val", "test"):
        for n in splits[s]["nodes"]:
            if n["type"] == "user" and n["id"] not in seen_u:
                all_user_ids.append(n["id"]); seen_u.add(n["id"])

    all_post_ids, seen_p = [], set()
    for s in ("train", "val", "test"):
        for n in splits[s]["nodes"]:
            if n["type"] == "post" and n["id"] not in seen_p:
                all_post_ids.append(n["id"]); seen_p.add(n["id"])

    user_map = {uid: i for i, uid in enumerate(all_user_ids)}
    post_map = {pid: i for i, pid in enumerate(all_post_ids)}
    num_u, num_p = len(user_map), len(post_map)

    labels     = torch.full((num_u,), -1, dtype=torch.long)
    train_mask = torch.zeros(num_u, dtype=torch.bool)
    val_mask   = torch.zeros(num_u, dtype=torch.bool)
    test_mask  = torch.zeros(num_u, dtype=torch.bool)
    mask_map   = {"train": train_mask, "val": val_mask, "test": test_mask}

    for s in ("train", "val", "test"):
        for n in splits[s]["nodes"]:
            if n["type"] == "user" and n["id"] in user_map:
                idx = user_map[n["id"]]
                if n.get("label") in flair_to_idx:
                    labels[idx] = flair_to_idx[n["label"]]
                    mask_map[s][idx] = True

    # collect edges — multi-edges allowed (different content = different fingerprint)
    # dedup only blocks the exact same raw interaction appearing in multiple split files
    def _fingerprint(emb):
        # first 4 values rounded to 4dp — fast, collision-free content identifier
        if emb and len(emb) >= 4:
            return tuple(round(v, 4) for v in emb[:4])
        return ()

    pub_edges = []
    com_edges, com_feats = [], []
    ucu_edges, ucu_feats, ucu_reply_feats = [], [], []
    seen_keys: set = set()
    _zero = torch.zeros(768)

    for s in ("train", "val", "test"):
        for e in splits[s]["edges"]:
            src, dst, et = e["source"], e["target"], e["type"]

            if et == "user_publish_post":
                if src in user_map and dst in post_map:
                    key = (src, dst, et)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        pub_edges.append((user_map[src], post_map[dst]))

            elif et == "user_comment_post":
                if src in user_map and dst in post_map:
                    emb = e.get("embedding", [])
                    key = (src, dst, et, _fingerprint(emb))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        com_edges.append((user_map[src], post_map[dst]))
                        com_feats.append(
                            torch.tensor(emb, dtype=torch.float)
                            if emb else _zero.clone())

            elif et == "user_comment_user":
                if src in user_map and dst in user_map:
                    emb       = e.get("embedding", [])
                    reply_emb = e.get("reply_embedding", [])
                    key = (src, dst, et, _fingerprint(emb), _fingerprint(reply_emb))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        ucu_edges.append((user_map[src], user_map[dst]))
                        ucu_feats.append(
                            torch.tensor(emb, dtype=torch.float)
                            if emb else _zero.clone())
                        ucu_reply_feats.append(
                            torch.tensor(reply_emb, dtype=torch.float)
                            if reply_emb else _zero.clone())

    graph_data = {}
    if pub_edges:  graph_data[("user", "publish",           "post")] = pub_edges
    if com_edges:  graph_data[("user", "comment",           "post")] = com_edges
    if ucu_edges:  graph_data[("user", "user_comment_user", "user")] = ucu_edges
    hetero_g = dgl.heterograph(graph_data,
                                num_nodes_dict={"user": num_u, "post": num_p})

    edge_features = {}
    if com_edges and "comment" in hetero_g.etypes:
        cf = torch.stack(com_feats)
        hetero_g.edges["comment"].data["feat"] = cf
        edge_features["comment"] = cf
    if ucu_edges and "user_comment_user" in hetero_g.etypes:
        uf = torch.stack(ucu_feats)
        hetero_g.edges["user_comment_user"].data["feat"] = uf
        edge_features["user_comment_user"] = uf
        rf = torch.stack(ucu_reply_feats)
        hetero_g.edges["user_comment_user"].data["reply_feat"] = rf

    # post features
    post_by_id = {}
    for s in ("train", "val", "test"):
        for n in splits[s]["nodes"]:
            if n["type"] == "post" and n["id"] not in post_by_id:
                post_by_id[n["id"]] = n

    post_feats_list = [_post_feat(post_by_id.get(pid, {})) for pid in all_post_ids]
    max_d = max(f.shape[0] for f in post_feats_list) if post_feats_list else 768
    pft   = torch.stack([
        torch.cat([f, torch.zeros(max_d - f.shape[0])]) if f.shape[0] < max_d else f
        for f in post_feats_list
    ]) if post_feats_list else torch.zeros(num_p, 768)

    hetero_g.nodes["post"].data["feat"] = pft
    hetero_g.nodes["user"].data["feat"] = torch.zeros(num_u, 768)

    node_features = {
        "user": hetero_g.nodes["user"].data["feat"],
        "post": hetero_g.nodes["post"].data["feat"],
    }

    print(f"  {dataset_name} (merged): {num_u} users, {num_p} posts | "
          f"train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()} "
          f"classes={num_classes}")

    return (hetero_g, edge_features, node_features,
            labels, train_mask, val_mask, test_mask, num_classes, idx_to_flair,
            user_map, splits)   # extra: user_map ordering + raw splits for EHR


# ── eval helpers ──────────────────────────────────────────────────────────────

def _f1(preds, truths):
    return (f1_score(truths, preds, average="macro",  zero_division=0),
            f1_score(truths, preds, average="micro",  zero_division=0))


def _eval_hetero(model, g, nf, ef, labels, mask):
    model.eval()
    with torch.no_grad():
        out = model(g, nf, ef)
        if isinstance(out, tuple): out = out[0]
        return _f1(out[mask].argmax(1).cpu().numpy(), labels[mask].cpu().numpy())


def _eval_homo(model, g_no_sl, g_raw, features, labels, mask, g_2hop=None):
    model.eval()
    with torch.no_grad():
        if model.__class__.__name__ == "H2GCN":
            logits = model(g_no_sl, g_2hop, features)
        else:
            logits = model(g_raw, features)
        return _f1(logits[mask].argmax(1).cpu().numpy(), labels[mask].cpu().numpy())


def _eval_separate(model, g, nf, ef, labels):
    mask = labels >= 0
    model.eval()
    with torch.no_grad():
        out = model(g, nf, ef)
        if isinstance(out, tuple): out = out[0]
        return _f1(out[mask].argmax(1).cpu().numpy(), labels[mask].cpu().numpy())


def _extract_ef(g, device):
    ef = {}
    if "feat" in g.edges["comment"].data:
        ef["comment"] = g.edges["comment"].data["feat"].to(device)
    if "feat" in g.edges["user_comment_user"].data:
        ef["user_comment_user"] = g.edges["user_comment_user"].data["feat"].to(device)
    return ef


def _class_weights(labels, mask, num_classes, device):
    c = torch.bincount(labels[mask], minlength=num_classes).float().clamp(min=1)
    w = 1.0 / c
    return (w / w.sum() * num_classes).to(device)


# ── training ──────────────────────────────────────────────────────────────────

def _training_loop(model, criterion, optimizer, scheduler, num_epochs, patience,
                   train_fn, eval_fn):
    """Generic train/eval loop. train_fn() → loss; eval_fn() → (macro, micro)."""
    best_macro, best_micro = -1.0, 0.0
    best_state = None
    ctr = 0
    for _ in range(num_epochs):
        model.train()
        loss = train_fn()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        vm, vmi = eval_fn()
        scheduler.step(vm)
        if vm > best_macro:
            best_macro, best_micro = vm, vmi
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ctr = 0
        else:
            ctr += 1
        if ctr >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return best_macro, best_micro


def train_separate(model_name, dataset_name,
                   train_g, val_g, test_g,
                   train_labels, val_labels, test_labels, num_classes,
                   hidden_size=256, n_layers=2, dropout=0.5, lr=0.005,
                   weight_decay=5e-4, num_heads=4,
                   num_epochs=200, patience=20, seed=42, **kwargs):
    """Inductive training for RGCN/HAN/HinSAGE."""
    set_seed(seed)
    device = torch.device("cpu")

    train_g = train_g.to(device); val_g = val_g.to(device); test_g = test_g.to(device)
    tl = train_labels.to(device); vl = val_labels.to(device); tel = test_labels.to(device)

    def _nf(g):
        return {"user": g.nodes["user"].data["feat"].to(device),
                "post": g.nodes["post"].data["feat"].to(device)}

    train_nf, val_nf, test_nf = _nf(train_g), _nf(val_g), _nf(test_g)
    train_ef, val_ef, test_ef = (_extract_ef(train_g, device),
                                  _extract_ef(val_g,   device),
                                  _extract_ef(test_g,  device))

    in_feats   = train_nf["user"].shape[1]
    post_feats = train_nf["post"].shape[1]
    edge_dim   = next(iter(train_ef.values())).shape[1] if train_ef else 768

    model = HETERO_MODELS[model_name](
        in_feats=in_feats, edge_feats=edge_dim, hidden_size=hidden_size,
        out_classes=num_classes, dropout=dropout, n_layers=n_layers,
        num_heads=num_heads, post_feats=post_feats,
    ).to(device)

    crit  = torch.nn.CrossEntropyLoss(weight=_class_weights(tl, tl >= 0, num_classes, device))
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=patience // 2, factor=0.5)
    mask  = tl >= 0

    def _train():
        out = model(train_g, train_nf, train_ef)
        if isinstance(out, tuple): main, aux = out; return crit(main[mask], tl[mask]) + 0.1 * crit(aux[mask], tl[mask])
        return crit(out[mask], tl[mask])

    bvm, bvmi = _training_loop(model, crit, opt, sched, num_epochs, patience,
                                _train, lambda: _eval_separate(model, val_g, val_nf, val_ef, vl))

    tm, tmi = _eval_separate(model, test_g, test_nf, test_ef, tel)
    return {"model": model_name, "dataset": dataset_name,
            "test_macro_f1": round(tm*100, 2), "test_micro_f1": round(tmi*100, 2),
            "val_macro_f1":  round(bvm*100, 2), "val_micro_f1":  round(bvmi*100, 2)}


def train_merged(model_name, dataset_name,
                 hetero_g, edge_features, node_features,
                 labels, train_mask, val_mask, test_mask, num_classes,
                 hidden_size=256, n_layers=2, dropout=0.5, lr=0.005,
                 weight_decay=5e-4, num_heads=4, split_ratio=0.5,
                 num_epochs=200, patience=20, seed=42, **kwargs):
    """Transductive training for all other models. Matches tune_homo/hetero."""
    set_seed(seed)
    device = torch.device("cpu")

    labels    = labels.to(device)
    hetero_g  = hetero_g.to(device)
    nf        = {k: v.to(device) for k, v in node_features.items()}
    ef        = {k: v.to(device) for k, v in edge_features.items()}
    if "reply_feat" in hetero_g.edges["user_comment_user"].data:
        hetero_g.edges["user_comment_user"].data["reply_feat"] = \
            hetero_g.edges["user_comment_user"].data["reply_feat"].to(device)

    in_feats   = nf["user"].shape[1]
    post_feats = nf["post"].shape[1]
    edge_dim   = next(iter(ef.values())).shape[1] if ef else 768
    is_homo    = model_name in HOMO_MODELS

    if is_homo:
        homo_no, homo_raw, features = build_homogeneous_user_graph(hetero_g, ef, nf)
        homo_no  = homo_no.to(device)
        homo_raw = homo_raw.to(device)
        g_2hop   = build_2hop_graph(homo_no).to(device)
        features = features.to(device)
        in_h     = features.shape[1]
        if model_name == "H2GCN":
            model = HOMO_MODELS["H2GCN"](in_h, hidden_size, num_classes, n_layers=n_layers, dropout=dropout)
        elif model_name == "ACMGNN":
            model = HOMO_MODELS["ACMGNN"](in_h, hidden_size, num_classes, n_layers=n_layers, dropout=dropout)
        else:
            model = GCNPlusGFS(in_h, hidden_size, num_classes, n_layers=n_layers,
                               dropout=dropout, split_ratio=split_ratio)
            model.set_feature_split(homo_raw, features, labels, train_mask)
    else:
        # build k-hop augmented graph for H2GFormer only
        if model_name == "H2GFormer":
            hetero_g_khop, ef_khop = build_khop_hetero_graph(hetero_g, ef, device)
        else:
            hetero_g_khop, ef_khop = hetero_g, ef

        model = HETERO_MODELS[model_name](
            in_feats=in_feats, edge_feats=edge_dim, hidden_size=hidden_size,
            out_classes=num_classes, dropout=dropout, n_layers=n_layers,
            num_heads=num_heads, post_feats=post_feats,
        )

    model = model.to(device)
    crit  = torch.nn.CrossEntropyLoss(weight=_class_weights(labels, train_mask, num_classes, device))
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=patience // 2, factor=0.5)

    def _train():
        if is_homo:
            if model_name == "H2GCN":
                return crit(model(homo_no, g_2hop, features)[train_mask], labels[train_mask])
            g = homo_raw
            return crit(model(g, features)[train_mask], labels[train_mask])
        g_use  = hetero_g_khop if model_name == "H2GFormer" else hetero_g
        ef_use = ef_khop        if model_name == "H2GFormer" else ef
        out = model(g_use, nf, ef_use)
        if isinstance(out, tuple):
            main, aux = out
            return crit(main[train_mask], labels[train_mask]) + 0.1 * crit(aux[train_mask], labels[train_mask])
        return crit(out[train_mask], labels[train_mask])

    def _eval():
        if is_homo: return _eval_homo(model, homo_no, homo_raw, features, labels, val_mask, g_2hop)
        g_use  = hetero_g_khop if model_name == "H2GFormer" else hetero_g
        ef_use = ef_khop        if model_name == "H2GFormer" else ef
        return _eval_hetero(model, g_use, nf, ef_use, labels, val_mask)

    bvm, bvmi = _training_loop(model, crit, opt, sched, num_epochs, patience, _train, _eval)

    if is_homo: tm, tmi = _eval_homo(model, homo_no, homo_raw, features, labels, test_mask, g_2hop)
    else:
        g_use  = hetero_g_khop if model_name == "H2GFormer" else hetero_g
        ef_use = ef_khop        if model_name == "H2GFormer" else ef
        tm, tmi = _eval_hetero(model, g_use, nf, ef_use, labels, test_mask)

    result = {"model": model_name, "dataset": dataset_name,
              "test_macro_f1": round(tm*100, 2), "test_micro_f1": round(tmi*100, 2),
              "val_macro_f1":  round(bvm*100, 2), "val_micro_f1":  round(bvmi*100, 2)}
    result["_trained_model"] = model   # internal — used by eval_homophily.py
    return result


# ── HP search ─────────────────────────────────────────────────────────────────

def hp_search(model_name, dataset_name, data_kwargs, n_trials=15, seed=42):
    grid   = {**BASE_GRID, **MODEL_EXTRA_GRID.get(model_name, {})}
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    rng = np.random.RandomState(seed)
    if len(combos) > n_trials:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), n_trials, replace=False))]

    print(f"\n  HP search: {model_name}/{dataset_name} — {len(combos)} trials")
    best_val, best_cfg = -1.0, None

    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        print(f"  Trial {i+1}/{len(combos)}: {cfg}")
        try:
            r = train_merged(model_name, dataset_name, num_epochs=50, patience=10,
                              seed=seed, **data_kwargs, **cfg)
            print(f"    val macro = {r['val_macro_f1']:.2f}%")
            if r["val_macro_f1"] > best_val:
                best_val, best_cfg = r["val_macro_f1"], cfg; print("    ✓ new best")
        except Exception as ex:
            print(f"    failed: {ex}")

    if best_cfg is None:
        best_cfg = {"hidden_size": 256, "n_layers": 2, "dropout": 0.5,
                    "lr": 0.005, "weight_decay": 5e-4}
    return best_cfg, best_val



def _hp_search_inductive(dataset_name, data_kwargs, n_trials=15, seed=42):
    """HP search for the inductive HinSAGE appendix experiment."""
    grid   = {**BASE_GRID}
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    rng = np.random.RandomState(seed)
    if len(combos) > n_trials:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), n_trials, replace=False))]

    print(f"\n  HP search (inductive): HinSAGE/{dataset_name} — {len(combos)} trials")
    best_val, best_cfg = -1.0, None

    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        print(f"  Trial {i+1}/{len(combos)}: {cfg}")
        try:
            r = train_separate("HinSAGE", dataset_name, num_epochs=50, patience=10,
                               seed=seed, **data_kwargs, **cfg)
            print(f"    val macro = {r['val_macro_f1']:.2f}%")
            if r["val_macro_f1"] > best_val:
                best_val, best_cfg = r["val_macro_f1"], cfg; print("    ✓ new best")
        except Exception as ex:
            print(f"    failed: {ex}")

    if best_cfg is None:
        best_cfg = {"hidden_size": 256, "n_layers": 2, "dropout": 0.5,
                    "lr": 0.005, "weight_decay": 5e-4}
    return best_cfg, best_val

# ── run one model × dataset ───────────────────────────────────────────────────

def run(model_name, dataset_name, data_dir=DATA_DIR, do_search=True,
        n_trials=15, seed=42, out_dir="tuning_results", inductive=False):
    """
    Main runner. All 9 models use merged transductive setup by default.
    inductive=True is only for the HinSAGE appendix experiment.
    """
    label = " [inductive]" if inductive else ""
    print(f"\n{'='*55}\n  {model_name} / {dataset_name}{label}\n{'='*55}")
    set_seed(seed)
    start = time.time()

    if inductive:
        if model_name != "HinSAGE":
            raise ValueError("--inductive is only supported for HinSAGE")
        raw = load_separate(dataset_name, data_dir)
        (train_g, val_g, test_g, tl, vl, tel, num_classes, _) = raw
        data_kw = dict(train_g=train_g, val_g=val_g, test_g=test_g,
                       train_labels=tl, val_labels=vl, test_labels=tel,
                       num_classes=num_classes)
        train_fn   = train_separate
        search_fn  = lambda: _hp_search_inductive(dataset_name, data_kw, n_trials, seed)
    else:
        raw = load_merged(dataset_name, data_dir)
        (hetero_g, edge_features, node_features,
         labels, train_mask, val_mask, test_mask, num_classes, _, _um, _sp) = raw
        data_kw = dict(hetero_g=hetero_g, edge_features=edge_features,
                       node_features=node_features, labels=labels,
                       train_mask=train_mask, val_mask=val_mask,
                       test_mask=test_mask, num_classes=num_classes)
        train_fn   = train_merged
        search_fn  = lambda: hp_search(model_name, dataset_name, data_kw, n_trials, seed)

    if do_search:
        best_cfg, best_val = search_fn()
        print(f"\n  Best HP: {best_cfg}  (val={best_val:.2f}%)")
    else:
        best_cfg = {"hidden_size": 256, "n_layers": 2, "dropout": 0.5,
                    "lr": 0.005, "weight_decay": 5e-4}
        best_val = 0.0

    print("\n  Final training with best config...")
    result = train_fn(model_name, dataset_name, num_epochs=200, patience=20,
                      seed=seed, **data_kw, **best_cfg)
    result.update({"best_config": best_cfg, "best_val_hp": best_val,
                   "elapsed_sec": round(time.time() - start, 1)})

    save_dir = os.path.join(out_dir, f"{model_name}_{dataset_name}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    # best_config.json for compatibility with train__2_.py
    with open(os.path.join(save_dir, "best_config.json"), "w") as f:
        json.dump({"model_type": model_name, "dataset": dataset_name,
                   "best_config": best_cfg, "best_val_f1": best_val}, f, indent=2)

    print(f"\n  Test  Macro F1: {result['test_macro_f1']:.2f}%  "
          f"Micro F1: {result['test_micro_f1']:.2f}%  ({result['elapsed_sec']}s)")
    return result


# ── summary output ────────────────────────────────────────────────────────────

def _save_summary(results, out_dir):
    rows = [{"Model": r["model"], "Dataset": r["dataset"],
             "Test Macro F1": r["test_macro_f1"], "Test Micro F1": r["test_micro_f1"],
             "Val Macro F1": r["val_macro_f1"],
             "Best Config": str(r.get("best_config", "")),
             "Time (s)": r.get("elapsed_sec", 0)}
            for r in results]
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*70}\n  FINAL RESULTS SUMMARY\n{'='*70}")
    try:
        print(df.pivot_table(index="Model", columns="Dataset",
                             values="Test Macro F1", aggfunc="first").to_string())
    except Exception:
        print(df[["Model", "Dataset", "Test Macro F1"]].to_string(index=False))
    print(f"\n  Full results: {csv_path}")

    try:
        models_u, datasets_u = df["Model"].unique(), df["Dataset"].unique()
        x = np.arange(len(datasets_u)); width = 0.8 / max(len(models_u), 1)
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, m in enumerate(models_u):
            vals = [df[(df["Model"]==m)&(df["Dataset"]==d)]["Test Macro F1"].values[0]
                    if len(df[(df["Model"]==m)&(df["Dataset"]==d)]) > 0 else 0.0
                    for d in datasets_u]
            ax.bar(x + i*width, vals, width, label=m)
        ax.set_xticks(x + width*(len(models_u)-1)/2); ax.set_xticklabels(datasets_u)
        ax.set_ylabel("Test Macro F1 (%)"); ax.set_title("ControBench GNN Results")
        ax.legend(); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
        plot_path = os.path.join(out_dir, "results_plot.png")
        plt.savefig(plot_path, dpi=300); plt.close()
        print(f"  Plot: {plot_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="RGCN")
    p.add_argument("--dataset",      default="trump")
    p.add_argument("--all_models",   action="store_true")
    p.add_argument("--all_datasets", action="store_true")
    p.add_argument("--models",       default=None, help="Comma-separated, e.g. RGCN,HAN")
    p.add_argument("--datasets",     default=None, help="Comma-separated, e.g. trump,abortion")
    p.add_argument("--n_trials",     type=int, default=15)
    p.add_argument("--no_search",    action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--data_dir",     default=DATA_DIR)
    p.add_argument("--out_dir",      default="tuning_results")
    p.add_argument("--inductive",    action="store_true",
                   help="HinSAGE optional: inductive separate-graph evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    models = (ALL_MODELS if args.all_models
              else [m.strip() for m in args.models.split(",")] if args.models
              else [args.model])
    datasets = (DATASETS if args.all_datasets
                else [d.strip() for d in args.datasets.split(",")] if args.datasets
                else [args.dataset])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Models   : {models}\nDatasets : {datasets}")
    print(f"HP search: {'ON ('+str(args.n_trials)+' trials)' if not args.no_search else 'OFF'}")
    print(f"Data dir : {args.data_dir}")

    good_results, all_rows = [], []
    for ds in datasets:
        for m in models:
            try:
                r = run(m, ds, data_dir=args.data_dir, do_search=not args.no_search,
                        n_trials=args.n_trials, seed=args.seed, out_dir=run_dir,
                        inductive=args.inductive)
                good_results.append(r); all_rows.append(r)
            except Exception as e:
                print(f"  ERROR {m}/{ds}: {e}")
                all_rows.append({"model": m, "dataset": ds, "error": str(e)})

    _save_summary(good_results, run_dir)

    print(f"\n{'='*65}\n  FINAL SUMMARY\n{'='*65}")
    print(f"{'Model/Dataset':<28} {'Test Macro':>12}  {'Test Micro':>12}")
    print("-"*55)
    for r in all_rows:
        key = f"{r['model']}/{r['dataset']}"
        if "error" in r: print(f"{key:<28}  ERROR: {r['error']}")
        else: print(f"{key:<28}  {r['test_macro_f1']:>10.2f}%  {r['test_micro_f1']:>10.2f}%")
    print("="*65)

    print(f"\nUsage examples:")
    print(f"  python tune.py --model RGCN --dataset trump")
    print(f"  python tune.py --all_models --all_datasets")
    print(f"  python tune.py --models RGCN,HAN --datasets trump,abortion --n_trials 10")
    print(f"  python tune.py --all_models --dataset trump --no_search")
    print(f"  python tune.py --model HinSAGE --dataset trump --inductive  # appendix")