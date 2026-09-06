# split_dataset.py
import json
import os
import sys
import numpy as np
from collections import Counter, defaultdict


def split_and_save_dataset(
    dataset_name,
    data_dir="embedded_data",
    output_dir="split_datasets_enriched_2",
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42,
    min_class_percentage=0.05,
):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio+val_ratio+test_ratio}")

    np.random.seed(seed)

    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    json_file = os.path.join(data_dir, f"{dataset_name}_graph_data_with_embeddings.json")
    print(f"Loading dataset from {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    user_nodes = [n for n in nodes if n["type"] == "user"]
    post_nodes = [n for n in nodes if n["type"] == "post"]

    print(f"Total nodes: {len(nodes)} ({len(user_nodes)} users, {len(post_nodes)} posts)")
    print(f"Total edges: {len(edges)}")

    user_node_by_id = {n["id"]: n for n in user_nodes}
    post_node_by_id = {n["id"]: n for n in post_nodes}

    # ── filter rare classes ────────────────────────────────────────────────────
    label_counts  = Counter(n["label"] for n in user_nodes)
    total_users   = len(user_nodes)
    valid_labels  = []
    ignored_labels = []
    for label, count in label_counts.items():
        pct = count / total_users
        if pct >= min_class_percentage:
            valid_labels.append(label)
        else:
            ignored_labels.append((label, count, pct * 100))

    if ignored_labels:
        print(f"\nIgnoring {len(ignored_labels)} rare classes "
              f"(below {min_class_percentage*100}% threshold):")
        for label, count, pct in ignored_labels:
            print(f"  - '{label}': {count} instances ({pct:.2f}%)")

    valid_user_nodes = [n for n in user_nodes if n["label"] in valid_labels]
    valid_user_ids   = set(n["id"] for n in valid_user_nodes)
    print(f"\nRetained {len(valid_user_nodes)}/{len(user_nodes)} users")

    # ── stratified 60/20/20 split ──────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    train_users, val_test_users = train_test_split(
        valid_user_nodes,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=[n["label"] for n in valid_user_nodes],
    )
    val_users, test_users = train_test_split(
        val_test_users,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=[n["label"] for n in val_test_users],
    )

    train_user_ids = set(n["id"] for n in train_users)
    val_user_ids   = set(n["id"] for n in val_users)
    test_user_ids  = set(n["id"] for n in test_users)

    print(f"Training users   : {len(train_users)} ({len(train_users)/len(valid_user_nodes)*100:.1f}%)")
    print(f"Validation users : {len(val_users)} ({len(val_users)/len(valid_user_nodes)*100:.1f}%)")
    print(f"Test users       : {len(test_users)} ({len(test_users)/len(valid_user_nodes)*100:.1f}%)")

    train_label_counts = Counter(n["label"] for n in train_users)
    val_label_counts   = Counter(n["label"] for n in val_users)
    test_label_counts  = Counter(n["label"] for n in test_users)

    print(f"\nLabel distribution:")
    print(f"{'Label':<25} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 50)
    for label in sorted(valid_labels):
        print(f"{label:<25} {train_label_counts.get(label,0):>6} "
              f"{val_label_counts.get(label,0):>6} "
              f"{test_label_counts.get(label,0):>6}")

    # ── index raw edges ────────────────────────────────────────────────────────
    # We track edges by their index in the raw list to allow multi-edges
    # (same src/dst/type with different content) while preventing the exact
    # same raw edge object from being added twice to the same split.
    edge_by_idx     = {i: e for i, e in enumerate(edges)}
    edge_idx_by_id  = {id(e): i for i, e in enumerate(edges)}

    # All comment/publish edges per user, for chain enrichment
    user_post_edges: dict = defaultdict(list)   # uid → list of (idx, edge)
    user_to_posts:   dict = defaultdict(set)    # uid → set of post_ids
    for i, edge in enumerate(edges):
        if edge["type"] in ("user_comment_post", "user_publish_post"):
            user_post_edges[edge["source"]].append((i, edge))
            user_to_posts[edge["source"]].add(edge["target"])

    def user_split(uid):
        if uid in train_user_ids: return "train"
        if uid in val_user_ids:   return "val"
        if uid in test_user_ids:  return "test"
        return None

    # ── main edge assignment ───────────────────────────────────────────────────
    # split_edges: list of edge dicts per split
    # added_raw_idx: set of raw edge indices already added per split
    #   → prevents adding the same raw interaction twice to the same split
    #   → but two different raw edges with same src/dst/type ARE both kept (multi-edge)
    split_edges   = {"train": [], "val": [], "test": []}
    added_raw_idx = {"train": set(), "val": set(), "test": set()}
    context_post_ids  = {"train": set(), "val": set(), "test": set()}
    context_user_ids  = {"train": set(), "val": set(), "test": set()}

    def _add_raw_edge(split, idx, edge):
        """Add a raw edge to a split if not already present (by raw index)."""
        if idx not in added_raw_idx[split]:
            split_edges[split].append(edge)
            added_raw_idx[split].add(idx)
            return True
        return False

    for i, edge in enumerate(edges):
        src, dst, etype = edge["source"], edge["target"], edge["type"]

        if etype == "user_publish_post":
            split = user_split(src)
            if split is None:
                continue
            _add_raw_edge(split, i, edge)
            context_post_ids[split].add(dst)

        elif etype == "user_comment_post":
            split = user_split(src)
            if split is None:
                continue
            _add_raw_edge(split, i, edge)
            context_post_ids[split].add(dst)

        elif etype == "user_comment_user":
            # A→B: A is original commenter, B is replier (who we predict)
            src_valid = src in valid_user_ids
            dst_valid = dst in valid_user_ids

            if not dst_valid:
                continue

            split = user_split(dst)
            if split is None:
                continue

            if src_valid:
                # Both valid → keep as UCU
                _add_raw_edge(split, i, edge)

                # Cross-split: add A as context node in B's split
                src_split = user_split(src)
                if src_split != split:
                    context_user_ids[split].add(src)

                # Pull in ALL of A's post edges into B's split (multi-edges allowed)
                for idx_a, edge_a in user_post_edges.get(src, []):
                    context_post_ids[split].add(edge_a["target"])
                    _add_raw_edge(split, idx_a, edge_a)

            else:
                # src invalid → demote: create a synthetic user_comment_post for dst
                # embedding = mean(A's comment, B's reply) to preserve conversation signal
                src_posts = list(user_to_posts.get(src, set()))
                dst_posts = list(user_to_posts.get(dst, set()))
                target_posts = src_posts if src_posts else dst_posts
                if not target_posts:
                    continue

                emb_a = edge.get("embedding", [])
                emb_b = edge.get("reply_embedding", [])

                if emb_a and emb_b and len(emb_a) == len(emb_b):
                    combined = [(a + b) / 2.0 for a, b in zip(emb_a, emb_b)]
                elif emb_b:
                    combined = emb_b
                else:
                    combined = emb_a

                # Each demotion is a new synthetic edge — always add (multi-edges)
                demoted = {
                    "source":    dst,
                    "target":    target_posts[0],
                    "type":      "user_comment_post",
                    "embedding": combined,
                }
                split_edges[split].append(demoted)
                context_post_ids[split].add(target_posts[0])

    # ── lazy B→P: pull in post edges for users whose posts are now in the split ─
    # After the main loop, some posts may have been added to context_post_ids
    # via chain enrichment. Any valid user in the split who has comment edges
    # to those posts should have those edges included too.
    for split in ("train", "val", "test"):
        primary_ids = (train_user_ids if split == "train"
                       else val_user_ids if split == "val"
                       else test_user_ids)
        all_split_user_ids = primary_ids | context_user_ids[split]

        for uid in all_split_user_ids:
            for idx, edge in user_post_edges.get(uid, []):
                if edge["target"] in context_post_ids[split]:
                    _add_raw_edge(split, idx, edge)

    # ── build final node lists ─────────────────────────────────────────────────
    final_user_nodes   = {}
    context_user_counts = {}

    for split in ("train", "val", "test"):
        primary = (train_users if split == "train"
                   else val_users if split == "val"
                   else test_users)
        primary_ids = (train_user_ids if split == "train"
                       else val_user_ids if split == "val"
                       else test_user_ids)

        ctx_users = [
            user_node_by_id[uid]
            for uid in context_user_ids[split]
            if uid not in primary_ids and uid in user_node_by_id
        ]
        final_user_nodes[split]    = list(primary) + ctx_users
        context_user_counts[split] = len(ctx_users)

    final_post_nodes = {}
    for split in ("train", "val", "test"):
        final_post_nodes[split] = [
            post_node_by_id[pid]
            for pid in context_post_ids[split]
            if pid in post_node_by_id
        ]

    # ── print split summary ────────────────────────────────────────────────────
    for split in ("train", "val", "test"):
        n_ucu = sum(1 for e in split_edges[split] if e["type"] == "user_comment_user")
        n_ucp = sum(1 for e in split_edges[split] if e["type"] == "user_comment_post")
        n_pub = sum(1 for e in split_edges[split] if e["type"] == "user_publish_post")
        print(f"\n{split.upper()} split:"
              f"\n  Users : {len(final_user_nodes[split])} "
              f"({context_user_counts[split]} cross-split context users)"
              f"\n  Posts : {len(final_post_nodes[split])}"
              f"\n  Edges : {len(split_edges[split])} "
              f"(publish={n_pub}, comment={n_ucp}, ucu={n_ucu})")

    print("\nCross-split UCU edge info:")
    for split in ("train", "val", "test"):
        split_user_node_ids = set(n["id"] for n in final_user_nodes[split])
        cross = sum(1 for e in split_edges[split]
                    if e["type"] == "user_comment_user"
                    and e["source"] not in split_user_node_ids)
        total = sum(1 for e in split_edges[split] if e["type"] == "user_comment_user")
        print(f"  {split}: {cross}/{total} UCU edges have User A in a different split")

    # ── save split files ───────────────────────────────────────────────────────
    file_names  = {"train": "train.json", "val": "validation.json", "test": "test.json"}
    split_label = {"train": "train", "val": "validation", "test": "test"}

    for split in ("train", "val", "test"):
        split_data = {
            "dataset":              dataset_name,
            "split":                split_label[split],
            "train_ratio":          train_ratio,
            "val_ratio":            val_ratio,
            "test_ratio":           test_ratio,
            "seed":                 seed,
            "min_class_percentage": min_class_percentage,
            "nodes": final_user_nodes[split] + final_post_nodes[split],
            "edges": split_edges[split],
        }
        out_path = os.path.join(dataset_output_dir, file_names[split])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split} → {out_path}")

    # ── metadata ───────────────────────────────────────────────────────────────
    metadata = {
        "dataset":               dataset_name,
        "train_ratio":           train_ratio,
        "val_ratio":             val_ratio,
        "test_ratio":            test_ratio,
        "seed":                  seed,
        "min_class_percentage":  min_class_percentage,
        "total_users":           len(user_nodes),
        "valid_users":           len(valid_user_nodes),
        "train_users":           len(train_users),
        "val_users":             len(val_users),
        "test_users":            len(test_users),
        "context_train_users":   context_user_counts["train"],
        "context_val_users":     context_user_counts["val"],
        "context_test_users":    context_user_counts["test"],
        "valid_labels":          valid_labels,
        "ignored_labels":        [(l, c) for l, c, _ in ignored_labels],
        "label_distribution": {
            "train":      dict(train_label_counts),
            "validation": dict(val_label_counts),
            "test":       dict(test_label_counts),
        },
    }
    meta_path = os.path.join(dataset_output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata → {meta_path}")
    print(f"\nSplit complete for {dataset_name}!")

    return {
        "train_file":    os.path.join(dataset_output_dir, "train.json"),
        "val_file":      os.path.join(dataset_output_dir, "validation.json"),
        "test_file":     os.path.join(dataset_output_dir, "test.json"),
        "metadata_file": meta_path,
        "train_users":   len(train_users),
        "val_users":     len(val_users),
        "test_users":    len(test_users),
    }


# ── entry point ────────────────────────────────────────────────────────────────

# Per-dataset thresholds: trump=5%, abortion=1%, religion=1%
DATASET_THRESHOLDS = {
    "trump":    0.05,
    "abortion": 0.01,
    "religion": 0.01,
}

if __name__ == "__main__":
    dataset              = "religion"
    data_dir             = "embedded_data"
    output_dir           = "split_datasets_enriched_2"
    train_ratio          = 0.6
    val_ratio            = 0.2
    test_ratio           = 0.2
    seed                 = 42

    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == "--dataset" and i + 1 < len(sys.argv):
                dataset = sys.argv[i + 1]
            elif sys.argv[i] == "--data_dir" and i + 1 < len(sys.argv):
                data_dir = sys.argv[i + 1]
            elif sys.argv[i] == "--output_dir" and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
            elif sys.argv[i] == "--seed" and i + 1 < len(sys.argv):
                seed = int(sys.argv[i + 1])
            elif sys.argv[i] == "--all":
                # Run all three datasets
                for ds, thresh in DATASET_THRESHOLDS.items():
                    print(f"\n{'='*60}\nProcessing {ds} (threshold={thresh*100}%)\n{'='*60}")
                    split_and_save_dataset(ds, data_dir, output_dir,
                                           train_ratio, val_ratio, test_ratio,
                                           seed, thresh)
                sys.exit(0)

    min_class_percentage = DATASET_THRESHOLDS.get(dataset, 0.01)

    print(f"Dataset              : {dataset}")
    print(f"Min class percentage : {min_class_percentage*100}%")
    print(f"Split ratios         : {train_ratio}/{val_ratio}/{test_ratio}")
    print(f"Output dir           : {output_dir}")
    print(f"Multi-edges          : ENABLED")

    try:
        from sklearn.model_selection import train_test_split
        split_and_save_dataset(
            dataset, data_dir, output_dir,
            train_ratio, val_ratio, test_ratio,
            seed, min_class_percentage,
        )
    except ImportError:
        print("scikit-learn required: pip install scikit-learn")