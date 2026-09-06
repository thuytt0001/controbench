"""
Shared graph construction utilities for ControBench models.

UCU edge semantics (consistent across all models):
  Edge A → B means A wrote a comment that B replied to.
  feat       (edge_features["user_comment_user"]) = A's comment embedding
  reply_feat (g.edges["user_comment_user"].data["reply_feat"]) = B's reply embedding
  → B (dst) receives both: context of what was said TO them + what THEY said back.
"""

import torch
import dgl


def build_metapath_graphs(hetero_g, h_user, device):
    """
    Build three user-user homogeneous subgraphs for meta-path convolutions.

    MP1 (UPU):  users who published the same post
    MP2 (UCpU): users who commented on the same post  (capped at 50k edges)
    MP3 (UCU):  direct user-comment-user edges
    """
    N = h_user.size(0)
    meta_edges = []

    def _empty():
        return (torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    # MP1
    if hetero_g.num_edges("publish") > 0:
        pu_src, pu_dst = hetero_g.edges(etype="publish")
        post_to_users = {}
        for u, p in zip(pu_src.tolist(), pu_dst.tolist()):
            post_to_users.setdefault(p, []).append(u)
        pairs = set()
        for users in post_to_users.values():
            for u1 in users:
                for u2 in users:
                    if u1 != u2:
                        pairs.add((u1, u2))
        srcs = [p[0] for p in pairs]
        dsts = [p[1] for p in pairs]
        meta_edges.append((
            torch.tensor(srcs, dtype=torch.long, device=device),
            torch.tensor(dsts, dtype=torch.long, device=device),
        ) if srcs else _empty())
    else:
        meta_edges.append(_empty())

    # MP2
    if hetero_g.num_edges("comment") > 0:
        cu_src, cu_dst = hetero_g.edges(etype="comment")
        post_to_commenters = {}
        for u, p in zip(cu_src.tolist(), cu_dst.tolist()):
            post_to_commenters.setdefault(p, []).append(u)
        pairs = set()
        for users in post_to_commenters.values():
            for u1 in users:
                for u2 in users:
                    if u1 != u2:
                        pairs.add((u1, u2))
        srcs = [p[0] for p in pairs]
        dsts = [p[1] for p in pairs]
        cap = min(len(srcs), 50000)
        meta_edges.append((
            torch.tensor(srcs[:cap], dtype=torch.long, device=device),
            torch.tensor(dsts[:cap], dtype=torch.long, device=device),
        ) if srcs else _empty())
    else:
        meta_edges.append(_empty())

    # MP3
    if hetero_g.num_edges("user_comment_user") > 0:
        s, d = hetero_g.edges(etype="user_comment_user")
        meta_edges.append((s.to(device), d.to(device)))
    else:
        meta_edges.append(_empty())

    return meta_edges


def make_homo_graph(src, dst, N, device, self_loops=True):
    """Build an undirected DGL homogeneous graph from src/dst edge lists."""
    if len(src) == 0:
        g = dgl.graph(([], []), num_nodes=N)
    else:
        all_src = torch.cat([src, dst])
        all_dst = torch.cat([dst, src])
        g = dgl.graph((all_src, all_dst), num_nodes=N)
        g = dgl.remove_self_loop(g)
    if self_loops:
        g = dgl.add_self_loop(g)
    return g.to(device)


def build_homogeneous_user_graph(hetero_g, edge_features, node_features):
    """
    Project ControBench heterogeneous graph → user-user homogeneous graph.
    Used by H2GCN, ACMGNN, GCNPlusGFS.

    Feature aggregation per user:
      - comment edge embeddings (own comment text)
      - post embeddings for posts commented on (stance context)
      - UCU feat (A's comment) aggregated to user A
      - UCU reply_feat (B's reply) aggregated to user B
      - post embeddings for published posts

    Returns:
      homo_g_no_sl : no self-loops  (for H2GCN)
      homo_g_sl    : with self-loops (for ACMGNN, GCNPlusGFS)
      features     : (N_users, 768) aggregated feature tensor
    """
    num_users = hetero_g.num_nodes("user")
    feat_dim  = 768
    user_feat_lists = [[] for _ in range(num_users)]

    def _post_768(pf):
        # post nodes have 1536-dim (title+content cat); reduce to 768 by averaging halves
        if pf.shape[1] == feat_dim * 2:
            return (pf[:, :feat_dim] + pf[:, feat_dim:]) / 2.0
        return pf[:, :feat_dim]

    post_feats_raw = node_features.get("post")

    # comment edge features → user A
    if "comment" in edge_features and edge_features["comment"] is not None:
        src, _ = hetero_g.edges(etype="comment")
        for i, u in enumerate(src.tolist()):
            user_feat_lists[u].append(edge_features["comment"][i])

    # post content for commenters
    if post_feats_raw is not None and "comment" in hetero_g.etypes:
        src, dst = hetero_g.edges(etype="comment")
        pf = _post_768(post_feats_raw)
        for u, p in zip(src.tolist(), dst.tolist()):
            user_feat_lists[u].append(pf[p])

    # UCU feat → user A (src)
    if "user_comment_user" in edge_features and edge_features["user_comment_user"] is not None:
        src, _ = hetero_g.edges(etype="user_comment_user")
        feats_a = edge_features["user_comment_user"]
        for i, s in enumerate(src.tolist()):
            user_feat_lists[s].append(feats_a[i])

    # UCU reply_feat → user B (dst)
    if "reply_feat" in hetero_g.edges["user_comment_user"].data:
        _, dst = hetero_g.edges(etype="user_comment_user")
        feats_b = hetero_g.edges["user_comment_user"].data["reply_feat"]
        for i, d in enumerate(dst.tolist()):
            user_feat_lists[d].append(feats_b[i])

    # published post features → publisher
    if "publish" in hetero_g.etypes and post_feats_raw is not None:
        src, dst = hetero_g.edges(etype="publish")
        pf = _post_768(post_feats_raw)
        for u, p in zip(src.tolist(), dst.tolist()):
            user_feat_lists[u].append(pf[p])

    features = torch.zeros(num_users, feat_dim)
    for u, flist in enumerate(user_feat_lists):
        if flist:
            features[u] = torch.stack(flist).mean(0)

    # UCU edges as undirected user-user graph
    if "user_comment_user" in hetero_g.etypes and hetero_g.num_edges("user_comment_user") > 0:
        src, dst = hetero_g.edges(etype="user_comment_user")
        all_src = torch.cat([src.long(), dst.long()])
        all_dst = torch.cat([dst.long(), src.long()])
    else:
        all_src = torch.zeros(0, dtype=torch.long)
        all_dst = torch.zeros(0, dtype=torch.long)

    # homo_g_raw: full multi-edge graph preserving natural UCU self-loops
    #   → used by ACMGNN and GCNPlusGFS directly
    # homo_g_no_sl: self-loops removed for H2GCN (ego/neighbour separation)
    homo_g_raw   = dgl.graph((all_src, all_dst), num_nodes=num_users)
    homo_g_no_sl = dgl.remove_self_loop(homo_g_raw)

    return homo_g_no_sl, homo_g_raw, features


def compute_adj_norm(g):
    """Symmetric normalised adjacency D^{-1/2} A D^{-1/2}."""
    src, dst = g.edges()
    deg = g.in_degrees().float().clamp(min=1)
    w   = deg.pow(-0.5)[src] * deg.pow(-0.5)[dst]
    return src, dst, w


import numpy as np
import scipy.sparse as sp


def build_2hop_graph(g_no_sl):
    """
    Build the true 2-hop graph from a no-self-loop homogeneous DGL graph.
    A_2 = A^2 with diagonal and 1-hop edges removed, then renormalised.
    Matches the original H2GCN paper's precomputed hop adjacencies.
    """
    import scipy.sparse as sp

    n = g_no_sl.num_nodes()
    src, dst = g_no_sl.edges()
    src, dst = src.numpy(), dst.numpy()

    # build sparse adjacency (binary, undirected already from build_homogeneous)
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))

    # A^2 gives reachability in exactly 0, 1, or 2 hops
    A2 = A @ A

    # remove diagonal (self-loops) and 1-hop edges
    A2 = A2 - sp.diags(A2.diagonal())  # remove diagonal
    A2 = A2 - A2.multiply(A > 0)       # remove 1-hop entries (sparse-efficient)
    A2.eliminate_zeros()

    # symmetric normalisation D^{-1/2} A2 D^{-1/2}
    deg = np.array(A2.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = deg[mask] ** -0.5  # avoid divide-by-zero warning
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A2_norm = D_inv_sqrt @ A2 @ D_inv_sqrt

    A2_coo = A2_norm.tocoo()
    g_2hop = dgl.graph(
        (torch.tensor(A2_coo.row, dtype=torch.long),
         torch.tensor(A2_coo.col, dtype=torch.long)),
        num_nodes=n
    )
    return g_2hop


def build_khop_hetero_graph(hetero_g, ef, device, cap=100_000):
    """
    Augment hetero graph with 2-hop user-user edges via posts.
    Paths: user→{publish,comment}→post←{publish,comment}←user

    For each new 2-hop edge u1→u2 via post p:
      feat       = u1's comment embedding on p (if u1 commented) else post p embedding
      reply_feat = u2's comment embedding on p (if u2 commented) else post p embedding

    This uses ControBench's rich text semantics rather than zeros.
    Capped at `cap` new edges to avoid combinatorial explosion on dense posts.
    """
    feat_dim = 768

    # post embeddings: reduce 1536-dim (title+content) to 768 by averaging halves
    post_feats_raw = hetero_g.nodes["post"].data.get("feat")
    if post_feats_raw is not None and post_feats_raw.shape[1] == feat_dim * 2:
        post_emb = (post_feats_raw[:, :feat_dim] + post_feats_raw[:, feat_dim:]) / 2.0
    elif post_feats_raw is not None:
        post_emb = post_feats_raw[:, :feat_dim]
    else:
        post_emb = torch.zeros(hetero_g.num_nodes("post"), feat_dim)
    post_emb = post_emb.to(device)

    # (user, post) → comment embedding lookup from user_comment_post edges
    user_post_comment_emb = {}
    if hetero_g.num_edges("comment") > 0 and "feat" in hetero_g.edges["comment"].data:
        com_src, com_dst = hetero_g.edges(etype="comment")
        com_feat = hetero_g.edges["comment"].data["feat"]
        for i, (u, p) in enumerate(zip(com_src.tolist(), com_dst.tolist())):
            # keep last comment if multiple — all are valid stance signals
            user_post_comment_emb[(u, p)] = com_feat[i].to(device)

    # collect post→users mapping from both publish and comment
    post_to_users = {}
    for etype in ("publish", "comment"):
        if hetero_g.num_edges(etype) > 0:
            src, dst = hetero_g.edges(etype=etype)
            for u, p in zip(src.tolist(), dst.tolist()):
                post_to_users.setdefault(p, set()).add(u)

    # existing UCU pairs for dedup
    existing_ucu = set()
    if hetero_g.num_edges("user_comment_user") > 0:
        ucu_s, ucu_d = hetero_g.edges(etype="user_comment_user")
        existing_ucu = set(zip(ucu_s.tolist(), ucu_d.tolist()))

    # generate 2-hop pairs, tracking the bridging post for embedding lookup
    new_src, new_dst, new_via_post = [], [], []
    seen_new = set()
    for p, users in post_to_users.items():
        users = list(users)
        for u1 in users:
            for u2 in users:
                if u1 == u2:
                    continue
                if (u1, u2) in existing_ucu or (u1, u2) in seen_new:
                    continue
                seen_new.add((u1, u2))
                new_src.append(u1)
                new_dst.append(u2)
                new_via_post.append(p)
                if len(new_src) >= cap:
                    break
            if len(new_src) >= cap:
                break
        if len(new_src) >= cap:
            break

    if not new_src:
        return hetero_g, ef

    N_u = hetero_g.num_nodes("user")
    N_p = hetero_g.num_nodes("post")
    n_new = len(new_src)

    # build rich embeddings for new edges
    new_feats, new_reply_feats = [], []
    for u1, u2, p in zip(new_src, new_dst, new_via_post):
        # feat: u1's comment on p, else post p embedding
        new_feats.append(user_post_comment_emb.get((u1, p), post_emb[p]))
        # reply_feat: u2's comment on p, else post p embedding
        new_reply_feats.append(user_post_comment_emb.get((u2, p), post_emb[p]))

    new_feat_t       = torch.stack(new_feats)        # (n_new, 768)
    new_reply_feat_t = torch.stack(new_reply_feats)  # (n_new, 768)

    # build augmented graph
    new_src_t = torch.tensor(new_src, dtype=torch.long, device=device)
    new_dst_t = torch.tensor(new_dst, dtype=torch.long, device=device)

    graph_data = {}
    if hetero_g.num_edges("publish") > 0:
        ps, pd = hetero_g.edges(etype="publish")
        graph_data[("user", "publish", "post")] = (ps.to(device), pd.to(device))
    if hetero_g.num_edges("comment") > 0:
        cs, cd = hetero_g.edges(etype="comment")
        graph_data[("user", "comment", "post")] = (cs.to(device), cd.to(device))

    if hetero_g.num_edges("user_comment_user") > 0:
        os_, od = hetero_g.edges(etype="user_comment_user")
        all_s = torch.cat([os_.to(device), new_src_t])
        all_d = torch.cat([od.to(device),  new_dst_t])
    else:
        all_s, all_d = new_src_t, new_dst_t
    graph_data[("user", "user_comment_user", "user")] = (all_s, all_d)

    g_khop = dgl.heterograph(graph_data, num_nodes_dict={"user": N_u, "post": N_p})

    # copy comment edge features
    if hetero_g.num_edges("comment") > 0 and "feat" in hetero_g.edges["comment"].data:
        g_khop.edges["comment"].data["feat"] =             hetero_g.edges["comment"].data["feat"].to(device)

    # copy original UCU features + append rich embeddings for new edges
    if hetero_g.num_edges("user_comment_user") > 0:
        orig_feat  = hetero_g.edges["user_comment_user"].data.get(
            "feat",       torch.zeros(hetero_g.num_edges("user_comment_user"), feat_dim))
        orig_reply = hetero_g.edges["user_comment_user"].data.get(
            "reply_feat", torch.zeros(hetero_g.num_edges("user_comment_user"), feat_dim))
        g_khop.edges["user_comment_user"].data["feat"]       = torch.cat([orig_feat.to(device),       new_feat_t])
        g_khop.edges["user_comment_user"].data["reply_feat"] = torch.cat([orig_reply.to(device), new_reply_feat_t])
    else:
        g_khop.edges["user_comment_user"].data["feat"]       = new_feat_t
        g_khop.edges["user_comment_user"].data["reply_feat"] = new_reply_feat_t

    # update ef dict with rich embeddings for new edges
    ef_khop = dict(ef)
    if "user_comment_user" in ef:
        ef_khop["user_comment_user"] = torch.cat([
            ef["user_comment_user"].to(device), new_feat_t
        ])

    return g_khop, ef_khop