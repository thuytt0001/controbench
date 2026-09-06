"""
Hetero²Net — Heterophily-aware Representation Learning on Heterogeneous Graphs
Li et al., TPAMI 2025. arXiv:2310.11664
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from .shared import build_metapath_graphs, make_homo_graph
import dgl.function as fn


class MetapathConv(nn.Module):
    """
    Node-level attention conv over a single meta-path subgraph.
    Uses scatter-softmax over incoming edges per destination node
    (proper graph attention, not sigmoid+degree-norm).
    """
    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        self.attn    = nn.Linear(hidden_size * 2, 1)
        self.dropout = dropout

    def forward(self, g, h):
        if g.num_edges() == 0:
            return h
        with g.local_scope():
            g.ndata["h"] = h
            src, dst = g.edges()
            N = h.size(0)
            device = h.device

            # raw attention score per edge
            a = self.attn(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)  # (E,)

            # scatter-softmax over incoming edges per destination node
            a_max = torch.full((N,), float("-inf"), device=device)
            a_max.scatter_reduce_(0, dst, a, reduce="amax", include_self=True)
            a_exp = (a - a_max[dst]).exp()
            a_sum = torch.zeros(N, device=device)
            a_sum.index_add_(0, dst, a_exp)
            a_norm = F.dropout(
                a_exp / (a_sum[dst] + 1e-9),
                p=self.dropout, training=self.training)  # (E,)

            # weighted sum
            msg = a_norm.unsqueeze(1) * h[src]           # (E, H)
            agg = torch.zeros(N, h.size(1), device=device)
            agg.index_add_(0, dst, msg)
            return agg


class SemanticFusion(nn.Module):
    """Semantic-level attention over multiple meta-path embeddings."""
    def __init__(self, hidden_size, n_paths):
        super().__init__()
        assert n_paths > 0
        self.query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.query, std=0.02)
        self.proj  = nn.Linear(hidden_size, hidden_size)

    def forward(self, embeddings):
        Z = torch.stack(embeddings, dim=1)                     # (N, M, H)
        w = (torch.tanh(self.proj(Z)) * self.query).sum(-1, keepdim=True)
        w = torch.softmax(w, dim=1)
        return (w * Z).sum(dim=1)


class HeteroConvLayer(nn.Module):
    """
    Per-layer heterogeneous message passing over the full hetero graph.
    Updates both h_user and h_post each layer so post nodes stay active.

    Faithful to DisenConv (Li et al., TPAMI 2025): each relation has separate
    lin_homo and lin_hetero projections applied to neighbour features, both
    aggregated independently and summed — not a residual approximation.
    """
    def __init__(self, hidden_size, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # DisenConv-style: separate homo/hetero projections per relation
        for rel in ("publish", "comment", "ucu"):
            setattr(self, f"lin_homo_{rel}", nn.Linear(hidden_size, hidden_size, bias=False))
            setattr(self, f"lin_hetero_{rel}", nn.Linear(hidden_size, hidden_size, bias=False))

        # reverse post→user direction (single projection, no disentanglement needed)
        self.W_post = nn.Linear(hidden_size, hidden_size, bias=False)

        # edge text projections (added to homo message — contextual signal)
        self.E_comment     = nn.Linear(hidden_size, hidden_size, bias=False)
        self.E_ucu_comment = nn.Linear(hidden_size, hidden_size, bias=False)
        self.E_ucu_reply   = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm_user   = nn.LayerNorm(hidden_size)
        self.norm_post   = nn.LayerNorm(hidden_size)
        # root (self) term — matching DisenConv's lin_r
        self.W_self_user = nn.Linear(hidden_size, hidden_size)
        self.W_self_post = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def _mean_agg(msg, dst_idx, N_dst, device):
        agg = torch.zeros(N_dst, msg.size(1), device=device)
        cnt = torch.zeros(N_dst, 1, device=device)
        dst_idx = dst_idx.to(device)
        agg.index_add_(0, dst_idx, msg)
        cnt.index_add_(0, dst_idx, torch.ones(len(dst_idx), 1, device=device))
        mask = cnt.squeeze(1) > 0
        agg[mask] = agg[mask] / cnt[mask]
        return agg

    def forward(self, g, h_user, h_post, ef):
        N_u    = h_user.size(0)
        N_p    = h_post.size(0)
        device = h_user.device

        # DisenConv: accumulate homo and hetero signals separately
        agg_homo_user   = torch.zeros_like(h_user)
        agg_hetero_user = torch.zeros_like(h_user)
        agg_post        = torch.zeros_like(h_post)

        if g.num_edges("publish") > 0:
            src, dst = g.edges(etype="publish")
            src, dst = src.to(device), dst.to(device)
            h_src = h_user[src]
            # forward: user→post (dst are post indices)
            agg_post        += self._mean_agg(self.lin_homo_publish(h_src),    dst, N_p, device)
            agg_post        += self._mean_agg(self.lin_hetero_publish(h_src),  dst, N_p, device)
            # reverse: post content flows back to publishing user
            agg_homo_user   += self._mean_agg(self.W_post(h_post[dst]),        src, N_u, device)

        if g.num_edges("comment") > 0:
            src, dst = g.edges(etype="comment")
            src, dst = src.to(device), dst.to(device)
            h_src = h_user[src]
            # edge text added to homo message (contextual/positive signal)
            homo_msg = self.lin_homo_comment(h_src)
            if "comment" in ef:
                homo_msg = homo_msg + self.E_comment(ef["comment"])
            # forward: user→post (dst are post indices)
            agg_post        += self._mean_agg(homo_msg,                            dst, N_p, device)
            agg_post        += self._mean_agg(self.lin_hetero_comment(h_src),      dst, N_p, device)
            # reverse: post content flows back to commenting user
            agg_homo_user   += self._mean_agg(self.W_post(h_post[dst]),            src, N_u, device)

        if g.num_edges("user_comment_user") > 0:
            src, dst = g.edges(etype="user_comment_user")
            src, dst = src.to(device), dst.to(device)
            h_src = h_user[src]
            # A→B: edge text added to homo message
            homo_msg = self.lin_homo_ucu(h_src)
            if "ucu_comment" in ef:
                homo_msg = homo_msg + self.E_ucu_comment(ef["ucu_comment"])
            if "ucu_reply" in ef:
                homo_msg = homo_msg + self.E_ucu_reply(ef["ucu_reply"])
            agg_homo_user   += self._mean_agg(homo_msg,                      dst, N_u, device)
            agg_hetero_user += self._mean_agg(self.lin_hetero_ucu(h_src),    dst, N_u, device)

        # DisenConv: out = out_homo + out_hetero + lin_r(self)
        h_user_new = self.norm_user(F.relu(
            self.W_self_user(h_user) + agg_homo_user + agg_hetero_user))
        h_post_new = self.norm_post(F.relu(
            self.W_self_post(h_post) + agg_post))

        h_user_new = F.dropout(h_user_new, p=self.dropout, training=self.training)
        h_post_new = F.dropout(h_post_new, p=self.dropout, training=self.training)
        return h_user_new, h_post_new


class Hetero2Net(nn.Module):
    """
    Hetero²Net for ControBench.

    Each layer:
      1. HeteroConvLayer — proper hetero message passing, keeps post nodes active
      2. Build meta-path graphs from updated h_user
      3. Homo + hetero meta-path convolutions with proper attention
      4. SemanticFusion + combine

    Auxiliary label-prediction head used as soft regularisation during training.
    """
    def __init__(self, in_feats=768, edge_feats=768, hidden_size=256,
                 out_classes=9, dropout=0.5, n_layers=2,
                 post_feats=1536, n_paths=3, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.n_paths  = n_paths
        self.dropout  = dropout

        self.user_proj = nn.Linear(in_feats,   hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)

        # separate projections per edge semantic role
        self.edge_proj_comment     = nn.Linear(edge_feats, hidden_size)
        self.edge_proj_ucu_comment = nn.Linear(edge_feats, hidden_size)
        self.edge_proj_ucu_reply   = nn.Linear(edge_feats, hidden_size)

        # per-layer hetero conv (updates h_user and h_post)
        self.hetero_layers = nn.ModuleList([
            HeteroConvLayer(hidden_size, dropout) for _ in range(n_layers)
        ])

        # per-layer meta-path convolutions
        self.homo_convs = nn.ModuleList([
            nn.ModuleList([MetapathConv(hidden_size, dropout) for _ in range(n_paths)])
            for _ in range(n_layers)
        ])
        self.hetero_convs = nn.ModuleList([
            nn.ModuleList([MetapathConv(hidden_size, dropout) for _ in range(n_paths)])
            for _ in range(n_layers)
        ])

        self.homo_fusion   = nn.ModuleList([
            SemanticFusion(hidden_size, n_paths) for _ in range(n_layers)])
        self.hetero_fusion = nn.ModuleList([
            SemanticFusion(hidden_size, n_paths) for _ in range(n_layers)])
        self.combine       = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size) for _ in range(n_layers)])

        self.label_pred = nn.Linear(hidden_size, out_classes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_classes),
        )

    def _build_mp3_directed(self, hetero_g, N_u, device):
        """MP3: directed A→B UCU graph, preserving directionality."""
        if hetero_g.num_edges("user_comment_user") > 0:
            src, dst = hetero_g.edges(etype="user_comment_user")
            return dgl.graph((src.to(device), dst.to(device)), num_nodes=N_u)
        return dgl.graph(([], []), num_nodes=N_u)

    def forward(self, hetero_g, node_features, edge_features=None):
        device = node_features["user"].device
        N_u    = hetero_g.num_nodes("user")

        h_user = F.relu(self.user_proj(node_features["user"]))
        h_post = F.relu(self.post_proj(node_features["post"]))

        # project edge features with separate projections per semantic role
        ef = {}
        if edge_features is not None:
            if "comment" in edge_features:
                ef["comment"] = self.edge_proj_comment(edge_features["comment"])
            if "user_comment_user" in edge_features:
                ef["ucu_comment"] = self.edge_proj_ucu_comment(
                    edge_features["user_comment_user"])
        if ("user_comment_user" in hetero_g.etypes
                and hetero_g.num_edges("user_comment_user") > 0
                and "reply_feat" in hetero_g.edges["user_comment_user"].data):
            ef["ucu_reply"] = self.edge_proj_ucu_reply(
                hetero_g.edges["user_comment_user"].data["reply_feat"])

        for i in range(self.n_layers):
            # step 1: proper hetero message passing — post nodes stay active
            h_user, h_post = self.hetero_layers[i](hetero_g, h_user, h_post, ef)

            # step 2: build meta-path graphs from updated h_user
            mp_edges = build_metapath_graphs(hetero_g, h_user, device)
            # MP1, MP2: undirected co-occurrence via make_homo_graph
            # MP3: directed A→B to preserve UCU directionality
            mp_graphs = [
                make_homo_graph(mp_edges[0][0], mp_edges[0][1], N_u, device),
                make_homo_graph(mp_edges[1][0], mp_edges[1][1], N_u, device),
                self._build_mp3_directed(hetero_g, N_u, device),
            ]

            # step 3: homo + hetero meta-path convolutions
            homo_embs, hetero_embs = [], []
            for j, sg in enumerate(mp_graphs):
                h_agg = self.homo_convs[i][j](sg, h_user)
                homo_embs.append(h_agg)
                hetero_embs.append(self.hetero_convs[i][j](sg, h_user - h_agg))

            # step 4: fuse
            h_user = F.relu(self.combine[i](torch.cat([
                self.homo_fusion[i](homo_embs),
                self.hetero_fusion[i](hetero_embs),
            ], dim=-1)))
            h_user = F.dropout(h_user, p=self.dropout, training=self.training)

        if self.training:
            return self.classifier(h_user), self.label_pred(h_user)
        return self.classifier(h_user)