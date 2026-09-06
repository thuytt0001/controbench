# """
# HAN — Heterogeneous Attention Network
# Wang et al., WWW 2019. arXiv:1903.07293
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# from dgl.nn import GATConv


# class HANLayer(nn.Module):
#     """GAT over a meta-path subgraph, projected back to hidden_size."""
#     def __init__(self, hidden_size, num_heads, dropout=0.5):
#         super().__init__()
#         head_dim = hidden_size // num_heads
#         self.gat = GATConv(
#             hidden_size, head_dim, num_heads=num_heads,
#             feat_drop=dropout, attn_drop=dropout,
#             activation=F.elu, allow_zero_in_degree=True,
#         )
#         self.proj = nn.Linear(head_dim * num_heads, hidden_size)
#         self.norm = nn.LayerNorm(hidden_size)

#     def forward(self, subg, h):
#         out = self.gat(subg, h).flatten(1)
#         return self.norm(h + self.proj(out))


# class SemanticAttention(nn.Module):
#     """Soft attention to fuse embeddings from multiple meta-paths."""
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.attn = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_size // 2, 1, bias=False),
#         )

#     def forward(self, embeds):
#         Z = torch.stack(embeds, dim=1)          # (N, M, H)
#         w = torch.softmax(self.attn(Z), dim=1)  # (N, M, 1)
#         return (w * Z).sum(dim=1)               # (N, H)


# class HANNodeClassifier(nn.Module):
#     """
#     HAN for ControBench. Three meta-paths:
#       MP1: user-publish-post (post content enrichment)
#       MP2: user-comment-post (comment context)
#       MP3: user-comment-user (direct interactions, dual edge features)
#     """
#     def __init__(self, in_feats=768, edge_feats=768, hidden_size=256,
#                  out_classes=9, dropout=0.5, n_layers=1,
#                  num_heads=8, post_feats=1536, **kwargs):
#         super().__init__()
#         self.n_layers = n_layers
#         self.dropout  = dropout

#         self.user_proj     = nn.Linear(in_feats,   hidden_size)
#         self.post_proj     = nn.Linear(post_feats, hidden_size)
#         self.edge_proj_com = nn.Linear(edge_feats, hidden_size)
#         self.edge_proj_ucu = nn.Linear(edge_feats, hidden_size)

#         n_paths = 3
#         self.han_layers = nn.ModuleList([
#             nn.ModuleList([HANLayer(hidden_size, num_heads, dropout) for _ in range(n_paths)])
#             for _ in range(n_layers)
#         ])
#         self.semantic_attn = nn.ModuleList([
#             SemanticAttention(hidden_size) for _ in range(n_layers)
#         ])
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, out_classes),
#         )

#     def _build_subgraphs(self, g, h_user, h_post, ef):
#         N      = h_user.size(0)
#         device = h_user.device

#         def _self_loop_graph():
#             idx = torch.arange(N, device=device)
#             return dgl.graph((idx, idx), num_nodes=N)

#         def _mean_pool(src_idx, vals, N_dst):
#             agg = torch.zeros(N_dst, vals.size(1), device=device)
#             cnt = torch.zeros(N_dst, 1, device=device)
#             agg.index_add_(0, src_idx, vals)
#             cnt.index_add_(0, src_idx, torch.ones(len(src_idx), 1, device=device))
#             mask = cnt.squeeze(1) > 0
#             agg[mask] = agg[mask] / cnt[mask]
#             return agg

#         subgraphs = []

#         # MP1: publish
#         if g.num_edges("publish") > 0:
#             src, dst = g.edges(etype="publish")
#             h_pub = h_user + _mean_pool(src, h_post[dst], N)
#             subgraphs.append((_self_loop_graph(), h_pub))
#         else:
#             subgraphs.append((_self_loop_graph(), h_user))

#         # MP2: comment
#         if g.num_edges("comment") > 0:
#             src, dst = g.edges(etype="comment")
#             h_com = h_user + _mean_pool(src, h_post[dst], N)
#             if "comment" in ef:
#                 h_com = h_com + _mean_pool(src, ef["comment"], N)
#             subgraphs.append((_self_loop_graph(), h_com))
#         else:
#             subgraphs.append((_self_loop_graph(), h_user))

#         # MP3: UCU with dual edge features
#         if g.num_edges("user_comment_user") > 0:
#             src, dst = g.edges(etype="user_comment_user")
#             edge_h = torch.zeros(len(src), h_user.size(1), device=device)
#             if "user_comment_user" in ef:
#                 edge_h = edge_h + ef["user_comment_user"]
#             if "reply_feat" in ef:
#                 edge_h = edge_h + ef["reply_feat"]

#             sg = dgl.graph((src, dst), num_nodes=N)
#             sg.edata["feat"] = edge_h
#             subgraphs.append((sg, h_user))
#         else:
#             subgraphs.append((_self_loop_graph(), h_user))

#         return subgraphs

#     def forward(self, g, node_features, edge_features=None):
#         h_user = F.relu(self.user_proj(node_features["user"]))
#         h_post = F.relu(self.post_proj(node_features["post"]))

#         ef = {}
#         if edge_features:
#             if "comment" in edge_features:
#                 ef["comment"] = self.edge_proj_com(edge_features["comment"])
#             if "user_comment_user" in edge_features:
#                 ef["user_comment_user"] = self.edge_proj_ucu(edge_features["user_comment_user"])
#         if "reply_feat" in g.edges["user_comment_user"].data:
#             ef["reply_feat"] = self.edge_proj_ucu(g.edges["user_comment_user"].data["reply_feat"])

#         for layer_idx in range(self.n_layers):
#             subgraphs   = self._build_subgraphs(g, h_user, h_post, ef)
#             path_embeds = [
#                 self.han_layers[layer_idx][i](sg, h_init)
#                 for i, (sg, h_init) in enumerate(subgraphs)
#             ]
#             h_user = self.semantic_attn[layer_idx](path_embeds)
#             h_user = F.dropout(h_user, p=self.dropout, training=self.training)

#         return self.classifier(h_user)
"""
HAN — Heterogeneous Attention Network
Wang et al., WWW 2019. arXiv:1903.07293
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
from .shared import build_metapath_graphs, make_homo_graph


class HANLayer(nn.Module):
    """GAT over a meta-path subgraph, projected back to hidden_size."""
    def __init__(self, hidden_size, num_heads, dropout=0.5):
        super().__init__()
        head_dim = hidden_size // num_heads
        self.gat = GATConv(
            hidden_size, head_dim, num_heads=num_heads,
            feat_drop=dropout, attn_drop=dropout,
            activation=F.elu, allow_zero_in_degree=True,
        )
        self.proj = nn.Linear(head_dim * num_heads, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, subg, h):
        out = self.gat(subg, h).flatten(1)
        return self.norm(h + self.proj(out))


class SemanticAttention(nn.Module):
    """
    Standard HAN semantic attention — shared path-level scalar weights.
    For each path Φ: w_Φ = (1/N) Σ_v tanh(W·z_v + b)·q
    then softmax over paths. All nodes share the same path weights.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.q, std=0.02)

    def forward(self, embeds):
        # embeds: list of M tensors each (N, H)
        Z = torch.stack(embeds, dim=1)              # (N, M, H)
        # per-node per-path score: tanh(W·z)·q → mean over nodes → shared weight
        e = torch.tanh(self.W(Z))                   # (N, M, H)
        scores = (e * self.q).sum(-1)               # (N, M)
        w = torch.softmax(scores.mean(0), dim=0)    # (M,) — shared across nodes
        return (w.unsqueeze(0).unsqueeze(-1) * Z).sum(dim=1)  # (N, H)


class HANNodeClassifier(nn.Module):
    """
    HAN for ControBench. Three meta-paths:
      MP1: UPU  — users who published the same post
      MP2: UCpU — users who commented on the same post
      MP3: UCU  — direct user-comment-user edges (dual edge features)
    """
    def __init__(self, in_feats=768, edge_feats=768, hidden_size=256,
                 out_classes=9, dropout=0.5, n_layers=1,
                 num_heads=8, post_feats=1536, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.dropout  = dropout

        self.user_proj     = nn.Linear(in_feats,   hidden_size)
        self.post_proj     = nn.Linear(post_feats, hidden_size)
        self.edge_proj_com         = nn.Linear(edge_feats, hidden_size)
        self.edge_proj_ucu_comment = nn.Linear(edge_feats, hidden_size)
        self.edge_proj_ucu_reply   = nn.Linear(edge_feats, hidden_size)

        # UCU edge features scattered into dst (B) node before GAT
        self.ucu_edge_proj = nn.Linear(hidden_size, hidden_size)

        n_paths = 3
        self.han_layers = nn.ModuleList([
            nn.ModuleList([HANLayer(hidden_size, num_heads, dropout) for _ in range(n_paths)])
            for _ in range(n_layers)
        ])
        self.semantic_attn = nn.ModuleList([
            SemanticAttention(hidden_size) for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_classes),
        )

    def _build_subgraphs(self, g, h_user, h_post, ef):
        N      = h_user.size(0)
        device = h_user.device

        # build real meta-path user-user graphs via shared utility
        mp_edges = build_metapath_graphs(g, h_user, device)

        subgraphs = []

        # MP1: UPU — real user-user co-publish graph
        # enrich node features with published post content before GAT
        h_mp1 = h_user.clone()
        if g.num_edges("publish") > 0:
            src, dst = g.edges(etype="publish")
            src, dst = src.to(device), dst.to(device)
            agg = torch.zeros(N, h_post.size(1), device=device)
            cnt = torch.zeros(N, 1, device=device)
            agg.index_add_(0, src, h_post[dst])
            cnt.index_add_(0, src, torch.ones(len(src), 1, device=device))
            mask = cnt.squeeze(1) > 0
            agg[mask] = agg[mask] / cnt[mask]
            h_mp1 = h_mp1 + agg
        sg1 = make_homo_graph(mp_edges[0][0], mp_edges[0][1], N, device)
        subgraphs.append((sg1, h_mp1))

        # MP2: UCpU — real user-user co-comment graph
        # enrich node features with commented post content + comment text
        h_mp2 = h_user.clone()
        if g.num_edges("comment") > 0:
            src, dst = g.edges(etype="comment")
            src, dst = src.to(device), dst.to(device)
            agg = torch.zeros(N, h_post.size(1), device=device)
            cnt = torch.zeros(N, 1, device=device)
            agg.index_add_(0, src, h_post[dst])
            cnt.index_add_(0, src, torch.ones(len(src), 1, device=device))
            mask = cnt.squeeze(1) > 0
            agg[mask] = agg[mask] / cnt[mask]
            h_mp2 = h_mp2 + agg
            if "comment" in ef:
                agg_c = torch.zeros(N, ef["comment"].size(1), device=device)
                cnt_c = torch.zeros(N, 1, device=device)
                agg_c.index_add_(0, src, ef["comment"])
                cnt_c.index_add_(0, src, torch.ones(len(src), 1, device=device))
                mask_c = cnt_c.squeeze(1) > 0
                agg_c[mask_c] = agg_c[mask_c] / cnt_c[mask_c]
                h_mp2 = h_mp2 + agg_c
        sg2 = make_homo_graph(mp_edges[1][0], mp_edges[1][1], N, device)
        subgraphs.append((sg2, h_mp2))

        # MP3: UCU — directed user-comment-user graph
        # incorporate dual edge features into src node representation before GAT
        h_mp3 = h_user.clone()
        if g.num_edges("user_comment_user") > 0:
            src, dst = g.edges(etype="user_comment_user")
            src, dst = src.to(device), dst.to(device)
            # scatter edge features into src nodes (A's comment context)
            edge_h = torch.zeros(len(src), h_user.size(1), device=device)
            if "user_comment_user" in ef:
                edge_h = edge_h + ef["user_comment_user"]
            if "reply_feat" in ef:
                edge_h = edge_h + ef["reply_feat"]
            # mean-scatter into dst (B) — B receives A's comment + B's reply
            dst_agg = torch.zeros(N, h_user.size(1), device=device)
            dst_cnt = torch.zeros(N, 1, device=device)
            dst_agg.index_add_(0, dst, self.ucu_edge_proj(edge_h))
            dst_cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=device))
            mask = dst_cnt.squeeze(1) > 0
            dst_agg[mask] = dst_agg[mask] / dst_cnt[mask]
            h_mp3 = h_mp3 + dst_agg
            sg3 = dgl.graph((src, dst), num_nodes=N)
        else:
            idx = torch.arange(N, device=device)
            sg3 = dgl.graph((idx, idx), num_nodes=N)
        subgraphs.append((sg3, h_mp3))

        return subgraphs

    def forward(self, g, node_features, edge_features=None):
        h_user = F.relu(self.user_proj(node_features["user"]))
        h_post = F.relu(self.post_proj(node_features["post"]))

        ef = {}
        if edge_features:
            if "comment" in edge_features:
                ef["comment"] = self.edge_proj_com(edge_features["comment"])
            if "user_comment_user" in edge_features:
                ef["user_comment_user"] = self.edge_proj_ucu_comment(
                    edge_features["user_comment_user"])
        if ("user_comment_user" in g.etypes
                and g.num_edges("user_comment_user") > 0
                and "reply_feat" in g.edges["user_comment_user"].data):
            ef["reply_feat"] = self.edge_proj_ucu_reply(
                g.edges["user_comment_user"].data["reply_feat"])

        for layer_idx in range(self.n_layers):
            subgraphs   = self._build_subgraphs(g, h_user, h_post, ef)
            path_embeds = [
                self.han_layers[layer_idx][i](sg, h_init)
                for i, (sg, h_init) in enumerate(subgraphs)
            ]
            h_user = self.semantic_attn[layer_idx](path_embeds)
            h_user = F.dropout(h_user, p=self.dropout, training=self.training)

        return self.classifier(h_user)