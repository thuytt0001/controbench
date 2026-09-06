# """
# H²G-former — When Heterophily Meets Heterogeneity
# Lin et al., 2024. HetGB benchmark paper.

# Type-aware multi-head attention with per-node heterophily mixing.
# """

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class HeteroAttentionLayer(nn.Module):
#     """
#     Heterogeneous graph attention.

#     Fixes vs. naive implementation:
#     - Relation-specific output projections (no gradient mixing between relations)
#     - dst_idx explicitly moved to correct device
#     - Scatter-softmax over incoming edges per destination node (not over heads)
#     """
#     def __init__(self, hidden_size, num_heads=4, dropout=0.3):
#         super().__init__()
#         assert hidden_size % num_heads == 0
#         self.num_heads = num_heads
#         self.d_k       = hidden_size // num_heads
#         self.dropout   = dropout

#         for rel in ("publish", "comment", "user_comment_user"):
#             setattr(self, f"W_Q_{rel}", nn.Linear(hidden_size, hidden_size))
#             setattr(self, f"W_K_{rel}", nn.Linear(hidden_size, hidden_size))
#             setattr(self, f"W_V_{rel}", nn.Linear(hidden_size, hidden_size))
#             setattr(self, f"W_out_user_{rel}", nn.Linear(hidden_size, hidden_size))

#         for rel in ("publish", "comment"):
#             setattr(self, f"W_out_post_{rel}", nn.Linear(hidden_size, hidden_size))

#         self.alpha_gate    = nn.Linear(hidden_size, 1)
#         self.norm_user     = nn.LayerNorm(hidden_size)
#         self.norm_post     = nn.LayerNorm(hidden_size)
#         self.E_ucu         = nn.Linear(hidden_size, hidden_size)
#         self.E_reply        = nn.Linear(hidden_size, hidden_size)

#     def _attend(self, W_Q, W_K, W_V, h_dst, h_src, dst_idx, N_dst, edge_bias=None):
#         """Neighbor-wise softmax attention aggregation."""
#         H      = self.num_heads
#         dk     = self.d_k
#         E      = len(dst_idx)
#         device = h_dst.device

#         if E == 0:
#             return torch.zeros(N_dst, H * dk, device=device)

#         dst_idx = dst_idx.to(device)
#         Q = W_Q(h_dst[dst_idx]).view(E, H, dk)
#         K = W_K(h_src).view(E, H, dk)
#         V = W_V(h_src).view(E, H, dk)

#         attn = (Q * K).sum(-1) / math.sqrt(dk)   # (E, H)
#         if edge_bias is not None:
#             attn = attn + edge_bias.view(E, H, dk).mean(-1)

#         # scatter-softmax over incoming edges per destination node
#         attn_max = torch.full((N_dst, H), float("-inf"), device=device)
#         attn_max.scatter_reduce_(0, dst_idx.unsqueeze(1).expand(E, H), attn,
#                                  reduce="amax", include_self=True)
#         attn_exp = (attn - attn_max[dst_idx]).exp()
#         attn_sum = torch.zeros(N_dst, H, device=device)
#         attn_sum.index_add_(0, dst_idx, attn_exp)
#         attn_norm = F.dropout(
#             attn_exp / (attn_sum[dst_idx] + 1e-9),
#             p=self.dropout, training=self.training)

#         msg = (attn_norm.unsqueeze(-1) * V).view(E, H * dk)
#         agg = torch.zeros(N_dst, H * dk, device=device)
#         agg.index_add_(0, dst_idx, msg)
#         return agg

#     def forward(self, g, h_user, h_post, ef):
#         N_u    = h_user.size(0)
#         N_p    = h_post.size(0)
#         device = h_user.device

#         agg_user = torch.zeros_like(h_user)
#         agg_post = torch.zeros_like(h_post)

#         if g.num_edges("publish") > 0:
#             src, dst = [t.to(device) for t in g.edges(etype="publish")]
#             agg_post = agg_post + self.W_out_post_publish(
#                 self._attend(self.W_Q_publish, self.W_K_publish, self.W_V_publish,
#                              h_post, h_user[src], dst, N_p))
#             agg_user = agg_user + self.W_out_user_publish(
#                 self._attend(self.W_Q_publish, self.W_K_publish, self.W_V_publish,
#                              h_user, h_post[dst], src, N_u))

#         if g.num_edges("comment") > 0:
#             src, dst = [t.to(device) for t in g.edges(etype="comment")]
#             bias     = ef.get("comment")
#             agg_post = agg_post + self.W_out_post_comment(
#                 self._attend(self.W_Q_comment, self.W_K_comment, self.W_V_comment,
#                              h_post, h_user[src], dst, N_p, bias))
#             agg_user = agg_user + self.W_out_user_comment(
#                 self._attend(self.W_Q_comment, self.W_K_comment, self.W_V_comment,
#                              h_user, h_post[dst], src, N_u))

#         if g.num_edges("user_comment_user") > 0:
#             src, dst = [t.to(device) for t in g.edges(etype="user_comment_user")]
#             # both dual edge features as edge bias for attention
#             bias = None
#             if "user_comment_user" in ef:
#                 bias = self.E_ucu(ef["user_comment_user"])
#             if "reply_feat" in ef:
#                 rb   = self.E_reply(ef["reply_feat"])
#                 bias = rb if bias is None else bias + rb
#             agg_user = agg_user + self.W_out_user_user_comment_user(
#                 self._attend(self.W_Q_user_comment_user, self.W_K_user_comment_user,
#                              self.W_V_user_comment_user,
#                              h_user, h_user[src], dst, N_u, bias))

#         alpha      = torch.sigmoid(self.alpha_gate(h_user))
#         h_user_new = self.norm_user(F.relu(
#             h_user + alpha * agg_user + (1.0 - alpha) * (h_user - agg_user)))
#         h_post_new = self.norm_post(F.relu(h_post + agg_post))

#         h_user_new = F.dropout(h_user_new, p=self.dropout, training=self.training)
#         h_post_new = F.dropout(h_post_new, p=self.dropout, training=self.training)
#         return h_user_new, h_post_new


# class H2GFormer(nn.Module):
#     def __init__(self, in_feats=768, edge_feats=768, hidden_size=256,
#                  out_classes=9, dropout=0.3, n_layers=2,
#                  num_heads=4, post_feats=1536, **kwargs):
#         super().__init__()
#         self.user_proj = nn.Linear(in_feats,   hidden_size)
#         self.post_proj = nn.Linear(post_feats, hidden_size)
#         self.edge_proj_comment     = nn.Linear(edge_feats, hidden_size)
#         self.edge_proj_ucu_comment = nn.Linear(edge_feats, hidden_size)
#         self.edge_proj_ucu_reply   = nn.Linear(edge_feats, hidden_size)
#         self.layers    = nn.ModuleList([
#             HeteroAttentionLayer(hidden_size, num_heads, dropout) for _ in range(n_layers)
#         ])
#         self.ffn       = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size * 2), nn.GELU(),
#                 nn.Dropout(dropout), nn.Linear(hidden_size * 2, hidden_size))
#             for _ in range(n_layers)
#         ])
#         self.ffn_norm  = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, out_classes),
#         )

#     def forward(self, hetero_g, node_features, edge_features=None):
#         h_user = F.relu(self.user_proj(node_features["user"]))
#         h_post = F.relu(self.post_proj(node_features["post"]))

#         ef = {}
#         if edge_features:
#             if "comment" in edge_features:
#                 ef["comment"] = self.edge_proj_comment(edge_features["comment"])
#             if "user_comment_user" in edge_features:
#                 ef["user_comment_user"] = self.edge_proj_ucu_comment(
#                     edge_features["user_comment_user"])
#         if ("user_comment_user" in hetero_g.etypes
#                 and hetero_g.num_edges("user_comment_user") > 0
#                 and "reply_feat" in hetero_g.edges["user_comment_user"].data):
#             ef["reply_feat"] = self.edge_proj_ucu_reply(
#                 hetero_g.edges["user_comment_user"].data["reply_feat"])

#         for i, layer in enumerate(self.layers):
#             h_user, h_post = layer(hetero_g, h_user, h_post, ef)
#             h_user = self.ffn_norm[i](h_user + self.ffn[i](h_user))

#         return self.classifier(h_user)


"""
H²G-former (Base) — runs directly on the original heterograph.
No k-hop augmentation. Same architecture as H2GFormer.
Use for ablation: original hetero graph vs k-hop augmented graph.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroAttentionLayer(nn.Module):
    """
    Heterogeneous graph attention.

    Fixes vs. naive implementation:
    - Relation-specific output projections (no gradient mixing between relations)
    - dst_idx explicitly moved to correct device
    - Scatter-softmax over incoming edges per destination node (not over heads)
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.d_k       = hidden_size // num_heads
        self.dropout   = dropout

        for rel in ("publish", "comment", "user_comment_user"):
            setattr(self, f"W_Q_{rel}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"W_K_{rel}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"W_V_{rel}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"W_out_user_{rel}", nn.Linear(hidden_size, hidden_size))

        for rel in ("publish", "comment"):
            setattr(self, f"W_out_post_{rel}", nn.Linear(hidden_size, hidden_size))

        self.alpha_gate    = nn.Linear(hidden_size, 1)
        self.norm_user     = nn.LayerNorm(hidden_size)
        self.norm_post     = nn.LayerNorm(hidden_size)
        self.E_ucu         = nn.Linear(hidden_size, hidden_size)
        self.E_reply        = nn.Linear(hidden_size, hidden_size)

    def _attend(self, W_Q, W_K, W_V, h_dst, h_src, dst_idx, N_dst, edge_bias=None):
        """Neighbor-wise softmax attention aggregation."""
        H      = self.num_heads
        dk     = self.d_k
        E      = len(dst_idx)
        device = h_dst.device

        if E == 0:
            return torch.zeros(N_dst, H * dk, device=device)

        dst_idx = dst_idx.to(device)
        Q = W_Q(h_dst[dst_idx]).view(E, H, dk)
        K = W_K(h_src).view(E, H, dk)
        V = W_V(h_src).view(E, H, dk)

        attn = (Q * K).sum(-1) / math.sqrt(dk)   # (E, H)
        if edge_bias is not None:
            attn = attn + edge_bias.view(E, H, dk).mean(-1)

        # scatter-softmax over incoming edges per destination node
        attn_max = torch.full((N_dst, H), float("-inf"), device=device)
        attn_max.scatter_reduce_(0, dst_idx.unsqueeze(1).expand(E, H), attn,
                                 reduce="amax", include_self=True)
        attn_exp = (attn - attn_max[dst_idx]).exp()
        attn_sum = torch.zeros(N_dst, H, device=device)
        attn_sum.index_add_(0, dst_idx, attn_exp)
        attn_norm = F.dropout(
            attn_exp / (attn_sum[dst_idx] + 1e-9),
            p=self.dropout, training=self.training)

        msg = (attn_norm.unsqueeze(-1) * V).view(E, H * dk)
        agg = torch.zeros(N_dst, H * dk, device=device)
        agg.index_add_(0, dst_idx, msg)
        return agg

    def forward(self, g, h_user, h_post, ef):
        N_u    = h_user.size(0)
        N_p    = h_post.size(0)
        device = h_user.device

        agg_user = torch.zeros_like(h_user)
        agg_post = torch.zeros_like(h_post)

        if g.num_edges("publish") > 0:
            src, dst = [t.to(device) for t in g.edges(etype="publish")]
            agg_post = agg_post + self.W_out_post_publish(
                self._attend(self.W_Q_publish, self.W_K_publish, self.W_V_publish,
                             h_post, h_user[src], dst, N_p))
            agg_user = agg_user + self.W_out_user_publish(
                self._attend(self.W_Q_publish, self.W_K_publish, self.W_V_publish,
                             h_user, h_post[dst], src, N_u))

        if g.num_edges("comment") > 0:
            src, dst = [t.to(device) for t in g.edges(etype="comment")]
            bias     = ef.get("comment")
            agg_post = agg_post + self.W_out_post_comment(
                self._attend(self.W_Q_comment, self.W_K_comment, self.W_V_comment,
                             h_post, h_user[src], dst, N_p, bias))
            agg_user = agg_user + self.W_out_user_comment(
                self._attend(self.W_Q_comment, self.W_K_comment, self.W_V_comment,
                             h_user, h_post[dst], src, N_u))

        if g.num_edges("user_comment_user") > 0:
            src, dst = [t.to(device) for t in g.edges(etype="user_comment_user")]
            # both dual edge features as edge bias for attention
            bias = None
            if "user_comment_user" in ef:
                bias = self.E_ucu(ef["user_comment_user"])
            if "reply_feat" in ef:
                rb   = self.E_reply(ef["reply_feat"])
                bias = rb if bias is None else bias + rb
            agg_user = agg_user + self.W_out_user_user_comment_user(
                self._attend(self.W_Q_user_comment_user, self.W_K_user_comment_user,
                             self.W_V_user_comment_user,
                             h_user, h_user[src], dst, N_u, bias))

        alpha      = torch.sigmoid(self.alpha_gate(h_user))
        h_user_new = self.norm_user(F.relu(
            h_user + alpha * agg_user + (1.0 - alpha) * (h_user - agg_user)))
        h_post_new = self.norm_post(F.relu(h_post + agg_post))

        h_user_new = F.dropout(h_user_new, p=self.dropout, training=self.training)
        h_post_new = F.dropout(h_post_new, p=self.dropout, training=self.training)
        return h_user_new, h_post_new


class H2GFormerBase(nn.Module):
    def __init__(self, in_feats=768, edge_feats=768, hidden_size=256,
                 out_classes=9, dropout=0.3, n_layers=2,
                 num_heads=4, post_feats=1536, **kwargs):
        super().__init__()
        self.user_proj = nn.Linear(in_feats,   hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)
        self.edge_proj_comment     = nn.Linear(edge_feats, hidden_size)
        self.edge_proj_ucu_comment = nn.Linear(edge_feats, hidden_size)
        self.edge_proj_ucu_reply   = nn.Linear(edge_feats, hidden_size)
        self.layers    = nn.ModuleList([
            HeteroAttentionLayer(hidden_size, num_heads, dropout) for _ in range(n_layers)
        ])
        self.ffn       = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_size * 2, hidden_size))
            for _ in range(n_layers)
        ])
        self.ffn_norm  = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_classes),
        )

    def forward(self, hetero_g, node_features, edge_features=None):
        h_user = F.relu(self.user_proj(node_features["user"]))
        h_post = F.relu(self.post_proj(node_features["post"]))

        ef = {}
        if edge_features:
            if "comment" in edge_features:
                ef["comment"] = self.edge_proj_comment(edge_features["comment"])
            if "user_comment_user" in edge_features:
                ef["user_comment_user"] = self.edge_proj_ucu_comment(
                    edge_features["user_comment_user"])
        if ("user_comment_user" in hetero_g.etypes
                and hetero_g.num_edges("user_comment_user") > 0
                and "reply_feat" in hetero_g.edges["user_comment_user"].data):
            ef["reply_feat"] = self.edge_proj_ucu_reply(
                hetero_g.edges["user_comment_user"].data["reply_feat"])

        for i, layer in enumerate(self.layers):
            h_user, h_post = layer(hetero_g, h_user, h_post, ef)
            h_user = self.ffn_norm[i](h_user + self.ffn[i](h_user))

        return self.classifier(h_user)