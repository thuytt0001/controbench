"""
RGCN — Relational Graph Convolutional Network
Schlichtkrull et al., ESWC 2018. arXiv:1703.06103
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.W_publish = nn.Linear(in_feats,  out_feats, bias=False)
        self.W_comment = nn.Linear(in_feats,  out_feats, bias=False)
        self.W_ucu     = nn.Linear(in_feats,  out_feats, bias=False)
        self.W_post_publish = nn.Linear(in_feats, out_feats, bias=False)
        self.W_post_comment = nn.Linear(in_feats, out_feats, bias=False)

        self.E_comment = nn.Linear(edge_feats, out_feats, bias=False)
        self.E_ucu     = nn.Linear(edge_feats, out_feats, bias=False)
        self.E_reply   = nn.Linear(edge_feats, out_feats, bias=False)

        self.W_self = nn.Linear(in_feats, out_feats)
        self.norm   = nn.LayerNorm(out_feats)

    def forward(self, g, h_user, h_post, ef):
        out_dim  = self.W_self.out_features
        agg_user = torch.zeros(h_user.size(0), out_dim, device=h_user.device)
        agg_post = torch.zeros(h_post.size(0), out_dim, device=h_post.device)

        if g.num_edges("publish") > 0:
            src, dst = g.edges(etype="publish")
            agg_post.index_add_(0, dst, self.W_publish(h_user[src]))
            agg_user.index_add_(0, src, self.W_post_publish(h_post[dst]))

        if g.num_edges("comment") > 0:
            src, dst = g.edges(etype="comment")
            msg = self.W_comment(h_user[src])
            if "comment" in ef:
                msg = msg + self.E_comment(ef["comment"])
            agg_post.index_add_(0, dst, msg)
            agg_user.index_add_(0, src, self.W_post_comment(h_post[dst]))

        if g.num_edges("user_comment_user") > 0:
            src, dst = g.edges(etype="user_comment_user")
            # A→B: B receives A's node features + A's comment + B's reply
            msg = self.W_ucu(h_user[src])
            if "user_comment_user" in ef:
                msg = msg + self.E_ucu(ef["user_comment_user"])
            if "reply_feat" in ef:
                msg = msg + self.E_reply(ef["reply_feat"])
            agg_user.index_add_(0, dst, msg)

        # degree normalisation
        deg_u = torch.zeros(h_user.size(0), device=h_user.device)
        deg_p = torch.zeros(h_post.size(0), device=h_post.device)
        for etype in g.etypes:
            _, _, d_type = g.to_canonical_etype(etype)
            _, d = g.edges(etype=etype)
            if d_type == "user":
                deg_u.index_add_(0, d, torch.ones_like(d, dtype=torch.float))
            elif d_type == "post":
                deg_p.index_add_(0, d, torch.ones_like(d, dtype=torch.float))

        h_user_new = self.norm(F.relu(
            self.W_self(h_user) + agg_user / deg_u.clamp(min=1).unsqueeze(1)))
        h_post_new = self.norm(F.relu(
            self.W_self(h_post) + agg_post / deg_p.clamp(min=1).unsqueeze(1)))

        h_user_new = F.dropout(h_user_new, p=self.dropout, training=self.training)
        h_post_new = F.dropout(h_post_new, p=self.dropout, training=self.training)
        return h_user_new, h_post_new


class RGCNNodeClassifier(nn.Module):
    def __init__(self, in_feats=768, edge_feats=768, hidden_size=256,
                 out_classes=9, dropout=0.5, n_layers=2,
                 post_feats=1536, **kwargs):
        super().__init__()
        self.user_proj = nn.Linear(in_feats,   hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)
        self.edge_proj = nn.Linear(edge_feats, hidden_size)
        self.layers    = nn.ModuleList([
            RGCNLayer(hidden_size, hidden_size, hidden_size, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_classes),
        )

    def forward(self, g, node_features, edge_features=None):
        h_user = F.relu(self.user_proj(node_features["user"]))
        h_post = F.relu(self.post_proj(node_features["post"]))

        ef = {}
        if edge_features:
            if "comment" in edge_features:
                ef["comment"] = self.edge_proj(edge_features["comment"])
            if "user_comment_user" in edge_features:
                ef["user_comment_user"] = self.edge_proj(edge_features["user_comment_user"])
        if ("user_comment_user" in g.etypes
                and g.num_edges("user_comment_user") > 0
                and "reply_feat" in g.edges["user_comment_user"].data):
            ef["reply_feat"] = self.edge_proj(g.edges["user_comment_user"].data["reply_feat"])

        for layer in self.layers:
            h_user, h_post = layer(g, h_user, h_post, ef)
        return self.classifier(h_user)