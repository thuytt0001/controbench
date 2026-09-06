# """
# HinSAGE — Heterogeneous GraphSAGE
# Hamilton et al. (GraphSAGE), NeurIPS 2017. arXiv:1706.02216
# Extended to heterogeneous graphs.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class HinSAGELayer(nn.Module):
#     def __init__(self, hidden_size, dropout=0.5):
#         super().__init__()
#         self.dropout = dropout

#         self.W_publish = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.W_comment = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.W_ucu     = nn.Linear(hidden_size, hidden_size, bias=False)

#         self.E_comment = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.E_ucu     = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.E_reply   = nn.Linear(hidden_size, hidden_size, bias=False)

#         # SAGE combine: concat(self, agg) → hidden
#         self.combine_user = nn.Linear(hidden_size * 2, hidden_size)
#         self.combine_post = nn.Linear(hidden_size * 2, hidden_size)

#         self.norm_user = nn.LayerNorm(hidden_size)
#         self.norm_post = nn.LayerNorm(hidden_size)

#     @staticmethod
#     def _mean_agg(msg, dst_idx, N_dst):
#         agg = torch.zeros(N_dst, msg.size(1), device=msg.device)
#         cnt = torch.zeros(N_dst, 1,           device=msg.device)
#         agg.index_add_(0, dst_idx, msg)
#         cnt.index_add_(0, dst_idx, torch.ones(len(dst_idx), 1, device=msg.device))
#         mask = cnt.squeeze(1) > 0
#         agg[mask] = agg[mask] / cnt[mask]
#         return agg

#     def forward(self, g, h_user, h_post, ef):
#         N_u = h_user.size(0)
#         N_p = h_post.size(0)

#         agg_post = torch.zeros(N_p, h_user.size(1), device=h_user.device)
#         agg_user = torch.zeros(N_u, h_user.size(1), device=h_user.device)

#         if g.num_edges("publish") > 0:
#             src, dst = g.edges(etype="publish")
#             agg_post = agg_post + self._mean_agg(self.W_publish(h_user[src]), dst, N_p)
#             agg_user = agg_user + self._mean_agg(self.W_publish(h_post[dst]), src, N_u)

#         if g.num_edges("comment") > 0:
#             src, dst = g.edges(etype="comment")
#             msg = self.W_comment(h_user[src])
#             if "comment" in ef:
#                 msg = msg + self.E_comment(ef["comment"])
#             agg_post = agg_post + self._mean_agg(msg, dst, N_p)
#             # reverse: post → commenter
#             msg_rev = self.W_comment(h_post[dst])
#             if "comment" in ef:
#                 msg_rev = msg_rev + self.E_comment(ef["comment"])
#             agg_user = agg_user + self._mean_agg(msg_rev, src, N_u)

#         if g.num_edges("user_comment_user") > 0:
#             src, dst = g.edges(etype="user_comment_user")
#             # A→B: B gets A's node features + A's comment + B's reply
#             msg = self.W_ucu(h_user[src])
#             if "user_comment_user" in ef:
#                 msg = msg + self.E_ucu(ef["user_comment_user"])
#             if "reply_feat" in ef:
#                 msg = msg + self.E_reply(ef["reply_feat"])
#             agg_user = agg_user + self._mean_agg(msg, dst, N_u)

#         h_user_new = self.norm_user(F.relu(
#             self.combine_user(torch.cat([h_user, agg_user], dim=1))))
#         h_post_new = self.norm_post(F.relu(
#             self.combine_post(torch.cat([h_post, agg_post], dim=1))))

#         h_user_new = F.dropout(h_user_new, p=self.dropout, training=self.training)
#         h_post_new = F.dropout(h_post_new, p=self.dropout, training=self.training)
#         return h_user_new, h_post_new


# class HinSAGENodeClassifier(nn.Module):
#     def __init__(self, in_feats=768, edge_feats=768, hidden_size=256,
#                  out_classes=9, dropout=0.5, n_layers=2,
#                  post_feats=1536, **kwargs):
#         super().__init__()
#         self.user_proj = nn.Linear(in_feats,   hidden_size)
#         self.post_proj = nn.Linear(post_feats, hidden_size)
#         self.edge_proj = nn.Linear(edge_feats, hidden_size)
#         self.layers    = nn.ModuleList([
#             HinSAGELayer(hidden_size, dropout) for _ in range(n_layers)
#         ])
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, out_classes),
#         )

#     def forward(self, g, node_features, edge_features=None):
#         h_user = F.relu(self.user_proj(node_features["user"]))
#         h_post = F.relu(self.post_proj(node_features["post"]))

#         ef = {}
#         if edge_features:
#             if "comment" in edge_features:
#                 ef["comment"] = self.edge_proj(edge_features["comment"])
#             if "user_comment_user" in edge_features:
#                 ef["user_comment_user"] = self.edge_proj(edge_features["user_comment_user"])
#         if "reply_feat" in g.edges["user_comment_user"].data:
#             ef["reply_feat"] = self.edge_proj(g.edges["user_comment_user"].data["reply_feat"])

#         for layer in self.layers:
#             h_user, h_post = layer(g, h_user, h_post, ef)
#         return self.classifier(h_user)

"""
HinSAGE — Heterogeneous GraphSAGE
Hamilton et al. (GraphSAGE), NeurIPS 2017. arXiv:1706.02216
Extended to heterogeneous graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HinSAGELayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.W_publish     = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_publish_rev = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_comment     = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_comment_rev = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ucu         = nn.Linear(hidden_size, hidden_size, bias=False)

        self.E_comment = nn.Linear(hidden_size, hidden_size, bias=False)
        self.E_ucu     = nn.Linear(hidden_size, hidden_size, bias=False)
        self.E_reply   = nn.Linear(hidden_size, hidden_size, bias=False)

        # SAGE combine: concat(self, agg) → hidden
        self.combine_user = nn.Linear(hidden_size * 2, hidden_size)
        self.combine_post = nn.Linear(hidden_size * 2, hidden_size)

        self.norm_user = nn.LayerNorm(hidden_size)
        self.norm_post = nn.LayerNorm(hidden_size)

    @staticmethod
    def _mean_agg(msg, dst_idx, N_dst):
        agg = torch.zeros(N_dst, msg.size(1), device=msg.device)
        cnt = torch.zeros(N_dst, 1,           device=msg.device)
        agg.index_add_(0, dst_idx, msg)
        cnt.index_add_(0, dst_idx, torch.ones(len(dst_idx), 1, device=msg.device))
        mask = cnt.squeeze(1) > 0
        agg[mask] = agg[mask] / cnt[mask]
        return agg

    def forward(self, g, h_user, h_post, ef):
        N_u = h_user.size(0)
        N_p = h_post.size(0)

        agg_post = torch.zeros(N_p, h_user.size(1), device=h_user.device)
        agg_user = torch.zeros(N_u, h_user.size(1), device=h_user.device)

        if g.num_edges("publish") > 0:
            src, dst = g.edges(etype="publish")
            agg_post = agg_post + self._mean_agg(self.W_publish(h_user[src]), dst, N_p)
            agg_user = agg_user + self._mean_agg(self.W_publish_rev(h_post[dst]), src, N_u)

        if g.num_edges("comment") > 0:
            src, dst = g.edges(etype="comment")
            msg = self.W_comment(h_user[src])
            if "comment" in ef:
                msg = msg + self.E_comment(ef["comment"])
            agg_post = agg_post + self._mean_agg(msg, dst, N_p)
            # reverse: post → commenter (comment text belongs to user→post only)
            agg_user = agg_user + self._mean_agg(self.W_comment_rev(h_post[dst]), src, N_u)

        if g.num_edges("user_comment_user") > 0:
            src, dst = g.edges(etype="user_comment_user")
            # A→B: B gets A's node features + A's comment + B's reply
            msg = self.W_ucu(h_user[src])
            if "user_comment_user" in ef:
                msg = msg + self.E_ucu(ef["user_comment_user"])
            if "reply_feat" in ef:
                msg = msg + self.E_reply(ef["reply_feat"])
            agg_user = agg_user + self._mean_agg(msg, dst, N_u)

        h_user_new = self.norm_user(F.relu(
            self.combine_user(torch.cat([h_user, agg_user], dim=1))))
        h_post_new = self.norm_post(F.relu(
            self.combine_post(torch.cat([h_post, agg_post], dim=1))))

        h_user_new = F.dropout(h_user_new, p=self.dropout, training=self.training)
        h_post_new = F.dropout(h_post_new, p=self.dropout, training=self.training)
        return h_user_new, h_post_new


class HinSAGENodeClassifier(nn.Module):
    def __init__(self, in_feats=768, edge_feats=768, hidden_size=256,
                 out_classes=9, dropout=0.5, n_layers=2,
                 post_feats=1536, **kwargs):
        super().__init__()
        self.user_proj = nn.Linear(in_feats,   hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)
        self.edge_proj = nn.Linear(edge_feats, hidden_size)
        self.layers    = nn.ModuleList([
            HinSAGELayer(hidden_size, dropout) for _ in range(n_layers)
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
        if "reply_feat" in g.edges["user_comment_user"].data:
            ef["reply_feat"] = self.edge_proj(g.edges["user_comment_user"].data["reply_feat"])

        for layer in self.layers:
            h_user, h_post = layer(g, h_user, h_post, ef)
        return self.classifier(h_user)