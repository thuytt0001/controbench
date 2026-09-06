"""
ACM-GNN — Revisiting Heterophily for Graph Neural Networks
Luan et al., NeurIPS 2022. arXiv:2206.09132
Official: https://github.com/SitaoLuan/ACM-GNN

Adaptive Channel Mixing: low-pass (aggregation), high-pass (diversification),
and identity channels mixed node-wisely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class ACMConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.W_lp = nn.Linear(in_feats, out_feats)
        self.W_hp = nn.Linear(in_feats, out_feats)
        self.W_id = nn.Linear(in_feats, out_feats)

        # per-channel scalar projections (att_vec_low/high/mlp in official code)
        self.att_lp  = nn.Linear(out_feats, 1, bias=False)
        self.att_hp  = nn.Linear(out_feats, 1, bias=False)
        self.att_id  = nn.Linear(out_feats, 1, bias=False)

        # learnable 3×3 mixing matrix with temperature T=3
        self.att_mix = nn.Parameter(torch.FloatTensor(3, 3))
        nn.init.uniform_(self.att_mix, -1.0 / 3.0 ** 0.5, 1.0 / 3.0 ** 0.5)
        self.T = 3

    def forward(self, g, h):
        with g.local_scope():
            deg  = g.in_degrees().float().clamp(min=1)
            norm = deg.pow(-0.5).unsqueeze(1)
            g.ndata["h"] = h * norm
            g.update_all(fn.copy_u("h", "m"), fn.sum("m", "agg"))
            agg = g.ndata["agg"] * norm

            lp_out = F.relu(self.W_lp(agg))
            hp_out = F.relu(self.W_hp(h - agg))
            id_out = F.relu(self.W_id(h))

            # sigmoid of per-channel scalar scores → (N, 3)
            scores = torch.sigmoid(torch.cat([
                self.att_lp(lp_out),
                self.att_hp(hp_out),
                self.att_id(id_out),
            ], dim=-1))

            # mix via learnable 3×3 matrix with temperature, then softmax
            w = torch.softmax(scores @ self.att_mix / self.T, dim=-1)  # (N, 3)

            return 3 * (w[:, 0:1] * lp_out + w[:, 1:2] * hp_out + w[:, 2:3] * id_out)


class ACMGNN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes,
                 n_layers=2, dropout=0.5):
        super().__init__()
        self.dropout    = dropout
        self.input_proj = nn.Linear(in_feats, hidden_size)
        self.convs      = nn.ModuleList([
            ACMConv(hidden_size, hidden_size) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, g, features):
        h = F.dropout(F.relu(self.input_proj(features)), p=self.dropout, training=self.training)
        for conv in self.convs:
            h = F.dropout(conv(g, h), p=self.dropout, training=self.training)
        return self.classifier(h)