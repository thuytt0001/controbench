"""
H2GCN — Beyond Homophily in Graph Neural Networks
Zhu et al., NeurIPS 2020. arXiv:2006.11468

Three key designs:
  1. Ego-neighbour separation
  2. Higher-order (2-hop) neighbourhood
  3. Concatenation of all intermediate representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class H2GCNConv(nn.Module):
    """
    Applies 1-hop and 2-hop adjacency independently to the same h.
    Matches original paper: h1 = A_1*h, h2 = A_2*h (not A*(A*h)).
    """
    def forward(self, g_1hop, g_2hop, h):
        def _agg(g, h):
            with g.local_scope():
                g.ndata["h"] = h
                g.update_all(fn.copy_u("h", "m"), fn.mean("m", "agg"))
                return g.ndata["agg"]
        return _agg(g_1hop, h), _agg(g_2hop, h)


class H2GCN(nn.Module):
    """
    H2GCN for ControBench.
    Operates on the user-user homogeneous projection (no self-loops).
    """
    def __init__(self, in_feats, hidden_size, num_classes,
                 n_layers=2, dropout=0.5):
        super().__init__()
        self.dropout  = dropout
        self.convs    = nn.ModuleList([H2GCNConv() for _ in range(n_layers)])
        self.ego_proj = nn.Linear(in_feats, hidden_size)
        # final dim: ego(H) + n_layers * [h1(H) + h2(H)]
        self.classifier = nn.Linear(hidden_size * (1 + 2 * n_layers), num_classes)

    def forward(self, g_1hop, g_2hop, features):
        h   = F.dropout(F.relu(self.ego_proj(features)), p=self.dropout, training=self.training)
        ego = h
        reps = [ego]
        for conv in self.convs:
            h1, h2 = conv(g_1hop, g_2hop, h)
            reps.extend([h1, h2])
            h = h1
        out = F.dropout(torch.cat(reps, dim=-1), p=self.dropout, training=self.training)
        return self.classifier(out)