"""
GCN + GFS — Is Graph Convolution Always Beneficial For Every Feature?
ICLR 2025.

Graph Feature Separation: splits node features into GNN-favored (high TFI)
and GNN-disfavored (low TFI) components. GNN stream runs GCN on favored
features; MLP stream bypasses the graph for disfavored features.

TFI (Train-set Feature Informativeness): measured on graph-aggregated
features (A*X) via homo_raw (multi-edge graph with natural UCU self-loops)
vs labels. Consistent with training graph for correct feature selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, h):
        with g.local_scope():
            norm = g.in_degrees().float().clamp(min=1).pow(-0.5).unsqueeze(1)
            g.ndata["h"] = h * norm
            g.update_all(fn.copy_u("h", "m"), fn.sum("m", "agg"))
            return F.relu(self.linear(g.ndata["agg"] * norm))


class GCNPlusGFS(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes,
                 n_layers=2, dropout=0.5, split_ratio=0.5):
        super().__init__()
        self.dropout     = dropout
        self.split_ratio = split_ratio
        self.gnn_feat_idx = None
        self.mlp_feat_idx = None

        n_gnn = max(1, int(in_feats * split_ratio))
        n_mlp = in_feats - n_gnn

        self.gcn_layers = nn.ModuleList([GCNLayer(n_gnn, hidden_size)])
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_size, hidden_size))

        if n_mlp > 0:
            self.mlp_stream = nn.Sequential(
                nn.Linear(n_mlp, hidden_size), nn.ReLU(), nn.Dropout(dropout))
            combined = hidden_size * 2
        else:
            self.mlp_stream = None
            combined = hidden_size

        self.classifier = nn.Linear(combined, num_classes)
        self._n_gnn = n_gnn
        self._n_mlp = n_mlp

    def set_feature_split(self, g, features, labels, train_mask):
        """
        Call once before training to compute TFI and set feature split.
        Aggregates features through one hop (A*X) via homo_raw first,
        then ranks by Pearson correlation with labels on training nodes.
        Uses same graph as training for consistent feature selection.
        """
        with torch.no_grad():
            # one-hop aggregation: D^{-1/2} A D^{-1/2} X
            norm = g.in_degrees().float().clamp(min=1).pow(-0.5).unsqueeze(1)
            g.ndata["h"] = features * norm
            import dgl.function as _fn
            with g.local_scope():
                g.ndata["h"] = features * norm
                g.update_all(_fn.copy_u("h", "m"), _fn.sum("m", "agg"))
                X_agg = (g.ndata["agg"] * norm).cpu().numpy()

        X  = X_agg[train_mask.cpu().numpy()]
        y  = labels[train_mask].detach().cpu().numpy()
        nc = int(y.max()) + 1

        Y = np.zeros((len(y), nc))
        Y[np.arange(len(y)), y] = 1

        tfi = np.zeros(X.shape[1])
        for c in range(nc):
            yc    = Y[:, c] - Y[:, c].mean()
            Xc    = X - X.mean(0, keepdims=True)
            denom = np.std(X, axis=0) * np.std(Y[:, c]) + 1e-8
            tfi  += np.abs((Xc * yc[:, None]).mean(0) / denom)

        idx = np.argsort(-tfi)
        self.gnn_feat_idx = torch.tensor(idx[:self._n_gnn], dtype=torch.long)
        self.mlp_feat_idx = torch.tensor(idx[self._n_gnn:], dtype=torch.long)

    def forward(self, g, features):
        if self.gnn_feat_idx is None:
            raise RuntimeError("Call set_feature_split() before forward()")

        dev = features.device
        gnn_idx = self.gnn_feat_idx.to(dev)
        mlp_idx = self.mlp_feat_idx.to(dev)

        h_gnn = features[:, gnn_idx]
        for layer in self.gcn_layers:
            h_gnn = F.dropout(layer(g, h_gnn), p=self.dropout, training=self.training)

        if self.mlp_stream is not None and len(mlp_idx) > 0:
            return self.classifier(torch.cat([h_gnn, self.mlp_stream(features[:, mlp_idx])], dim=-1))
        return self.classifier(h_gnn)