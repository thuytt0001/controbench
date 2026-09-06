from .rgcn       import RGCNNodeClassifier
from .han        import HANNodeClassifier
from .hinsage    import HinSAGENodeClassifier
from .h2gcn      import H2GCN
from .acmgnn     import ACMGNN
from .gcn_gfs    import GCNPlusGFS
from .hetero2net import Hetero2Net
# from .h2gformer  import H2GFormer
from .h2gformer import H2GFormerBase
from .shared     import build_homogeneous_user_graph, build_2hop_graph, build_khop_hetero_graph

# Hetero models (operate on DGL heterogeneous graph directly)
HETERO_MODELS = {
    "RGCN":       RGCNNodeClassifier,
    "HAN":        HANNodeClassifier,
    "HinSAGE":    HinSAGENodeClassifier,
    "Hetero2Net": Hetero2Net,
    # "H2GFormer":  H2GFormer,
    "H2GFormer": H2GFormerBase
}

# Homo models (need build_homogeneous_user_graph projection first)
HOMO_MODELS = {
    "H2GCN":      H2GCN,
    "ACMGNN":     ACMGNN,
    "GCNPlusGFS": GCNPlusGFS
}

ALL_MODELS = list(HETERO_MODELS) + list(HOMO_MODELS)
