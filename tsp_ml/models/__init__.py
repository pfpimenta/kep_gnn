# -*- coding: utf-8 -*-
from models.dtsp_gnn_prates import DTSP_GNN_Prates
from models.kep_gcn import KEP_GCN
from models.tsp_ggcn import TSP_GGCN
from models.tsp_ggcn_large import TSP_GGCN_large
from models.tsp_ggcn_v4_weights import TSP_GGCN_v4_weights

AVAILABLE_MODELS = {
    "TSP_GGCN": TSP_GGCN,
    "TSP_GGCN_large": TSP_GGCN_large,
    "TSP_GGCN_v4_weights": TSP_GGCN_v4_weights,
    "DTSP_GNN_Prates": DTSP_GNN_Prates,
    "KEP_GCN": KEP_GCN,
}
