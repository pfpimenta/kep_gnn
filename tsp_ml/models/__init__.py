# -*- coding: utf-8 -*-
from models.dtsp_gnn_prates import DTSP_GNN_Prates
from models.greedy_cycles_model import GreedyCyclesModel
from models.greedy_paths_model import GreedyPathsModel
from models.kep_gat_pna_ce import KEP_GAT_PNA_CE
from models.kep_gcn import KEP_GCN
from models.kepce_gat import KEPCE_GAT
from models.kepce_gat_pna import KEPCE_GAT_PNA
from models.kepce_gcn import KEPCE_GCN
from models.tsp_ggcn import TSP_GGCN
from models.tsp_ggcn_v4_weights import TSP_GGCN_v4_weights

AVAILABLE_MODELS = {
    "TSP_GGCN": TSP_GGCN,
    "TSP_GGCN_v4_weights": TSP_GGCN_v4_weights,
    "DTSP_GNN_Prates": DTSP_GNN_Prates,
    "KEP_GCN": KEP_GCN,
    "KEP_GAT_PNA_CE": KEP_GAT_PNA_CE,
    "KEPCE_GCN": KEPCE_GCN,
    "KEPCE_GAT": KEPCE_GAT,
    "KEPCE_GAT_PNA": KEPCE_GAT_PNA,
    "GreedyPathsModel": GreedyPathsModel,
    "GreedyCyclesModel": GreedyCyclesModel,
}
