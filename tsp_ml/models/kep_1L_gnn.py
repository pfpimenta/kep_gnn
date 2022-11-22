# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from models.gnn_layers.node_wise_softmax import node_wise_softmax
from models.kep_gnn import KEP_GNN
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, Linear


class KEP_1L_GNN(KEP_GNN):
    """Simple GNN with only 1 GCN layer for KEP dataset.
    It may be useful for debugging and comparing with more complex GNNs.
    """

    def __init__(self, predict_method: str = "greedy_paths"):
        super().__init__()
        self.predict_method = predict_method

        # binary classification -> one score for each class
        output_size = 2  # TODO consider 3 options: output_size==1 OR output_size==2 OR softmax on this dim as well
        # node feature sizes for each layer
        node_features_size = [5, 16, 16]
        # edge feature sizes for each layer
        # (edge_features are initialized as concatenation of src and dst node features,
        # as well as the edge weight)
        hidden_edge_features_size = node_features_size[-1] * 2 + 1
        edge_features_size = [1, hidden_edge_features_size, output_size]

        # message passing layers
        self.gcn_layer = GCNConv(
            in_channels=node_features_size[0], out_channels=node_features_size[1]
        )

        # fully connected layers
        self.fully_connected_nodes = Linear(
            node_features_size[1], node_features_size[2]
        )
        self.fully_connected_edges = Linear(
            edge_features_size[1], edge_features_size[2]
        )

    def forward(self, data: Batch) -> Tensor:
        edge_index = data.edge_index
        node_features = data.x
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]
        edge_features = data.edge_weights.view(-1, 1)

        # message passing phase
        node_features = self.gcn_layer(
            x=node_features.to(torch.float32),
            edge_index=edge_index,
        )
        node_features = F.relu(node_features)
        node_features = F.dropout(node_features, training=self.training)
        node_features = self.fully_connected_nodes(node_features)
        node_features = F.relu(node_features)

        # calculate one score for each edge
        src, dst = data.edge_index
        edge_features = torch.cat(
            (
                edge_features,
                node_features[src],
                node_features[dst],
            ),
            1,
        )
        edge_scores = self.fully_connected_edges(edge_features)

        # src_node-wise softmax
        edge_scores[:, 0] = node_wise_softmax(
            edge_scores=edge_scores[:, 0], node_indexes=src, num_nodes=num_nodes
        )
        edge_scores[:, 1] = node_wise_softmax(
            edge_scores=edge_scores[:, 1], node_indexes=src, num_nodes=num_nodes
        )
        return edge_scores
