# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear


class KEP_GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # binary classification -> one score for each class
        output_size = 2
        # node feature sizes for each layer
        node_features_size = [8, 16, 32, 32]
        # edge feature sizes for each layer
        # (edge_features are initialized as concatenation of src and dst node features, as well as the edge weight)
        edge_features_size = [node_features_size[-1] * 2 + 1, 32, output_size]

        # MLP to initalize node features
        self.init_node_features = Linear(
            in_channels=1,
            out_channels=node_features_size[0],
        )

        # message passing layers
        # TODO incorporate edge_weight information before the message passing layers
        # - sum, mean, std of edge_weight of incoming edges ?
        # - sum, mean, std of edge_weight of outcoming edges ?
        # - both ?
        self.conv_1 = GCNConv(node_features_size[0], node_features_size[1])
        self.conv_2 = GCNConv(node_features_size[1], node_features_size[2])

        # fully connected layers
        self.fully_connected_nodes = Linear(
            node_features_size[2], node_features_size[3]
        )
        self.fully_connected_edges_1 = Linear(
            edge_features_size[0], edge_features_size[1]
        )
        self.fully_connected_edges_2 = Linear(
            edge_features_size[1], edge_features_size[2]
        )

    def forward(self, data):
        edge_index = data.edge_index
        edge_weights = data.edge_weights
        num_nodes = data.num_nodes

        # initialize node features with the same values
        ones = torch.ones(size=(num_nodes, 1))
        node_features = self.init_node_features(ones)

        # message passing phase
        node_features = self.conv_1(node_features, edge_index)
        node_features = F.relu(node_features)
        node_features = F.dropout(node_features, training=self.training)
        node_features = self.conv_2(node_features, edge_index)
        node_features = F.relu(node_features)
        node_features = F.dropout(node_features, training=self.training)
        node_features = self.fully_connected_nodes(node_features)
        node_features = F.relu(node_features)

        # calculate one score for each edge
        src, dst = data.edge_index
        edge_features = torch.cat(
            (
                edge_weights.view(-1, 1),
                node_features[src],
                node_features[dst],
            ),
            1,
        )
        edge_features = self.fully_connected_edges_1(edge_features)
        edge_scores = self.fully_connected_edges_2(edge_features)

        return edge_scores
