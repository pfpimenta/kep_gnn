# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear


class KEPCE_GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # binary classification -> one score for each class
        output_size = 2
        # node feature sizes for each layer
        gat_attention_heads = 4
        node_features_size = [5, 8, 16, 32, 32]
        # edge feature sizes for each layer
        # (edge_features are initialized as concatenation of src and dst node features,
        # as well as the edge weight and the counter_edge binary flag)
        hidden_edge_features_size = node_features_size[-1] * 2 + 2
        edge_features_size = [2, hidden_edge_features_size, 32, output_size]

        # MLP to initalize node features
        self.init_node_features = Linear(
            in_channels=node_features_size[0],
            out_channels=node_features_size[1],
        )

        # message passing layers
        # TODO incorporate edge_weight information before the message passing layers
        # - sum, mean, std of edge_weight of incoming edges ?
        # - sum, mean, std of edge_weight of outcoming edges ?
        # - both ?
        self.conv_1 = GATv2Conv(
            in_channels=(node_features_size[1]),
            edge_dim=edge_features_size[0],
            out_channels=int(node_features_size[2] / gat_attention_heads),
            heads=gat_attention_heads,
        )
        self.conv_2 = GATv2Conv(
            in_channels=node_features_size[2],
            edge_dim=edge_features_size[0],
            out_channels=int(node_features_size[3] / gat_attention_heads),
            heads=gat_attention_heads,
        )

        # fully connected layers
        self.fully_connected_nodes = Linear(
            node_features_size[3], node_features_size[4]
        )
        self.fully_connected_edges_1 = Linear(
            edge_features_size[1], edge_features_size[2]
        )
        self.fully_connected_edges_2 = Linear(
            edge_features_size[2], edge_features_size[3]
        )

    def forward(self, data):
        edge_index = data.edge_index
        edge_weights = data.edge_weights
        counter_edge = data.counter_edge
        node_features = data.x
        edge_features = torch.cat(
            (
                edge_weights.view(-1, 1),
                counter_edge.view(-1, 1),
            ),
            1,
        )
        # import pdb

        # pdb.set_trace()

        # pass node featuires through a fully connected layer
        node_features = self.init_node_features(node_features.to(torch.float32))
        node_features = F.relu(node_features)

        # message passing phase
        node_features = self.conv_1(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
        )
        node_features = F.relu(node_features)
        node_features = F.dropout(node_features, training=self.training)
        node_features = self.conv_2(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
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
        edge_features = self.fully_connected_edges_1(edge_features)
        edge_scores = self.fully_connected_edges_2(edge_features)

        return edge_scores
