# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear, PNAConv


class KEPCE_GAT_PNA(torch.nn.Module):
    # def __init__(self):
    def __init__(self, pna_deg: torch.Tensor):
        super().__init__()
        # binary classification -> one score for each class
        output_size = 2  # TODO consider 3 options: output_size==1 OR output_size==2 OR softmax on this dim as well
        # node feature sizes for each layer
        gat_attention_heads = 4
        node_features_size = [5, 8, 16, 32, 32]
        # edge feature sizes for each layer
        # (edge_features are initialized as concatenation of src and dst node features,
        # as well as the edge weight and the counter_edge binary flag)
        hidden_edge_features_size = node_features_size[-1] * 2 + 2
        edge_features_size = [2, hidden_edge_features_size, 32, output_size]

        # message passing layers
        # TODO incorporate edge_weight information before the message passing layers
        # - sum, mean, std of edge_weight of incoming edges ?
        # - sum, mean, std of edge_weight of outcoming edges ?
        # - both ?
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.pna_conv = PNAConv(
            in_channels=node_features_size[0],
            out_channels=node_features_size[1],
            aggregators=aggregators,
            scalers=scalers,
            deg=pna_deg,
            edge_dim=edge_features_size[0],
        )
        self.gat_conv_1 = GATv2Conv(
            in_channels=(node_features_size[1]),
            edge_dim=edge_features_size[0],
            out_channels=int(node_features_size[2] / gat_attention_heads),
            heads=gat_attention_heads,
        )
        self.gat_conv_2 = GATv2Conv(
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
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]
        edge_features = torch.cat(
            (
                edge_weights.view(-1, 1),
                counter_edge.view(-1, 1),
            ),
            1,
        )

        # message passing phase
        node_features = self.pna_conv(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
        )
        node_features = self.gat_conv_1(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
        )
        node_features = F.relu(node_features)
        node_features = F.dropout(node_features, training=self.training)
        node_features = self.gat_conv_2(
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

        # src_node-wise softmax
        # TODO encapsulate to a GNN layer
        # column_indexes = torch.range(0, num_nodes - 1).repeat(num_edges, 1)
        # repeated_node_src = src.repeat(num_nodes, 1).t()
        # mask = (repeated_node_src == column_indexes).to(torch.int16)
        # edge_scores = edge_scores.repeat(1, num_nodes)
        # edge_scores = mask * edge_scores
        # edge_scores[edge_scores == 0.0] = float("-inf")
        # edge_scores = torch.softmax(input=edge_scores, dim=0)
        # edge_scores[edge_scores != edge_scores] = 0  # substitute NaN for 0.0
        # edge_scores = torch.sum(input=edge_scores, dim=1)
        # TODO test now kk
        # breakpoint()
        return edge_scores
