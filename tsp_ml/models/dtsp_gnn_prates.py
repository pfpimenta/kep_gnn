# -*- coding: utf-8 -*-
import torch
from torch_geometric.nn import Linear


class DTSP_GNN_Prates(torch.nn.Module):
    """Implementation of the GNN model described at the article titled
    'Learning to Solve NP-Complete Problems - A Graph Neural Network
    for Decision TSP', by Prates et al.
    The article can be found at https://arxiv.org/abs/1809.02721
    """

    def __init__(self):
        super().__init__()
        input_edge_features_size = 2
        hidden_edge_features_size = 10

        # MLP edge features
        self.fully_connected_nodes = Linear(
            in_channels=input_edge_features_size, out_channels=hidden_edge_features_size
        )

        # node message passing
        # receive messages from adjacent edges

        # edge message passing
        # receive messages from adjacent nodes

        # translate edge embeddings to logits probabilities
        # edge_logits =

        output_size = 2

        # DEBUG
        self.fully_connected_nodes = Linear(in_channels=2, out_channels=output_size)

    def forward(self, data):
        edge_index = data.edge_index
        row, col = edge_index
        node_features = data.node_features
        edge_features = data.edge_features

        import pdb

        pdb.set_trace()
        # pass edge_features through MLP
        # loop:
        # - update node features with messages from edges
        # - update edge features with messages from nodes
        # edge_features = self.update_edge_features(
        #     node_features[row], node_features[col], edge_features
        # )
        # translate edge embeddings to logits probabilities

        # average logits and translate to probability:
        # pred = sigmoid(mean(edge_logits))
        # return pred

        # DEBUG
        return self.fully_connected_nodes(node_features)
