# -*- coding: utf-8 -*-
import torch
from models.gnn_layers.dtsp_edge_update import DTSP_EdgeUpdate
from torch.nn import RNN, Linear


class DTSP_GNN_Prates(torch.nn.Module):
    """Implementation of the GNN model described at the article titled
    'Learning to Solve NP-Complete Problems - A Graph Neural Network
    for Decision TSP', by Prates et al.
    The article can be found at https://arxiv.org/abs/1809.02721
    """

    def __init__(self):
        super().__init__()
        input_edge_features_size = 2  # edge_weight and input cost
        input_node_features_size = 2
        hidden_edge_features_sizes = [10, 10]
        # hidden_node_features_sizes = [10, 10]
        edge_message_size = 10
        node_message_size = 10

        # MLP edge features
        self.fully_connected_edges = Linear(
            in_features=input_edge_features_size,
            out_features=hidden_edge_features_sizes[0],
            # in_channels=input_edge_features_size,
            # out_channels=hidden_edge_features_size,
        )

        # node message passing
        # receive messages from adjacent edges
        self.update_edge_features = DTSP_EdgeUpdate(
            input_edge_feature_size=hidden_edge_features_sizes[0],
            input_node_feature_size=input_node_features_size,
            output_edge_feature_size=hidden_edge_features_sizes[1],
        )

        # self.edges_message_layer = Linear(
        #     in_features=hidden_edge_features_size,
        #     out_features=edge_message_size,
        #     # in_channels=hidden_edge_features_size,
        #     # out_channels=edge_message_size,
        # )

        # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        # self.rnn_seila_renomear = RNN(input_size=, layer_size)

        # edge message passing
        # receive messages from adjacent nodes
        self.nodes_message_layer = Linear(
            in_features=input_node_features_size,
            out_features=node_message_size,
            # in_channels=input_node_features_size,
            # out_channels=node_message_size,
        )

        # translate edge embeddings to logits probabilities
        # edge_logits =

        output_size = 2

        # DEBUG
        self.fully_connected_nodes = Linear(in_features=2, out_features=output_size)
        # self.fully_connected_nodes = Linear(in_channels=2, out_channels=output_size)

    def forward(self, data):
        edge_index = data.edge_index
        row, col = edge_index
        node_features = data.node_features
        edge_features = data.edge_features
        # pass edge_features through MLP
        edge_features = self.fully_connected_edges(edge_features)

        # import pdb

        # pdb.set_trace()
        # loop:
        # - update node features with messages from edges
        # - update edge features with messages from nodes
        edge_features = self.update_edge_features(
            edge_index=edge_index,
            edge_features=edge_features,
            node_features=node_features,
            # node_features[row], node_features[col], edge_features
        )

        # translate edge embeddings to logits probabilities
        # softmax
        # TODO
        # average logits and translate to probability:
        # pred = sigmoid(mean(edge_logits))
        # return pred

        # DEBUG
        return self.fully_connected_nodes(node_features)
