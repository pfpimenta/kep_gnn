# -*- coding: utf-8 -*-
import torch
from models.gnn_layers import DTSP_EdgeUpdate, DTSP_NodeUpdate
from torch.nn import RNN, Linear
from torch_geometric.nn import global_mean_pool


class DTSP_GNN_Prates(torch.nn.Module):
    """Implementation of the GNN model described at the article titled
    'Learning to Solve NP-Complete Problems - A Graph Neural Network
    for Decision TSP', by Prates et al.
    The article can be found at https://arxiv.org/abs/1809.02721
    """

    def __init__(self, tmax: int = 10):
        super().__init__()
        input_edge_features_size = 2  # edge_weight and input cost
        init_node_features_size = 2
        hidden_edge_features_sizes = [13, 14]
        hidden_node_features_sizes = [10, 10]
        edge_message_size = 10
        node_message_size = 10

        # MLP to initalize node features
        self.init_node_features = Linear(
            in_features=1,
            out_features=init_node_features_size,
        )

        # MLP edge features
        self.fully_connected_edges = Linear(
            in_features=input_edge_features_size,
            out_features=hidden_edge_features_sizes[0],
        )

        # node message passing
        # (nodes receive messages from adjacent edges)
        self.update_node_features = DTSP_NodeUpdate(
            input_node_feature_size=init_node_features_size,
            input_edge_feature_size=hidden_edge_features_sizes[0],
            output_node_feature_size=hidden_node_features_sizes[0],
        )
        # TODO use this layer {tmax} times

        # TODO use RNN
        # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        # self.rnn_seila_renomear = RNN(input_size=, layer_size)

        # edge message passing
        # (edges receive messages from adjacent nodes)
        self.update_edge_features = DTSP_EdgeUpdate(
            input_edge_feature_size=hidden_edge_features_sizes[0],
            input_node_feature_size=hidden_node_features_sizes[0],
            output_edge_feature_size=hidden_edge_features_sizes[1],
        )
        # TODO use this layer {tmax} times

        # TODO
        # translate edge embeddings to logits probabilities
        # edge_logits =
        self.fully_connected_edges_logit_prob = Linear(
            in_features=hidden_edge_features_sizes[1],
            out_features=1,
        )

    def forward(self, data):
        edge_index = data.edge_index
        # row, col = edge_index
        # node_features = data.node_features
        edge_features = data.edge_features
        num_nodes = data.node_features.shape[0]
        # num_nodes  = max(max(data.edge_index.tolist())) + 1

        # initialize node features with the same values
        ones = torch.ones(size=(num_nodes, 1))
        node_features = self.init_node_features(ones)

        # pass edge_features through MLP
        edge_features = self.fully_connected_edges(edge_features)

        # loop:
        # TODO actually loop this part: tmax times
        # - update node features with messages from edges
        node_features = self.update_node_features(
            edge_index=edge_index,
            edge_features=edge_features,
            node_features=node_features,
        )
        # - update edge features with messages from nodes
        edge_features = self.update_edge_features(
            edge_index=edge_index,
            edge_features=edge_features,
            node_features=node_features,
            # node_features[row], node_features[col], edge_features
        )

        # translate edge embeddings to logits probabilities
        # TODO try to use softmax and 2 logits, so that CrossEntropyLoss
        # for binary classification can be used
        edge_features = self.fully_connected_edges_logit_prob(edge_features)

        # average logits
        row, col = edge_index
        edge_batch = data.batch[row]
        logits = global_mean_pool(x=edge_features, batch=edge_batch)
        # translate to probability
        prediction = torch.sigmoid(logits).flatten()
        return prediction
