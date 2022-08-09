# -*- coding: utf-8 -*-
from typing import Optional

import torch
from models.gnn_layers.aggregations import AGGREGATIONS
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing


class DTSP_EdgeUpdate(MessagePassing):
    """Layer for edges to receive messages from adjacent nodes"""

    def __init__(
        self,
        input_edge_feature_size: int,
        output_edge_feature_size: int,
        input_node_feature_size: int,
        node_message_feature_size: Optional[int] = None,
        aggregation: str = "add",
    ):
        super().__init__(aggr=aggregation)
        if node_message_feature_size is None:
            node_message_feature_size = input_edge_feature_size

        # MLP for generating the message features for each node
        self.node_message_mlp = Linear(
            in_features=input_node_feature_size,
            out_features=node_message_feature_size,
            # bias=False,
        )

        # aggregation of node messages
        if aggregation in AGGREGATIONS.keys():
            self.aggregation = AGGREGATIONS[aggregation]
        else:
            raise ValueError(
                f"Given aggregation '{aggregation}' is not valid."
                f" Supported aggregations: {AGGREGATIONS.keys()}"
            )

        # MLP for the concatenation of the aggregated node messages and the edge current feature
        edge_mlp_input_size = input_edge_feature_size + node_message_feature_size
        self.edge_mlp = Linear(
            in_features=edge_mlp_input_size,
            out_features=output_edge_feature_size,
            # bias=False,
        )
        # self.bias = Parameter(torch.Tensor(output_edge_feature_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.node_message_mlp.reset_parameters()
        self.edge_mlp.reset_parameters()
        # self.bias.data.zero_()

    def forward(
        self, edge_index: Tensor, edge_features: Tensor, node_features: Tensor
    ) -> Tensor:
        """
        edge_features has shape [num_edges, input_edge_feature_size]
        node_features has shape [num_nodes, input_node_feature_size]
        edge_index has shape [2, num_edges]
        """
        # compute messages for each node
        node_features = self.node_message_mlp(node_features)
        # aggregate node messages
        row, col = edge_index
        src_node_features = node_features[row]
        dst_node_features = node_features[col]
        aggregated_messages = self.aggregation(
            feature_tensors=[src_node_features, dst_node_features]
        )
        # concatenate with the input edge features and compute output edge features
        edge_features = torch.cat((edge_features, aggregated_messages), dim=1)
        edge_features = self.edge_mlp(edge_features)
        return edge_features

        # TODO delete:
        # apply a final bias vector.
        # out += self.bias
        # return out

    # def message(self, x_j, norm):
    #     # x_j has shape [E, out_channels]

    #     # Step 4: Normalize node features.
    #     return norm.view(-1, 1) * x_j
