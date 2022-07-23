# -*- coding: utf-8 -*-
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class DTSP_EdgeUpdate(MessagePassing):
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

        # MLP for the concatenation of the aggregated node messages and the edge current feature
        edge_mlp_input_size = input_edge_feature_size + node_message_feature_size
        self.edge_mlp = Linear(
            in_features=edge_mlp_input_size,
            out_features=output_edge_feature_size,
            # bias=False,
        )

        self.bias = Parameter(torch.Tensor(output_edge_feature_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.node_message_mlp.reset_parameters()
        self.edge_mlp.reset_parameters()
        self.bias.data.zero_()

    def edge_update(
        self, edge_features: Tensor, node_features: Tensor, edge_index: Tensor
    ) -> Tensor:
        r"""Computes or updates features for each edge in the graph.
        This function can take any argument as input which was initially passed
        to :meth:`edge_updater`.
        Furthermore, tensors passed to :meth:`edge_updater` can be mapped to
        the respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        # compute messages for each node
        node_features = self.node_message_mlp(node_features)
        # aggregate node messages
        # TODO for each edge, aggregate the 2 node messages
        aggregated_messages = None
        # concatenate with the input edge features and compute output edge features
        edge_features = self.edge_mlp(edge_features + aggregated_messages)
        return edge_features

    def forward(
        self, edge_index: Tensor, edge_features: Tensor, node_features: Tensor
    ) -> Tensor:
        # edge_features has shape [TODO, TODO]
        # node_features has shape [TODO, TODO]
        # edge_index has shape [2, num_edges]
        import pdb

        pdb.set_trace()

        # apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
