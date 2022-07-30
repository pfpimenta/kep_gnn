# -*- coding: utf-8 -*-
from typing import Optional

import torch
from models.gnn_layers.aggregations import AGGREGATIONS
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing


class DTSP_NodeUpdate(MessagePassing):
    """Layer for nodes to receive messages from adjacent edges"""

    def __init__(
        self,
        input_node_feature_size: int,
        output_node_feature_size: int,
        input_edge_feature_size: int,
        edge_message_feature_size: Optional[int] = None,
        aggregation: str = "add",
    ):
        super().__init__(aggr=aggregation)
        if edge_message_feature_size is None:
            self.edge_message_feature_size = input_node_feature_size
        else:
            self.edge_message_feature_size = edge_message_feature_size

        # MLP for generating the message features for each edge
        self.edge_message_mlp = Linear(
            in_features=input_edge_feature_size,
            out_features=self.edge_message_feature_size,
            # bias=False,
        )

        # aggregation of edge messages
        if aggregation in AGGREGATIONS.keys():
            self.aggregation = AGGREGATIONS[aggregation]
        else:
            raise ValueError(
                f"Given aggregation '{aggregation}' is not valid."
                f" Supported aggregations: {AGGREGATIONS.keys()}"
            )

        # MLP for the concatenation of the aggregated edge messages and the current node features
        node_mlp_input_size = input_node_feature_size + self.edge_message_feature_size
        self.node_mlp = Linear(
            in_features=node_mlp_input_size,
            out_features=output_node_feature_size,
            # bias=False,
        )

        # self.bias = Parameter(torch.Tensor(output_edge_feature_size))

        self.reset_parameters()

        # TODO https://discuss.pytorch.org/t/tensor-indexing-with-another-tensor/114485/5
        # f = torch.Tensor.__getitem__
        # g = lambda *a, **kw: f(*a, **kw)
        # torch.Tensor.__getitem__ = g

    def reset_parameters(self):
        self.edge_message_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        # self.bias.data.zero_()

    def forward(
        self,
        edge_index: Tensor,
        edge_features: Tensor,
        node_features: Tensor,
    ) -> Tensor:
        # edge_features has shape [num_edges, input_edge_feature_size]
        # node_features has shape [num_nodes, input_node_feature_size]
        # edge_index has shape [2, num_edges]
        num_nodes = node_features.shape[0]

        # compute messages for each edge
        edge_features = self.edge_message_mlp(edge_features)

        # select edge indexes for each node
        edge_indexes = []
        for i in range(num_nodes):
            edge_indexes.append((edge_index == i).nonzero(as_tuple=True)[1])

        # select edge messages for each node
        edge_messages = []
        for node_id in range(num_nodes):
            messages = torch.index_select(
                input=edge_features, dim=0, index=edge_indexes[node_id]
            )
            edge_messages.append(messages)

        # aggregate edge messages for each node
        aggregated_messages_list = []
        for node_id in range(num_nodes):
            msg_list = [t for t in edge_messages[node_id]]
            aggregated_messages = self.aggregation(feature_tensors=msg_list).reshape(
                (1, self.edge_message_feature_size)
            )
            aggregated_messages_list.append(aggregated_messages)

        # concatenate with the input edge features and compute output edge features
        aggregated_messages = torch.cat(aggregated_messages_list, dim=0)
        node_features = torch.cat((node_features, aggregated_messages), dim=1)
        node_features = self.node_mlp(node_features)
        return node_features
