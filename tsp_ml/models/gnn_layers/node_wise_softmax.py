# -*- coding: utf-8 -*-
import torch
from torch import Tensor


def node_wise_softmax(
    edge_scores: Tensor, node_indexes: Tensor, num_nodes: int, device: torch.device
):
    """Applies an independent softmax for each group of edges that have the same
    source/destination node, depending on what is passed in node_indexes tensor.
    """
    # first, create a new dimension where each position corresponds to a node
    num_edges = len(edge_scores)
    column_indexes = torch.arange(start=0, end=num_nodes, device=device).repeat(
        num_edges, 1
    )
    repeated_node_src = node_indexes.repeat(num_nodes, 1).t()
    # breakpoint()
    mask = (repeated_node_src == column_indexes).to(torch.int16)
    edge_scores = edge_scores.repeat(num_nodes, 1).t()
    # place 0s where the edge is not associated to the node
    edge_scores = mask * edge_scores

    # put -1e10 where the edge is not associated to the node.
    # then, softmax will (almost) not consider this value when applied to the 0 dimension
    minus_inf = torch.full_like(
        input=edge_scores, fill_value=float(-1e10), device=device
    )
    edge_scores = torch.where(
        condition=(edge_scores != 0.0), input=edge_scores, other=minus_inf
    )

    edge_scores = torch.softmax(input=edge_scores, dim=0)

    # 'remove' the new dimension created in the beggining of the function
    edge_scores = torch.sum(input=edge_scores, dim=1)
    return edge_scores
