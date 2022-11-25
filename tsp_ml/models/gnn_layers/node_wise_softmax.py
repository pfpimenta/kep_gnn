# -*- coding: utf-8 -*-
import torch
from torch import Tensor


def node_wise_softmax(edge_scores: Tensor, node_indexes: Tensor, num_nodes: int):
    """Applies an independent softmax for each group of edges that have the same
    source/destination node, depending on what is passed in node_indexes tensor.
    """
    torch.autograd.set_detect_anomaly(True)  # DEBUG
    num_edges = len(edge_scores)
    column_indexes = torch.arange(0, num_nodes).repeat(num_edges, 1)
    repeated_node_src = node_indexes.repeat(num_nodes, 1).t()
    mask = (repeated_node_src == column_indexes).to(torch.int16)
    edge_scores = edge_scores.repeat(num_nodes, 1).t()
    edge_scores = mask * edge_scores
    edge_scores[edge_scores == 0.0] = float("-inf")
    edge_scores = torch.softmax(input=edge_scores, dim=0)
    edge_scores[edge_scores != edge_scores] = 0  # substitute NaN for 0.0
    edge_scores = torch.sum(input=edge_scores, dim=1)
    # TODO RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [5333, 300]], which is output 0 of SoftmaxBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
    return edge_scores
