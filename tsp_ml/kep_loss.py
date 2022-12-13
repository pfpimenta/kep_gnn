# -*- coding: utf-8 -*-
from typing import Optional

import torch
from greedy import greedy
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

EPSILON = 0.0000000001
KEP_LOSS_COEFFICIENT = 1.0
INCOMING_EDGE_NODES_COEFFICIENT = 0.0
OUTCOMING_EDGE_NODES_COEFFICIENT = 0.0


def edges_restriction_loss(pred: Tensor, edge_node_ids: Tensor) -> Tensor:
    """Loss regularization value modelling the restriction that
    each node must have at maximum 1 outcoming/incoming
    (depending on what tensor is passed) edge in the solution.
    'pred' and 'edge_node_ids' params must have the same shape:
    torch.Size([num_edges])
    """
    solution_edge_indexes = torch.nonzero(pred).flatten()
    solution_node_ids = torch.index_select(
        input=edge_node_ids, dim=0, index=solution_edge_indexes
    )
    unique_solution_node_ids = torch.unique(solution_node_ids)
    num_nodes_in_solution = solution_node_ids.shape[0]
    num_unique_nodes_in_solution = unique_solution_node_ids.shape[0]

    # invalid solution: num_nodes_in_solution > unique_num_nodes_in_solution --> loss > 0
    # valid solution: num_nodes_in_solution == unique_num_nodes_in_solution --> loss == 0
    loss = Tensor(
        [(num_nodes_in_solution + EPSILON) / (num_unique_nodes_in_solution + EPSILON)]
    )
    loss = torch.log(loss)
    return loss


def unsupervised_kep_loss(
    scores: Tensor,
    pred: Tensor,
    edge_weights: Tensor,
):
    """sum of weights of ALL edges over
    sum of weights of edges IN predicted solution.
    Args:
        pred: the predicted class for each edge
        edge_weights: the priority weight pre-assigned to each edge
    """

    masked_scores = pred * scores[:, 0]
    solution_weight_sum = torch.sum(edge_weights * masked_scores)
    total_solution_weight_sum = torch.sum(edge_weights)

    kep_loss = torch.log(
        (total_solution_weight_sum + EPSILON) / (solution_weight_sum + EPSILON)
    )
    return kep_loss


def kep_loss(
    scores: Tensor,
    pred: Tensor,
    edge_weights: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
    counter_edges: Optional[Tensor] = None,
) -> Tensor:
    """Computes the custom KEP loss for a given edge-wise prediction.
    Args:
        scores: the predicted score for each edge
        pred: the predicted class for each edge (0s and 1s)
        edge_weights: the priority weight pre-assigned to each edge
        edge_index: the pytorch_geometric edge_index tensor (see documentation)
        node_types: the type of each node ('PDP', 'NDD', or 'P')
        counter_edges: tensor 1 if the corresponding edge is a counter_edge
            i.e. if it was put there artificially for passing messages
            from dst to src nodes, 0 otherwise
    """
    kep_loss = unsupervised_kep_loss(
        scores=scores, pred=pred, edge_weights=edge_weights
    )

    # add regularization terms modelling restrictions
    # src, dst = edge_index
    # outcoming_edges_loss = edges_restriction_loss(
    #     pred=pred,
    #     edge_node_ids=src,
    # )
    # incoming_edges_loss = edges_restriction_loss(
    #     pred=pred,
    #     edge_node_ids=dst,
    # )

    loss = (
        KEP_LOSS_COEFFICIENT
        * kep_loss
        # + OUTCOMING_EDGE_NODES_COEFFICIENT * outcoming_edges_loss
        # + INCOMING_EDGE_NODES_COEFFICIENT * incoming_edges_loss
    )

    loss = Variable(loss, requires_grad=True)
    return loss


class KEPLoss(_Loss):
    """Loss class for the custom KEP loss"""

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()

    def forward(
        self,
        scores: Tensor,
        edge_weights: Tensor,
        edge_index: Tensor,
        pred: Tensor,
        node_types: Tensor,
        counter_edges: Optional[Tensor] = None,
    ) -> Tensor:
        loss = unsupervised_kep_loss(
            scores=scores, pred=pred, edge_weights=edge_weights
        )
        return loss
