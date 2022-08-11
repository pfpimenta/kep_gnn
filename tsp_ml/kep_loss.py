# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

EPSILON = 0.0000000001
INCOMING_EDGE_NODES_COEFFICIENT = 10
OUTCOMING_EDGE_NODES_COEFFICIENT = 10


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

    # print("\nDEBUG  incoming_edges_restriction_loss")
    # print(f"DEBUG num_nodes_in_solution: {num_nodes_in_solution}")
    # print(f"DEBUG num_unique_nodes_in_solution: {num_unique_nodes_in_solution}")
    # print(f"DEBUG loss before log: {loss}")
    loss = torch.log(loss)
    return loss


def kep_loss(scores: Tensor, edge_weights: Tensor, edge_index: Tensor) -> Tensor:
    """sum of weights of edges NOT IN predicted solution over
    sum of weights of edges IN predicted solution"""
    pred = torch.argmax(scores, dim=1)
    solution_weight_sum = torch.sum(edge_weights * pred)
    total_solution_weight_sum = torch.sum(edge_weights)
    # not_solution_weight_sum = total_solution_weight_sum - solution_weight_sum
    # loss = not_solution_weight_sum / (solution_weight_sum + 0.0000001)
    # loss = torch.log(not_solution_weight_sum / (solution_weight_sum + 0.0000001))
    kep_loss = torch.log(
        (total_solution_weight_sum + EPSILON) / (solution_weight_sum + EPSILON)
    )

    # add regularization terms modelling restrictions
    src, dst = edge_index
    outcoming_edges_loss = edges_restriction_loss(
        pred=pred,
        edge_node_ids=src,
    )
    incoming_edges_loss = edges_restriction_loss(
        pred=pred,
        edge_node_ids=dst,
    )
    loss = (
        kep_loss
        + OUTCOMING_EDGE_NODES_COEFFICIENT * outcoming_edges_loss
        + INCOMING_EDGE_NODES_COEFFICIENT * incoming_edges_loss
    )

    # print(f"\n\nDEBUG outcoming_edges_loss: {outcoming_edges_loss}")
    # print(f"DEBUG incoming_edges_loss: {incoming_edges_loss}")
    # print(f"DEBUG kep_loss: {kep_loss}")
    # print(f"DEBUG loss: {loss}")

    # print(f"DEBUG solution_weight_sum: {solution_weight_sum}")
    # print(f"DEBUG not_solution_weight_sum: {not_solution_weight_sum}")
    # print(f"DEBUG total_solution_weight_sum: {total_solution_weight_sum}")
    # print(
    #     f"DEBUG not_solution_weight_sum / (solution_weight_sum + 0.0000001): {not_solution_weight_sum / (solution_weight_sum + 0.0000001)}"
    # )
    # print(f"DEBUG loss: {loss}")
    loss = Variable(loss, requires_grad=True)
    return loss


class KEPLoss(_Loss):
    """TODO description"""

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(KEPLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, edge_index) -> Tensor:
        return kep_loss(scores=input, edge_weights=target, edge_index=edge_index)
