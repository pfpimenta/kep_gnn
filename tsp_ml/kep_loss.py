# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


def kep_loss(scores: Tensor, edge_weights: Tensor) -> Tensor:
    """sum of weights of edges NOT IN predicted solution over
    sum of weights of edges IN predicted solution"""
    pred = torch.argmax(scores, dim=1)
    solution_weight_sum = torch.sum(edge_weights * pred)
    total_solution_weight_sum = torch.sum(edge_weights)
    not_solution_weight_sum = total_solution_weight_sum - solution_weight_sum
    # TODO repensar a loss
    # loss = not_solution_weight_sum / (solution_weight_sum + 0.0000001)
    # loss = torch.log(not_solution_weight_sum / (solution_weight_sum + 0.0000001))
    loss = torch.log(
        (total_solution_weight_sum + 0.0000000001)
        / (solution_weight_sum + 0.0000000001)
    )
    loss = Variable(loss, requires_grad=True)

    # TODO fazer um mini script pra simular loss em varias situações

    # print(f"DEBUG solution_weight_sum: {solution_weight_sum}")
    # print(f"DEBUG not_solution_weight_sum: {not_solution_weight_sum}")
    # print(f"DEBUG total_solution_weight_sum: {total_solution_weight_sum}")
    # print(
    #     f"DEBUG not_solution_weight_sum / (solution_weight_sum + 0.0000001): {not_solution_weight_sum / (solution_weight_sum + 0.0000001)}"
    # )
    # print(f"DEBUG loss: {loss}")
    return loss


class KEPLoss(_Loss):
    """TODO description"""

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(KEPLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return kep_loss(scores=input, edge_weights=target)
