# -*- coding: utf-8 -*-
import torch
from torch import Tensor


def get_unavailable_edge_mask(chosen_edge_index: int, edge_index: Tensor) -> Tensor:
    """Returns a mask with the same shape as 'edge_index',
    with 1s where the associated edge has become unavailable after chosing
    edge with id=='chosen_edge_index', and 0s otherwise."""
    src, dst = edge_index
    unavailable_edge_mask = torch.zeros_like(src)
    for node_list in [src, dst]:
        node_id = node_list[chosen_edge_index]
        unavailable_edge_mask += (node_list == node_id).to(int)
    # eliminate doubles
    unavailable_edge_mask[unavailable_edge_mask == 2] = 1
    # # repeat once along dim=1 to have the same shape as edge_index
    # unavailable_edge_mask = unavailable_edge_mask.repeat(2, 1)
    return unavailable_edge_mask


def greedy(edge_scores: Tensor, edge_index: Tensor):
    # TODO description
    num_edges = len(edge_scores)
    solution = torch.zeros_like(edge_scores[:, 0])
    # TODO: subtraction or division ?
    # edge_scores = edge_scores[:,1] - edge_scores[:,0]
    edge_scores = edge_scores[:, 1] / edge_scores[:, 0]
    while (edge_scores == torch.zeros_like(edge_scores)).all() == False:
        chosen_edge_index = torch.argmax(edge_scores)
        solution[chosen_edge_index] = 1

        # mask edges that have become unavailable
        # (already src/dst of a chosen edge)
        unavailable_edge_mask = get_unavailable_edge_mask(
            chosen_edge_index=chosen_edge_index, edge_index=edge_index
        )
        edge_scores[unavailable_edge_mask == 1] = 0
    return solution
