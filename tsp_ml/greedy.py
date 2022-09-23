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


def greedy(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
) -> Tensor:
    """TODO description"""
    solution = greedy_only_paths(
        edge_scores=edge_scores,
        edge_index=edge_index,
        node_types=node_types,
    )
    return solution
    # num_edges = len(edge_scores)
    # solution = torch.zeros_like(edge_scores[:, 0])
    # # TODO: subtraction or division ?
    # # edge_scores = edge_scores[:,1] - edge_scores[:,0]
    # edge_scores = edge_scores[:, 1] / edge_scores[:, 0]
    # while (edge_scores == torch.zeros_like(edge_scores)).all() == False:
    #     chosen_edge_index = torch.argmax(edge_scores)
    #     solution[chosen_edge_index] = 1

    #     # mask edges that have become unavailable
    #     # (already src/dst of a chosen edge)
    #     unavailable_edge_mask = get_unavailable_edge_mask(
    #         chosen_edge_index=chosen_edge_index, edge_index=edge_index
    #     )
    #     edge_scores[unavailable_edge_mask == 1] = 0
    # return solution


def greedy_only_cycles(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
) -> Tensor:
    # TODO returns a solution formed only by kidney exchange cycles
    pass


def greedy_choose_path(
    edge_index: Tensor,
    edge_scores: Tensor,
    ndd_out_edge_scores: Tensor,
    current_solution: Tensor,
) -> Tensor:

    src, _ = edge_index
    # choose max score out of NDD-out-edges
    chosen_edge_index = torch.argmax(ndd_out_edge_scores)
    current_solution[chosen_edge_index] = 1

    # mask edges that have become unavailable
    # (already src/dst of a chosen edge)
    unavailable_edge_mask = get_unavailable_edge_mask(
        chosen_edge_index=chosen_edge_index, edge_index=edge_index
    )
    edge_scores[unavailable_edge_mask == 1] = 0

    # get max score out-edge of next node in the path
    end_of_path = False
    while end_of_path == False:
        current_node_id = edge_index[1, chosen_edge_index]
        node_mask = (src == current_node_id).to(int)
        node_edge_scores = edge_scores * node_mask
        end_of_path = (node_edge_scores == torch.zeros_like(node_edge_scores)).all()
        if end_of_path:
            break  # TODO refactor

        # choose next edge
        chosen_edge_index = torch.argmax(node_edge_scores)
        current_solution[chosen_edge_index] = 1

        # mask edges that have become unavailable
        # (already src/dst of a chosen edge)
        unavailable_edge_mask = get_unavailable_edge_mask(
            chosen_edge_index=chosen_edge_index, edge_index=edge_index
        )
        edge_scores[unavailable_edge_mask == 1] = 0
    return current_solution, edge_scores


def greedy_only_paths(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
) -> Tensor:
    """
    repeats until there are no more valid edges
    coming from an NDD node:
    starts a path on a NDD node,
    then follows the best scoring edge from node to node
    until no more edges are available.
    """
    edge_scores = edge_scores[:, 1] / edge_scores[:, 0]
    src, _ = edge_index
    solution = torch.zeros_like(edge_scores)

    # select edges of NDD nodes
    is_node_ndd_mask = torch.Tensor([int(type == "NDD") for type in node_types])
    if torch.sum(is_node_ndd_mask) == 0:
        # raise ValueError("No NDD node found in instance.")
        print("No NDD node found in instance.")  # TODO
    ndd_node_ids = (is_node_ndd_mask == 1).nonzero(as_tuple=True)[0]
    ndd_out_edge_ids = torch.Tensor()
    for ndd_node_id in ndd_node_ids:
        edge_ids = (src == ndd_node_id).nonzero(as_tuple=True)[0]
        ndd_out_edge_ids = torch.cat((ndd_out_edge_ids, edge_ids))
    is_edge_ndd_mask = torch.zeros_like(edge_scores)
    is_edge_ndd_mask.scatter_(dim=0, index=ndd_out_edge_ids.to(torch.int64), value=1.0)

    # add paths to the solution until there are no more valid edges
    # coming from an NDD node
    ndd_out_edge_scores = edge_scores * is_edge_ndd_mask
    while (ndd_out_edge_scores == torch.zeros_like(ndd_out_edge_scores)).all() == False:
        solution, edge_scores = greedy_choose_path(
            edge_index=edge_index,
            edge_scores=edge_scores,
            ndd_out_edge_scores=ndd_out_edge_scores,
            current_solution=solution,
        )
        ndd_out_edge_scores = edge_scores * is_edge_ndd_mask

    return solution
