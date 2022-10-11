# -*- coding: utf-8 -*-
from typing import Tuple

import torch
from torch import Tensor

EPSILON = 1e-10


def get_node_type_edge_mask(
    edge_index: Tensor,
    node_types: Tensor,
    node_type: str,
    direction: str = "src",
) -> Tensor:
    """Returns a mask for edge_scores
    with 1s where the edge's source/destination/both/either node is of type node_type
    and 0s elsewhere.
    TODO? optimize:"""
    src, dst = edge_index
    is_node_pdp_mask = torch.Tensor([int(type == node_type) for type in node_types])
    if torch.sum(is_node_pdp_mask) == 0:
        raise ValueError(f"No {node_type} node found in instance.")
        # print(f"No {node_type} node found in instance.")
    pdp_node_ids = (is_node_pdp_mask == 1).nonzero(as_tuple=True)[0]
    edge_ids = torch.Tensor()
    for pdp_node_id in pdp_node_ids:
        if direction == "src":
            src_edge_ids = (src == pdp_node_id).nonzero(as_tuple=True)[0]
            edge_ids = torch.cat((edge_ids, src_edge_ids))
        elif direction == "dst":
            dst_edge_ids = (dst == pdp_node_id).nonzero(as_tuple=True)[0]
            edge_ids = torch.cat((edge_ids, dst_edge_ids))
        else:
            ValueError(f"Invalid 'direction' parameter: {direction}")
    edge_mask = torch.zeros_like(src)
    edge_mask.scatter_(dim=0, index=edge_ids.to(torch.int64), value=1.0)
    # breakpoint()
    return edge_mask


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
    greedy_algorithm: str = "greedy_paths",
) -> Tensor:
    """TODO description"""
    if greedy_algorithm == "greedy_cycles_and_paths":
        solution = greedy_cycles_and_paths(
            edge_scores=edge_scores,
            edge_index=edge_index,
            node_types=node_types,
        )
    elif greedy_algorithm == "greedy_paths":
        solution = greedy_paths(
            edge_scores=edge_scores,
            edge_index=edge_index,
            node_types=node_types,
        )
    elif greedy_algorithm == "greedy_cycles":
        solution = greedy_cycles(
            edge_scores=edge_scores,
            edge_index=edge_index,
            node_types=node_types,
        )
    elif greedy_algorithm == "greedy_1_path":
        solution = greedy_1_path(
            edge_scores=edge_scores,
            edge_index=edge_index,
            node_types=node_types,
        )
    elif greedy_algorithm == "greedy_1_cycle":
        solution = greedy_1_cycle(
            edge_scores=edge_scores,
            edge_index=edge_index,
            node_types=node_types,
        )
    else:
        raise ValueError(
            f"Greedy algorithm '{greedy_algorithm}' not found."
            " Please check if you typed the name correctly."
        )
    return solution


def greedy_cycles_and_paths(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
) -> Tensor:
    raise NotImplemented()  # TODO


def greedy_cycles(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
) -> Tensor:
    # TODO description: returns a solution formed only by kidney exchange cycles
    edge_scores = edge_scores[:, 0] / (edge_scores[:, 1] + EPSILON)
    solution = torch.zeros_like(edge_scores)

    # select only PDP->PDP edges
    is_pdp_src_edge_mask = get_node_type_edge_mask(
        edge_index=edge_index,
        node_types=node_types,
        node_type="PDP",
        direction="src",
    )
    is_pdp_dst_edge_mask = get_node_type_edge_mask(
        edge_index=edge_index,
        node_types=node_types,
        node_type="PDP",
        direction="dst",
    )
    is_pdp_edge_mask = is_pdp_src_edge_mask * is_pdp_dst_edge_mask
    edge_scores = edge_scores * is_pdp_edge_mask
    oi = edge_scores * is_pdp_edge_mask

    # add cycles to the solution until there are no more valid edges
    # coming from an NDD node
    while (edge_scores == torch.zeros_like(edge_scores)).all() == False:
        solution, edge_scores, dead_end = greedy_choose_cycle(
            edge_index=edge_index,
            edge_scores=edge_scores,
            current_solution=solution,
        )
        breakpoint()
        if dead_end:
            break
    return solution


def greedy_paths(
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
    edge_scores = edge_scores[:, 0] / (edge_scores[:, 1] + EPSILON)
    src, _ = edge_index
    solution = torch.zeros_like(edge_scores)

    # select edges of NDD nodes
    # TODO use get_node_type_edge_mask (DRY refactor)
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


def greedy_1_path(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
) -> Tensor:
    raise NotImplemented()  # TODO


def greedy_1_cycle(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
) -> Tensor:
    raise NotImplemented()  # TODO


def greedy_get_next_edge(
    edge_scores: Tensor,
    edge_index: Tensor,
    current_solution: Tensor,
) -> Tuple[Tensor, Tensor]:
    # TODO description
    chosen_edge_index = torch.argmax(edge_scores)
    current_solution[chosen_edge_index] = 1
    current_node_id = edge_index[1, chosen_edge_index]
    return current_solution, current_node_id


def greedy_choose_cycle(
    edge_index: Tensor,
    edge_scores: Tensor,
    current_solution: Tensor,
) -> Tuple[Tensor, Tensor]:
    # TODO
    src, dst = edge_index
    cycle_node_ids = []

    # keep a copy of the solution and edge_scores in case no cycles are found
    unchanged_solution = current_solution.clone()
    unchanged_edge_scores = edge_scores.clone()

    # TODO loop: try max score, then 2nd max score, etc.

    # choose max score out of PDP-out-edges
    chosen_edge_index = torch.argmax(edge_scores)
    first_node_id = src[chosen_edge_index]
    current_node_id = dst[chosen_edge_index]
    current_solution[chosen_edge_index] = 1
    cycle_node_ids.append(first_node_id.item())
    print(
        f"DEBUG first edge: id: {chosen_edge_index}, nodes: {edge_index[:, chosen_edge_index]}"
    )

    # mask edges that have become unavailable:
    # the ones that have first_node_id as src
    edge_scores[src == first_node_id] = 0

    # get max score out-edge of next node in the cycle.
    # if next node is already in the cycle, close it
    # breakpoint()
    dead_end = False
    while not current_node_id in cycle_node_ids:
        cycle_node_ids.append(current_node_id.item())

        current_node_edge_mask = (src == current_node_id).to(int)
        next_edge_scores = edge_scores * current_node_edge_mask

        # if unable to finish cycle (a dead end was reached), return unchanged solition
        dead_end = (next_edge_scores == torch.zeros_like(next_edge_scores)).all()
        if dead_end:
            print("\n\n\nDEBUG dead_end !\n\n")
            dead_end = True
            return unchanged_solution, unchanged_edge_scores, dead_end

        # choose next edge
        chosen_edge_index = torch.argmax(next_edge_scores)
        current_solution[chosen_edge_index] = 1
        current_node_id = edge_index[1, chosen_edge_index]
        print(
            f"DEBUG chosen edge: id: {chosen_edge_index}, nodes: {edge_index[:, chosen_edge_index]}"
        )

        # mask edges that have become unavailable:
        # the ones that have current_node_id as src
        edge_scores[src == current_node_id] = 0
        # breakpoint()

    breakpoint()
    # TODO

    # breakpoint()
    # close cycle
    # add last edge
    # current_node_edge_mask = (src == current_node_id).to(int)
    # next_edge_scores = edge_scores * (1 - current_node_edge_mask)
    # chosen_edge_index = torch.argmax(next_edge_scores)
    # current_solution[chosen_edge_index] = 1
    # current_node_id = edge_index[1, chosen_edge_index]

    # TODO
    # remove nodes added before the one where the loop closed

    return current_solution, edge_scores, dead_end


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
