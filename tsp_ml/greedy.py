# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

EPSILON = 1e-10


def get_ndd_edge_mask(
    edge_index: Tensor,
    node_types: Tensor,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tensor:
    # select edges of NDD nodes
    # TODO use get_node_type_edge_mask (DRY refactor)
    src, _ = edge_index
    is_node_ndd_mask = torch.tensor(
        [int(type == "NDD") for type in node_types], device=device
    )
    if torch.sum(is_node_ndd_mask) == 0:
        # raise ValueError("No NDD node found in instance.")
        print("No NDD node found in instance.")  # TODO
    ndd_node_ids = (is_node_ndd_mask == 1).nonzero(as_tuple=True)[0]
    ndd_out_edge_ids = torch.tensor(data=[], device=device)
    for ndd_node_id in ndd_node_ids:
        edge_ids = (src == ndd_node_id).nonzero(as_tuple=True)[0]
        ndd_out_edge_ids = torch.cat((ndd_out_edge_ids, edge_ids))
    is_edge_ndd_mask = torch.zeros_like(src, device=device)
    is_edge_ndd_mask.scatter_(dim=0, index=ndd_out_edge_ids.to(torch.int64), value=1.0)
    return is_edge_ndd_mask


def get_node_type_edge_mask(
    edge_index: Tensor,
    node_types: Tensor,
    node_type: str,
    direction: str = "src",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tensor:
    """Returns a mask for edge_scores with 1s where the edge's
    source/destination/both/either node is of type node_type
    and 0s elsewhere."""
    src, dst = edge_index
    is_node_pdp_mask = torch.Tensor(
        [int(type == node_type) for type in node_types], device=device
    )
    if torch.sum(is_node_pdp_mask) == 0:
        raise ValueError(f"No {node_type} node found in instance.")
        # print(f"No {node_type} node found in instance.")
    pdp_node_ids = (is_node_pdp_mask == 1).nonzero(as_tuple=True)[0]
    edge_ids = torch.tensor(data=[], device=device)
    for pdp_node_id in pdp_node_ids:
        if direction == "src":
            src_edge_ids = (src == pdp_node_id).nonzero(as_tuple=True)[0]
            edge_ids = torch.cat((edge_ids, src_edge_ids))
        elif direction == "dst":
            dst_edge_ids = (dst == pdp_node_id).nonzero(as_tuple=True)[0]
            edge_ids = torch.cat((edge_ids, dst_edge_ids))
        else:
            ValueError(f"Invalid 'direction' parameter: {direction}")
    edge_mask = torch.zeros_like(src, device=device)
    edge_mask.scatter_(dim=0, index=edge_ids.to(torch.int64), value=1.0)
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
    return unavailable_edge_mask


def greedy(
    kep_instance: Data,
    # edge_scores: Tensor,
    # edge_index: Tensor,
    # node_types: Tensor,
    greedy_algorithm: str = "greedy_paths",
    cycle_path_size_limit: Optional[int] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tensor:
    """TODO description
    greedy_algorithm, by default, is set to 'greedy_paths' its options are:
    'greedy_cycles_and_paths', 'greedy_paths', 'greedy_cycles',
    'greedy_1_path', and 'greedy_1_cycle'.
    """
    if greedy_algorithm == "greedy_cycles_and_paths":
        solution = greedy_cycles_and_paths(
            kep_instance=kep_instance,
            # edge_scores=edge_scores,
            # edge_index=edge_index,
            # node_types=node_types,
            cycle_path_size_limit=cycle_path_size_limit,
            device=device,
        )
    elif greedy_algorithm == "greedy_paths":
        solution = greedy_paths(
            kep_instance=kep_instance,
            # edge_scores=edge_scores,
            # edge_index=edge_index,
            # node_types=node_types,
            path_size_limit=cycle_path_size_limit,
            device=device,
        )
    elif greedy_algorithm == "greedy_cycles":
        solution = greedy_cycles(
            kep_instance=kep_instance,
            # edge_scores=edge_scores,
            # edge_index=edge_index,
            # node_types=node_types,
            cycle_size_limit=cycle_path_size_limit,
            device=device,
        )
    elif greedy_algorithm == "greedy_1_path":
        solution = greedy_1_path(
            kep_instance=kep_instance,
            # edge_scores=edge_scores,
            # edge_index=edge_index,
            # node_types=node_types,
            path_size_limit=cycle_path_size_limit,
            device=device,
        )
    elif greedy_algorithm == "greedy_1_cycle":
        solution = greedy_1_cycle(
            kep_instance=kep_instance,
            # edge_scores=edge_scores,
            # edge_index=edge_index,
            # node_types=node_types,
            cycle_size_limit=cycle_path_size_limit,
            device=device,
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
    device: torch.device,
    cycle_path_size_limit: Optional[int] = None,
) -> Tensor:
    raise NotImplemented()  # TODO


def greedy_cycles(
    kep_instance: Data,
    # edge_scores: Tensor,
    # edge_index: Tensor,
    # node_types: Tensor,
    device: torch.device,
    cycle_size_limit: Optional[int] = None,
) -> Tensor:
    # TODO description: returns a solution formed only by kidney exchange cycles
    # edge_scores = edge_scores[:, 0] / (edge_scores[:, 1] + EPSILON)
    # edge_scores = edge_scores[:, 0]
    # check if there are scores to use
    if "scores" in kep_instance.keys:
        # print('DEBUG greedy_cycles using edge scores !')
        edge_scores = kep_instance.scores
        assert kep_instance.scores.shape[0] == kep_instance.edge_weights.shape[0]
        edge_scores = edge_scores[:, 0] / (edge_scores[:, 1] + EPSILON)
        # breakpoint()
    else:
        # print('DEBUG greedy_cycles using edge weights!!!!')
        edge_scores = kep_instance.edge_weights

    node_types = kep_instance.type[0]

    solution = torch.zeros_like(edge_scores)

    # select only PDP->PDP edges
    is_pdp_src_edge_mask = get_node_type_edge_mask(
        edge_index=kep_instance.edge_index,
        node_types=node_types,
        node_type="PDP",
        direction="src",
        device=device,
    )
    is_pdp_dst_edge_mask = get_node_type_edge_mask(
        edge_index=kep_instance.edge_index,
        node_types=node_types,
        node_type="PDP",
        direction="dst",
        device=device,
    )
    is_pdp_edge_mask = is_pdp_src_edge_mask * is_pdp_dst_edge_mask
    edge_scores = edge_scores * is_pdp_edge_mask

    # add cycles to the solution until there are no more valid edges
    # coming from an NDD node
    num_cycles_found = 0
    while (edge_scores == torch.zeros_like(edge_scores)).all() == False:
        solution, edge_scores, dead_end = greedy_choose_cycle(
            edge_index=kep_instance.edge_index,
            edge_scores=edge_scores,
            current_solution=solution,
            cycle_size_limit=cycle_size_limit,
        )
        if dead_end:
            # print("DEBUG dead_end was found.")
            break
        else:
            num_cycles_found += 1

    # # DEBUG:
    kep_instance.pred = solution
    from kep_evaluation import evaluate_kep_instance_prediction

    eval_dict = evaluate_kep_instance_prediction(kep_instance)
    if eval_dict["is_solution_valid"] == 0:
        # TODO DEBUG
        breakpoint()
    return solution


# TODO mudar params pra
# kep_instance: Data,
# device: torch.device,
# path_size_limit: Optional[int] = None,
def greedy_paths(
    kep_instance: Data,
    # edge_scores: Tensor,
    # edge_index: Tensor,
    # node_types: Tensor,
    device: torch.device,
    path_size_limit: Optional[int] = None,
) -> Tensor:
    """
    repeats until there are no more valid edges
    coming from an NDD node:
    starts a path on a NDD node,
    then follows the best scoring edge from node to node
    until no more edges are available.
    """
    # check if there are scores to use
    if "scores" in kep_instance.keys:
        edge_scores = kep_instance.scores
        assert kep_instance.scores.shape[0] == kep_instance.edge_weights.shape[0]
        edge_scores = edge_scores[:, 0] / (edge_scores[:, 1] + EPSILON)
    else:
        edge_scores = kep_instance.edge_weights

    node_types = kep_instance.type[0]
    solution = torch.zeros_like(edge_scores)
    # solution.requires_grad = True  # DEBUG

    # select edges of NDD nodes
    is_edge_ndd_mask = get_ndd_edge_mask(
        edge_index=kep_instance.edge_index, node_types=node_types, device=device
    )

    # add paths to the solution until there are no more valid edges
    # coming from an NDD node
    ndd_out_edge_scores = edge_scores * is_edge_ndd_mask
    while not (ndd_out_edge_scores == torch.zeros_like(ndd_out_edge_scores)).all():
        solution, edge_scores = greedy_choose_path(
            edge_index=kep_instance.edge_index,
            edge_scores=edge_scores,
            ndd_out_edge_scores=ndd_out_edge_scores,
            current_solution=solution,
            path_size_limit=path_size_limit,
        )
        ndd_out_edge_scores = edge_scores * is_edge_ndd_mask
    return solution


def greedy_1_path(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
    device: torch.device,
    path_size_limit: Optional[int] = None,
) -> Tensor:
    # TODO testar !!! validar
    edge_scores = edge_scores[:, 0] / (edge_scores[:, 1] + EPSILON)

    # select edges of NDD nodes
    is_edge_ndd_mask = get_ndd_edge_mask(
        edge_index=edge_index, node_types=node_types, device=device
    )
    ndd_out_edge_scores = edge_scores * is_edge_ndd_mask

    solution = torch.zeros_like(edge_scores)
    solution, edge_scores = greedy_choose_path(
        edge_index=edge_index,
        edge_scores=edge_scores,
        ndd_out_edge_scores=ndd_out_edge_scores,
        current_solution=solution,
    )
    return solution


def greedy_1_cycle(
    edge_scores: Tensor,
    edge_index: Tensor,
    node_types: Tensor,
    device: torch.device,
    cycle_size_limit: Optional[int] = None,
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
    cycle_size_limit: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    # TODO description
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

    # mask edges that have become unavailable:
    # the ones that have first_node_id as src
    edge_scores[src == first_node_id] = 0

    # get max score out-edge of next node in the cycle.
    # if next node is already in the cycle, close it
    dead_end = False
    while not current_node_id in cycle_node_ids:
        cycle_node_ids.append(current_node_id.item())

        # if cycle size limit is passed,
        # then return unchanged solution
        if isinstance(cycle_size_limit, int):
            if len(cycle_node_ids) > cycle_size_limit:
                dead_end = True
                return unchanged_solution, unchanged_edge_scores, dead_end

        current_node_edge_mask = (src == current_node_id).to(int)
        next_edge_scores = edge_scores * current_node_edge_mask

        # mask edges that become unavailable:
        # the ones that have current_node_id as src
        edge_scores[src == current_node_id] = 0

        # if unable to finish cycle (a dead end was reached),
        # then return unchanged solition
        dead_end = (next_edge_scores == torch.zeros_like(next_edge_scores)).all()
        if dead_end:
            dead_end = True
            return unchanged_solution, unchanged_edge_scores, dead_end

        # choose next edge
        chosen_edge_index = torch.argmax(next_edge_scores)
        current_solution[chosen_edge_index] = 1
        current_node_id = edge_index[1, chosen_edge_index]

    # mask edges that have become unavailable:
    # the ones that have current_node_id as src
    edge_scores[src == current_node_id] = 0

    # close cycle: delete nodes left before the node where the cycle ended
    cycle_beggining = cycle_node_ids.index(current_node_id)
    nodes_before_cycle = cycle_node_ids[:cycle_beggining]
    nodes_before_cycle_mask = torch.zeros_like(current_solution)
    for node_id in nodes_before_cycle:
        node_mask = (src == node_id).to(int)
        node_mask += (dst == node_id).to(int)
        nodes_before_cycle_mask += node_mask
    # only 0s and 1s:
    nodes_before_cycle_mask = nodes_before_cycle_mask.to(bool).to(int)
    nodes_before_cycle_mask = 1 - nodes_before_cycle_mask
    current_solution = current_solution * nodes_before_cycle_mask

    return current_solution, edge_scores, dead_end


def greedy_choose_path(
    edge_index: Tensor,
    edge_scores: Tensor,
    ndd_out_edge_scores: Tensor,
    current_solution: Tensor,
    path_size_limit: Optional[int] = None,
) -> Tensor:

    src, _ = edge_index
    # choose max score out of NDD-out-edges
    chosen_edge_index = torch.argmax(ndd_out_edge_scores)
    current_solution[chosen_edge_index] = 1
    current_path_size = 1

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
            # print(f"DEBUG end_of_path reached: {torch.sum(current_solution)}")
            break  # TODO refactor?

        # if path size limit is reached,
        # then return current solution
        if isinstance(path_size_limit, int):
            if current_path_size >= path_size_limit:
                break  # TODO refactor?

        # choose next edge
        chosen_edge_index = torch.argmax(node_edge_scores)
        current_solution[chosen_edge_index] = 1
        current_path_size += 1

        # mask edges that have become unavailable
        # (already src/dst of a chosen edge)
        unavailable_edge_mask = get_unavailable_edge_mask(
            chosen_edge_index=chosen_edge_index, edge_index=edge_index
        )
        edge_scores[unavailable_edge_mask == 1] = 0

    return current_solution, edge_scores
