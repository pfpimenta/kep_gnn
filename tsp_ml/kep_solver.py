# -*- coding: utf-8 -*-
# functions for solving KEP instances with integer programming
import time
from typing import List

import torch
from pycsp3 import *  # TODO tentar importar só o necessario
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj


def solve_kep_recursive(kep_instance: Data) -> List[int]:
    """Returns the received instance, but with a new Tensor 'pred'
    containing the optimal solution. Uses the 'Recusive Algorithm'
    formulation presented at https://www.pnas.org/doi/epdf/10.1073/pnas.1421853112
    """
    clear()  # clears vars, constraints, etc from previous PyCSP3 executions
    timestamp_start = time.time()

    # get adjecency matrix
    adj_matrix_tensor = to_dense_adj(kep_instance.edge_index)
    adj_matrix = adj_matrix_tensor[0].numpy()
    # get edge weights (multiply weights by 10000 to force them to integers)
    edge_weights = (kep_instance.edge_weights * 10000).to(torch.int32).numpy()
    # get node types (PDP, NDD, or P)
    node_types = kep_instance.type
    ndd_nodes = [i for i in range(len(node_types)) if node_types[i] == "NDD"]
    pdp_nodes = [i for i in range(len(node_types)) if node_types[i] == "PDP"]
    p_nodes = [i for i in range(len(node_types)) if node_types[i] == "P"]
    num_edges = int(edge_weights.shape[0])
    num_nodes = int(adj_matrix.shape[0])
    in_edges = [
        torch.where(kep_instance.edge_index[1] == v)[0].tolist()
        for v in range(num_nodes)
    ]
    out_edges = [
        torch.where(kep_instance.edge_index[0] == v)[0].tolist()
        for v in range(num_nodes)
    ]
    # print(f"num_nodes: {num_nodes}")
    # print(f"num_edges: {num_edges}")
    # print(f"node_types: {node_types}")
    # print(f"edge_weights: {edge_weights}")
    # print(f"kep_instance.edge_index:\n {kep_instance.edge_index}")
    # print(f"in_edges: {in_edges}")
    # print(f"out_edges: {out_edges}")

    # CSP variables
    domain = 0, 1
    y = VarArray(size=num_edges, dom=domain)
    f_in = VarArray(size=num_nodes, dom=range(num_nodes))
    f_out = VarArray(size=num_nodes, dom=range(num_nodes))
    # print("\nDomain of y[0]: ", y[0].dom)
    # print("Domain of f_in[0]: ", f_in[0].dom)
    # print("Domain of f_out[0]: ", f_out[0].dom)

    # constraints
    # constraint 1a: sum(y[e], paratodo e em in(v)) = f_in[v], paratodo v em V
    constraint_1a = [
        (Sum([y[e] for e in in_edges[v]]) == f_in[v]) for v in range(num_nodes)
    ]
    # constraint 1b
    # sum(y[e], paratodo e em out(v)) = f_out[v], paratodo v em V
    constraint_1b = [
        (Sum([y[e] for e in out_edges[v]]) == f_out[v]) for v in range(num_nodes)
    ]
    # constraint 2
    # f_out[v] <= f_in[v] <= 1, paratodo v em PDP(V)
    constraint_2a = [(f_out[v] <= f_in[v]) for v in pdp_nodes]
    constraint_2b = [(f_in[v] <= 1) for v in pdp_nodes]
    # constraint 3: for NND nodes, flow out is at most 1
    # f_out[v] <= 1, paratodo v em NDD(V)
    constraint_3a = [(f_out[v] <= 1) for v in ndd_nodes]
    constraint_3b = [(f_in[v] == 0) for v in ndd_nodes]
    # constraint bonus: for P nodes, flow in is at most 1
    constraint_Pnodes_a = [(f_in[v] <= 1) for v in p_nodes]
    constraint_Pnodes_b = [(f_out[v] == 0) for v in p_nodes]
    # TODO constraint 4
    # TODO generate cycles list: https://stackoverflow.com/questions/40833612/find-all-cycles-in-a-graph-implementation
    # constraint 4: prohibits cycles with length longer than k
    # sum(y[e], paratodo e em C) <= abs_size(c), paratodo C em C\Ck
    # constraint_4 = [
    #     (<= len(c)) for c in cycles
    # ]

    # constraint 5: not necessary
    # o dominio dos y é {0,1}
    satisfy(
        constraint_1a,
        constraint_1b,
        constraint_2a,
        constraint_2b,
        constraint_3a,
        constraint_3b,
        constraint_Pnodes_a,
        constraint_Pnodes_b
        # TODO: usar constraint 4?
    )

    # objective
    # maximize:  sum(w[e] * y[e], paratodo e em E)
    maximize(Sum((edge_weights[e] * y[e]) for e in range(num_edges)))

    timestamp_before_solver = time.time()
    # print(
    #     f"time passed until now: {timestamp_before_solver - timestamp_start} seconds")

    # solve instance
    solved_instance = solve()

    # measure time TODO remove?
    timestamp_end = time.time()
    # print(
    #     f"solver time elapsed: {timestamp_end - timestamp_before_solver} seconds")
    print(f"total time elapsed: {timestamp_end - timestamp_start} seconds")

    # add solution to PyG object as a torch.Tensor
    solution = values(y)
    kep_instance.pred = torch.Tensor(solution)
    return kep_instance
