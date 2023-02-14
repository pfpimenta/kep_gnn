# -*- coding: utf-8 -*-
# solucao exata pro KEP
# formulacao recursive alg Roth2014

from os import listdir

import numpy as np
import torch
from paths import get_dataset_folder_path
from pycsp3 import *
from torch_geometric.utils import to_dense_adj

# get KEP instance filepath
dataset_dir = get_dataset_folder_path(dataset_name="KEP", step="custom")
filename = listdir(dataset_dir)[0]
filepath = dataset_dir / filename
# load KEP instance
kep_instance = torch.load(filepath)
print(f"Loaded {filepath}")
# get adjecency matrix
adj_matrix_tensor = to_dense_adj(kep_instance.edge_index)
adj_matrix = adj_matrix_tensor[0].numpy()
# get edge weights (multiply weights by 10000 to force them to integers)
# edge_weights = (kep_instance.edge_weights * 10000).to(torch.int32).numpy()
edge_weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # DEBUG
num_edges = int(edge_weights.shape[0])
num_nodes = int(adj_matrix.shape[0])
# DELETE (getting node indexes instead of edge indexes)
# in_edges = [np.where(adj_matrix[:, n] == 1)[0].tolist() for n in range(num_nodes)]
# out_edges = [np.where(adj_matrix[n] == 1)[0].tolist() for n in range(num_nodes)]
in_edges = [
    torch.where(kep_instance.edge_index[0] == v)[0].tolist() for v in range(num_nodes)
]
out_edges = [
    torch.where(kep_instance.edge_index[1] == v)[0].tolist() for v in range(num_nodes)
]
print(f"num_nodes: {num_nodes}")
print(f"num_edges: {num_edges}")
print(f"edge_weights: {edge_weights}")
print(f"kep_instance.edge_index:\n {kep_instance.edge_index}")
print(f"in_edges: {in_edges}")
print(f"out_edges: {out_edges}")

# breakpoint()

# variables
domain = 0, 1
y = VarArray(size=num_edges, dom=domain)
f_in = VarArray(size=num_nodes, dom=range(num_nodes))
f_out = VarArray(size=num_nodes, dom=range(num_nodes))
print("\nDomain of y[0]: ", y[0].dom)
print("Domain of f_in[0]: ", f_in[0].dom)
print("Domain of f_out[0]: ", f_out[0].dom)

# TODO constraints
# constraint 1a: sum(y[e], paratodo e em in(v)) = f_in[v], paratodo v em V
constraint_1a = [
    (Sum([y[e] for e in in_edges[v]]) == f_in[v]) for v in range(num_nodes)
]
# constraint 1b
# sum(y[e], paratodo e em out(v)) = f_out[v], paratodo v em V
constraint_1b = [
    (Sum([y[e] for e in out_edges[v]]) == f_out[v]) for v in range(num_nodes)
]
satisfy(
    constraint_1a,
    constraint_1b
    # constraint 2
    # f_out[v] <= f_in[v] <= 1, paratodo v em V
    # constraint 3
    # f_out[v] <= 1
    # constraint 4
    # TODO entender kk
    # sum(y[e], paratodo e em C) <= abs_size(c), paratodo C em C\Ck
    # constraint 5
    # o dominio dos y é {0,1}
)
print(f"\nDEBUG posted:\n{posted()}\n")
# breakpoint()


# TODO objectives
# sintaxe da documentação:
# minimize(
#     Sum(
#         (x[i] > 1) * c[i] for i in range(n)
#     )
# )
# maximize:  sum(w[e] * y[e], paratodo e em E)
# maximize(Sum((edge_weights[e] * y[e]) for e in range(num_edges)))
# maximize(Sum(y))
# maximize(Sum((y[e]) for e in range(num_edges)))


if solve(sols=ALL) is SAT:
    for i in range(n_solutions()):
        print(f" Solution {i}: {values(y, sol=i)}")
else:
    print("Not solved!")


# solve
# if solve() is SAT:
#     print(f"Solved!\n y values: {values(y)}")
#     print(f"f_in values: {values(f_in)}")
#     print(f"f_out values: {values(f_out)}")
# else:
#     print("Not solved!")
