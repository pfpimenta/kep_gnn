# -*- coding: utf-8 -*-
# solucao exata pro KEP
# formulacao recursive alg Roth2014

import time
from os import listdir

import torch
from paths import get_dataset_folder_path
from pycsp3 import *
from torch_geometric.utils import to_dense_adj

timestamp_start = time.time()

# get KEP instance filepath
# dataset_dir = get_dataset_folder_path(dataset_name="KEP", step="custom")
dataset_dir = get_dataset_folder_path(dataset_name="KEP", step="test")
filename = listdir(dataset_dir)[0]
# TODO custom_instance_A tem 2 soluções igualmente otimas! achar as duas?
# filename = "kep_instance_4fefaa1110bf0c99a7cb172264c477c5.pt" # A
# filename = "kep_instance_0dc9871f2973964c58fba8e5bba31971.pt" # B
# filename = "kep_instance_aa039ce150e27524ac16d541af320cbf.pt" # C
# filename = "kep_instance_c9059962f747c0240ec9d456b88cb421.pt" # D

filepath = dataset_dir / filename
# load KEP instance
kep_instance = torch.load(filepath)
print(f"Loaded {filepath}")
# get adjecency matrix
adj_matrix_tensor = to_dense_adj(kep_instance.edge_index)
adj_matrix = adj_matrix_tensor[0].numpy()
# get edge weights (multiply weights by 10000 to force them to integers)
edge_weights = (kep_instance.edge_weights * 10000).to(torch.int32).numpy()
# edge_weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # DEBUG
# get node types (PDP, NDD, or P)
node_types = kep_instance.type
ndd_nodes = [i for i in range(len(node_types)) if node_types[i] == "NDD"]
pdp_nodes = [i for i in range(len(node_types)) if node_types[i] == "PDP"]
p_nodes = [i for i in range(len(node_types)) if node_types[i] == "P"]
num_edges = int(edge_weights.shape[0])
num_nodes = int(adj_matrix.shape[0])
# TODO delete this (getting node indexes instead of edge indexes):
# in_edges = [np.where(adj_matrix[:, n] == 1)[0].tolist() for n in range(num_nodes)]
# out_edges = [np.where(adj_matrix[n] == 1)[0].tolist() for n in range(num_nodes)]
in_edges = [
    torch.where(kep_instance.edge_index[1] == v)[0].tolist() for v in range(num_nodes)
]
out_edges = [
    torch.where(kep_instance.edge_index[0] == v)[0].tolist() for v in range(num_nodes)
]
print(f"num_nodes: {num_nodes}")
print(f"num_edges: {num_edges}")
print(f"node_types: {node_types}")
print(f"edge_weights: {edge_weights}")
print(f"kep_instance.edge_index:\n {kep_instance.edge_index}")
print(f"in_edges: {in_edges}")
print(f"out_edges: {out_edges}")

# CSP variables
domain = 0, 1
y = VarArray(size=num_edges, dom=domain)
f_in = VarArray(size=num_nodes, dom=range(num_nodes))
f_out = VarArray(size=num_nodes, dom=range(num_nodes))
print("\nDomain of y[0]: ", y[0].dom)
print("Domain of f_in[0]: ", f_in[0].dom)
print("Domain of f_out[0]: ", f_out[0].dom)

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
    # constraint 4
    # TODO entender kk
    # sum(y[e], paratodo e em C) <= abs_size(c), paratodo C em C\Ck
)
constraints = posted()
print(f"\nDEBUG posted:\n{constraints}\n")
print(f"Number of restrictions for this instance: {len(constraints)}")


# objective
# maximize:  sum(w[e] * y[e], paratodo e em E)
maximize(Sum((edge_weights[e] * y[e]) for e in range(num_edges)))
print(f"\nDEBUG objective:\n{objective()}\n")

timestamp_before_solver = time.time()
print(f"time passed until now: {timestamp_before_solver - timestamp_start} seconds")

# solve instance
solved_instance = solve()

timestamp_end = time.time()
print(f"solver time elapsed: {timestamp_end - timestamp_before_solver} seconds")
print(f"total time elapsed: {timestamp_end - timestamp_start} seconds")


# print solution
breakpoint()
if solved_instance is OPTIMUM:
    # breakpoint()
    # if n_solutions() == 1:
    print(f" Solution:\n\ty: {values(y)}, f_in: {values(f_in)}, f_out: {values(f_out)}")
    # breakpoint()
else:
    print("Not solved!")
    # breakpoint()


# variation that shows ALL FEASIBLE solutions (not only the optimal ones!)
# solved_instance = solve(sols=ALL)
# if solved_instance is OPTIMUM:
#     for i in range(n_solutions()):
#         print(f" Solution {i}:\n\ty: {values(y, sol=i)}, f_in: {values(f_in, sol=i)}, f_out: {values(f_out, sol=i)}")
#     breakpoint()
# else:
#     print("Not solved!")
#     breakpoint()

# solve
# if solve() is SAT:
#     print(f"Solved!\n y values: {values(y)}")
#     print(f"f_in values: {values(f_in)}")
#     print(f"f_out values: {values(f_out)}")
# else:
#     print("Not solved!")
