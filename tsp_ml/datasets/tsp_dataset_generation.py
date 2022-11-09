# -*- coding: utf-8 -*-
import random
import sys
from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import torch
from python_tsp.exact import solve_tsp_dynamic_programming
from torch_geometric.utils.convert import from_networkx

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from dataset_utils import filter_tensors

# random seed to make the same dataset everytime
random.seed(42)


# TODO use second, third, fourth, etc. best routes as labels as well
# (1/x instead of 1, where x=2, 3, 4, etc.)


def solve_tsp_instance(
    tsp_instance_graph: nx.Graph, verbose: bool = False
) -> Tuple[List[int], float]:
    distance_matrix = nx.floyd_warshall_numpy(G=tsp_instance_graph, weight="distance")
    path_nodes, path_distance = solve_tsp_dynamic_programming(distance_matrix)
    if verbose:
        print(
            f"Solved TSP instance. Path nodes: {path_nodes} ...\npath distance: {path_distance}"
        )
    path_edges = [
        [path_nodes[i], path_nodes[i + 1]] for i in range(len(path_nodes) - 1)
    ]
    last_path_edge = [path_nodes[-1], path_nodes[0]]
    path_edges.append(last_path_edge)
    return path_edges, path_distance


def generate_tsp_instance(
    num_nodes: Optional[int] = None, verbose: bool = False
) -> nx.Graph:
    if num_nodes is None:
        # num_nodes = int(random.uniform(20, 40)) # uniform distribution
        num_nodes = int(random.uniform(5, 16))  # DEBUG
    if verbose:
        print(f"Generating a TSP instance graph with {num_nodes} nodes ...")
    nodes = range(num_nodes)
    x1_values = [random.uniform(0, sqrt(2) / 2) for _ in nodes]
    x2_values = [random.uniform(0, sqrt(2) / 2) for _ in nodes]
    g = nx.Graph()
    # add nodes
    for node in nodes:
        # node features are its 2D coordinates (x1,x2)
        g.add_node(node, node_features=(x1_values[node], x2_values[node]))
    # add edges
    for src_node_id in nodes:
        for dst_node_id in nodes:
            if src_node_id != dst_node_id:
                x1_node1 = g.nodes[src_node_id]["node_features"][0]
                x2_node1 = g.nodes[src_node_id]["node_features"][1]
                x1_node2 = g.nodes[dst_node_id]["node_features"][0]
                x2_node2 = g.nodes[dst_node_id]["node_features"][1]
                euclidian_distance = sqrt(
                    (x1_node1 - x1_node2) ** 2 + (x2_node1 - x2_node2) ** 2
                )
                g.add_edge(src_node_id, dst_node_id, distance=euclidian_distance)
    return g


def generate_tsp_dataset(num_samples: int, output_dir: str):
    for i in range(num_samples):
        # generate TSP instance graph
        tsp_instance_nx_graph = generate_tsp_instance()
        # get optimal solution for this TSP instance
        path_edges, _ = solve_tsp_instance(tsp_instance_graph=tsp_instance_nx_graph)
        # add solution to graph as an edge feature/attribute
        # (1 if edge is part of the optimal route, 0 otherwise)
        solution_dict = {edge: 0 for edge in list(tsp_instance_nx_graph.edges())}
        for edge in path_edges:
            solution_dict[tuple(edge)] = 1
        nx.set_edge_attributes(G=tsp_instance_nx_graph, values=solution_dict, name="y")
        # convert from nx.Graph to torch_geometric.data.Data
        tsp_instance_pyg_graph = from_networkx(tsp_instance_nx_graph)
        # set instance ID
        instance_id = nx.weisfeiler_lehman_graph_hash(
            G=tsp_instance_nx_graph, edge_attr="distance"
        )
        tsp_instance_pyg_graph.id = instance_id
        # delete tensors that are not used for the TSP
        tensors_names = [
            "id",
            "edge_index",
            "node_features",
            "edge_features",
            "y",
            "distance",
        ]
        tsp_instance_pyg_graph = filter_tensors(
            data=tsp_instance_pyg_graph, tensor_names=tensors_names
        )
        # save TSP instance on output_dir
        filename = f"tsp_instance_{instance_id}.pt"
        filepath = output_dir / filename
        torch.save(tsp_instance_pyg_graph, filepath)
        print(f"[{i+1}/{num_samples}] Saved {filepath}")
