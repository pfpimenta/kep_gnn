# -*- coding: utf-8 -*-
import random
from math import sqrt
from typing import List, Optional, Tuple

import networkx as nx
from python_tsp.exact import solve_tsp_dynamic_programming

# random seed to make the same dataset everytime
random.seed(42)


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
