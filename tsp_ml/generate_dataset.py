import os
import random
from math import sqrt
from typing import List, Optional, Tuple

import networkx as nx
import torch
from python_tsp.exact import solve_tsp_dynamic_programming
from torch_geometric.utils.convert import from_networkx
from definitions import NX_GRAPHS_FOLDER_PATH, PYG_GRAPHS_FOLDER_PATH


def solve_tsp_instance(tsp_instance_graph: nx.Graph) -> Tuple[List[int], float]:
    distance_matrix = nx.floyd_warshall_numpy(G=tsp_instance_graph, weight="distance")
    path_nodes, path_distance = solve_tsp_dynamic_programming(distance_matrix)
    print(
        f"Solved TSP instance. Path nodes: {path_nodes} ...\npath distance: {path_distance}"
    )
    path_edges = [
        [path_nodes[i], path_nodes[i + 1]] for i in range(len(path_nodes) - 1)
    ]
    last_path_edge = [path_nodes[-1], path_nodes[0]]
    path_edges.append(last_path_edge)
    return path_edges, path_distance


def generate_tsp_instance(num_nodes: Optional[int] = None) -> nx.Graph:
    if num_nodes is None:
        # num_nodes = int(random.uniform(20, 40)) # uniform distribution
        num_nodes = int(random.uniform(5, 16))  # DEBUG
    print(f"Generating a TSP instance graph with {num_nodes} nodes ...")
    nodes = range(num_nodes)
    x1_values = [random.uniform(0, sqrt(2) / 2) for n in nodes]
    x2_values = [random.uniform(0, sqrt(2) / 2) for n in nodes]
    g = nx.Graph()
    # add nodes
    for node in nodes:
        # node features are its 2D coordinates (x1,x2)
        g.add_node(node, features=(x1_values[node], x2_values[node]))
    # add edges
    for src_node_id in nodes:
        for dst_node_id in nodes:
            if src_node_id != dst_node_id:
                x1_node1 = g.nodes[src_node_id]["features"][0]
                x2_node1 = g.nodes[src_node_id]["features"][1]
                x1_node2 = g.nodes[dst_node_id]["features"][0]
                x2_node2 = g.nodes[dst_node_id]["features"][1]
                euclidian_distance = sqrt(
                    (x1_node1 - x1_node2) ** 2 + (x2_node1 - x2_node2) ** 2
                )
                g.add_edge(src_node_id, dst_node_id, distance=euclidian_distance)
    return g


def generate_tsp_dataset(num_samples: int):
    for i in range(num_samples):
        # generate TSP instance graph
        tsp_instance_graph = generate_tsp_instance()
        # get optimal solution for this TSP instance
        path_edges, _ = solve_tsp_instance(tsp_instance_graph=tsp_instance_graph)
        # add solution to nx.Graph as an edge feature/attribute
        solution_dict = {edge: 0 for edge in list(tsp_instance_graph.edges())}
        for edge in path_edges:
            solution_dict[tuple(edge)] = 1
        nx.set_edge_attributes(G=tsp_instance_graph, values=solution_dict, name="y")
        # save nx graph to GML file
        nx_graph_filename = f"tsp_instance_{i}.gml"
        nx_graph_filepath = NX_GRAPHS_FOLDER_PATH / nx_graph_filename
        nx.write_gml(G=tsp_instance_graph, path=nx_graph_filepath)
        print(f"[{i+1}/{num_samples}] Saved {nx_graph_filename}")


def generate_pyg_graphs(nx_graphs_folder_path: str = NX_GRAPHS_FOLDER_PATH):
    filepath_list = sorted(os.listdir(nx_graphs_folder_path))
    num_graphs = len(filepath_list)
    for i, nx_graph_filename in enumerate(sorted(os.listdir(nx_graphs_folder_path))):
        # load nx graph from GML file
        nx_graph_filepath = NX_GRAPHS_FOLDER_PATH / nx_graph_filename
        nx_graph = nx.read_gml(path=nx_graph_filepath, destringizer=int)
        # save into a PyG graph file
        pyg_graph_filename = os.path.splitext(nx_graph_filename)[0] + ".pt"
        pyg_graph_filepath = PYG_GRAPHS_FOLDER_PATH / pyg_graph_filename
        pyg_graph = from_networkx(nx_graph)
        torch.save(pyg_graph, pyg_graph_filepath)
        print(f"[{i+1}/{num_graphs}] Saved {pyg_graph_filepath}")


if __name__ == "__main__":
    os.makedirs(NX_GRAPHS_FOLDER_PATH, exist_ok=True)
    os.makedirs(PYG_GRAPHS_FOLDER_PATH, exist_ok=True)
    num_samples = 1000  # usaram 2**20 no paper
    generate_tsp_dataset(num_samples=num_samples)
    generate_pyg_graphs()
