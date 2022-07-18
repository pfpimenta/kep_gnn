# -*- coding: utf-8 -*-
# script to randomly generate n instances of the TSP problem
# and their optimal solution routes
import os
import sys
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

# to allow imports from outside the tsp_ml/datasets/ package
upper_dir = Path(__file__).parent.parent.resolve()
print(upper_dir)
# sys.path.insert(0, upper_dir)
sys.path.insert(0, "/home/pimenta/tsp_ml/tsp_ml")

from paths import NX_GRAPHS_FOLDER_PATH, PYG_GRAPHS_FOLDER_PATH
from tsp_instance import generate_tsp_instance, solve_tsp_instance


def generate_tsp_nx_graphs(num_samples: int, output_dir: str):
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
        nx_graph_filepath = output_dir / nx_graph_filename
        nx.write_gml(G=tsp_instance_graph, path=nx_graph_filepath)
        print(f"[{i+1}/{num_samples}] Saved {nx_graph_filename}")


def generate_tsp_pyg_graphs(nx_graphs_folder_path: str = NX_GRAPHS_FOLDER_PATH):
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


def generate_tsp_dataset(output_dir: str):
    pass  # TODO


if __name__ == "__main__":
    num_samples = 1000  # usaram 2**20 no paper
    generate_tsp_nx_graphs(num_samples=num_samples, output_dir=NX_GRAPHS_FOLDER_PATH)
    generate_tsp_pyg_graphs()
