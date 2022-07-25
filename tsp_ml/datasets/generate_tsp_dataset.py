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
sys.path.insert(0, "/home/pimenta/tsp_ml/tsp_ml")

from dataset_utils import filter_tensors
from paths import (
    TSP_TEST_DATASET_FOLDER_PATH,
    TSP_TRAIN_DATASET_FOLDER_PATH,
    TSP_VAL_DATASET_FOLDER_PATH,
)
from tsp_instance import generate_tsp_instance, solve_tsp_instance


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
        graph_hash = hash(tsp_instance_pyg_graph)
        tsp_instance_pyg_graph.id = f"{i}_{graph_hash}"
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
        filename = f"tsp_instance_{i}.pt"
        filepath = output_dir / filename
        torch.save(tsp_instance_pyg_graph, filepath)
        print(f"[{i+1}/{num_samples}] Saved {filepath}")


if __name__ == "__main__":
    num_samples = 1000  # usaram 2**20 no paper
    generate_tsp_dataset(
        num_samples=num_samples, output_dir=TSP_TEST_DATASET_FOLDER_PATH
    )
    generate_tsp_dataset(
        num_samples=num_samples, output_dir=TSP_TRAIN_DATASET_FOLDER_PATH
    )
    generate_tsp_dataset(
        num_samples=num_samples, output_dir=TSP_VAL_DATASET_FOLDER_PATH
    )
