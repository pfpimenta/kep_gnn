# -*- coding: utf-8 -*-
# script to randomly generate NUM_SAMPLES instances of the Travelling Salesperson Problem (TSP)
# and their optimal solution routes
import sys
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)


from dataset_utils import filter_tensors
from paths import get_dataset_folder_path
from tsp_instance import generate_tsp_instance, solve_tsp_instance

NUM_SAMPLES = 1000  # usaram 2**20 no paper

# TODO use second, third, fourth, etc. best routes as labels as well
# (1/x instead of 1, where x=2, 3, 4, etc.)


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


if __name__ == "__main__":
    # train dataset
    tsp_train_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="train")
    generate_tsp_dataset(
        num_samples=NUM_SAMPLES,
        output_dir=tsp_train_dataset_dir,
    )
    # test dataset
    tsp_test_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="test")
    generate_tsp_dataset(
        num_samples=NUM_SAMPLES,
        output_dir=tsp_test_dataset_dir,
    )
    # validation dataset
    tsp_val_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="val")
    generate_tsp_dataset(
        num_samples=NUM_SAMPLES,
        output_dir=tsp_val_dataset_dir,
    )
