# -*- coding: utf-8 -*-
# script to randomly generate n instances of the Decision TSP problem
# and their optimal solution routes
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch_geometric


# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)

from dataset_utils import filter_tensors
from paths import get_dataset_folder_path


def tsp_to_dtsp(
    tsp_graph: torch_geometric.data.Data, cost_deviation: float = 0.02
) -> Tuple[torch_geometric.data.Data, torch_geometric.data.Data]:
    """Generates a pair of Decision TSP pytorch instances (Y and N)
    from a TSP pytorch instance.
    """
    # add 'optimal_cost' (total cost of the optimal solution route)
    tsp_graph.optimal_cost = (tsp_graph.distance * tsp_graph.y).sum()
    y_dtsp_graph = tsp_graph.clone()  # instance with feasible cost
    n_dtsp_graph = tsp_graph.clone()  # instance with unfeasible cost
    # and 'cost' (value that is {cost_deviation} higher or lower than {optimal_cost})
    y_dtsp_graph.cost = tsp_graph.optimal_cost * (1 - cost_deviation)
    n_dtsp_graph.cost = tsp_graph.optimal_cost * (1 + cost_deviation)
    # add cost as an edge feature
    y_costs = torch.ones(tsp_graph.num_edges) * y_dtsp_graph.cost
    n_costs = torch.ones(tsp_graph.num_edges) * n_dtsp_graph.cost
    y_dtsp_graph.edge_features = torch.cat(
        (tsp_graph.distance.view(-1, 1), y_costs.view(-1, 1)), 1
    )
    n_dtsp_graph.edge_features = torch.cat(
        (tsp_graph.distance.view(-1, 1), n_costs.view(-1, 1)), 1
    )
    # rename optimal_route tensor
    y_dtsp_graph.optimal_route = y_dtsp_graph.y
    n_dtsp_graph.optimal_route = n_dtsp_graph.y
    # set ground truth values (i.e. values to be predicted)
    y_dtsp_graph.y = 1
    n_dtsp_graph.y = 0
    # set instance ID
    y_dtsp_graph.id = f"{y_dtsp_graph.id}_y_{cost_deviation}"
    n_dtsp_graph.id = f"{n_dtsp_graph.id}_n_{cost_deviation}"
    # delete tensors that are not used for the DTSP
    tensors_names = ["id", "edge_index", "node_features", "edge_features", "y"]
    y_dtsp_graph = filter_tensors(data=y_dtsp_graph, tensor_names=tensors_names)
    n_dtsp_graph = filter_tensors(data=n_dtsp_graph, tensor_names=tensors_names)
    return y_dtsp_graph, n_dtsp_graph


def generate_dtsp_dataset(
    tsp_instances_dir: str, output_dir: str, cost_deviation: float = 0.02
) -> None:
    """For each TSP instance in {tsp_instances_dir}, creates two instances of the
    Decision TSP problem: one with the 'cost' tensor {cost_deviation} lower than
    the optimal, therefore feasible/Y/1, and another one with the 'cost' tensor
    {cost_deviation} higher than the optimal, therefore unfeasible/N/0.
    All the generated DTSP instances are saved in .PT files at {output_dir}
    """
    filepath_list = sorted(os.listdir(tsp_instances_dir))
    num_samples = len(filepath_list) * 2
    for i, tsp_instance_filename in enumerate(filepath_list):
        # load TSP instance
        filepath = Path(tsp_instances_dir) / tsp_instance_filename
        tsp_graph = torch.load(filepath)
        # generate a pair of DTSP instances
        dtsp_graph_pair: Tuple = tsp_to_dtsp(
            tsp_graph=tsp_graph, cost_deviation=cost_deviation
        )
        for j, dtsp_graph in enumerate(dtsp_graph_pair):
            # save DTSP instance in a .PT file
            instance_index = 2 * i + j
            dtsp_graph_filename = f"d{tsp_instance_filename[:-3]}_{dtsp_graph.y}.pt"
            dtsp_graph_filepath = Path(output_dir) / dtsp_graph_filename
            torch.save(dtsp_graph, dtsp_graph_filepath)
            print(f"[{instance_index + 1}/{num_samples}] Saved {dtsp_graph_filepath}")


if __name__ == "__main__":
    cost_deviation = 0.02
    tsp_train_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="train")
    dtsp_train_dataset_dir = get_dataset_folder_path(dataset_name="DTSP", step="train")
    generate_dtsp_dataset(
        tsp_instances_dir=tsp_train_dataset_dir,
        output_dir=dtsp_train_dataset_dir,
        cost_deviation=cost_deviation,
    )

    tsp_test_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="test")
    dtsp_test_dataset_dir = get_dataset_folder_path(dataset_name="DTSP", step="test")
    generate_dtsp_dataset(
        tsp_instances_dir=tsp_test_dataset_dir,
        output_dir=dtsp_test_dataset_dir,
        cost_deviation=cost_deviation,
    )

    tsp_val_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="val")
    dtsp_val_dataset_dir = get_dataset_folder_path(dataset_name="DTSP", step="val")
    generate_dtsp_dataset(
        tsp_instances_dir=tsp_val_dataset_dir,
        output_dir=dtsp_val_dataset_dir,
        cost_deviation=cost_deviation,
    )
