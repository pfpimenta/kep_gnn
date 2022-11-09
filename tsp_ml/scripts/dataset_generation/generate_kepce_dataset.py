# -*- coding: utf-8 -*-
# script to randomly generate instances of the Kidney Exchange Problem (KEP)
# BUT with counter edges (edges that connect destination nodes to source nodes).
# This updated KEP dataset is refered to as KEP-CE, where CE stands for Counter Edges
import sys
from pathlib import Path
from typing import List, Optional

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

# TODO:
from datasets.generate_kep_dataset import generate_kep_instance
from paths import get_dataset_folder_path

## script parameters
NUM_INSTANCES = 5000
NUM_NODES = 15
NUM_EDGES = 45
NODE_TYPES = [
    "NDD",  # non-directed donors
    "PDP",  # patient-donor pair
    "P",  # pacient without a pair
]
NODE_TYPE_DISTRIBUTION = [0.05, 0.9, 0.05]


def generate_kepce_instance(
    num_nodes: int,
    num_edges: int,
    node_types: List[str],
    node_type_distribution: List[float],
    add_node_features: bool = True,
) -> nx.DiGraph:
    kep_instance = generate_kep_instance(
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_types=node_types,
        node_type_distribution=node_type_distribution,
        add_node_features=add_node_features,
    )
    # copy graph without the edges
    kepce_instance = nx.create_empty_copy(kep_instance)
    for kep_edge in kep_instance.edges(data=True):
        src, dst, attr = kep_edge
        # add edge to KEPCE instance
        kepce_instance.add_edge(
            src,
            dst,
            edge_weights=attr["edge_weights"],
            counter_edge=0,
        )
        # add counter edge to KEPCE instance
        kepce_instance.add_edge(
            dst,
            src,
            edge_weights=attr["edge_weights"],
            counter_edge=1,
        )
    return kepce_instance


def generate_kepce_dataset(
    num_instances: int,
    output_dir: str,
    node_types: Optional[List[str]] = None,
    node_type_distribution: Optional[List[float]] = None,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
):
    """Generates 'num_instances' instances of the Kidney-Exchange Problem
    with counter edges (KEP-CE) and saves them in .PT files inside 'output_dir'.
    """
    for i in range(num_instances):
        kepce_instance_nx_graph = generate_kepce_instance(
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_types=node_types,
            node_type_distribution=node_type_distribution,
        )
        # convert from nx.DiGraph to torch_geometric.data.Data
        node_feature_names = [
            "num_edges_in",
            "num_edges_out",
            "is_NDD",
            "is_PDP",
            "is_P",
        ]
        kep_instance_pyg_graph = from_networkx(
            G=kepce_instance_nx_graph,
            group_node_attrs=node_feature_names,
        )
        # set instance ID
        instance_id = nx.weisfeiler_lehman_graph_hash(
            G=kepce_instance_nx_graph, edge_attr="edge_weights"
        )
        kep_instance_pyg_graph.id = instance_id
        # save KEPCE instance on output_dir
        filename = f"kepce_instance_{instance_id}.pt"
        filepath = output_dir / filename
        torch.save(kep_instance_pyg_graph, filepath)
        print(f"[{i+1}/{num_instances}] Saved {filepath}")


if __name__ == "__main__":
    # generate and save train, test, and val KEP-CE datasets
    for step in ["train", "test", "val"]:
        kep_dataset_dir = get_dataset_folder_path(
            dataset_name="KEPCE",
            step=step,
        )
        generate_kepce_dataset(
            num_instances=NUM_INSTANCES,
            num_nodes=NUM_NODES,
            num_edges=NUM_EDGES,
            node_types=NODE_TYPES,
            node_type_distribution=NODE_TYPE_DISTRIBUTION,
            output_dir=kep_dataset_dir,
        )
