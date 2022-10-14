# -*- coding: utf-8 -*-
# script to randomly generate instances of the Kidney Exchange Problem (KEP)
# TODO: generate their optimal solution routes
import random
import sys
from typing import List, Optional

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)

from paths import get_dataset_folder_path

## script parameters
NUM_INSTANCES = 10000
NUM_NODES = 300
NUM_EDGES = 5500
NODE_TYPES = [
    "NDD",  # non-directed donors
    "PDP",  # patient-donor pair
    "P",  # pacient without a pair
]
NODE_TYPE_DISTRIBUTION = [0.05, 0.9, 0.05]


def generate_kep_instance(
    num_nodes: int,
    num_edges: int,
    node_types: List[str],
    node_type_distribution: List[float],
    add_node_features: bool = True,
) -> nx.DiGraph:
    """Generates one instance of the Kidney-Exchange Problem:
    a directed graph, where nodes represent patients, donnors, or patient-donnor-pairs,
    and edges represent biological compatibility for kidney donnation.
    The choice of which node is connected to which node is made randomly.
    """
    node_ids = range(num_nodes)
    kep_instance = nx.DiGraph()

    # add nodes
    for node in node_ids:
        node_type = random.choices(
            population=node_types, weights=node_type_distribution
        )[0]
        # TODO weight should be in node ???
        # NAO (!?) ... no artigo ta nas arestas msm...
        if add_node_features:
            # num_edges_in and num_edges_out will be added later
            num_edges_in = 0
            num_edges_out = 0
            is_NDD = 1 if node_type == "NDD" else 0
            is_PDP = 1 if node_type == "PDP" else 0
            is_P = 1 if node_type == "P" else 0
            kep_instance.add_node(
                node,
                type=node_type,
                num_edges_in=num_edges_in,
                num_edges_out=num_edges_out,
                is_NDD=is_NDD,
                is_PDP=is_PDP,
                is_P=is_P,
            )
        else:
            kep_instance.add_node(node, type=node_type)

    # add edges (no features, directed)
    # TODO refactor
    for edge_id in range(num_edges):
        src_node_id = random.choice(node_ids)
        # avoid having patient nodes as source of edges
        while kep_instance.nodes[src_node_id]["type"] == "P":
            src_node_id = random.choice(node_ids)
        dst_node_id = random.choice(node_ids)
        # avoid self loops
        # AND avoid NDD nodes as destination of edges
        while (
            src_node_id == dst_node_id
            or kep_instance.nodes[dst_node_id]["type"] == "NDD"
        ):
            dst_node_id = random.choice(node_ids)
        # add some random weight to the edge
        edge_weight = random.random()
        kep_instance.add_edge(src_node_id, dst_node_id, edge_weights=edge_weight)
        if add_node_features:
            # add num_edges_in and num_edges_out node features
            kep_instance.nodes[src_node_id]["num_edges_out"] += 1
            kep_instance.nodes[dst_node_id]["num_edges_in"] += 1

    return kep_instance


def generate_kep_dataset(
    num_instances: int,
    output_dir: str,
    node_types: List[str] = NODE_TYPES,
    node_type_distribution: List[float] = NODE_TYPE_DISTRIBUTION,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    add_node_features: bool = True,
):
    """Generates 'num_instances' instances of the Kidney-Exchange Problem
    and saves them in .PT files inside 'output_dir'."""
    for i in range(num_instances):
        kep_instance_nx_graph = generate_kep_instance(
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_types=node_types,
            node_type_distribution=node_type_distribution,
            add_node_features=add_node_features,
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
            G=kep_instance_nx_graph, group_node_attrs=node_feature_names
        )
        # set instance ID
        instance_id = nx.weisfeiler_lehman_graph_hash(
            G=kep_instance_nx_graph, edge_attr="edge_weights"
        )
        kep_instance_pyg_graph.id = instance_id
        # save KEP instance on output_dir
        filename = f"kep_instance_{instance_id}.pt"
        filepath = output_dir / filename
        torch.save(kep_instance_pyg_graph, filepath)
        print(f"[{i+1}/{num_instances}] Saved {filepath}")


if __name__ == "__main__":
    # generate and save train, test, and val KEP datasets
    for step in ["train", "test", "val"]:
        kep_dataset_dir = get_dataset_folder_path(dataset_name="KEP", step=step)
        generate_kep_dataset(
            num_instances=NUM_INSTANCES,
            num_nodes=NUM_NODES,
            num_edges=NUM_EDGES,
            node_types=NODE_TYPES,
            node_type_distribution=NODE_TYPE_DISTRIBUTION,
            add_node_features=True,
            output_dir=kep_dataset_dir,
        )
