# -*- coding: utf-8 -*-
# script to randomly generate 1 instance of the Kidney Exchange Problem (KEP)
# and their optimal solution routes
import random
import sys
from typing import List, Optional

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

# to allow imports from outside the tsp_ml/datasets/ package
sys.path.insert(0, "/home/pimenta/tsp_ml/tsp_ml")
from paths import (
    KEP_TEST_DATASET_FOLDER_PATH,
    KEP_TRAIN_DATASET_FOLDER_PATH,
    KEP_VAL_DATASET_FOLDER_PATH,
)

## script parameters
NUM_INSTANCES = 100
NUM_NODES = 100
NUM_EDGES = 200
NODE_TYPES = [
    "NDD",  # non-directed donors
    "PDP",  # patient-donor pair
    "P",  # pacient without a pair
]
NODE_TYPE_DISTRIBUTION = [0.2, 0.4, 0.4]


# TODO : create graphs with num_nodes
# mirroring data described at Tables S3 and S4


def generate_kep_instance(
    num_nodes: int,
    num_edges: int,
    node_types: List[str],
    node_type_distribution: List[float],
) -> nx.DiGraph:
    """TODO description"""
    node_ids = range(num_nodes)
    kep_instance = nx.DiGraph()

    # add nodes
    for node in node_ids:
        node_type = random.choices(
            population=node_types, weights=node_type_distribution
        )[0]
        # TODO weight should be in node ???
        # NAO (!?) ... no artigo ta nas arestas msm...
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

    return kep_instance


def generate_kep_dataset(
    num_instances: int,
    output_dir: str,
    node_types: Optional[List[str]] = None,
    node_type_distribution: Optional[List[float]] = None,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
):
    for instance_id in range(num_instances):
        kep_instance_nx_graph = generate_kep_instance(
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_types=node_types,
            node_type_distribution=node_type_distribution,
        )
        # convert from nx.DiGraph to torch_geometric.data.Data
        kep_instance_pyg_graph = from_networkx(kep_instance_nx_graph)
        # import pdb

        # pdb.set_trace()
        # set instance ID
        instance_id = nx.weisfeiler_lehman_graph_hash(
            G=kep_instance_nx_graph, edge_attr="edge_weights"
        )
        # save KEP instance on output_dir
        filename = f"kep_instance_{instance_id}.pt"
        filepath = output_dir / filename
        torch.save(kep_instance_pyg_graph, filepath)


if __name__ == "__main__":
    # test dataset
    generate_kep_dataset(
        num_instances=NUM_INSTANCES,
        num_nodes=NUM_NODES,
        num_edges=NUM_EDGES,
        node_types=NODE_TYPES,
        node_type_distribution=NODE_TYPE_DISTRIBUTION,
        output_dir=KEP_TEST_DATASET_FOLDER_PATH,
    )
    # train dataset
    generate_kep_dataset(
        num_instances=NUM_INSTANCES,
        num_nodes=NUM_NODES,
        num_edges=NUM_EDGES,
        node_types=NODE_TYPES,
        node_type_distribution=NODE_TYPE_DISTRIBUTION,
        output_dir=KEP_TRAIN_DATASET_FOLDER_PATH,
    )
    # validation dataset
    generate_kep_dataset(
        num_instances=NUM_INSTANCES,
        num_nodes=NUM_NODES,
        num_edges=NUM_EDGES,
        node_types=NODE_TYPES,
        node_type_distribution=NODE_TYPE_DISTRIBUTION,
        output_dir=KEP_VAL_DATASET_FOLDER_PATH,
    )
