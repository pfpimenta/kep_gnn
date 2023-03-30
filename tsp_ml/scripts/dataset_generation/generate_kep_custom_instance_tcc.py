# -*- coding: utf-8 -*-
import random
import sys
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from paths import get_dataset_folder_path

""" create custom example instance to use in the
figures that will be put in the TCC and KEP GNN article.
It has 10 PDP nodes connected with ? specific edges;
edge weights are random, with the exception of 0->5,
which is less than 0->3
"""
kep_instance = nx.DiGraph()
random.seed(42)

# add 8 PDP nodes, that will form a cycle
for node_id in range(8):
    kep_instance.add_node(
        node_id,
        type="PDP",
        num_edges_in=0,
        num_edges_out=0,
        is_NDD=0,
        is_PDP=1,
        is_P=0,
    )


# add a NDD node connecting to a PDP node
kep_instance.add_node(
    8,
    type="NDD",
    num_edges_in=0,
    num_edges_out=0,
    is_NDD=1,
    is_PDP=0,
    is_P=0,
)
# and a P node connected from PDP nodes
kep_instance.add_node(
    9,
    type="P",
    num_edges_in=0,
    num_edges_out=0,
    is_NDD=0,
    is_PDP=0,
    is_P=1,
)

# add edges (no features, directed)
edges = [
    (0, 8, 0.9),
    (0, 4, 0.42),
    # (0, 5, 0.23),
    (0, 1, 0.51),
    (1, 5, random.random()),
    (9, 0, random.random()),
    (2, 0, random.random()),
    (2, 6, random.random()),
    (2, 7, random.random()),
    (2, 1, random.random()),
    (4, 2, random.random()),
    (4, 1, random.random()),
    (4, 6, random.random()),
    (5, 8, 0.85),
    (5, 3, random.random()),
    (6, 7, random.random()),
    (6, 3, random.random()),
    (7, 1, random.random()),
    # (1, 0, random.random()),
    (1, 7, random.random()),
    (1, 3, random.random()),
    (3, 2, random.random()),
    (3, 1, random.random()),
]
num_edges = len(edges)
for edge in edges:
    src_node_id, dst_node_id, weight = edge
    kep_instance.add_edge(src_node_id, dst_node_id, edge_weights=weight)
    # add num_edges_in and num_edges_out node features
    kep_instance.nodes[src_node_id]["num_edges_out"] += 1
    kep_instance.nodes[dst_node_id]["num_edges_in"] += 1


# convert from nx.DiGraph to torch_geometric.data.Data
node_feature_names = [
    "num_edges_in",
    "num_edges_out",
    "is_NDD",
    "is_PDP",
    "is_P",
]
kep_instance_pyg_graph = from_networkx(
    G=kep_instance, group_node_attrs=node_feature_names
)
# set instance ID
instance_id = nx.weisfeiler_lehman_graph_hash(G=kep_instance, edge_attr="edge_weights")
kep_instance_pyg_graph.id = instance_id

# breakpoint()
# save KEP instance on output_dir
kep_dataset_dir = get_dataset_folder_path(dataset_name="KEP", step="custom")
filename = f"kep_instance_{instance_id}.pt"
filepath = kep_dataset_dir / filename
torch.save(kep_instance_pyg_graph, filepath)
print(f"Saved instance at {filepath}")
