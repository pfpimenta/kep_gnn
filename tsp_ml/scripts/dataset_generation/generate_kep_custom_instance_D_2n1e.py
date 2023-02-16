# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from paths import get_dataset_folder_path

""" create custom example instance
with 2 nodes: 1 NDD node connected to 1 P node,
the edge has weight = 1.
"""
kep_instance = nx.DiGraph()

# add a NDD node connecting to a PDP node
kep_instance.add_node(
    0,
    type="NDD",
    num_edges_in=0,
    num_edges_out=0,
    is_NDD=1,
    is_PDP=0,
    is_P=0,
)
# and a P node connected from a PDP node
kep_instance.add_node(
    1,
    type="P",
    num_edges_in=0,
    num_edges_out=0,
    is_NDD=0,
    is_PDP=0,
    is_P=1,
)

# add edges (no features, directed)
num_edges = 1
edges = [(0, 1)]
for edge in edges:
    src_node_id, dst_node_id = edge
    kep_instance.add_edge(src_node_id, dst_node_id, edge_weights=1)
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
