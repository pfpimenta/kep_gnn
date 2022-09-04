# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def generate_kep_plot(predicted_instance: Data, folderpath: str) -> None:
    """Generates a plot of a KEP predicted instance,
    then saves it as a PNG file at the given folderpath."""
    # cast to nx.Graph
    nx_kep_graph = to_networkx(
        data=predicted_instance,
        node_attrs=["type"],
        edge_attrs=["edge_weights", "pred", "scores"],
    )
    color_map = get_kep_color_map(nx_kep_graph)
    posisions_dict = nx.spring_layout(nx_kep_graph)
    predicted_edges = get_predicted_edges_list(nx_kep_graph)
    nx.draw(G=nx_kep_graph, pos=posisions_dict, node_color=color_map, with_labels=True)
    # plot predicted route in red
    nx.draw_networkx_edges(
        G=nx_kep_graph,
        pos=posisions_dict,
        edgelist=predicted_edges,
        edge_color="r",
        width=3,
    )
    # save plot as a PNG
    plot_filename = f"{predicted_instance.id}.png"
    filepath = Path(folderpath) / plot_filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def get_predicted_edges_list(nx_kep_graph: nx.DiGraph) -> List[Tuple[int, int]]:
    """Return a list of the edges included in the prediction (pred==1)"""
    predicted_edges = []
    for (edge_src, edge_dst, edge_features) in nx_kep_graph.edges(data=True):
        if edge_features["pred"] == 1:
            predicted_edges.append((edge_src, edge_dst))
    return predicted_edges


def get_kep_color_map(nx_kep_graph: nx.DiGraph) -> List[str]:
    """Return node colors according to their type"""
    color_map = []
    for node_id in nx_kep_graph:
        node_type = nx_kep_graph.nodes[node_id]["type"]
        if node_type == "NDD":
            color_map.append("lightgreen")
        elif node_type == "PDP":
            color_map.append("lightblue")
        elif node_type == "P":
            color_map.append("red")
        else:
            raise ValueError(f"Type {node_type} is not valid.")
    return color_map
