import networkx as nx
import pandas as pd
import torch
from torch_geometric.utils.convert import to_networkx
from paths import TSP_TEST_DATASET_FOLDER_PATH


def load_tsp_instance_with_prediction(
    predictions_filepath: str,
    tsp_instance_id: str,
) -> nx.Graph:
    """Load the TSP instance graph, then load its predictions,
    add the predictions to the graph as an edge feature, and return
    the resulting graph in a nx.Graph object.
    """

    # load CSV
    predictions_df = pd.read_csv(
        predictions_filepath, header=0, names=["id", "predictions", "truth"]
    )
    # get predictions from CSV
    predictions = predictions_df[predictions_df["id"] == tsp_instance_id][
        "predictions"
    ].tolist()

    # load pytorch TSP instance
    # TODO: create load_tsp_graph(tsp_instance_id: str) that infers where the instance is (train/val/test)
    tsp_instance_filename = f"tsp_instance_{tsp_instance_id}.pt"
    tsp_instance_filepath = TSP_TEST_DATASET_FOLDER_PATH / tsp_instance_filename
    tsp_graph = torch.load(tsp_instance_filepath)
    # cast to nx.Graph
    nx_tsp_graph = to_networkx(
        data=tsp_graph,
        node_attrs=["node_features"],
        edge_attrs=["y", "distance"],
    )

    # add predictions to graph...
    # create Dict: edge_tuple -> pred
    edge_tuples = nx_tsp_graph.edges()
    predictions_list = predictions[id]["predictions"]
    pred_attr_dict = {}
    for edge_tuple, pred in zip(edge_tuples, predictions_list):
        pred_attr_dict[edge_tuple] = pred
    # add predictions to graph:
    nx.set_edge_attributes(G=nx_tsp_graph, values=pred_attr_dict, name="prediction")

    # cast to undirected graph: nx.Graph
    nx_tsp_graph = nx_tsp_graph.to_undirected()

    return nx_tsp_graph


def plot_tsp_solution(tsp_graph_with_preds: nx.Graph):
    """Plots the given graph with its optimal route in green and
    its predicted route in red. The optimal route must be encoded
    as edge features "y" and the predicted route must be encoded
    as edge features "prediction".
    """

    # get node_positions
    node_positions = {}
    for node_id, node_features in tsp_graph_with_preds.nodes(data=True):
        node_positions[node_id] = node_features["node_features"]

    # get optimal and predicted routes
    optimal_route_edges = []
    predicted_route_edges = []
    for (edge_src, edge_dst, edge_features) in tsp_graph_with_preds.edges(data=True):
        if edge_features["y"] == 1:
            optimal_route_edges.append((edge_src, edge_dst))
        if edge_features["prediction"] == 1:
            predicted_route_edges.append((edge_src, edge_dst))

    # plot graph
    nx.draw(G=tsp_graph_with_preds, pos=node_positions)
    # plot optimal route in green
    nx.draw_networkx_edges(
        G=tsp_graph_with_preds,
        pos=node_positions,
        edgelist=optimal_route_edges,
        edge_color="g",
        width=12,
    )
    # plot predicted route in red
    nx.draw_networkx_edges(
        G=tsp_graph_with_preds,
        pos=node_positions,
        edgelist=predicted_route_edges,
        edge_color="r",
        width=3,
    )

    # TODO option to save somewhere OR return the plot
