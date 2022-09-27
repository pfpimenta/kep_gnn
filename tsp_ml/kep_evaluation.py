# -*- coding: utf-8 -*-
# Script with functions to evaluate KEP predictions
import os
import sys
from random import randint
from typing import Any, Dict

import pandas as pd
import torch
from paths import get_eval_results_folder_path, get_predictions_folder_path
from plot_kep import generate_kep_plot
from torch import Tensor, tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.degree import degree
from tqdm import tqdm

DATASET_NAME = "KEP"
# TRAINED_MODEL_NAME = "2022_09_19_23h55_GreedyPathsModel"
TRAINED_MODEL_NAME = "2022_09_22_02h42_KEP_GAT_PNA_CE"
# weather to redo evaluation and save it overwriting the pre-existing CSV:
OVERWRITE_RESULTS = True


def minor_kep_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    save_plot: bool = False,
):
    print("\n\nMinor evaluation: predicting on 3 random instances...")
    # predict 3 random instances
    num_instances = len(dataloader.dataset)
    num_random_instances = 3
    folderpath = "/home/pimenta/tsp_ml/results"  # TODO change it
    for i in range(num_random_instances):
        # randomly choose an instance
        instance_index = randint(0, num_instances - 1)
        instance = dataloader.dataset[instance_index]

        # predict
        instance.scores = model(instance)
        instance.pred = model.predict(data=instance)

        # print prediction info
        prediction_info = evaluate_kep_instance_prediction(
            predicted_instance=instance,
        )
        print(f"\nPrediction info: {prediction_info}")

        if save_plot:
            generate_kep_plot(predicted_instance=instance, folderpath=folderpath)


def evaluate_kep_instance_prediction(
    predicted_instance: Data,
) -> Dict[str, Any]:
    """Evaluates a predicted instance of the Kidney-Exchange Problem,
    where the prediction is edge binary classification (an edge may be in the solution or not).
    Checks if the solution is valid, how many of the solution edges are valid,
    the number of invalid PDP nodes (PDP nodes that donate a kidney without receiving one),
    the total weight sum of the solution, and the sum of the weights not used in the solution."""
    # TODO DRY refactor (kep_loss.edges_restriction_loss)
    pred = predicted_instance.pred
    edge_index = predicted_instance.edge_index
    edge_weights = predicted_instance.edge_weights
    num_nodes = predicted_instance.x.shape[0]
    src, dst = edge_index

    solution_edge_indexes = torch.nonzero(pred).flatten()
    total_num_edges_in_solution = torch.sum(pred)
    total_num_edges = edge_index.shape[1]

    is_solution_valid = True
    node_degrees = {}
    num_valid_edges = {}
    num_invalid_edges = {}
    valid_edges_percentage = {}

    for (name, edge_node_ids) in [("src", src), ("dst", dst)]:
        solution_edge_node_ids = torch.index_select(
            input=edge_node_ids, dim=0, index=solution_edge_indexes
        )
        unique_solution_node_ids = torch.unique(solution_edge_node_ids)
        num_nodes_in_solution = solution_edge_node_ids.shape[0]
        num_unique_nodes_in_solution = unique_solution_node_ids.shape[0]

        # get degree considering ONLY edges in the solution
        node_degrees[name] = degree(index=solution_edge_node_ids, num_nodes=num_nodes)

        # get num_valid_edges and num_invalid_edges
        # TODO this may be optimized: use node_degrees approach instead (not yet necessary)
        num_valid_edges[name] = num_unique_nodes_in_solution
        num_invalid_edges[name] = num_nodes_in_solution - num_unique_nodes_in_solution
        valid_edges_percentage[name] = (
            num_valid_edges[name] / total_num_edges_in_solution
        )
        # solution is only valid if there are no invalid edges
        if num_invalid_edges[name] > 0:
            is_solution_valid = False

    # check conditional donation restriction for PDP nodes
    pdp_node_mask = predicted_instance.x[:, 3]
    # degree in must be >= than degree out. check if any is <
    num_invalid_pdp_nodes = torch.sum(
        (node_degrees["dst"] < node_degrees["src"]).to(torch.int16) * pdp_node_mask
    )
    if num_invalid_pdp_nodes > 0:
        is_solution_valid = False
    # sum of all edge weights
    total_weight_sum = sum(edge_weights)
    # sum of all edge weights in solution
    solution_weight_sum = sum(edge_weights[solution_edge_indexes])
    # sum of all edge weights outside solution
    not_solution_weight_sum = total_weight_sum - solution_weight_sum

    # return evaluation in a dict
    kep_prediction_evaluation = {
        "instance_id": str(predicted_instance.id),
        "is_solution_valid": int(is_solution_valid),
        "total_num_edges": int(total_num_edges),
        "total_num_nodes": num_nodes,
        "num_invalid_pdp_nodes": num_invalid_pdp_nodes,
        "num_valid_edges_src": int(num_valid_edges["src"]),
        "num_invalid_edges_src": int(num_invalid_edges["src"]),
        "valid_edges_percentage_src": float(valid_edges_percentage["src"]),
        "num_valid_edges_dst": int(num_valid_edges["dst"]),
        "num_invalid_edges_dst": int(num_invalid_edges["dst"]),
        "valid_edges_percentage_dst": float(valid_edges_percentage["dst"]),
        "total_weight_sum": float(total_weight_sum),
        "solution_weight_sum": float(solution_weight_sum),
        "not_solution_weight_sum": float(not_solution_weight_sum),
    }
    return kep_prediction_evaluation


def evaluate_kep_predicted_instances(predictions_dir: str) -> pd.DataFrame:
    """Given a folder with predicted KEP instances, evaluates each one of them,
    gathers the evaluation results on a pandas DataFrame, and returns it."""
    predicted_instances_dir = predictions_dir / "predicted_instances"
    eval_df = pd.DataFrame()
    # load and evaluate one predicted instance at a time
    for filename in tqdm(
        os.listdir(predicted_instances_dir), desc="Evaluation", file=sys.stdout
    ):
        filepath = predicted_instances_dir / filename
        predicted_instance = torch.load(filepath)
        kep_prediction_evaluation = evaluate_kep_instance_prediction(
            predicted_instance=predicted_instance,
        )
        instance_eval_df = pd.DataFrame([kep_prediction_evaluation])
        eval_df = pd.concat([eval_df, instance_eval_df], ignore_index=True)
    return eval_df


def print_evaluation_overview(eval_df: pd.DataFrame) -> None:
    # dataset information
    print(f"Total number of instances: {len(eval_df)}")
    print(f"Mean total_num_edges: {eval_df['total_num_edges'].mean():.2f}")
    print(f"Mean total_num_nodes: {eval_df['total_num_nodes'].mean():.2f}")
    # instance validity information:
    print(f"Number of valid solutions: {eval_df['is_solution_valid'].sum()}")
    print(f"Valid solution percentage: {100*eval_df['is_solution_valid'].mean():.2f}%")
    # edge validity information:
    print(f"Mean num_valid_edges_src: {eval_df['num_valid_edges_src'].mean():.2f}")
    print(f"Mean num_invalid_edges_src: {eval_df['num_invalid_edges_src'].mean():.2f}")
    print(
        f"Mean valid_edges_percentage_src: {eval_df['valid_edges_percentage_src'].mean():.2f}"
    )
    print(f"Mean num_valid_edges_dst: {eval_df['num_valid_edges_dst'].mean():.2f}")
    print(f"Mean num_invalid_edges_dst: {eval_df['num_invalid_edges_dst'].mean():.2f}")
    print(
        f"Mean valid_edges_percentage_dst: {eval_df['valid_edges_percentage_dst'].mean():.2f}"
    )
    # PDP conditional donation validity:
    print(f"Mean num_invalid_pdp_nodes: {eval_df['num_invalid_pdp_nodes'].mean():.2f}")
    # edge weight information:
    print(f"Mean total_weight_sum: {eval_df['total_weight_sum'].mean():.2f}")
    print(f"Mean solution_weight_sum: {eval_df['solution_weight_sum'].mean():.2f}")
    print(
        f"Mean not_solution_weight_sum: {eval_df['not_solution_weight_sum'].mean():.2f}"
    )
    eval_df["solution_weight_percentage"] = (
        eval_df["solution_weight_sum"] / eval_df["total_weight_sum"]
    )
    eval_df["not_solution_weight_percentage"] = (
        eval_df["not_solution_weight_sum"] / eval_df["total_weight_sum"]
    )
    print(
        f"Mean solution_weight_percentage: {eval_df['solution_weight_percentage'].mean():.2f}"
    )
    print(
        f"Mean not_solution_weight_percentage: {eval_df['not_solution_weight_percentage'].mean():.2f}"
    )


def kep_evaluation(
    step: str,
    trained_model_name: str,
    print_overview: bool = True,
    dataset_name: str = "KEP",
) -> None:
    """Evaluates the predictions made by a model trained for the
    Kidney-Exchange Problem (KEP) on the test, train or val dataset
    (indicated by 'step' param).
    The evaluation results for each instance are saved in a CSV file.
    If 'print_overview' is set to True, a summary of the evaluation is printed.
    """

    print(
        f"\n\nEvaluating {trained_model_name}'s predictions on the {step} KEP dataset"
    )
    predictions_dir = get_predictions_folder_path(
        dataset_name=dataset_name,
        step=step,
        trained_model_name=trained_model_name,
    )
    eval_df = evaluate_kep_predicted_instances(predictions_dir=predictions_dir)

    # get output CSV filepath
    output_dir = get_eval_results_folder_path(
        dataset_name=dataset_name,
        step=step,
        trained_model_name=trained_model_name,
    )
    csv_filepath = output_dir / f"{step}_eval.csv"
    # save dataframe in a CSV
    eval_df.to_csv(csv_filepath)
    print(f"\n\Saved results at {csv_filepath}")

    # print evaluation overview (stats on the whole dataset)
    # TODO save this in a JSON
    if print_overview:
        print(f"\nEvaluation overview for {trained_model_name} on {step} dataset:")
        print_evaluation_overview(eval_df=eval_df)


if __name__ == "__main__":
    # compute and save evaluation for predictions on the train, test and val datasets
    # steps_to_predict = ["train", "test", "val"]
    steps_to_predict = ["test"]
    # steps_to_predict = ["test_small"]
    for step in steps_to_predict:
        kep_evaluation(
            step=step,
            trained_model_name=TRAINED_MODEL_NAME,
            dataset_name=DATASET_NAME,
        )
