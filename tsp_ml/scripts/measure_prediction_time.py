# -*- coding: utf-8 -*-
# predicts the solutions on every instance of the test dataset
# while keeping how much time it took for each of them
import os
import sys
import time
from pathlib import Path

import pandas as pd

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)
import torch
from dataset_utils import get_dataset, print_dataset_information
from model_utils import load_model
from paths import PREDICTION_TIME_FOLDER_PATH
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

DATASET_NAME = "KEP"
#  gambi genial 220 e poucos
# TRAINED_MODEL_NAME = "2022_12_09_01h15_KEP_GAT_PNA_CE"  # e006_s03500 ou e009_09000
# TRAINED_MODEL_NAME = "2023_03_16_23h16_KEP_GAT_PNA_CE"  # nova GNN+GreedyCycles pro tcc
# TRAINED_MODEL_NAME = "2023_03_17_12h09_KEP_GAT_PNA_CE" # UnsupervisedGNN sem regularization loss
# TRAINED_MODEL_NAME = "2023_03_17_15h55_KEP_GAT_PNA_CE" # UnsupervisedGNN com regularization loss
TRAINED_MODELS = [
    # (trained_model_name, checkpoint, predict_method)
    ("2022_09_19_23h55_GreedyPathsModel", None, "greedy_paths"),  # GreedyPaths
    ("2022_10_17_19h17_GreedyCyclesModel", None, "greedy_cycles"),  # GreedyCycles
    (
        "2022_12_09_01h15_KEP_GAT_PNA_CE",
        "e006_s03500",
        "greedy_paths",
    ),  # GNN+GreedyPaths
    ("2023_03_16_23h16_KEP_GAT_PNA_CE", None, "greedy_cycles"),  # GNN+GreedyCycles
    (
        "2023_03_17_12h09_KEP_GAT_PNA_CE",
        None,
        None,
    ),  # UnsupervisedGNN sem regularization loss
    (
        "2023_03_17_15h55_KEP_GAT_PNA_CE",
        None,
        None,
    ),  # UnsupervisedGNN com regularization loss
]

# CHECKPOINT = None
# CHECKPOINT = "e006_s03500"
STEP = "test"
# PREDICT_METHOD = "greedy_paths"


def measure_prediction_time(
    # step,
    trained_model_name: str,
    device: torch.device,
    dataset: Dataset,
) -> None:
    """predicts the solutions on every instance of the dataset while
    keeping how much time it took for each of them,
    then saves the results in a CSV.
    """
    # load model
    model = load_model(
        trained_model_name=trained_model_name,
        dataset=dataset,
        predict_method=predict_method,
        checkpoint=checkpoint,
        device=device,
    )

    csv_filepath = (
        PREDICTION_TIME_FOLDER_PATH
        / f"{trained_model_name}_{device}_{STEP}_prediction_time.csv"
    )
    print(f"Predictions times will be saved to {csv_filepath}.")

    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4
    )

    # predict and save elapsed time for predictions
    for i, batch in enumerate(tqdm(dataloader, desc="Prediction", file=sys.stdout)):
        start_time = time.time()

        batch = batch.to(device)

        # predict
        batch.scores = model(data=batch)
        pred = model.predict(data=batch)
        end_time = time.time()
        prediction_elapsed_time = end_time - start_time  # in seconds

        # save prediction time info in CSV (append if it already exist)
        pred_time_dict = {
            "num_nodes": batch.num_nodes,
            "instance_id": batch.id[0],
            "prediction_elapsed_time": prediction_elapsed_time,
        }
        df = pd.DataFrame.from_records([pred_time_dict])
        df.to_csv(csv_filepath, mode="a", header=not os.path.exists(csv_filepath))
        # print(f"Saved prediction time ({prediction_elapsed_time} seconds) to {csv_filepath}")

    print(f"Saved prediction times to {csv_filepath}.")


if __name__ == "__main__":
    # select either CPU or GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using {device}")

    # setup data
    dataset = get_dataset(dataset_name=DATASET_NAME, step=STEP)
    print_dataset_information(dataset=dataset)

    for trained_model_name, checkpoint, predict_method in TRAINED_MODELS:
        # measure prediction times and save them to a CSV
        measure_prediction_time(
            trained_model_name=trained_model_name,
            device=device,
            dataset=dataset,
        )

    # dataframe for the prediction time results
    # df_cols = ["num_nodes", "instance_id", "prediction_elapsed_time"]
    # df = pd.DataFrame(columns=df_cols)

    # csv_filepath = (
    #     RESULTS_FOLDER_PATH / f"{STEP}_{TRAINED_MODEL_NAME}_prediction_time.csv"
    # )

    # # predict and save elapsed time for predictions
    # for i, batch in enumerate(tqdm(dataloader, desc="Prediction", file=sys.stdout)):
    #     start_time = time.time()

    #     batch = batch.to(device)

    #     # predict
    #     batch.scores = model(data=batch)
    #     pred = model.predict(data=batch)
    #     end_time = time.time()
    #     prediction_elapsed_time = end_time - start_time  # in seconds

    #     # save prediction time info in CSV (append if it already exist)
    #     pred_time_dict = {
    #         "num_nodes": batch.num_nodes,
    #         "instance_id": batch.id[0],
    #         "prediction_elapsed_time": prediction_elapsed_time,
    #     }
    #     df = pd.DataFrame.from_records([pred_time_dict])
    #     df.to_csv(csv_filepath, mode="a", header=not os.path.exists(csv_filepath))
    #     # print(f"Saved prediction time ({prediction_elapsed_time} seconds) to {csv_filepath}")

    # print(f"Saved prediction times to {csv_filepath}.")
