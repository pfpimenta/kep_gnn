# -*- coding: utf-8 -*-
# script to evaluate the solver completion time
# in relation to the size of the input
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)

from kep_solver import solve_kep_recursive
from paths import (
    RESULTS_FOLDER_PATH,
    get_dataset_folder_path,
    get_predictions_folder_path,
)

DATASET_NAME = "KEP"
START_WITH_NUM_NODE = 15

if __name__ == "__main__":
    max_instance_size = 150  # num_nodes
    df_cols = ["num_nodes", "instance_id", "prediction_elapsed_time"]
    df = pd.DataFrame(columns=df_cols)
    csv_filepath = RESULTS_FOLDER_PATH / "recursive_solver_time.csv"
    for num_nodes in range(max_instance_size + 1)[START_WITH_NUM_NODE:]:
        step = f"solver_exp/{num_nodes}"
        # get folder paths
        kep_dataset_dir = get_dataset_folder_path(dataset_name=DATASET_NAME, step=step)
        predictions_dir = get_predictions_folder_path(
            dataset_name=DATASET_NAME,
            step=step,
            trained_model_name="RecursiveSolver",
        )
        predicted_instances_dir = predictions_dir / "predicted_instances"
        predicted_instances_dir.mkdir(parents=True, exist_ok=True)
        # run solver only for the instances that were not yet ran
        all_instances = [filepath.name for filepath in kep_dataset_dir.iterdir()]
        predicted_instances = [
            "kep_instance_" + filepath.name[:-8] + filepath.name[-3:]
            for filepath in predicted_instances_dir.iterdir()
        ]
        remaining_instances = [
            instance_filename
            for instance_filename in all_instances
            if instance_filename not in predicted_instances
        ]
        for i, filename in enumerate(remaining_instances):
            filepath = kep_dataset_dir / filename
            print(
                f"[num_nodes=={num_nodes}, {i+1}/{len(remaining_instances)}] solving: {filepath}"
            )
            # load KEP instance
            kep_instance = torch.load(filepath)
            # use solver, measure time
            start_time = time.time()
            solved_instance = solve_kep_recursive(kep_instance=kep_instance)
            end_time = time.time()
            prediction_elapsed_time = end_time - start_time  # in seconds
            # save prediction
            solved_instance_filepath = predicted_instances_dir / (
                solved_instance.id + "_pred.pt"
            )
            torch.save(solved_instance, solved_instance_filepath)
            print(f"Saved {solved_instance_filepath}")
            # collect elapsed time info
            pred_time_dict = {
                "num_nodes": num_nodes,
                "instance_id": solved_instance.id,
                "prediction_elapsed_time": prediction_elapsed_time,
            }
            df = pd.DataFrame.from_records([pred_time_dict])
            # print(
            #     f"\n\n\nDEBUG mean time per instance size: {df.groupby('num_nodes')['prediction_elapsed_time'].mean()}\n\n"
            # )
            # save time info (append if it already exist)
            df.to_csv(csv_filepath, mode="a", header=not os.path.exists(csv_filepath))
