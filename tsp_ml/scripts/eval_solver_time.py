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

if __name__ == "__main__":
    max_instance_size = 150  # num_nodes
    df_cols = ["num_nodes", "instance_id", "prediction_elapsed_time"]
    df = pd.DataFrame(columns=df_cols)
    csv_filepath = RESULTS_FOLDER_PATH / "recursive_solver_time.csv"
    for num_nodes in range(max_instance_size + 1)[5:]:
        step = f"solver_exp/{num_nodes}"
        kep_dataset_dir = get_dataset_folder_path(dataset_name=DATASET_NAME, step=step)
        pred_times_list = []
        for i, filepath in enumerate(kep_dataset_dir.iterdir()):
            # load KEP instance
            kep_instance = torch.load(filepath)
            # use solver, measure time
            start_time = time.time()
            solved_instance = solve_kep_recursive(kep_instance=kep_instance)
            end_time = time.time()
            prediction_elapsed_time = end_time - start_time  # in seconds
            # save prediction
            predictions_dir = get_predictions_folder_path(
                dataset_name=DATASET_NAME,
                step=step,
                trained_model_name="RecursiveSolver",
            )
            predicted_instances_dir = predictions_dir / "predicted_instances"
            predicted_instances_dir.mkdir(parents=True, exist_ok=True)
            solved_instance_filepath = predicted_instances_dir / (
                solved_instance.id + "_pred.pt"
            )
            torch.save(solved_instance, solved_instance_filepath)
            print(
                f"[num_nodes=={num_nodes}, {i+1}/100] Saved {solved_instance_filepath}"
            )
            # collect elapsed time info
            pred_time_dict = {
                "num_nodes": num_nodes,
                "instance_id": solved_instance.id,
                "prediction_elapsed_time": prediction_elapsed_time,
            }
            pred_times_list.append(pred_time_dict)

            # df = df.append(pred_time_dict, ignore_index=True)
        df_new_row = pd.DataFrame.from_records(pred_times_list)
        df = pd.concat([df, df_new_row])
        print(
            f"\n\n\nDEBUG mean time per instance size: {df.groupby('num_nodes')['prediction_elapsed_time'].mean()}\n\n"
        )
        # save time info (append if it already exist)
        df.to_csv(csv_filepath, mode="a", header=not os.path.exists(csv_filepath))
