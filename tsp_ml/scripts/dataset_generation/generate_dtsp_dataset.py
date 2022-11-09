# -*- coding: utf-8 -*-
# script to randomly generate n instances of the Decision TSP problem
# and their optimal solution routes
import sys
from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from datasets.dtsp_dataset_generation import generate_dtsp_dataset
from paths import get_dataset_folder_path

if __name__ == "__main__":
    cost_deviation = 0.02
    tsp_train_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="train")
    dtsp_train_dataset_dir = get_dataset_folder_path(dataset_name="DTSP", step="train")
    generate_dtsp_dataset(
        tsp_instances_dir=tsp_train_dataset_dir,
        output_dir=dtsp_train_dataset_dir,
        cost_deviation=cost_deviation,
    )

    tsp_test_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="test")
    dtsp_test_dataset_dir = get_dataset_folder_path(dataset_name="DTSP", step="test")
    generate_dtsp_dataset(
        tsp_instances_dir=tsp_test_dataset_dir,
        output_dir=dtsp_test_dataset_dir,
        cost_deviation=cost_deviation,
    )

    tsp_val_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="val")
    dtsp_val_dataset_dir = get_dataset_folder_path(dataset_name="DTSP", step="val")
    generate_dtsp_dataset(
        tsp_instances_dir=tsp_val_dataset_dir,
        output_dir=dtsp_val_dataset_dir,
        cost_deviation=cost_deviation,
    )
