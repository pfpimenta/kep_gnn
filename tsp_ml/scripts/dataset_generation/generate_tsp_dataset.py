# -*- coding: utf-8 -*-
# script to randomly generate NUM_SAMPLES instances of the Travelling Salesperson Problem (TSP)
# and their optimal solution routes
import sys
from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)


from datasets.tsp_dataset_generation import generate_tsp_dataset
from paths import get_dataset_folder_path

NUM_SAMPLES = 1000  # usaram 2**20 no paper


if __name__ == "__main__":
    # train dataset
    tsp_train_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="train")
    generate_tsp_dataset(
        num_samples=NUM_SAMPLES,
        output_dir=tsp_train_dataset_dir,
    )
    # test dataset
    tsp_test_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="test")
    generate_tsp_dataset(
        num_samples=NUM_SAMPLES,
        output_dir=tsp_test_dataset_dir,
    )
    # validation dataset
    tsp_val_dataset_dir = get_dataset_folder_path(dataset_name="TSP", step="val")
    generate_tsp_dataset(
        num_samples=NUM_SAMPLES,
        output_dir=tsp_val_dataset_dir,
    )
