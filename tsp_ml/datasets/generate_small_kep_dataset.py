# -*- coding: utf-8 -*-
# script to generate small instances for debugging.
# The instances are small enough for the predictions
# to be visualized clearly.
# Also, less instances are generated, to accelerate debugging time.
import sys

# to allow imports from outside the tsp_ml/datasets/ package
sys.path.insert(0, "/home/pimenta/tsp_ml/tsp_ml")
from generate_kep_dataset import generate_kep_dataset
from paths import get_dataset_folder_path

if __name__ == "__main__":
    step = "test_small"
    kep_dataset_dir = get_dataset_folder_path(dataset_name="KEP", step=step)
    generate_kep_dataset(
        num_instances=100,
        num_nodes=10,
        num_edges=30,
        add_node_features=True,
        output_dir=kep_dataset_dir,
    )
