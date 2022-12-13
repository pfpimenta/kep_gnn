# -*- coding: utf-8 -*-
# script to randomly generate instances of the Kidney Exchange Problem (KEP)
# TODO: generate their optimal solution routes
import sys
from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from datasets.kep_dataset_generation import generate_kep_dataset
from paths import get_dataset_folder_path

## script parameters
NUM_INSTANCES = {"train": 10000, "test": 10000, "val": 100}
NUM_NODES = 300
NUM_EDGES = 5500
NODE_TYPES = [
    "NDD",  # non-directed donors
    "PDP",  # patient-donor pair
    "P",  # pacient without a pair
]
NODE_TYPE_DISTRIBUTION = [0.05, 0.9, 0.05]


if __name__ == "__main__":
    # generate and save train, test, and val KEP datasets
    for step in ["train", "test", "val"]:
        # for step in ["val"]:
        kep_dataset_dir = get_dataset_folder_path(dataset_name="KEP", step=step)
        generate_kep_dataset(
            num_instances=NUM_INSTANCES[step],
            num_nodes=NUM_NODES,
            num_edges=NUM_EDGES,
            node_types=NODE_TYPES,
            node_type_distribution=NODE_TYPE_DISTRIBUTION,
            add_node_features=True,
            output_dir=kep_dataset_dir,
        )
