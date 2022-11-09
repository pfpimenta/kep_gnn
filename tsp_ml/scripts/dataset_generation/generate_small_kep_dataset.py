# -*- coding: utf-8 -*-
# script to generate small instances for debugging.
# The instances are small enough for the predictions
# to be visualized clearly.
# Also, less instances are generated, to accelerate debugging time.
import sys
from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from generate_kep_dataset import generate_kep_dataset
from paths import get_dataset_folder_path

if __name__ == "__main__":
    step = "test_small"
    kep_dataset_dir = get_dataset_folder_path(dataset_name="KEP", step=step)
    generate_kep_dataset(
        num_instances=100,
        num_nodes=10,
        num_edges=30,
        node_type_distribution=[0.15, 0.7, 0.15],
        add_node_features=True,
        output_dir=kep_dataset_dir,
    )
