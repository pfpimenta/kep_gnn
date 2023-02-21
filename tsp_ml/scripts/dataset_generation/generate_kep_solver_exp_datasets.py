# -*- coding: utf-8 -*-
# script to generate datasets with instances of varying sizes
# to evaluate how much time the solver takes
import sys
from pathlib import Path

from generate_kep_dataset import generate_kep_dataset

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from paths import get_dataset_folder_path

if __name__ == "__main__":
    max_instance_size = 150  # num_nodes
    for num_nodes in range(max_instance_size + 1)[5:]:
        step = f"solver_exp/{num_nodes}"
        kep_dataset_dir = get_dataset_folder_path(dataset_name="KEP", step=step)
        # approximate the proportion of 300 nodes to 5500 edges:
        num_edges = int(num_nodes * 18.3)
        generate_kep_dataset(
            num_instances=100,
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_type_distribution=[0.15, 0.7, 0.15],
            add_node_features=True,
            output_dir=kep_dataset_dir,
        )
