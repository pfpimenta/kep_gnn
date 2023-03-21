# -*- coding: utf-8 -*-
"""
script to generate datasets of several sizes,
which will be measured with the script
tsp.scripts.check_dataset_size_consistency.py
"""
import sys
from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

# TODO move this to tsp/datasets
from datasets.kep_dataset_generation import generate_kep_dataset
from paths import get_dataset_folder_path

dataset_sizes = [10, 50, 100, 250, 500, 1000, 5000]
for dataset_size in dataset_sizes:
    output_dir = get_dataset_folder_path(
        dataset_name="KEP",
        step=f"debug_{dataset_size}",
    )
    generate_kep_dataset(
        num_instances=dataset_size,
        output_dir=output_dir,
    )
