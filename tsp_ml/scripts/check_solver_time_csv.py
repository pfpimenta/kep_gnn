# -*- coding: utf-8 -*-
# script to check the CSV with results from eval_solver_time.py
import sys
from pathlib import Path

import pandas as pd

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)

from paths import RESULTS_FOLDER_PATH

csv_filepath = RESULTS_FOLDER_PATH / "recursive_solver_time.csv"
df = pd.read_csv(csv_filepath)

print(
    f"Mean time per num_nodes: {df.groupby('num_nodes')['prediction_elapsed_time'].mean()}"
)
breakpoint()
