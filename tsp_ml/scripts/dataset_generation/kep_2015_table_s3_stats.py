# -*- coding: utf-8 -*-
""" script to print basic stats about kep_2015_table_s3.csv
which contains the data from Table S3 of https://www.pnas.org/doi/epdf/10.1073/pnas.1421853112
"""

import sys
from pathlib import Path

import pandas as pd

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)
from paths import PROJECT_FOLDER_PATH

# load CSV
csv_filename = "kep_2015_table_s3.csv"
csv_filepath = PROJECT_FOLDER_PATH / "data" / "KEP" / csv_filename
df = pd.read_csv(csv_filepath)

# print stats
print(df)
print(f"Mean num_nodes: {df['num_nodes'].mean()}")
print(f"Mean num_edges: {df['num_edges'].mean()}")
breakpoint()
