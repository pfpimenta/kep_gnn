# -*- coding: utf-8 -*-
# script to generate kep_2015_table_s3.csv
# which contains the data from Table S3 of https://www.pnas.org/doi/epdf/10.1073/pnas.1421853112
import sys
from pathlib import Path

import pandas as pd

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from paths import PROJECT_FOLDER_PATH

# create dataframe
first_entry = {
    "NDDs": [
        3,
        10,
        6,
        5,
        6,
        6,
        6,
        10,
        3,
        10,
        7,
        6,
        10,
        1,
        11,
        6,
        10,
        6,
        10,
        7,
        8,
        4,
        3,
        7,
        2,
    ],
    "PDPs": [
        202,
        156,
        263,
        284,
        324,
        328,
        312,
        152,
        269,
        257,
        255,
        215,
        255,
        310,
        257,
        261,
        256,
        330,
        256,
        291,
        275,
        289,
        199,
        198,
        289,
    ],
    "num_edges": [
        4706,
        1109,
        8939,
        10126,
        13175,
        13711,
        13045,
        1125,
        2642,
        2461,
        2390,
        6145,
        2550,
        4463,
        2502,
        8915,
        2411,
        13399,
        2347,
        3771,
        3158,
        3499,
        2581,
        4882,
        8346,
    ],
    # "recursive_time": [0.148],
    # "tsp_time": [0.031]
}
df = pd.DataFrame(first_entry)
df["num_nodes"] = df["NDDs"] + df["PDPs"]

# save to CSV
csv_filename = "kep_2015_table_s3.csv"
csv_filepath = PROJECT_FOLDER_PATH / "data" / "KEP" / csv_filename
df.to_csv(csv_filepath)
