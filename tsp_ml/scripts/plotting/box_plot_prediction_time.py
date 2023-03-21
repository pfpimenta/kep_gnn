# -*- coding: utf-8 -*-
# generates a box/scatter plot from the results
# of prediction time measured for each method for each instance
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from paths import RESULTS_FOLDER_PATH

DEVICE = "cpu"
STEP = "test"
TRAINED_MODELS = [
    # (trained_model_name, plot_model_name)
    # ("2022_09_19_23h55_GreedyPathsModel", "GreedyPaths"),  # GreedyPaths
    ("2022_10_17_19h17_GreedyCyclesModel", "GreedyCycles"),  # GreedyCycles
    ("2022_12_09_01h15_KEP_GAT_PNA_CE", "GNN+GreedyPaths"),  # GNN+GreedyPaths
    # ("", None, "greedy_paths"), # GNN+GreedyCycles
    # ("2023_03_16_23h16_KEP_GAT_PNA_CE", None, "greedy_cycles"),  # nova GNN+GreedyCycles pro tcc
    # ("2023_03_17_12h09_KEP_GAT_PNA_CE", None, "greedy_paths"), # UnsupervisedGNN sem regularization loss
    # ("2023_03_17_15h55_KEP_GAT_PNA_CE", None, "greedy_paths"), # UnsupervisedGNN com regularization loss
]


if __name__ == "__main__":
    df_list = []
    for trained_model_name, plot_model_name in TRAINED_MODELS:
        # load prediction time CSV
        csv_filepath = (
            RESULTS_FOLDER_PATH
            / f"{trained_model_name}_{DEVICE}_{STEP}_prediction_time.csv"
        )
        print(f"Loading predictions times from {csv_filepath}.")
        df = pd.read_csv(csv_filepath)

        # remove duplicates
        df = df.drop_duplicates("instance_id")

        # add plot_model_name to a col
        df["plot_model_name"] = plot_model_name

        df_list.append(df)

    # gather all results in a single dataframe
    plot_df = pd.concat(df_list)

# breakpoint()
plot = sns.boxplot(plot_df, x="plot_model_name", y="prediction_elapsed_time")
plot.set_ylabel("Prediction time in seconds")
# plot.set_xlabel("Number of nodes")

plt.show()
