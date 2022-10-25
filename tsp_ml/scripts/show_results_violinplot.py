# -*- coding: utf-8 -*-
"""Script that plots the distribution of a given model's performance
(i.e. the distribution of the solution_weight_sum of the predictions
made on a test dataset) using a violin plot"""
import sys
from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from paths import get_evaluation_folder_path

# configure matplotlib to automatically adjust the plot size
rcParams.update({"figure.autolayout": True})

# load evaluation dataframes
trained_model_names = [
    "2022_10_17_19h17_GreedyCyclesModel",  # GreedyCycles
    "2022_10_21_23h55_KEP_GAT_PNA_CE",  # GNN + GreedyCycles
    "2022_09_19_23h55_GreedyPathsModel",  # GreedyPaths
    "2022_09_22_02h42_KEP_GAT_PNA_CE",  # GNN + GreedyPaths
]
df_list = []
for name in trained_model_names:
    eval_folder_path = get_evaluation_folder_path(
        dataset_name="KEP",
        trained_model_name=name,
        step="test",
    )
    csv_filepath = eval_folder_path / "test_eval.csv"
    print(f"Reading csv: {csv_filepath} ...")
    df = pd.read_csv(csv_filepath)
    df["trained_model_name"] = name
    df = df[["trained_model_name", "solution_weight_sum"]]
    df_list.append(df)
results_df = pd.concat(df_list)

# generate and save plot
plot = sns.violinplot(
    x=results_df["trained_model_name"], y=results_df["solution_weight_sum"]
)
# plot.set_xticklabels(rotation=30)
plt.xticks(rotation=30)
plt.gcf().subplots_adjust(bottom=0.15)
fig = plot.get_figure()
fig.savefig("kep_results_violinplot.png")
