# -*- coding: utf-8 -*-
"""Script that plots the distribution of a given model's performance
(i.e. the distribution of the solution_weight_sum of the predictions
made on a test dataset) using a violin plot"""
import sys
from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from paths import PLOTS_FOLDER_PATH, get_evaluation_folder_path

# configure matplotlib to automatically adjust the plot size
rcParams.update({"figure.autolayout": True})

# load evaluation dataframes
trained_model_names = [
    "2022_10_17_19h17_GreedyCyclesModel",  # GreedyCycles
    "2022_10_21_23h55_KEP_GAT_PNA_CE",  # GNN + GreedyCycles tcc old
    "2022_12_10_19h00_KEP_GAT_PNA_CE",  # GNN + GreedyCycles tcc new?
    "2022_09_19_23h55_GreedyPathsModel",  # GreedyPaths
    # "2022_09_22_02h42_KEP_GAT_PNA_CE",  # GNN + GreedyPaths
    # "2022_10_25_16h24_KEP_GAT_PNA_CE",  # GNN + GreedyPaths
    "2022_12_09_01h15_KEP_GAT_PNA_CE",  # GNN + GreedyPaths
]
method_names = [
    "GreedyCycles",
    "GNN+GreedyCycles",
    "GreedyPaths",
    "GNN+GreedyPaths",
]

df_list = []
for trained_model_name, method_name in zip(trained_model_names, method_names):
    eval_folder_path = get_evaluation_folder_path(
        dataset_name="KEP",
        trained_model_name=trained_model_name,
        step="test",
    )
    csv_filepath = eval_folder_path / "test_eval.csv"
    print(f"Reading csv: {csv_filepath}")
    df = pd.read_csv(csv_filepath)
    df["trained_model_name"] = trained_model_name
    df["method_name"] = method_name
    df = df[["solution_weight_sum", "method_name"]]
    df_list.append(df)
results_df = pd.concat(df_list)

# generate and save plot
plt.clf()
# plot = sns.violinplot(
plot = sns.boxplot(x=results_df["method_name"], y=results_df["solution_weight_sum"])
plot.set(xlabel="", ylabel="Score")
# plot.set_xticklabels(rotation=30)
plt.gcf().subplots_adjust(bottom=0.15)
fig = plot.get_figure()
filename = "kep_results_violinplot.png"
filepath = PLOTS_FOLDER_PATH / filename
fig.savefig(filepath)
print(f"Saved {filepath}")
