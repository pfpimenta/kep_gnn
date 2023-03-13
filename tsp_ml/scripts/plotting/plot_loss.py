# -*- coding: utf-8 -*-
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)
from paths import PLOTS_FOLDER_PATH, TRAINED_MODELS_FOLDER_PATH

# TRAINED_MODEL_NAME = "2022_12_10_19h00_KEP_GAT_PNA_CE"
TRAINED_MODEL_NAME = "2022_12_09_01h15_KEP_GAT_PNA_CE"
# TRAINED_MODEL_NAME = "2022_12_05_23h44_KEP_1L_GNN"
PLOT_FULL_TRAINING_LOSS = False
SAVE_PLOTS = False
SHOW_PLOTS = True

# TODO load all training loss values
if PLOT_FULL_TRAINING_LOSS:
    raise NotImplementedError()


def plot_line_graph(
    checkpoints: List[str],
    val_values: List[float],
    train_values: Optional[List[float]] = None,
    plot_content: str = "Loss",
    show_plots: bool = SHOW_PLOTS,
    save_plot: bool = SAVE_PLOTS,
):
    x_values = range(len(checkpoints))
    x_labels = checkpoints
    # breakpoint()
    plt.plot(x_values, val_values, color="red", marker="o", label="validation")
    if train_values:
        plt.plot(x_values, train_values, color="blue", marker="o", label="train")
    plt.title(f"Evolution of {plot_content} during training", fontsize=14)
    plt.xlabel("Epoch and Step", fontsize=14)
    plt.ylabel(plot_content, fontsize=14)
    plt.xticks(x_values, x_labels)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()

    if show_plots:
        plt.show()
    if save_plot:

        loss_plot_filename = TRAINED_MODEL_NAME + f"_{plot_content}.png".replace(
            " ", "_"
        )
        loss_plot_filepath = PLOTS_FOLDER_PATH / loss_plot_filename
        plt.savefig(loss_plot_filepath, dpi=300)
        print(f"Saved {loss_plot_filepath}")


model_checkpoints_folderpath = (
    TRAINED_MODELS_FOLDER_PATH / f"{TRAINED_MODEL_NAME}/checkpoints/"
)
checkpoint_list = sorted(os.listdir(model_checkpoints_folderpath))

val_loss_list = []
train_loss_list = []
val_score_list = []
train_loss_std_list = []
val_score_std_list = []  # TODO validar, plotar
for checkpoint in checkpoint_list:
    performance_json_filepath = (
        model_checkpoints_folderpath / checkpoint / "performance.json"
    )
    with open(performance_json_filepath, "r") as file:
        performance_dict = json.load(file)
        train_loss_list.append(performance_dict["mean_training_loss"])
        val_loss_list.append(performance_dict["mean_validation_loss"])
        val_score_list.append(performance_dict["mean_validation_solution_weight_sum"])
        train_loss_std_list.append(performance_dict["std_training_loss"])
        val_score_std_list.append(
            performance_dict["std_validation_solution_weight_sum"]
        )

plot_line_graph(
    checkpoints=checkpoint_list,
    val_values=val_loss_list,
    train_values=train_loss_list,
    plot_content="loss",
)
plot_line_graph(
    checkpoints=checkpoint_list,
    val_values=val_score_list,
    plot_content="score",
)

plot_line_graph(
    checkpoints=checkpoint_list,
    val_values=train_loss_std_list,
    plot_content="training loss STD",
)

# TODO validar, plotar

plot_line_graph(
    checkpoints=checkpoint_list,
    val_values=val_score_std_list,
    plot_content="validation score standard deviation",
)
