# -*- coding: utf-8 -*-
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)
from paths import TRAINED_MODELS_FOLDER_PATH

TRAINED_MODEL_NAME = "2022_12_10_19h00_KEP_GAT_PNA_CE"
# TRAINED_MODEL_NAME = "2022_12_09_01h15_KEP_GAT_PNA_CE"
# TRAINED_MODEL_NAME = "2022_12_05_23h44_KEP_1L_GNN"
PLOT_FULL_TRAINING_LOSS = False

# TODO load all training loss values
if PLOT_FULL_TRAINING_LOSS:
    raise NotImplementedError()


def plot_line_graph(
    checkpoints: List[str],
    val_values: List[float],
    train_values: Optional[List[float]] = None,
    plot_content: str = "loss",
):
    x_values = range(len(checkpoints))
    x_labels = checkpoints
    plt.plot(x_values, val_values, color="red", marker="o")
    if train_values:
        plt.plot(x_values, train_values, color="blue", marker="o")
    plt.title(f"Evolution of {plot_content} during training", fontsize=14)
    plt.xlabel("Epoch and Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(x_values, x_labels)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()
    # TODO save
    # plt.savefig(loss_plot_filepath)


model_checkpoints_folderpath = (
    TRAINED_MODELS_FOLDER_PATH / f"{TRAINED_MODEL_NAME}/checkpoints/"
)
checkpoint_list = sorted(os.listdir(model_checkpoints_folderpath))

val_loss_list = []
train_loss_list = []
val_score_list = []
train_loss_std_list = []
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
