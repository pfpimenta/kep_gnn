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
# TRAINED_MODEL_NAME = "2022_10_21_23h55_KEP_GAT_PNA_CE" # GNN+GreedyCycles do tcc
TRAINED_MODEL_NAME = "2022_12_10_19h00_KEP_GAT_PNA_CE"  # outra GNN+GreedyCycles
# TRAINED_MODEL_NAME = "2022_12_09_01h15_KEP_GAT_PNA_CE"  # GNN+paths 230
# TRAINED_MODEL_NAME = "2022_12_05_23h44_KEP_1L_GNN"
PLOT_FULL_TRAINING_LOSS = False
SAVE_PLOTS = True
SHOW_PLOTS = False

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
    num_epochs: Optional[int] = None,
):
    x_values = range(len(checkpoints))
    x_labels = sorted(
        [checkpoint for checkpoint in checkpoints if checkpoint[-5:] == "00500"]
    )
    # x_labels = []
    # x_labels = checkpoint_labels
    if num_epochs:
        x_labels = [epoch + 1 for epoch in range(num_epochs)]
        x_labels_locations = [
            position * num_epochs for position in range(len(x_labels))
        ]
    else:
        x_labels = sorted(
            [checkpoint for checkpoint in checkpoints if checkpoint[-5:] == "00500"]
        )
        x_labels_locations = [position * 20 for position in range(len(x_labels))]
    # breakpoint()
    plt.clf()  # clear previous figures
    plt.plot(x_values, val_values, color="red", marker="o", label="validation")
    if train_values:
        plt.plot(x_values, train_values, color="blue", marker="o", label="train")
    # plt.title(f"Evolution of {plot_content} during training", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(plot_content, fontsize=14)
    plt.xticks(x_labels_locations, x_labels)
    # plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()

    if show_plots:
        plt.show()
    if save_plot:

        loss_plot_filename = (
            TRAINED_MODEL_NAME + f"_{plot_content.lower()}.png".replace(" ", "_")
        )
        loss_plot_filepath = PLOTS_FOLDER_PATH / loss_plot_filename
        plt.savefig(loss_plot_filepath, dpi=300)
        print(f"Saved {loss_plot_filepath}")


model_checkpoints_folderpath = (
    TRAINED_MODELS_FOLDER_PATH / f"{TRAINED_MODEL_NAME}/checkpoints/"
)
checkpoint_list = sorted(os.listdir(model_checkpoints_folderpath))
training_report_filepath = (
    TRAINED_MODELS_FOLDER_PATH / f"{TRAINED_MODEL_NAME}" / "training_report.json"
)
with open(training_report_filepath, "r") as file:
    training_report = json.load(file)
num_epochs = training_report["epoch"]

train_loss_list = []
train_loss_std_list = []
val_loss_list = []
val_loss_std_list = []
val_score_list = []
val_score_std_list = []  # TODO validar, plotar
invalid_checkpoints = []
for checkpoint in checkpoint_list:
    performance_json_filepath = (
        model_checkpoints_folderpath / checkpoint / "performance.json"
    )
    # print(f"Reading {performance_json_filepath}")
    with open(performance_json_filepath, "r") as file:
        performance_dict = json.load(file)
        if performance_dict["std_training_loss"] == 0.0:
            # fix to avoid plotting 1-batch validations
            invalid_checkpoints.append(checkpoint)
        else:
            train_loss_list.append(performance_dict["mean_training_loss"])
            val_loss_list.append(performance_dict["mean_validation_loss"])
            val_score_list.append(
                performance_dict["mean_validation_solution_weight_sum"]
            )
            train_loss_std_list.append(performance_dict["std_training_loss"])
            val_loss_std_list.append(performance_dict["std_validation_loss"])
            val_score_std_list.append(
                performance_dict["std_validation_solution_weight_sum"]
            )

# fix to avoid plotting 1-batch validations:
checkpoint_list = list(set(checkpoint_list) - set(invalid_checkpoints))


plot_line_graph(
    checkpoints=checkpoint_list,
    val_values=val_loss_list,
    train_values=train_loss_list,
    plot_content="Loss",
    num_epochs=num_epochs,
)
plot_line_graph(
    checkpoints=checkpoint_list,
    val_values=val_score_list,
    plot_content="Score in validation set",
    num_epochs=num_epochs,
)
plot_line_graph(
    checkpoints=checkpoint_list,
    train_values=train_loss_std_list,
    val_values=val_loss_std_list,
    plot_content="Loss standard deviation",
    num_epochs=num_epochs,
)
plot_line_graph(
    checkpoints=checkpoint_list,
    val_values=val_score_std_list,
    plot_content="Validation score standard deviation",
    num_epochs=num_epochs,
)
