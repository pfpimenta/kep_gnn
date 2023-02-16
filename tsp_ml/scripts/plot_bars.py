# -*- coding: utf-8 -*-
"""Simple script top show results graph,
with the standard deviation as error bars."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)
from paths import RESULTS_FOLDER_PATH

# configure matplotlib to automatically adjust the plot size
rcParams.update({"figure.autolayout": True})

# data necessary for the plot
dataset_sizes = [10, 11, 50, 51, 100, 101, 250, 251, 500, 501, 1000, 1001, 5000, 5001]
bar_labels = [f"{size}" for size in dataset_sizes]
mean_weight_sum = [
    173.33,
    202.59,
    201.47,
    205.45,
    204.51,
    199.55,
    202.79,
    207.07,
    202.81,
    203.49,
    203.23,
    201.47,
    203.24,
    204.51,
]
std_weight_sum = [
    59.10,
    37.10,
    37.86,
    29.82,
    35.72,
    33.14,
    37.64,
    28.34,
    34.72,
    37.15,
    35.93,
    37.23,
    35.23,
    34.30,
]
y_pos = np.arange(len(bar_labels))

# create plot
# plot = plt.bar(y_pos, mean_weight_sum)
plot = plt.bar(y_pos, std_weight_sum)

# add labels to the x-axis
plt.xticks(y_pos, bar_labels)
plt.xticks(rotation=30)

# save figure as a PNG
filename = "dataset_size_barplot.png"
filepath = RESULTS_FOLDER_PATH / filename
plt.savefig(filepath)
