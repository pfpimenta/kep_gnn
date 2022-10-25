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
bar_labels = [
    "GreedyCycles",
    "GNN+GreedyCycles",
    "GreedyPaths",
    "GNN+GreedyPaths",
]
height = [13.71, 0.85, 203.26, 223.99]
y_pos = np.arange(len(bar_labels))

# create plot
plot = plt.bar(y_pos, height)

# add labels to the x-axis
plt.xticks(y_pos, bar_labels)
plt.xticks(rotation=30)

# save figure as a PNG
filename = "kep_results_barplot.png"
filepath = RESULTS_FOLDER_PATH / filename
plt.savefig(filepath)
