# -*- coding: utf-8 -*-
"""
script to measure how much the dataset changes when subsampled
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from matplotlib import rcParams

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)
from dataset_utils import get_dataset
from kep_evaluation import kep_evaluation
from model_utils import load_model
from paths import PLOTS_FOLDER_PATH, get_predictions_folder_path
from predict import predict


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """
    # TODO normalize vector size
    while len(p) > len(q):
        q.append(0)
    while len(q) > len(p):
        p.append(0)
    # vector_size = max(len(p), len(q))
    # print(vector_size)

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using {device}")

# load 10k instances dataset
train_dataset = get_dataset(dataset_name="KEP", step="train")
train_in_degree_histogram = [int(x) for x in list(train_dataset.in_degree_histogram)]
# breakpoint()

# load GreedyPaths model
trained_model_name = "2022_09_19_23h55_GreedyPathsModel"
model = load_model(
    trained_model_name=trained_model_name,
    dataset=train_dataset,
    device=device,
)

dataset_sizes = [10, 50, 100, 250, 500, 1000, 5000]
js_distances = []
for dataset_size in dataset_sizes:
    # load dataset
    step = f"debug_{dataset_size}"
    print(f"\nLoading dataset with {dataset_size} nodes...")
    dataset = get_dataset(dataset_name="KEP", step=f"debug_{dataset_size}")
    in_degree_histogram = [int(x) for x in list(dataset.in_degree_histogram)]

    # plot dataset in degree histogram
    # y_pos = np.arange(len(in_degree_histogram))
    # plt.bar(y_pos, in_degree_histogram)
    # plt.show()

    # print percentage of 0 in-degrees
    in_degree_0_percentage = in_degree_histogram[0] / np.sum(in_degree_histogram)
    print(f"in_degree_histogram[0]: {in_degree_histogram[0]}")
    print(f"np.sum(in_degree_histogram): {np.sum(in_degree_histogram)}")
    print(f"in_degree_0_percentage: {in_degree_0_percentage} %")

    # print JS distance (dissimilarity) between distributions (the lower the better)
    js_distance = jensen_shannon_distance(
        train_in_degree_histogram, in_degree_histogram
    )
    print(f"js_distance from 10k dataset: {js_distance}")
    js_distances.append(js_distance)

    # evaluate greedy_path's performance on this dataset
    # print(f"\n\nPredicting on the {step} dataset using GreedyPaths method")
    # predictions_dir = get_predictions_folder_path(
    #     dataset_name="KEP",
    #     step=step,
    #     trained_model_name=trained_model_name,
    # )
    # predict(
    #     model=model,
    #     device=device,
    #     dataset=dataset,
    #     output_dir=predictions_dir,
    #     batch_size=1,
    #     save_as_pt=True,
    # )
    # ## then, compute and save evaluation for the predictions done.
    # kep_evaluation(
    #     step=step,
    #     trained_model_name=trained_model_name,
    #     dataset_name="KEP",
    # )


## plot js_distances per dataset size figure and save
# print(f"DEBUG js_distances: {js_distances}")
# configure matplotlib to automatically adjust the plot size
rcParams.update({"figure.autolayout": True})
# create bar plot
y_pos = np.arange(len(dataset_sizes))
plot = plt.bar(y_pos, js_distances)
# set axis labels
plt.xlabel("Dataset size\n(Number of instances)")
plt.ylabel("J.S distance")
# add labels to the x-axis
plt.xticks(y_pos, dataset_sizes)
plt.xticks(rotation=30)
# save figure as a PNG
filename = "JS_distances_per_dataset_size.png"
filepath = PLOTS_FOLDER_PATH / filename
plt.savefig(filepath)
print(f"Saved {filepath}")
