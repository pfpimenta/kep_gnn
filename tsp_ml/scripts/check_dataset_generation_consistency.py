# -*- coding: utf-8 -*-
""" script to measure how much the dataset changes between 2 random generations
(in this case, train and test datasets)"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from dataset_utils import get_dataset


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """
    # normalize vector size
    while len(p) > len(q):
        q.append(0)
    while len(q) > len(p):
        p.append(0)

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


# load datasets
train_dataset = get_dataset(dataset_name="KEP", step="train")
test_dataset = get_dataset(dataset_name="KEP", step="test")

## TRAIN
train_in_degree_histogram = [int(x) for x in list(train_dataset.in_degree_histogram)]
y_pos = np.arange(len(train_in_degree_histogram))
plt.bar(y_pos, train_in_degree_histogram)
plt.show()

# print percentage of 0 in-degrees
total_in_degrees = np.sum(train_in_degree_histogram)
in_degree_0_percentage = train_in_degree_histogram[0] / total_in_degrees
print(f"train_in_degree_histogram[0]: {train_in_degree_histogram[0]}")
print(f"np.sum(train_in_degree_histogram): {np.sum(train_in_degree_histogram)}")
print(f"in_degree_0_percentage: {in_degree_0_percentage} %")

## TEST
test_in_degree_histogram = [int(x) for x in list(test_dataset.in_degree_histogram)]
y_pos = np.arange(len(test_in_degree_histogram))
plt.bar(y_pos, test_in_degree_histogram)
plt.show()

# print percentage of 0 in-degrees
total_in_degrees = np.sum(test_in_degree_histogram)
in_degree_0_percentage = test_in_degree_histogram[0] / total_in_degrees
print(f"test_in_degree_histogram[0]: {test_in_degree_histogram[0]}")
print(f"np.sum(test_in_degree_histogram): {np.sum(test_in_degree_histogram)}")
print(f"in_degree_0_percentage: {in_degree_0_percentage} %")

## QUANTITATIVE COMPARISON
# print JS distance (dissimilarity) between distributions (the lower the better)
js_distance = jensen_shannon_distance(
    train_in_degree_histogram, test_in_degree_histogram
)
print(f"js_distance: {js_distance}")
