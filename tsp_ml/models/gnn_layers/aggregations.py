# -*- coding: utf-8 -*-
from typing import List

import torch
from torch import Tensor


def sum_agg(feature_tensors: List[Tensor]):
    if len(feature_tensors) == 0:
        return []
    aggregated_tensor = torch.zeros_like(feature_tensors[0])
    for tensor in feature_tensors:
        aggregated_tensor += tensor
    return aggregated_tensor


def mean_agg(feature_tensors: List[Tensor]):
    num_tensors = len(feature_tensors)
    if num_tensors == 0:
        return []
    aggregated_tensor = sum_agg(feature_tensors)
    aggregated_tensor = aggregated_tensor / num_tensors
    return aggregated_tensor


AGGREGATIONS = {
    "add": sum_agg,
    "mean": mean_agg,
}
