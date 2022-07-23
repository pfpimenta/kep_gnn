# -*- coding: utf-8 -*-
from typing import Tuple

from datasets.dtsp_dataset import DTSPDataset
from datasets.tsp_dataset import TSPDataset
from paths import (
    DTSP_TRAIN_DATASET_FOLDER_PATH,
    DTSP_VAL_DATASET_FOLDER_PATH,
    TSP_TRAIN_DATASET_FOLDER_PATH,
    TSP_VAL_DATASET_FOLDER_PATH,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def get_class_weights(dataloader: DataLoader) -> Tuple[float, float]:
    """calculates class weights to adjust the loss function
    based on the class distribution of the given dataset
    """
    class_0_count = 0
    class_1_count = 0
    for batch in dataloader:
        batch_class_0_count = (batch.y == 0).sum()
        class_0_count += batch_class_0_count
        class_1_count += batch.num_edges - batch_class_0_count
    total_num_edges = class_0_count + class_1_count
    class_0_weight = 1 / (class_0_count / total_num_edges)
    class_1_weight = 1 / (class_1_count / total_num_edges)
    # normalize weights
    class_0_weight = class_0_weight / (class_0_weight + class_1_weight)
    class_1_weight = class_1_weight / (class_0_weight + class_1_weight)
    return class_0_weight, class_1_weight


def get_dataloaders(
    dataset_name: str = "TSP", batch_size: int = 10
) -> Tuple[DataLoader, DataLoader]:
    # setup dataset into training and validation dataloaders
    # TODO refactor
    if dataset_name == "TSP":
        train_dataset = TSPDataset(dataset_folderpath=TSP_TRAIN_DATASET_FOLDER_PATH)
        val_dataset = TSPDataset(dataset_folderpath=TSP_VAL_DATASET_FOLDER_PATH)
    elif dataset_name == "DTSP":
        train_dataset = DTSPDataset(dataset_folderpath=DTSP_TRAIN_DATASET_FOLDER_PATH)
        val_dataset = DTSPDataset(dataset_folderpath=DTSP_VAL_DATASET_FOLDER_PATH)
    else:
        raise ValueError(f"No dataset named '{dataset_name}' found.")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    return train_dataloader, val_dataloader


# TODO consider moving this function somewhere else
def filter_tensors(data: Data, tensor_names: str):
    # keep only the specified tensors (tensor_names) and delete the rest
    for key in data.keys:
        if key not in tensor_names:
            del data[key]
    return data
