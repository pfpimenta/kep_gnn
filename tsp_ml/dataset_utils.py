# -*- coding: utf-8 -*-
from typing import Tuple

from datasets import DTSPDataset, KEPDataset, TSPDataset
from paths import get_dataset_folder_path
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def get_dataloaders(
    dataset_name: str = "TSP", batch_size: int = 10
) -> Tuple[DataLoader, DataLoader]:
    # setup dataset into training and validation dataloaders
    train_dataset = get_dataset(dataset_name=dataset_name, step="train")
    val_dataset = get_dataset(dataset_name=dataset_name, step="val")
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


def get_dataset(dataset_name: str, step: str) -> Dataset:
    folder_path = get_dataset_folder_path(dataset_name=dataset_name, step=step)
    if dataset_name == "TSP":
        dataset = TSPDataset(dataset_folderpath=folder_path)
    elif dataset_name == "DTSP":
        dataset = DTSPDataset(dataset_folderpath=folder_path)
    elif dataset_name == "KEP":
        dataset = KEPDataset(dataset_folderpath=folder_path)
    else:
        raise ValueError(f"No dataset named '{dataset_name}' found.")
    return dataset


# TODO consider moving this function somewhere else
def filter_tensors(data: Data, tensor_names: str) -> Data:
    # keep only the specified tensors (tensor_names) and delete the rest
    for key in data.keys:
        if key not in tensor_names:
            del data[key]
    return data
