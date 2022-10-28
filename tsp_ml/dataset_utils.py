# -*- coding: utf-8 -*-
from typing import List, Tuple

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
    elif dataset_name == "KEP" or dataset_name == "KEPCE":
        dataset = KEPDataset(dataset_folderpath=folder_path)
        dataset.save_properties()
    else:
        raise ValueError(f"No dataset named '{dataset_name}' found.")
    return dataset


def print_dataset_information(dataset: Dataset) -> None:
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    print(f"Dataset total num_nodes: {dataset.num_nodes}")
    print(f"Dataset total num_edges: {dataset.num_edges}")
    avg_num_edges = int(dataset.num_edges) / int(dataset_size)
    print(f"Mean num_edges per graph: {avg_num_edges}")
    print(f"Dataset maximum_in_degree: {dataset.maximum_in_degree}")
    print(f"Dataset in_degree_histogram: {dataset.in_degree_histogram}")


# TODO consider moving this function somewhere else
def filter_tensors(data: Data, tensor_names: str) -> Data:
    # keep only the specified tensors (tensor_names) and delete the rest
    for key in data.keys:
        if key not in tensor_names:
            del data[key]
    return data


# TODO consider moving this function somewhere else
def get_instance_ids(batch: Data) -> List[int]:
    row, _ = batch.edge_index
    edge_batch = batch.batch[row]
    instance_ids = [batch.id[idx] for idx in edge_batch.tolist()]
    return instance_ids
