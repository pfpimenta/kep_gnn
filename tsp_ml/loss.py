# -*- coding: utf-8 -*-
from typing import Optional

import torch
from kep_loss import KEPLoss
from torch import Tensor
from torch.nn.functional import one_hot
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader


def calculate_loss(
    batch: Batch,
    dataset_name: str,
    loss_function: torch.nn.modules.loss._Loss,
) -> Tensor:
    """Calculates loss value for the given batch"""
    if dataset_name == "TSP" or dataset_name == "DTSP":
        label = one_hot(batch.y).to(torch.float32)
        loss = loss_function(batch.scores, label)
    elif dataset_name == "KEP":
        loss = loss_function(
            batch.scores,
            batch.edge_weights,
            batch.edge_index,
            pred=batch.pred,
            node_types=batch.type[0],
        )
    elif dataset_name == "KEPCE":
        loss = loss_function(
            batch.scores,
            batch.edge_weights,
            batch.edge_index,
            batch.counter_edge,
        )
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")

    return loss


def get_loss_function(dataset_name: str, train_dataloader: Optional[DataLoader] = None):
    if dataset_name == "TSP":
        if train_dataloader is None:
            raise ValueError(
                f"A dataloader for the training data must be passed when using the {dataset_name} dataset."
            )
        class_weights = train_dataloader.dataset.get_class_weights
        loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    elif dataset_name == "DTSP":
        loss_function = torch.nn.L1Loss()
    elif dataset_name == "KEP" or dataset_name == "KEPCE":
        loss_function = KEPLoss()
    else:
        raise ValueError(f"No dataset named '{dataset_name}' found.")
    return loss_function
