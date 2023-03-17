# -*- coding: utf-8 -*-
# This script contains functions for training a model
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
from dataset_utils import get_dataloaders
from loss import calculate_loss, get_loss_function
from model_utils import get_model, save_model, set_torch_seed
from torch.nn.functional import one_hot
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from validation import validation

BATCH_SIZE = 10
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
MODEL_NAME = "TSP_GGCN_large"
DATASET_NAME = "TSP"
PREDICT_METHOD = "greedy_paths"
VALIDATION_PERIOD = 1000  # how many batch predictions on the training set will be executed between each validation


def training_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    dataset_name: str,
) -> float:
    model.train()  # set the model to training mode
    optimizer.zero_grad()
    # debug_model_state_dict = str(model.state_dict())
    # predict
    model = model.to(device=device)
    batch = batch.to(device=device)
    batch.scores = model(batch).to(torch.float32)
    batch.pred = model.predict(batch).to(torch.float32)
    loss = calculate_loss(
        batch=batch, dataset_name=dataset_name, loss_function=loss_function
    )
    loss.backward()
    optimizer.step()
    # debug_model_state_dict_2 = str(model.state_dict())
    # has_state_dict_changed = not (debug_model_state_dict == debug_model_state_dict_2)
    # print(f"\nDEBUG training_step updated the model? {has_state_dict_changed}\n")
    # breakpoint()
    return loss.detach().item()


def training_epoch(
    epoch: int,
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    validation_dataloader: Optional[DataLoader] = None,
    validation_period: int = 1000,
) -> float:
    dataset_size = len(train_dataloader.dataset)
    training_loss_list = np.empty((dataset_size))
    for i, batch in enumerate(tqdm(train_dataloader, desc="Training", file=sys.stdout)):
        loss = training_step(
            model=model,
            device=device,
            batch=batch,
            optimizer=optimizer,
            loss_function=loss_function,
            dataset_name=train_dataloader.dataset.dataset_name,
        )
        training_loss_list[i] = loss
        if not validation_dataloader is None and (i % (validation_period) == 0):
            validation(
                model=model,
                device=device,
                loss_function=loss_function,
                step_i=i,
                epoch=epoch,
                training_loss_list=training_loss_list,
                validation_dataloader=validation_dataloader,
                validation_period=validation_period,
            )


def get_training_report(
    num_epochs: int,
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
) -> Dict[str, Any]:
    """Returns a dictionary containing information about the current training run"""
    loss_dict = dict(loss_function.state_dict())
    loss_params = {k: v.tolist() for k, v in loss_dict.items()}
    model_architecture_name = model.__class__.__name__
    training_report = {
        "dataset_name": train_dataloader.dataset.dataset_name,
        "num_epochs": num_epochs,
        "model_architecture_name": model_architecture_name,
        "model_architecture": str(model),
        "prediction_method": model.predict_method,
        "batch_size": train_dataloader.batch_size,
        "device": str(device),
        "optimizer": optimizer.__class__.__name__,
        "optimizer_params": optimizer.state_dict()["param_groups"],
        "loss_function": loss_function.__class__.__name__,
        "loss_params": loss_params,
        "training_start_time": str(datetime.now().strftime("%Y_%m_%d_%Hh%M")),
    }
    return training_report


def train_model(
    num_epochs: int,
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    validation_dataloader: Optional[DataLoader] = None,
    validation_period: int = 1000,
) -> torch.nn.Module:
    # TODO description
    model.training_report = get_training_report(
        num_epochs=num_epochs,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
    )
    start = time.time()
    print(
        f"\nTraining started at {datetime.now()}"
        f" with the following params: {model.training_report}"
    )
    for ep in range(1, num_epochs + 1):
        optimizer.zero_grad()
        print(f"Epoch [{ep}/{num_epochs}]")
        training_epoch(
            model=model,
            device=device,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            validation_dataloader=validation_dataloader,
            validation_period=validation_period,
            epoch=ep,
        )
    end = time.time()
    elapsed_time = end - start
    model.training_report["total_training_time"] = elapsed_time
    print(f"Total training time: {elapsed_time} seconds")
    return model


def train(
    device: torch.device,
    model_name: str = "TSP_GGCN",
    predict_method: Optional[str] = None,
    dataset_name: str = "KEP",
    batch_size: int = 10,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
    optimizer_weight_decay: float = 0.0,
    use_validation: bool = True,
    validation_period: int = 1000,
):
    """
    TODO description:
    Trains a model....
    Args:
        device: torch object selecting the device
            where the training will be run (CPU or CUDA).
        model_name: the name of the model class to be trained.
            It must be one of the classes of the files inside tsp.models
            ('KEP_GAT_PNA_CE', 'KEP_GCN', etc).
        predict_method: name of the method with which the model
            will compute the binary predictions.
        dataset_name: the name of the dataset ('KEP', 'TSP', or 'DTSP')
        batch_size: the number of instances to be loaded at each time
            a training step (forward + backward pass) occurs.
        num_epochs: number of epochs (which is a pass through
            the whole dataset) that the training will last.
        learning_rate: learning rate to be used in training by the optimizer.
        use_validation: if True, runs a validation after each epoch
        validation_period: how many batch predictions will be executed between each validation
    """
    set_torch_seed()

    # initialize dataloaders
    train_dataloader, val_dataloader = get_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size
    )
    if use_validation is False:
        val_dataloader = None

    # initialize model, optimizer, and loss function
    model = get_model(
        model_name=model_name,
        dataset=train_dataloader.dataset,
        predict_method=predict_method,
        device=device,
    ).to(device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=optimizer_weight_decay,
    )
    loss_function = get_loss_function(
        dataset_name=dataset_name, train_dataloader=train_dataloader
    )
    # train model
    model = train_model(
        num_epochs=num_epochs,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        validation_period=validation_period,
    )
    # save model
    save_model(model=model)


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    train(
        device=device,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
    )
