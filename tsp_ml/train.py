# -*- coding: utf-8 -*-
# This script contains functions for training a model
import sys
import time
from typing import Optional

import torch
from average_meter import AverageMeter
from dataset_utils import get_dataloaders
from kep_evaluation import minor_kep_evaluation
from kep_loss import KEPLoss
from model_utils import get_model, save_model, set_torch_seed
from torch.nn.functional import one_hot
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

BATCH_SIZE = 10
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
MODEL_NAME = "TSP_GGCN_large"
DATASET_NAME = "TSP"
MINOR_EVAL = True  # set to True to print extra eval information during training


def validation_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    loss_function: torch.nn.modules.loss._Loss,
    epoch_loss: AverageMeter,
) -> AverageMeter:
    # predict
    batch = batch.to(device)
    scores = model(batch)
    label = batch.y
    # calculate loss
    loss = loss_function(scores, label)
    # save batch loss in AverageMeter object
    numInputs = scores.view(-1, 1).size(0)
    epoch_loss.update(loss.detach().item(), numInputs)
    return epoch_loss


def validation_epoch(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
) -> float:
    epoch_loss = AverageMeter()
    model.eval()  # set the model to evaluation mode (freeze weights)
    for _, batch in enumerate(tqdm(dataloader, desc="Validation", file=sys.stdout)):
        epoch_loss = validation_step(
            model=model,
            device=device,
            batch=batch,
            loss_function=loss_function,
            epoch_loss=epoch_loss,
        )
    return epoch_loss.average


def training_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    epoch_loss: AverageMeter,
    dataset_name: str,
) -> AverageMeter:
    optimizer.zero_grad()
    # predict
    batch = batch.to(device)
    scores = model(batch).to(torch.float32)
    # calculate loss
    if dataset_name == "TSP" or dataset_name == "DTSP":
        label = one_hot(batch.y).to(torch.float32)
        loss = loss_function(scores, label)
    elif dataset_name == "KEP":
        loss = loss_function(
            scores,
            batch.edge_weights,
            batch.edge_index,
            batch.type,
        )
    elif dataset_name == "KEPCE":
        loss = loss_function(
            scores,
            batch.edge_weights,
            batch.edge_index,
            batch.counter_edge,
        )
    # backpropagate
    loss.backward()
    optimizer.step()
    # save batch loss in AverageMeter object
    numInputs = scores.view(-1, 1).size(0)
    epoch_loss.update(loss.detach().item(), numInputs)
    return epoch_loss


def training_epoch(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    minor_eval: bool = False,
) -> float:
    epoch_loss = AverageMeter()
    model.train()  # set the model to training mode
    for i, batch in enumerate(tqdm(dataloader, desc="Training", file=sys.stdout)):
        epoch_loss = training_step(
            model=model,
            device=device,
            batch=batch,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch_loss=epoch_loss,
            dataset_name=dataloader.dataset.dataset_name,
        )
        if (i % 2500 == 0) and minor_eval:
            print(f"Minor evaluation at step {i}")
            minor_kep_evaluation(model=model, dataloader=dataloader)

    num_batches = int(len(dataloader.dataset) / dataloader.batch_size)
    avg_loss_per_batch = epoch_loss.sum / num_batches
    print(
        f"current training loss: {epoch_loss.average}"
        f" (total: {epoch_loss.sum} over {epoch_loss.count} predictions,"
        f" {len(dataloader.dataset)} graphs)."
        f" Average loss per batch: {avg_loss_per_batch}"
    )
    if minor_eval:
        minor_kep_evaluation(model=model, dataloader=dataloader)
    return epoch_loss.average


def get_training_report(
    num_epochs: int,
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
):
    """Returns a dictionary containing information about the current training run"""
    loss_dict = dict(loss_function.state_dict())
    loss_params = {k: v.tolist() for k, v in loss_dict.items()}
    model_architecture_name = model.__class__.__name__
    training_report = {
        "dataset_name": train_dataloader.dataset.dataset_name,
        "num_epochs": num_epochs,
        "model_architecture_name": model_architecture_name,
        "model_architecture": str(model),
        "device": str(device),
        "optimizer": optimizer.__class__.__name__,
        "optimizer_params": optimizer.state_dict()["param_groups"],
        "loss_function": loss_function.__class__.__name__,
        "loss_params": loss_params,
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
    minor_eval: bool = False,
) -> torch.nn.Module:
    # TODO description
    print("\nTraining..")
    training_report = get_training_report(
        num_epochs=num_epochs,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
    )
    start = time.time()
    for ep in range(1, num_epochs + 1):
        optimizer.zero_grad()
        print(f"Epoch [{ep}/{num_epochs}]")
        train_loss = training_epoch(
            model=model,
            device=device,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            minor_eval=minor_eval,
        )
        if validation_dataloader is not None and ep % 5 == 0:
            validation_loss = validation_epoch(
                model=model,
                device=device,
                dataloader=validation_dataloader,
                loss_function=loss_function,
            )
            print(f"current validation loss: {validation_loss}")

    end = time.time()
    elapsed_time = end - start
    training_report["training_time"] = elapsed_time
    training_report["training_timestamp"] = start
    print(f"Total training time: {elapsed_time} seconds")
    return model, training_report


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


def train(
    device: torch.device,
    model_name: str = "TSP_GGCN",
    dataset_name: str = "TSP",
    batch_size: int = 10,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
    use_validation: bool = True,
    minor_eval: bool = False,
):
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
    )
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4
    )
    loss_function = get_loss_function(
        dataset_name=dataset_name, train_dataloader=train_dataloader
    )

    # train model
    model, training_report = train_model(
        num_epochs=num_epochs,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        optimizer=adam_optimizer,
        loss_function=loss_function,
        minor_eval=minor_eval,
    )

    # save model
    save_model(model=model, training_report=training_report)


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train(
        device=device,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        minor_eval=MINOR_EVAL,
    )
