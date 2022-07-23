# -*- coding: utf-8 -*-
import sys
import time
from typing import Optional

import torch
from average_meter import AverageMeter
from dataset_utils import get_class_weights, get_dataloaders
from model_utils import get_model, save_model
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# from training_report import TrainingReport


BATCH_SIZE = 10
NUM_EPOCHS = 20
LEARNING_RATE = 0.005


def set_torch_seed(seed: int = 1234):
    """manually choose seed to allow for deterministic reproduction of results"""
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validation_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    loss_function: torch.nn.modules.loss._Loss,
    epoch_loss: AverageMeter,
) -> AverageMeter:
    # predict
    batch = batch.to(device)
    edge_scores = model(batch)
    label = batch.y
    # calculate loss
    loss = loss_function(edge_scores, label)
    # save batch loss in AverageMeter object
    numInputs = edge_scores.view(-1, 1).size(0)
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
    # TODO optionally save predictions
    return epoch_loss.average


def training_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    epoch_loss: AverageMeter,
) -> AverageMeter:
    optimizer.zero_grad()
    # predict
    batch = batch.to(device)
    edge_scores = model(batch)
    label = batch.y
    # calculate loss
    loss = loss_function(edge_scores, label)
    # backpropagate
    loss.backward()
    optimizer.step()
    # save batch loss in AverageMeter object
    numInputs = edge_scores.view(-1, 1).size(0)
    epoch_loss.update(loss.detach().item(), numInputs)
    return epoch_loss


def training_epoch(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
) -> float:
    epoch_loss = AverageMeter()
    model.train()  # set the model to training mode
    for _, batch in enumerate(tqdm(dataloader, desc="Training", file=sys.stdout)):
        epoch_loss = training_step(
            model=model,
            device=device,
            batch=batch,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch_loss=epoch_loss,
        )
    # TODO optionally save predictions
    return epoch_loss.average


def train_model(
    num_epochs: int,
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    validation_dataloader: Optional[DataLoader] = None,
) -> torch.nn.Module:
    # TODO description
    print("\nTraining..")
    start = time.time()
    # training_report = TrainingReport() # TODO
    for ep in range(1, num_epochs + 1):
        optimizer.zero_grad()
        print(f"Epoch [{ep}/{num_epochs}]")
        train_loss = training_epoch(
            model=model,
            device=device,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
        )
        print(f"current training loss: {train_loss}")
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
    # training_report.training_time = elapsed_time
    print(f"Total training time: {elapsed_time} seconds")
    # training_report.save()
    return model


def train(
    device: torch.device, model_name: str = "TSP_GGCN", dataset_name: str = "TSP"
):
    set_torch_seed()

    # initialize dataloaders
    train_dataloader, val_dataloader = get_dataloaders(
        dataset_name=dataset_name, batch_size=BATCH_SIZE
    )

    # initialize model, optimizer, and loss function
    model = get_model(model_name=model_name)
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4
    )
    class_weights = get_class_weights(dataloader=train_dataloader)
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    # train model
    model = train_model(
        num_epochs=NUM_EPOCHS,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        optimizer=adam_optimizer,
        loss_function=loss_function,
    )

    # save model
    save_model(model=model)


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train(device=device, model_name="TSP_GGCN_v4_weights", dataset_name="TSP")
