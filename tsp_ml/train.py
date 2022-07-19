# -*- coding: utf-8 -*-
import sys
import time
from datetime import datetime
from operator import mod
from statistics import mode
from typing import Optional, Tuple

import torch
from average_meter import AverageMeter
from datasets.tsp_dataset import TSPDataset
from model_utils import save_model
from models.dtsp_gnn_prates import DTSP_GNN_Prates
from models.tsp_ggcn import TSP_GGCN
from models.tsp_ggcn_v2 import TSP_GGCN_v2
from paths import (
    DTSP_TEST_DATASET_FOLDER_PATH,
    DTSP_TRAIN_DATASET_FOLDER_PATH,
    DTSP_VAL_DATASET_FOLDER_PATH,
    TRAINED_MODELS_FOLDER_PATH,
    TSP_TEST_DATASET_FOLDER_PATH,
    TSP_TRAIN_DATASET_FOLDER_PATH,
    TSP_VAL_DATASET_FOLDER_PATH,
)
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
    # import pdb

    # pdb.set_trace()
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


def train(
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


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    set_torch_seed()

    # setup data
    # TODO refactor
    tsp_train_dataset = TSPDataset(dataset_folderpath=TSP_TRAIN_DATASET_FOLDER_PATH)
    tsp_val_dataset = TSPDataset(dataset_folderpath=TSP_TRAIN_DATASET_FOLDER_PATH)
    train_dataloader = DataLoader(
        tsp_train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        tsp_val_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=4,
    )

    # initalize model and optimizer
    model = TSP_GGCN().to(device)
    # model = TSP_GGCN_v2().to(device)
    # model = DTSP_GNN_Prates().to(device)
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4
    )
    class_weights = get_class_weights(dataloader=train_dataloader)
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    # train
    model = train(
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
