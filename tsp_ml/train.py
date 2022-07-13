# -*- coding: utf-8 -*-
import sys
import time
from datetime import datetime
from operator import mod
from statistics import mode
from typing import Tuple

import torch
from average_meter import AverageMeter
from definitions import TRAIN_DATASET_FOLDER_PATH, TRAINED_MODELS_FOLDER_PATH
from models.tsp_ggcn import TSP_GGCN
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tsp_dataset import TSPDataset

# from training_report import TrainingReport


BATCH_SIZE = 10
NUM_EPOCHS = 3  # 100
LEARNING_RATE = 0.005


def set_torch_seed(seed: int = 1234):
    # manually choose seed to allow for deterministic reproduction of results
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model: torch.nn.Module):
    # save model state in the trained models folder
    model_name = get_model_name(model)
    model_filepath = TRAINED_MODELS_FOLDER_PATH / (model_name + ".pt")
    torch.save(model.state_dict(), model_filepath)
    print(f"Saved {model_filepath}")


def get_model_name(model: torch.nn.Module) -> str:
    # generates a model name based on the name of the architecture and
    # the current date and time
    model_architecture_name = model.__class__.__name__
    training_timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    model_name = f"{model_architecture_name}_{training_timestamp}"
    return model_name


def get_class_weights(dataloader: DataLoader) -> Tuple[float, float]:
    # calculates class weights to adjust the loss function
    # based on the class distribution of the given dataset
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
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    epoch_loss: AverageMeter,
) -> torch.nn.Module:
    pass  # TODO


def validation_epoch(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
):
    pass  # TODO


def training_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    epoch_loss: AverageMeter,
) -> torch.nn.Module:
    # TODO description
    optimizer.zero_grad()
    # predict
    batch = batch.to(device)
    # score = model(batch)
    edge_scores = model(batch)
    # score = score.to(torch.float32)
    label = batch.y
    # label = label.to(torch.float32)
    # treatment for cross entropy loss
    # score_1 = score
    # score_0 = - score
    # scores = torch.cat((score_0.reshape(-1,1), score_1.reshape(-1, 1)), 1)
    # calculate loss
    loss = loss_function(edge_scores, label)
    # loss = loss.to(torch.float32)
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
) -> torch.nn.Module:
    # TODO description
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
    # TODO save
    return epoch_loss.average


def train(
    num_epochs: int,
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
) -> torch.nn.Module:
    # TODO description
    print("\nTraining..")
    start = time.time()
    # training_report = TrainingReport() # TODO
    for ep in range(1, num_epochs + 1):
        optimizer.zero_grad()
        print(f"Epoch [{ep}/{num_epochs}]")
        trainLoss = training_epoch(
            model=model,
            device=device,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
        )
        print(f"current training loss: {trainLoss}")
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
    tsp_dataset = TSPDataset(dataset_folderpath=TRAIN_DATASET_FOLDER_PATH)
    train_dataloader = DataLoader(
        tsp_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4
    )

    # initalize model and optimizer
    model = TSP_GGCN().to(device)
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4
    )
    class_weights = get_class_weights(dataloader=train_dataloader)
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    ## train
    model = train(
        num_epochs=NUM_EPOCHS,
        model=model,
        device=device,
        dataloader=train_dataloader,
        optimizer=adam_optimizer,
        loss_function=loss_function,
    )

    # save model
    save_model(model=model)
