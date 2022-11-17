# -*- coding: utf-8 -*-
# This script contains functions for training a model
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from average_meter import AverageMeter
from dataset_utils import get_dataloaders
from kep_evaluation import minor_kep_evaluation
from kep_loss import KEPLoss
from model_utils import get_model, save_model, save_model_checkpoint, set_torch_seed
from torch import Tensor
from torch.nn.functional import one_hot
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

BATCH_SIZE = 10
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
MODEL_NAME = "TSP_GGCN_large"
DATASET_NAME = "TSP"
PREDICT_METHOD = "greedy_paths"
VALIDATION_PERIOD = (
    1000  # how many batch predictions will be executed between each validation
)

# TODO remove:
MINOR_EVAL = False  # set to True to print extra eval information during training


def validation_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    loss_function: torch.nn.modules.loss._Loss,
    epoch_loss: AverageMeter,
    dataset_name: str,
) -> AverageMeter:
    # predict
    batch = batch.to(device)
    batch.scores = model(batch).to(torch.float32)
    batch.pred = model.predict(batch).to(torch.float32)

    # calculate loss
    loss = calculate_loss(
        batch=batch, dataset_name=dataset_name, loss_function=loss_function
    )

    # save batch loss in AverageMeter object
    numInputs = batch.scores.view(-1, 1).size(0)
    epoch_loss.update(loss.detach().item(), numInputs)
    # TODO botar algo do minor eval aqui
    return epoch_loss


def validation_epoch(
    epoch: int,
    step: int,
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
    dataset_name: str,
) -> float:
    epoch_loss = AverageMeter()
    # TODO criar outra classe que guarda TODAS as loss
    # TODO ter um objeto desse pra validation e outro pra train
    # TODO metodos: avg loss per edge, avg per instance, avg per epoch
    # TODO usar isso pra fazer early stopping
    model.eval()  # set the model to evaluation mode (freeze weights)
    for _, batch in enumerate(tqdm(dataloader, desc="Validation", file=sys.stdout)):
        epoch_loss = validation_step(
            model=model,
            device=device,
            batch=batch,
            loss_function=loss_function,
            epoch_loss=epoch_loss,
            dataset_name=dataset_name,
        )
    save_model_checkpoint(model=model, epoch=epoch, step=step)
    return epoch_loss.average


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
            node_types=batch.type,
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


def training_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    epoch_loss: AverageMeter,
    dataset_name: str,
) -> AverageMeter:
    model.train()  # set the model to training mode
    optimizer.zero_grad()
    # predict
    batch = batch.to(device)
    batch.scores = model(batch).to(torch.float32)
    batch.pred = model.predict(batch).to(torch.float32)

    if isinstance(batch.type[0], list):
        # print(f"DEBUG batch.type is LIST OF LISTS: {batch.type}")
        # -> cai quase sempre aqui! 99.85% das instancias
        # TODO re generate dataset,
        # make sure everything is consistent,
        # and then delete this if else
        batch.type = batch.type[0]
    else:
        print(f"DEBUG batch.type is a SIMPLE LIST: {batch.type}")
        # -> 0.15% das instancias
        # node_types = batch.type
    loss = calculate_loss(
        batch=batch, dataset_name=dataset_name, loss_function=loss_function
    )
    # backpropagate
    loss.backward()
    optimizer.step()
    # save batch loss in AverageMeter object
    numInputs = batch.scores.view(-1, 1).size(0)
    epoch_loss.update(loss.detach().item(), numInputs)
    return epoch_loss


def training_epoch(
    epoch: int,
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    validation_dataloader: Optional[DataLoader] = None,
    validation_period: int = 1000,
    minor_eval: bool = False,
) -> float:
    epoch_loss = AverageMeter()
    for i, batch in enumerate(tqdm(train_dataloader, desc="Training", file=sys.stdout)):
        epoch_loss = training_step(
            model=model,
            device=device,
            batch=batch,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch_loss=epoch_loss,
            dataset_name=train_dataloader.dataset.dataset_name,
        )
        if not validation_dataloader is None and (i % validation_period == 0):
            print(f"Validation at step {i}")
            validation_loss = validation_epoch(
                epoch=epoch,
                step=i,
                model=model,
                device=device,
                dataloader=validation_dataloader,
                loss_function=loss_function,
                dataset_name=train_dataloader.dataset.dataset_name,
            )
            print(f"current validation loss: {validation_loss}")
            # TODO save validation loss in an array? then, save it in a file?

        # TODO : delete minor_eval?
        # if (i % 2500 == 0) and minor_eval:
        #     model.eval()
        #     print(f"Minor evaluation at step {i}")
        #     minor_kep_evaluation(model=model, dataloader=train_dataloader)

    num_batches = int(len(train_dataloader.dataset) / train_dataloader.batch_size)
    avg_loss_per_batch = epoch_loss.sum / num_batches
    print(
        f"current training loss: {epoch_loss.average}"
        f" (total: {epoch_loss.sum} over {epoch_loss.count} predictions,"
        f" {len(train_dataloader.dataset)} graphs)."
        f" Average loss per batch: {avg_loss_per_batch}"
    )
    # if minor_eval:
    #     minor_kep_evaluation(model=model, dataloader=train_dataloader)
    return epoch_loss.average


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
    minor_eval: bool = False,
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
        # TODO tem q retornar model tbm?
        train_loss = training_epoch(
            model=model,
            device=device,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            minor_eval=minor_eval,
            validation_dataloader=validation_dataloader,
            validation_period=validation_period,
            epoch=ep,
        )
    end = time.time()
    elapsed_time = end - start
    model.training_report["total_training_time"] = elapsed_time
    print(f"Total training time: {elapsed_time} seconds")
    return model


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
    predict_method: Optional[str] = None,
    dataset_name: str = "KEP",
    batch_size: int = 10,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
    use_validation: bool = True,
    validation_period: int = 1000,
    minor_eval: bool = False,
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
        minor_eval: if True, runs the kep_evaluation.minor_kep_evaluation
            function at each n predictions. It predicts on 3 random instances
            and prints on screen the evaluation results for each of them.
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
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4
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
        minor_eval=minor_eval,
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
        minor_eval=MINOR_EVAL,
    )
