# -*- coding: utf-8 -*-
# This script contains functions for training a model
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from dataset_utils import get_dataloaders
from greedy import greedy_paths
from kep_evaluation import evaluate_kep_instance_prediction, get_eval_overview_string
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

# TODO refatorar, botar umas funcoes pra fora desse arquivo


def validation_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: Batch,
    loss_function: torch.nn.modules.loss._Loss,
    dataset_name: str,
) -> Dict[str, Any]:
    # predict
    batch = batch.to(device)
    batch.scores = model(batch)
    batch.pred = model.predict(batch).to(torch.float32)
    # calculate loss
    loss = calculate_loss(
        batch=batch, dataset_name=dataset_name, loss_function=loss_function
    )
    # evaluate predictions
    prediction_info = evaluate_kep_instance_prediction(predicted_instance=batch)
    # save batch loss in prediction_info dict
    prediction_info["loss"] = loss.detach().item()
    return prediction_info


def validation_epoch(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
    dataset_name: str,
) -> pd.DataFrame:
    # epoch_loss = AverageMeter() # TODO delete this
    # TODO criar outra classe que guarda TODAS as loss
    # TODO ter um objeto desse pra validation e outro pra train
    # TODO metodos: avg loss per edge, avg per instance, avg per epoch
    # TODO usar isso pra fazer early stopping
    model.eval()  # set the model to evaluation mode (freeze weights)
    eval_df = pd.DataFrame()
    for _, batch in enumerate(tqdm(dataloader, desc="Validation", file=sys.stdout)):
        prediction_info = validation_step(
            model=model,
            device=device,
            batch=batch,
            loss_function=loss_function,
            dataset_name=dataset_name,
        )
        instance_eval_df = pd.DataFrame([prediction_info])
        eval_df = pd.concat([eval_df, instance_eval_df], ignore_index=True)
    return eval_df


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

    # predict
    batch = batch.to(device)
    batch.scores = model(batch).to(torch.float32)
    # batch.pred = model.predict(batch).to(torch.float32)
    # TODO DEBUG use predict directly from greedy
    batch.pred = greedy_paths(
        edge_scores=batch.scores, edge_index=batch.edge_index, node_types=batch.type[0]
    )

    loss = calculate_loss(
        batch=batch, dataset_name=dataset_name, loss_function=loss_function
    )
    # TODO os pesos não tao atualizando! wtf
    # backpropagate
    print(f"DEBUG loss: {loss}")
    loss.backward()
    optimizer.step()
    return loss.detach().item()


def get_performance_dict(
    checkpoint_training_loss_list: np.ndarray,
    validation_eval_df: pd.DataFrame,
) -> Dict[str, Any]:
    performance_dict = {
        "mean_training_loss": np.mean(checkpoint_training_loss_list),
        "std_training_loss": np.std(checkpoint_training_loss_list),
        "min_training_loss": np.min(checkpoint_training_loss_list),
        "max_training_loss": np.max(checkpoint_training_loss_list),
    }
    # mean/std/min/max validation loss (per instance)
    performance_dict.update(
        {
            "mean_validation_loss": validation_eval_df["loss"].mean(),
            "std_validation_loss": validation_eval_df["loss"].std(),
            "min_validation_loss": validation_eval_df["loss"].min(),
            "max_validation_loss": validation_eval_df["loss"].max(),
        }
    )
    # mean/std/min/max validation solution_weight_sum (per instance)
    performance_dict.update(
        {
            "mean_validation_solution_weight_sum": validation_eval_df[
                "solution_weight_sum"
            ].mean(),
            "std_validation_solution_weight_sum": validation_eval_df[
                "solution_weight_sum"
            ].std(),
            "min_validation_solution_weight_sum": validation_eval_df[
                "solution_weight_sum"
            ].min(),
            "max_validation_solution_weight_sum": validation_eval_df[
                "solution_weight_sum"
            ].max(),
        }
    )
    # mean/std/min/max validation solution_weight_percentage (per instance)
    performance_dict.update(
        {
            "mean_validation_solution_weight_percentage": validation_eval_df[
                "solution_weight_percentage"
            ].mean(),
            "std_validation_solution_weight_percentage": validation_eval_df[
                "solution_weight_percentage"
            ].std(),
            "min_validation_solution_weight_percentage": validation_eval_df[
                "solution_weight_percentage"
            ].min(),
            "max_validation_solution_weight_percentage": validation_eval_df[
                "solution_weight_percentage"
            ].max(),
        }
    )
    return performance_dict


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
    # TODO DEBUG testar se modelo muda:
    from copy import deepcopy

    model_copy = deepcopy(model)
    dataset_size = len(train_dataloader.dataset)
    training_loss_list = np.empty((dataset_size))
    for i, batch in enumerate(tqdm(train_dataloader, desc="Training", file=sys.stdout)):
        debug_model_state_dict = str(model.state_dict())
        loss = training_step(
            model=model,
            device=device,
            batch=batch,
            optimizer=optimizer,
            loss_function=loss_function,
            dataset_name=train_dataloader.dataset.dataset_name,
        )
        training_loss_list[i] = loss
        debug_model_state_dict_2 = str(model.state_dict())
        has_state_dict_changed = not (
            debug_model_state_dict == debug_model_state_dict_2
        )
        print(f"DEBUG training_step updated the model? {has_state_dict_changed}")
        if not validation_dataloader is None and (i % (validation_period - 1) == 0):
            # measure total time for validation epoch
            start = time.time()
            print(f"Validation at step {i}")
            validation_eval_df = validation_epoch(
                model=model,
                device=device,
                dataloader=validation_dataloader,
                loss_function=loss_function,
                dataset_name=train_dataloader.dataset.dataset_name,
            )
            checkpoint_training_loss_list = training_loss_list[
                (i % validation_period) : i + 1
            ]
            performance_dict = get_performance_dict(
                checkpoint_training_loss_list=checkpoint_training_loss_list,
                validation_eval_df=validation_eval_df,
            )
            # measure total time for validation epoch
            end = time.time()
            elapsed_time = end - start
            # validation evaluation overview string to be saved in a markdown file
            eval_overview = get_eval_overview_string(
                eval_df=validation_eval_df,
                trained_model_name=model.trained_model_name,
                step="validation",
                eval_time=elapsed_time,
            )

            save_model_checkpoint(
                model=model,
                epoch=epoch,
                step=i,
                evaluation_overview=eval_overview,
                performance_dict=performance_dict,
                training_loss_list=checkpoint_training_loss_list,
            )
            # TODO print validation summary:
            validation_overview_string = (
                f"# Validation overview (epoch {epoch}, step {i}):"
            )
            validation_overview_string += (
                f"\n* Mean training loss: {performance_dict['mean_training_loss']}"
            )
            validation_overview_string += (
                f"\n* Mean validation loss: {performance_dict['mean_validation_loss']}"
            )
            validation_overview_string += f"\n* Mean validation solution_weight_sum: {performance_dict['mean_validation_solution_weight_sum']}"
            validation_overview_string += f"\n* Mean validation solution_weight_percentage: {performance_dict['mean_validation_solution_weight_percentage']}"
            print(validation_overview_string)
            # print(f"current validation loss: {validation_loss}")

    num_batches = int(len(train_dataloader.dataset) / train_dataloader.batch_size)

    # TODO remake this part with the new structure:
    # avg_loss_per_batch = epoch_loss.sum / num_batches
    # print(
    #     f"current training loss: {epoch_loss.average}"
    #     f" (total: {epoch_loss.sum} over {epoch_loss.count} predictions,"
    #     f" {len(train_dataloader.dataset)} graphs)."
    #     f" Average loss per batch: {avg_loss_per_batch}"
    # )
    # return epoch_loss.average
    return None  # TODO


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
        # TODO tem q retornar model tbm?
        train_loss = training_epoch(
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
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4
    )
    loss_function = get_loss_function(
        dataset_name=dataset_name, train_dataloader=train_dataloader
    )

    # train model
    debug_model_state_dict = str(model.state_dict())
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
    debug_model_state_dict_2 = str(model.state_dict())
    has_state_dict_changed = not (debug_model_state_dict == debug_model_state_dict_2)
    print(f"DEBUG training worked? {has_state_dict_changed}")

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
