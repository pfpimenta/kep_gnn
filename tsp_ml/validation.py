# -*- coding: utf-8 -*-
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from kep_evaluation import evaluate_kep_instance_prediction, get_eval_overview_string
from loss import calculate_loss
from model_utils import save_model_checkpoint
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


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


def print_validation_summary(
    epoch: int,
    step_i: int,
    performance_dict: Dict[str, Any],
) -> None:
    validation_overview_string = (
        f"# Validation overview (epoch {epoch}, step {step_i}):"
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


def validation(
    model: torch.nn.Module,
    device: torch.device,
    loss_function: torch.nn.modules.loss._Loss,
    step_i: int,
    epoch: int,
    training_loss_list: np.ndarray,
    validation_dataloader: Optional[DataLoader] = None,
    validation_period: int = 1000,
) -> None:
    # measure total time for validation epoch
    start = time.time()
    print(f"Validation at step {step_i}")
    validation_eval_df = validation_epoch(
        model=model,
        device=device,
        dataloader=validation_dataloader,
        loss_function=loss_function,
        dataset_name=validation_dataloader.dataset.dataset_name,
    )
    checkpoint_training_loss_list = training_loss_list[
        (step_i % validation_period) : step_i + 1
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
        step=step_i,
        evaluation_overview=eval_overview,
        performance_dict=performance_dict,
        training_loss_list=checkpoint_training_loss_list,
    )
    # print validation summary:
    print_validation_summary(
        epoch=epoch, step_i=step_i, performance_dict=performance_dict
    )
