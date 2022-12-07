# -*- coding: utf-8 -*-
import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
from models import AVAILABLE_MODELS
from paths import TRAINED_MODELS_FOLDER_PATH
from torch_geometric.data import Dataset


def get_checkpoint_name(
    epoch: int,
    step: int,
) -> str:
    checkpoint_name = f"e{epoch:0>3d}_s{step:0>5d}"
    return checkpoint_name


def save_dict_to_json(dict: Dict[str, Any], json_filepath: str):
    """Saves dictionary in a JSON file formatted with 'pretty-print' style"""
    with open(json_filepath, "w") as file:
        json_string = json.dumps(
            dict, default=lambda o: o.__dict__, sort_keys=True, indent=2
        )
        file.write(json_string)


# def save_model(model: torch.nn.Module, training_report: Dict[str, Any]):
def save_model(model: torch.nn.Module) -> None:
    """Creates a folder and saves in it the model state and a JSON file with
    information about the training process"""
    training_end_time = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    model.training_report["training_end_time"] = training_end_time
    # save model state dict
    trained_model_dir = model.trained_model_dir
    trained_model_filepath = trained_model_dir / "model.pt"
    torch.save(model.state_dict(), trained_model_filepath)
    print(f"Saved {trained_model_filepath}")
    # save training information JSON
    json_filepath = trained_model_dir / "training_report.json"
    save_dict_to_json(dict=model.training_report, json_filepath=json_filepath)
    print(f"Saved {json_filepath}")


def save_model_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    step: int,
    evaluation_overview: str,
    performance_dict: Dict[str, Any],
    training_loss_list: np.ndarray,
) -> None:
    """TODO description"""
    # save model checkpoint
    checkpoint_name = get_checkpoint_name(epoch=epoch, step=step)
    checkpoint_dir = model.trained_model_dir / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trained_model_filepath = checkpoint_dir / "model.pt"
    torch.save(model.state_dict(), trained_model_filepath)
    print(f"Saved {trained_model_filepath}")
    # prepare training information JSON
    checkpoint_training_report = model.training_report
    checkpoint_training_report["epoch"] = epoch
    checkpoint_training_report["step"] = step
    checkpoint_training_report["checkpoint_time"] = datetime.now().strftime(
        "%Y_%m_%d_%Hh%M"
    )
    # save training information JSON
    report_json_filepath = checkpoint_dir / "training_report.json"
    save_dict_to_json(
        dict=checkpoint_training_report, json_filepath=report_json_filepath
    )
    print(f"Saved {report_json_filepath}")
    # save performance JSON
    performance_json_filepath = checkpoint_dir / "performance.json"
    save_dict_to_json(dict=performance_dict, json_filepath=performance_json_filepath)
    print(f"Saved {performance_json_filepath}")
    # save evaluation overview
    eval_overview_filepath = checkpoint_dir / "eval_overview.md"
    with open(eval_overview_filepath, "w") as f:
        f.write(evaluation_overview)
    print(f"Saved {eval_overview_filepath}")
    # save training loss list
    training_loss_list_filepath = checkpoint_dir / "training_loss_list.npy"
    with open(training_loss_list_filepath, "wb") as file:
        np.save(file, training_loss_list)
    print(f"Saved {training_loss_list_filepath}")


def load_model(
    trained_model_name: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    print_training_report: bool = True,
    dataset: Optional[Dataset] = None,
    predict_method: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> torch.nn.Module:
    model_name = trained_model_name[17:]
    model = get_model(
        model_name=model_name,
        dataset=dataset,
        predict_method=predict_method,
    ).to(device)
    if checkpoint is None:
        model_filepath = TRAINED_MODELS_FOLDER_PATH / f"{trained_model_name}/model.pt"
    else:
        model_filepath = (
            TRAINED_MODELS_FOLDER_PATH
            / f"{trained_model_name}/checkpoints/{checkpoint}/model.pt"
        )
    print(f"...Loading model from file {model_filepath}")
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    if print_training_report and not "Greedy" in model_name:
        training_report_filepath = (
            TRAINED_MODELS_FOLDER_PATH / f"{trained_model_name}/training_report.json"
        )
        with open(training_report_filepath, "r") as file:
            training_report = json.load(file)
            print(f"Training report:\n{training_report}")
    return model


def get_model(
    model_name: str,
    dataset: Optional[Dataset] = None,
    predict_method: Optional[str] = None,
) -> torch.nn.Module:
    """Returns an initialized model object"""
    try:
        Model = AVAILABLE_MODELS[model_name]
    except KeyError:
        raise ValueError(
            f"No model named '{model_name}' found."
            f"Currently available models: {list(AVAILABLE_MODELS.keys())}"
        )
    model_args = {}
    if "PNA" in model_name:
        model_args["pna_deg"] = dataset.in_degree_histogram
    if predict_method and not "Greedy" in model_name:
        model_args["predict_method"] = predict_method
    model = Model(**model_args)
    return model


# TODO move somewhere else?
def set_torch_seed(seed: int = 42):
    """manually choose seed to allow for deterministic reproduction of results"""
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
