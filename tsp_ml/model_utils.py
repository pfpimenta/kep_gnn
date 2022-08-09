# -*- coding: utf-8 -*-
import json
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from models import AVAILABLE_MODELS
from paths import TRAINED_MODELS_FOLDER_PATH


def save_dict_to_json(dict: Dict[str, Any], json_filepath: str):
    """Saves dictionary in a JSON file formatted with 'pretty-print' style"""
    with open(json_filepath, "w") as file:
        json_string = json.dumps(
            dict, default=lambda o: o.__dict__, sort_keys=True, indent=2
        )
        file.write(json_string)


def old_save_model(model: torch.nn.Module):
    # save model state in the trained models folder
    model_name = get_trained_model_name(model)
    model_filepath = TRAINED_MODELS_FOLDER_PATH / f"{model_name}.pt"
    torch.save(model.state_dict(), model_filepath)
    print(f"Saved {model_filepath}")


def save_model(model: torch.nn.Module, training_report: Dict[str, Any]):
    """Creates a folder and saves in it the model state and a JSON file with
    information about the training process"""
    training_timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    trained_model_name = get_trained_model_name(
        model=model, training_timestamp=training_timestamp
    )
    training_report["training_timestamp"] = training_timestamp
    training_report["trained_model_name"] = trained_model_name
    # create folder
    trained_model_dir = TRAINED_MODELS_FOLDER_PATH / f"{trained_model_name}"
    trained_model_dir.mkdir(parents=True, exist_ok=True)
    # save model state dict
    trained_model_filepath = trained_model_dir / "model.pt"
    torch.save(model.state_dict(), trained_model_filepath)
    print(f"Saved {trained_model_filepath}")
    # save training information JSON
    json_filepath = trained_model_dir / "training_info.json"
    save_dict_to_json(dict=training_report, json_filepath=json_filepath)
    print(f"Saved {json_filepath}")


def load_model(
    trained_model_name: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    print_training_report: bool = True,
) -> torch.nn.Module:
    model_name = trained_model_name[17:]
    model = get_model(model_name=model_name).to(device)
    model_filepath = TRAINED_MODELS_FOLDER_PATH / f"{trained_model_name}/model.pt"
    print(f"...Loading model from file {model_filepath}")
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    if print_training_report:
        training_report_filepath = (
            TRAINED_MODELS_FOLDER_PATH / f"{trained_model_name}/training_info.json"
        )
        with open(training_report_filepath, "r") as file:
            training_report = json.load(file)
            print(training_report)
    return model


def get_trained_model_name(
    model: torch.nn.Module, training_timestamp: Optional[str] = None
) -> str:
    # generates a model name based on the name of the architecture and
    # the current date and time
    model_architecture_name = model.__class__.__name__
    if training_timestamp is None:
        training_timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    model_name = f"{training_timestamp}_{model_architecture_name}"
    return model_name


def get_model(model_name: str = "TSP_GGCN") -> torch.nn.Module:
    # returns an initialized model object
    try:
        Model = AVAILABLE_MODELS[model_name]
    except KeyError:
        raise ValueError(f"No model named '{model_name}' found.")
    model = Model()
    return model


# TODO move somewhere else?
def set_torch_seed(seed: int = 42):
    """manually choose seed to allow for deterministic reproduction of results"""
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
