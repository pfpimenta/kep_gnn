# -*- coding: utf-8 -*-
from datetime import datetime

import torch
from definitions import TRAINED_MODELS_FOLDER_PATH
from models.tsp_ggcn_v2 import TSP_GGCN_v2 as TSP_GGCN


def save_model(model: torch.nn.Module):
    # save model state in the trained models folder
    model_name = get_model_name(model)
    model_filepath = TRAINED_MODELS_FOLDER_PATH / f"{model_name}.pt"
    torch.save(model.state_dict(), model_filepath)
    print(f"Saved {model_filepath}")


def load_model(
    model_name: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.nn.Module:
    # TODO: save model with class name & refactor load_model and save_model
    # to infer the model class to be used (TSP_GGCN, TSP_GGCN_v2, etc)

    model = TSP_GGCN().to(device)
    model_filepath = TRAINED_MODELS_FOLDER_PATH / f"{model_name}.pt"
    print(f"...Loading model from file {model_filepath}")
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model


def get_model_name(model: torch.nn.Module) -> str:
    # generates a model name based on the name of the architecture and
    # the current date and time
    model_architecture_name = model.__class__.__name__
    training_timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    model_name = f"{model_architecture_name}_{training_timestamp}"
    return model_name
