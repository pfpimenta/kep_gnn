# -*- coding: utf-8 -*-
from datetime import datetime

import torch
from models.dtsp_gnn_prates import DTSP_GNN_Prates
from models.tsp_ggcn import TSP_GGCN
from models.tsp_ggcn_v2 import TSP_GGCN_v2
from models.tsp_ggcn_v4_weights import TSP_GGCN_v4_weights
from paths import TRAINED_MODELS_FOLDER_PATH

__AVAILABLE_MODELS = {
    "TSP_GGCN": TSP_GGCN,
    "TSP_GGCN_v2": TSP_GGCN_v2,
    "TSP_GGCN_v4_weights": TSP_GGCN_v4_weights,
    "DTSP_GNN_Prates": DTSP_GNN_Prates,
}


def save_model(model: torch.nn.Module):
    # save model state in the trained models folder
    model_name = get_model_name(model)
    model_filepath = TRAINED_MODELS_FOLDER_PATH / f"{model_name}.pt"
    torch.save(model.state_dict(), model_filepath)
    print(f"Saved {model_filepath}")


def load_model(
    trained_model_name: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.nn.Module:
    model_name = trained_model_name[:-16]
    model = get_model(model_name=model_name[:-1]).to(device)
    model_filepath = TRAINED_MODELS_FOLDER_PATH / f"{trained_model_name}.pt"
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


def get_model(model_name: str = "TSP_GGCN") -> torch.nn.Module:
    # returns an initialized model object
    try:
        Model = __AVAILABLE_MODELS[model_name]
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
