# -*- coding: utf-8 -*-
import pathlib
from typing import List

PROJECT_FOLDER_PATH = pathlib.Path(__file__).parent.parent.resolve()
TRAINED_MODELS_FOLDER_PATH = PROJECT_FOLDER_PATH / "trained_models"
MODEL_PERFORMANCE_FOLDER_PATH = PROJECT_FOLDER_PATH / "model_performance"


def get_dataset_folder_path(dataset_name: str, step: str = "train") -> pathlib.Path:
    folder_path = PROJECT_FOLDER_PATH / "data" / dataset_name / "dataset" / step
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def get_predictions_folder_path(
    dataset_name: str,
    trained_model_name: str,
    step: str = "train",
) -> pathlib.Path:
    folder_path = (
        PROJECT_FOLDER_PATH
        / "data"
        / dataset_name
        / "predictions"
        / trained_model_name
        / step
    )
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def get_eval_results_folder_path(
    dataset_name: str,
    trained_model_name: str,
    step: str = "train",
) -> pathlib.Path:
    folder_path = (
        PROJECT_FOLDER_PATH
        / "data"
        / dataset_name
        / "evaluation"
        / trained_model_name
        / step
    )
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


# create all folders that do not exist yet:
folder_path_list = [
    PROJECT_FOLDER_PATH,
    TRAINED_MODELS_FOLDER_PATH,
    MODEL_PERFORMANCE_FOLDER_PATH,
]
for folder_path in folder_path_list:
    folder_path.mkdir(parents=True, exist_ok=True)
