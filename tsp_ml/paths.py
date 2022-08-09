# -*- coding: utf-8 -*-
import pathlib
from typing import List

PROJECT_FOLDER_PATH = pathlib.Path(__file__).parent.parent.resolve()
TRAINED_MODELS_FOLDER_PATH = PROJECT_FOLDER_PATH / "trained_models"
MODEL_PERFORMANCE_FOLDER_PATH = PROJECT_FOLDER_PATH / "model_performance"


def get_dataset_folder_path(dataset_name: str, step: str = "train") -> pathlib.Path:
    folder_path = PROJECT_FOLDER_PATH / "data" / dataset_name / "dataset" / step
    return folder_path


def get_predictions_folder_path(dataset_name: str, step: str = "train") -> pathlib.Path:
    folder_path = PROJECT_FOLDER_PATH / "data" / dataset_name / "predictions" / step
    return folder_path


def get_all_folder_paths() -> List[pathlib.Path]:
    """put all folder paths in a list"""
    folder_path_list = [
        PROJECT_FOLDER_PATH,
        TRAINED_MODELS_FOLDER_PATH,
        MODEL_PERFORMANCE_FOLDER_PATH,
    ]
    for dataset_name in ["TSP", "DTSP", "KEP"]:
        for step in ["train", "test", "val"]:
            # dataset folder paths:
            dataset_folder_path = get_dataset_folder_path(
                dataset_name=dataset_name, step=step
            )
            folder_path_list.append(dataset_folder_path)
            # predictions folder paths:
            predictions_folder_path = get_predictions_folder_path(
                dataset_name=dataset_name, step=step
            )
            folder_path_list.append(predictions_folder_path)
    return folder_path_list


# create all folders that do not exist yet:
folder_path_list = get_all_folder_paths()
for folder_path in folder_path_list:
    folder_path.mkdir(parents=True, exist_ok=True)
