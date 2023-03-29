# -*- coding: utf-8 -*-
import pathlib
from typing import Optional

PROJECT_FOLDER_PATH = pathlib.Path(__file__).parent.parent.resolve()
TRAINED_MODELS_FOLDER_PATH = PROJECT_FOLDER_PATH / "trained_models"
RESULTS_FOLDER_PATH = PROJECT_FOLDER_PATH / "results"
PLOTS_FOLDER_PATH = RESULTS_FOLDER_PATH / "plots"
PREDICTION_TIME_FOLDER_PATH = RESULTS_FOLDER_PATH / "prediction_time"
MODEL_PERFORMANCE_FOLDER_PATH = PROJECT_FOLDER_PATH / "model_performance"


def get_dataset_folder_path(dataset_name: str, step: str = "train") -> pathlib.Path:
    folder_path = PROJECT_FOLDER_PATH / "data" / dataset_name / "dataset" / step
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def get_predictions_folder_path(
    dataset_name: str,
    trained_model_name: str,
    step: str = "train",
    cycle_path_size_limit: Optional[int] = None,
) -> pathlib.Path:
    if cycle_path_size_limit is None:
        cycle_path_size_limit = "no"
    folder_path = (
        PROJECT_FOLDER_PATH
        / "data"
        / dataset_name
        / "predictions"
        / trained_model_name
        / step
        / f"{cycle_path_size_limit}_size"
    )
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def get_evaluation_folder_path(
    dataset_name: str,
    trained_model_name: Optional[str] = None,
    step: str = "train",
    cycle_path_size_limit: Optional[int] = None,
) -> pathlib.Path:
    if cycle_path_size_limit is None:
        cycle_path_size_limit = "no"
    if trained_model_name is None:
        # return base folder for evaluation data
        folder_path = PROJECT_FOLDER_PATH / "data" / dataset_name / "evaluation"
    else:
        folder_path = (
            PROJECT_FOLDER_PATH
            / "data"
            / dataset_name
            / "evaluation"
            / trained_model_name
            / step
            / f"{cycle_path_size_limit}_size"
        )
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


# create all folders that do not exist yet:
folder_path_list = [
    PROJECT_FOLDER_PATH,
    TRAINED_MODELS_FOLDER_PATH,
    MODEL_PERFORMANCE_FOLDER_PATH,
    RESULTS_FOLDER_PATH,
    PLOTS_FOLDER_PATH,
    PREDICTION_TIME_FOLDER_PATH,
]
for folder_path in folder_path_list:
    folder_path.mkdir(parents=True, exist_ok=True)
