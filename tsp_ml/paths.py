# -*- coding: utf-8 -*-
import pathlib

PROJECT_FOLDER_PATH = pathlib.Path(__file__).parent.parent.resolve()
TRAINED_MODELS_FOLDER_PATH = PROJECT_FOLDER_PATH / "trained_models"
MODEL_PERFORMANCE_FOLDER_PATH = PROJECT_FOLDER_PATH / "model_performance"

TSP_TRAIN_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/tsp_dataset/train"
TSP_TEST_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/tsp_dataset/test"
TSP_VAL_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/tsp_dataset/val"
DTSP_TRAIN_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/dtsp_dataset/train"
DTSP_TEST_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/dtsp_dataset/test"
DTSP_VAL_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/dtsp_dataset/val"


folder_path_list = [
    PROJECT_FOLDER_PATH,
    TRAINED_MODELS_FOLDER_PATH,
    MODEL_PERFORMANCE_FOLDER_PATH,
    TSP_TRAIN_DATASET_FOLDER_PATH,
    TSP_TEST_DATASET_FOLDER_PATH,
    TSP_VAL_DATASET_FOLDER_PATH,
    DTSP_TRAIN_DATASET_FOLDER_PATH,
    DTSP_TEST_DATASET_FOLDER_PATH,
    DTSP_VAL_DATASET_FOLDER_PATH,
]

# create all folders that do not exist yet
for folder_path in folder_path_list:
    folder_path.mkdir(parents=True, exist_ok=True)
