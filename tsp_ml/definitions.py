# -*- coding: utf-8 -*-
import pathlib

PROJECT_FOLDER_PATH = pathlib.Path(__file__).parent.parent.resolve()
NX_GRAPHS_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/nx_graphs/"
PYG_GRAPHS_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/pyg_graphs/"
TRAIN_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/train/pyg_graphs/"
TEST_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/test/pyg_graphs/"
TRAINED_MODELS_FOLDER_PATH = PROJECT_FOLDER_PATH / "trained_models"
MODEL_PERFORMANCE_FOLDER_PATH = PROJECT_FOLDER_PATH / "model_performance"

folder_path_list = [
    PROJECT_FOLDER_PATH,
    NX_GRAPHS_FOLDER_PATH,
    PYG_GRAPHS_FOLDER_PATH,
    TRAIN_DATASET_FOLDER_PATH,
    TEST_DATASET_FOLDER_PATH,
    TRAINED_MODELS_FOLDER_PATH,
    MODEL_PERFORMANCE_FOLDER_PATH,
]

# create all folders that do not exist yet
for folder_path in folder_path_list:
    folder_path.mkdir(parents=True, exist_ok=True)
