import pathlib

PROJECT_FOLDER_PATH = pathlib.Path(__file__).parent.parent.resolve()
NX_GRAPHS_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/nx_graphs/"
PYG_GRAPHS_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/pyg_graphs/"
TRAIN_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/train/pyg_graphs/"
TEST_DATASET_FOLDER_PATH = PROJECT_FOLDER_PATH / "data/test/pyg_graphs/"
TRAINED_MODELS_FOLDER_PATH = PROJECT_FOLDER_PATH / "trained_models"
