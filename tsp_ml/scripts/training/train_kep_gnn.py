# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import torch

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)
from train import DATASET_NAME, PREDICT_METHOD, train

BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
OPTIMIZER_WEIGHT_DECAY = 0.0  # L2 regularization
MODEL_NAME = "KEP_GAT_PNA_CE"
# MODEL_NAME = "KEP_1L_GNN"
DATASET_NAME = "KEP"
PREDICT_METHOD = "greedy_paths"
# PREDICT_METHOD = "greedy_cycles"
VALIDATION_PERIOD = 100
USE_VALIDATION = True


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train(
        device=device,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer_weight_decay=OPTIMIZER_WEIGHT_DECAY,
        use_validation=USE_VALIDATION,
        validation_period=VALIDATION_PERIOD,
        predict_method=PREDICT_METHOD,
    )
