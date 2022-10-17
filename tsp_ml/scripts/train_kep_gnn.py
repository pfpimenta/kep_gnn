# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import torch

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)
from train import DATASET_NAME, train

BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
MODEL_NAME = "KEP_GAT_PNA_CE"
DATASET_NAME = "KEP"
MINOR_EVAL = True

# TODO save training params with model

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
        use_validation=False,
        minor_eval=MINOR_EVAL,
    )
