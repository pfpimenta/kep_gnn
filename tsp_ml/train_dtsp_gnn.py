# -*- coding: utf-8 -*-
import torch
from train import train

BATCH_SIZE = 2
NUM_EPOCHS = 2
LEARNING_RATE = 0.005
MODEL_NAME = "DTSP_GNN_Prates"

if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train(
        device=device,
        model_name=MODEL_NAME,
        dataset_name="DTSP",
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )
