# -*- coding: utf-8 -*-
import sys
from typing import Tuple

import torch
from definitions import (
    TEST_DATASET_FOLDER_PATH,
    TRAIN_DATASET_FOLDER_PATH,
    TRAINED_MODELS_FOLDER_PATH,
)
from model_performance import ModelPerformance
from models.tsp_ggcn import TSP_GGCN
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tsp_dataset import TSPDataset

# MODEL_FILENAME = "tsp_gcn_model.pt"
# MODEL_FILENAME = "TSP_GGCN_2022_07_05_16h31.pt"
# MODEL_FILENAME = "TSP_GGCN_2022_07_07_17h56.pt"
# MODEL_FILENAME = "TSP_GGCN_2022_07_07_19h40.pt"
# MODEL_FILENAME = "TSP_GGCN_2022_07_08_17h00.pt"
MODEL_FILENAME = "TSP_GGCN_2022_07_12_12h27.pt"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


def evaluate(model: torch.nn.Module, dataset: TSPDataset) -> ModelPerformance:
    model_performance = ModelPerformance()
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4
    )
    model.eval()  # set the model to evaluation mode
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluation", file=sys.stdout)):
        batch = batch.to(device)
        label = batch.y
        label = label.to(torch.float32)
        scores = model(batch)
        pred = torch.argmax(scores, 1).to(int)
        model_performance.update(pred=pred, truth=batch.y)
        if i == 30:
            pass
            # import pdb
            # pdb.set_trace()

    dataset_size = len(dataset)
    print(f"dataset_size: {dataset_size}")
    print(f"dataset.num_edges: {dataset.num_edges}")
    avg_num_edges = int(dataset.num_edges) / int(dataset_size)
    print(f"avg_num_edges: {avg_num_edges}")
    model_performance.print()


# load model
model = TSP_GGCN().to(device)
model_filepath = TRAINED_MODELS_FOLDER_PATH / MODEL_FILENAME
print(f"...Loading model from file {model_filepath}")
model.load_state_dict(torch.load(model_filepath, map_location=device))

# setup data
batch_size = 10
train_dataset = TSPDataset(dataset_folderpath=TRAIN_DATASET_FOLDER_PATH)
test_dataset = TSPDataset(dataset_folderpath=TEST_DATASET_FOLDER_PATH)

print("\n\nEvaluating the model on the train dataset")
train_model_performance = evaluate(model=model, dataset=train_dataset)
print("\n\nEvaluating the model on the test dataset")
test_model_performance = evaluate(model=model, dataset=test_dataset)
