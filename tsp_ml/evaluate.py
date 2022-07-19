# -*- coding: utf-8 -*-
import sys

import torch
from datasets.tsp_dataset import TSPDataset
from model_utils import get_model_name, load_model
from paths import TSP_TEST_DATASET_FOLDER_PATH, TSP_TRAIN_DATASET_FOLDER_PATH
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model_performance import ModelPerformance

MODEL_NAME = "TSP_GGCN_2022_07_18_22h43"


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
    print(f"Dataset size: {dataset_size}")
    print(f"Dataset total num_edges: {dataset.num_edges}")
    avg_num_edges = int(dataset.num_edges) / int(dataset_size)
    print(f"Mean num_edges per graph: {avg_num_edges}")
    model_performance.print()
    return model_performance


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # setup data
    batch_size = 10
    train_dataset = TSPDataset(dataset_folderpath=TSP_TRAIN_DATASET_FOLDER_PATH)
    test_dataset = TSPDataset(dataset_folderpath=TSP_TEST_DATASET_FOLDER_PATH)

    model_name = MODEL_NAME
    model = load_model(model_name=model_name)
    print("\n\nEvaluating the model on the train dataset")
    train_model_performance = evaluate(model=model, dataset=train_dataset)
    train_model_performance.save(output_filename="train_" + model_name)
    print("\n\nEvaluating the model on the test dataset")
    test_model_performance = evaluate(model=model, dataset=test_dataset)
    test_model_performance.save(output_filename="test_" + model_name)
