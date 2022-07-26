# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets.tsp_dataset import TSPDataset
from model_utils import load_model, set_torch_seed
from paths import (
    TSP_TEST_DATASET_FOLDER_PATH,
    TSP_TEST_PREDICTIONS_FOLDER_PATH,
    TSP_TRAIN_DATASET_FOLDER_PATH,
    TSP_TRAIN_PREDICTIONS_FOLDER_PATH,
)
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model_performance import ModelPerformance

TRAINED_MODEL_NAME = "TSP_GGCN_2022_07_25_22h40"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


def evaluate(
    model: torch.nn.Module, dataset: TSPDataset, output_dir: Optional[str] = None
) -> ModelPerformance:
    """Evaluates the model by making predictions on the given dataset,
    and then returns a performance report."""

    set_torch_seed()

    # check if predictions file already exist
    if output_dir is not None:
        save_predictions = True
        filepath = Path(output_dir) / "predictions.csv"
        if filepath.exists():
            replace_file = input(
                f"File already exists: {filepath} ... Replace it? (Y to confirm): "
            ).lower()
            if replace_file == "y":
                print(f"Deleting {filepath} ...")
                filepath.unlink()
                print(f"Will save predictions at {filepath}.")
            else:
                print("Predictions will not be saved.")
                save_predictions = False

    model_performance = ModelPerformance()
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4
    )
    model.eval()  # set the model to evaluation mode
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluation", file=sys.stdout)):
        batch = batch.to(device)
        label = batch.y
        label = label.to(torch.float32)
        scores = model(data=batch)
        pred = torch.argmax(scores, 1).to(int)
        model_performance.update(pred=pred, truth=batch.y)
        if save_predictions:
            # save predictions
            y_predictions = pred.detach().cpu().tolist()
            y_truths = batch.y.detach().cpu().tolist()
            row, _ = batch.edge_index
            edge_batch = batch.batch[row]
            instance_ids = [batch.id[idx] for idx in edge_batch.tolist()]
            with open(filepath, "a") as file:
                for (id, pred, truth) in zip(instance_ids, y_predictions, y_truths):
                    csv_row = f"{id},{int(pred)},{int(truth)}\n"
                    file.write(csv_row)

    # print dataset information
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

    # load model
    trained_model_name = TRAINED_MODEL_NAME
    model = load_model(trained_model_name=trained_model_name)

    print("\n\nEvaluating the model on the train dataset")
    train_model_performance = evaluate(
        model=model,
        dataset=train_dataset,
        output_dir=TSP_TRAIN_PREDICTIONS_FOLDER_PATH,
    )
    train_model_performance.save(output_filename="train_" + trained_model_name)

    print("\n\nEvaluating the model on the test dataset")
    test_model_performance = evaluate(
        model=model,
        dataset=test_dataset,
        output_dir=TSP_TEST_PREDICTIONS_FOLDER_PATH,
    )
    test_model_performance.save(output_filename="test_" + trained_model_name)
