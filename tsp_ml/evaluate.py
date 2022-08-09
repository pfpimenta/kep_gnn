# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from dataset_utils import get_dataset
from model_utils import load_model, set_torch_seed
from paths import get_predictions_folder_path
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model_performance import ModelPerformance

DATASET_NAME = "KEP"
TRAINED_MODEL_NAME = "2022_08_08_17h35_KEP_GCN"
BATCH_SIZE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


def print_dataset_information(dataset: Dataset) -> None:
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    print(f"Dataset total num_edges: {dataset.num_edges}")
    avg_num_edges = int(dataset.num_edges) / int(dataset_size)
    print(f"Mean num_edges per graph: {avg_num_edges}")


def create_predictions_CSV(
    output_dir: str,
) -> Tuple[bool, str]:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
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
        if save_predictions:
            with open(filepath, "a") as file:
                # create CSV file with header and 0 rows
                csv_header = "id, predictions, truth\n"
                file.write(csv_header)
    return save_predictions, filepath


def evaluate(
    model: torch.nn.Module,
    dataset: torch.Dataset,
    batch_size: int,
    output_dir: Optional[str] = None,
) -> ModelPerformance:
    """Evaluates the model by making predictions on the given dataset,
    and then returns a performance report."""

    set_torch_seed()

    # check if predictions file already exist
    save_predictions, filepath = create_predictions_CSV(output_dir=output_dir)

    model_performance = ModelPerformance()
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4
    )
    model.eval()  # set the model to evaluation mode
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluation", file=sys.stdout)):
        batch = batch.to(device)
        label = batch.y
        label = label.to(torch.float32)
        scores = model(data=batch)
        # TODO evaluate KEP
        if dataset.dataset_name == "DTSP":
            pred = scores.to(int)  # DTSP
        elif dataset.dataset_name == "TSP":
            pred = torch.argmax(scores, 1).to(int)  # TSP
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

    print_dataset_information(dataset=dataset)
    model_performance.print()
    return model_performance


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # setup data
    train_dataset = get_dataset(dataset_name=DATASET_NAME, step="train")
    test_dataset = get_dataset(dataset_name=DATASET_NAME, step="test")

    # load model
    model = load_model(trained_model_name=TRAINED_MODEL_NAME)

    print("\n\nEvaluating the model on the train dataset")
    predictions_dir = get_predictions_folder_path(
        dataset_name=DATASET_NAME, step="train"
    )
    train_model_performance = evaluate(
        model=model,
        dataset=train_dataset,
        output_dir=predictions_dir,
        batch_size=BATCH_SIZE,
    )
    train_model_performance.save(output_filename="train_" + TRAINED_MODEL_NAME)

    print("\n\nEvaluating the model on the test dataset")
    predictions_dir = get_predictions_folder_path(
        dataset_name=DATASET_NAME, step="train"
    )
    test_model_performance = evaluate(
        model=model,
        dataset=test_dataset,
        output_dir=predictions_dir,
        batch_size=BATCH_SIZE,
    )
    test_model_performance.save(output_filename="test_" + TRAINED_MODEL_NAME)
