# -*- coding: utf-8 -*-
# this script loads a model and a dataset, predicts, and saves predictions in a CSV
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from dataset_utils import get_dataset, print_dataset_information
from model_utils import load_model, set_torch_seed
from paths import get_predictions_folder_path
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# TODO refactor with evaluate.py (Don't Repeat Yourself principle)

DATASET_NAME = "KEP"
TRAINED_MODEL_NAME = "2022_08_10_22h49_KEP_GCN"
BATCH_SIZE = 10


def create_predictions_CSV(
    output_dir: str,
    trained_model_name: str,
    dataset_name: str,
) -> Tuple[bool, str]:
    if output_dir is not None:
        save_predictions = True
        output_dir = Path(output_dir) / trained_model_name
        output_dir.mkdir(parents=True, exist_ok=True)
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
                if dataset_name == "TSP" or dataset_name == "DTSP":
                    csv_header = "id, predictions, truth\n"
                elif dataset_name == "KEP":
                    csv_header = "id, predictions\n"
                file.write(csv_header)
    return save_predictions, filepath


def save_predictions_to_csv(
    instance_ids: List[str], pred: torch.Tensor, truth: torch.Tensor, filepath: str
):
    """Saves the predictions given in a CSV file"""
    y_predictions = pred.detach().cpu().tolist()
    if truth is not None:
        # with ground truth Y values
        y_truths = truth.detach().cpu().tolist()
        with open(filepath, "a") as file:
            for (id, pred, truth) in zip(instance_ids, y_predictions, y_truths):
                csv_row = f"{id},{int(pred)},{int(truth)}\n"
                file.write(csv_row)
    else:
        # without ground truth Y values
        with open(filepath, "a") as file:
            for (id, pred) in zip(instance_ids, y_predictions):
                csv_row = f"{id},{int(pred)}\n"
                file.write(csv_row)


def predict(
    model: torch.nn.Module,
    trained_model_name: str,
    dataset: Dataset,
    batch_size: int,
    output_dir: str,
) -> None:
    """Uses the model to make predictions on the given dataset,
    and then saves the predictions on 'output_dir'"""

    set_torch_seed()

    # check if predictions file already exist
    save_predictions, filepath = create_predictions_CSV(
        output_dir=output_dir,
        trained_model_name=trained_model_name,
        dataset_name=DATASET_NAME,
    )
    if save_predictions is False:
        return None

    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4
    )

    model.eval()  # set the model to evaluation mode to freeze the weights

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluation", file=sys.stdout)):
        batch = batch.to(device)
        if batch.y is not None:
            label = batch.y
            label = label.to(torch.float32)

        scores = model(data=batch)
        if dataset.dataset_name == "DTSP":
            # graph+cost classification
            pred = scores.to(int)  # DTSP
        elif dataset.dataset_name == "TSP" or "KEP":
            # edge classification
            pred = torch.argmax(scores, 1).to(int)
        # get instance IDs
        row, _ = batch.edge_index
        edge_batch = batch.batch[row]
        instance_ids = [batch.id[idx] for idx in edge_batch.tolist()]
        # save predictions in a CSV
        save_predictions_to_csv(
            filepath=filepath, instance_ids=instance_ids, pred=pred, truth=batch.y
        )


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # setup data
    train_dataset = get_dataset(dataset_name=DATASET_NAME, step="train")
    print_dataset_information(dataset=train_dataset)
    test_dataset = get_dataset(dataset_name=DATASET_NAME, step="test")
    print_dataset_information(dataset=test_dataset)
    val_dataset = get_dataset(dataset_name=DATASET_NAME, step="val")
    print_dataset_information(dataset=val_dataset)

    # load model
    model = load_model(trained_model_name=TRAINED_MODEL_NAME)

    print("\n\nPredicting on the train dataset")
    predictions_dir = get_predictions_folder_path(
        dataset_name=DATASET_NAME, step="train"
    )
    predict(
        model=model,
        trained_model_name=TRAINED_MODEL_NAME,
        dataset=train_dataset,
        output_dir=predictions_dir,
        batch_size=BATCH_SIZE,
    )

    print("\n\nPredicting on the test dataset")
    predictions_dir = get_predictions_folder_path(
        dataset_name=DATASET_NAME, step="test"
    )
    predict(
        model=model,
        trained_model_name=TRAINED_MODEL_NAME,
        dataset=test_dataset,
        output_dir=predictions_dir,
        batch_size=BATCH_SIZE,
    )

    print("\n\nPredicting on the validation dataset")
    predictions_dir = get_predictions_folder_path(dataset_name=DATASET_NAME, step="val")
    predict(
        model=model,
        trained_model_name=TRAINED_MODEL_NAME,
        dataset=val_dataset,
        output_dir=predictions_dir,
        batch_size=BATCH_SIZE,
    )
