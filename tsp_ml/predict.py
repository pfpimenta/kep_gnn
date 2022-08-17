# -*- coding: utf-8 -*-
# this script loads a model and a dataset, predicts, and saves predictions in a CSV
import pathlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from dataset_utils import get_dataset, print_dataset_information
from model_utils import load_model, set_torch_seed
from paths import get_predictions_folder_path
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# TODO refactor with evaluate.py (Don't Repeat Yourself principle)

DATASET_NAME = "KEPCE"
TRAINED_MODEL_NAME = "2022_08_17_17h30_KEPCE_GCN"
BATCH_SIZE = 10


def delete_predictions_CSV(filepath: pathlib.Path):
    print(f"Deleting {filepath} ...")
    filepath.unlink()


def create_predictions_CSV(filepath: str, dataset_name: str):
    with open(filepath, "a") as file:
        # create CSV file with header and 0 rows
        if dataset_name == "TSP" or dataset_name == "DTSP":
            csv_header = "id, predictions, truth\n"
        elif dataset_name == "KEP" or dataset_name == "KEPCE":
            csv_header = "id, predictions\n"
        else:
            raise ValueError(f"No dataset named '{dataset_name}' found.")
        file.write(csv_header)


def initialize_predictions_CSV(
    output_dir: str,
    dataset_name: str,
    overwrite_results: bool = True,
) -> Tuple[bool, str]:
    if output_dir is None:
        raise ValueError("No output_dir was passed.")
    # create output_dir if it does not exist yet
    output_dir.mkdir(parents=True, exist_ok=True)
    # check if CSV file already exist
    filepath = Path(output_dir) / "predictions.csv"
    if not filepath.exists():
        save_predictions = True
    else:
        if overwrite_results:
            delete_predictions_CSV(filepath=filepath)
            save_predictions = True
        else:
            # ask user if they want the file replaced
            replace_file = input(
                f"File already exists: {filepath} ... Replace it? (Y to confirm): "
            ).lower()
            if replace_file == "y":
                delete_predictions_CSV(filepath=filepath)
                save_predictions = True
            else:
                print("Predictions will not be saved.")
                save_predictions = False
    # create CSV file with header
    if save_predictions:
        print(f"Will save predictions at {filepath}")
        create_predictions_CSV(filepath=filepath, dataset_name=dataset_name)

    return save_predictions, filepath


def save_predictions_to_csv(
    instance_ids: List[str],
    pred: torch.Tensor,
    filepath: str,
    truth: Optional[torch.Tensor] = None,
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
    dataset: Dataset,
    batch_size: int,
    output_dir: str,
    save_as_pt: bool = True,
    save_csv: bool = False,
) -> None:
    """Uses the model to make predictions on the given dataset,
    and then saves the predictions on 'output_dir' in a CSV.
    If 'save_as_pt' is True, each instance is predicted separately
    and each prediction is saved with the rest of the instance data in
    a .PT file."""

    set_torch_seed()

    # check if predictions CSV file already exist
    if save_csv:
        save_csv, csv_filepath = initialize_predictions_CSV(
            output_dir=output_dir,
            dataset_name=DATASET_NAME,
        )
    if save_as_pt:
        print("Setting batch_size to 1 in order to save each predicted instance...")
        batch_size = 1
        predicted_instances_dir = output_dir / "predicted_instances"
        predicted_instances_dir.mkdir(parents=True, exist_ok=True)
    if save_csv is False and save_as_pt is False:
        print(
            "Since predictions would not be saved,"
            " the prediction will be skipped altogheter."
        )
        return None

    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4
    )

    num_instances = len(dataset)

    model.eval()  # set the model to evaluation mode to freeze the weights

    for i, batch in enumerate(tqdm(dataloader, desc="Prediction", file=sys.stdout)):
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

        # save predicted instance in a .PT file
        if save_as_pt:
            graph = batch[0]
            graph.scores = scores
            graph.pred = pred
            graph_filepath = predicted_instances_dir / (graph.id + "_pred.pt")
            torch.save(graph, graph_filepath)
            # print(f"[{i+1}/{num_instances}] Saved {graph_filepath}")

        # save predictions in a CSV
        if save_csv:
            save_predictions_to_csv(
                filepath=csv_filepath,
                instance_ids=instance_ids,
                pred=pred,
                truth=batch.y,
            )


if __name__ == "__main__":
    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # load model
    model = load_model(trained_model_name=TRAINED_MODEL_NAME)

    # steps_to_predict = ["train", "test", "val"]
    steps_to_predict = ["val"]
    for step in steps_to_predict:
        # setup data
        dataset = get_dataset(dataset_name=DATASET_NAME, step=step)
        print_dataset_information(dataset=dataset)

        print(f"\n\nPredicting on the {step} dataset")
        predictions_dir = get_predictions_folder_path(
            dataset_name=DATASET_NAME,
            step=step,
            trained_model_name=TRAINED_MODEL_NAME,
        )
        predict(
            model=model,
            dataset=dataset,
            output_dir=predictions_dir,
            batch_size=BATCH_SIZE,
            save_as_pt=True,
        )
