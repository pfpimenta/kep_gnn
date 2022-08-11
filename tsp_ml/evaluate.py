# -*- coding: utf-8 -*-
import sys
from typing import Optional

import torch
from dataset_utils import get_dataset, print_dataset_information
from model_utils import load_model, set_torch_seed
from paths import get_predictions_folder_path
from predict import create_predictions_CSV, save_predictions_to_csv
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model_performance import ModelPerformance

DATASET_NAME = "KEP"
TRAINED_MODEL_NAME = "2022_08_08_17h35_KEP_GCN"
BATCH_SIZE = 10


def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    output_dir: Optional[str] = None,
) -> ModelPerformance:
    """Evaluates the model by making predictions on the given dataset,
    and then returns a performance report."""

    set_torch_seed()

    # check if predictions file already exist
    save_predictions, filepath = create_predictions_CSV(
        output_dir=output_dir, dataset_name=dataset.name
    )

    model_performance = ModelPerformance()
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4
    )
    model.eval()  # set the model to evaluation mode

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluation", file=sys.stdout)):
        batch = batch.to(device)

        if batch.y is not None:
            label = batch.y
            label = label.to(torch.float32)

        # get predictions
        scores = model(data=batch)
        if dataset.dataset_name == "DTSP":
            # graph+cost classification
            pred = scores.to(int)  # DTSP
        elif dataset.dataset_name == "TSP" or "KEP":
            # edge classification
            pred = torch.argmax(scores, 1).to(int)
        model_performance.update(pred=pred, truth=batch.y)

        if save_predictions:
            # get instance IDs
            row, _ = batch.edge_index
            edge_batch = batch.batch[row]
            instance_ids = [batch.id[idx] for idx in edge_batch.tolist()]
            # save predictions in a CSV
            save_predictions_to_csv(
                filepath=filepath, instance_ids=instance_ids, pred=pred, truth=batch.y
            )

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
