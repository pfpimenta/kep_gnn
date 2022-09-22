# -*- coding: utf-8 -*-
# first, predicts the solutions on every instance of the test dataset;
# then, compute and save evaluation for the predictions done.

import torch
from dataset_utils import get_dataset, print_dataset_information
from kep_evaluation import kep_evaluation
from model_utils import load_model
from paths import get_predictions_folder_path
from predict import predict

DATASET_NAME = "KEP"
TRAINED_MODEL_NAME = "2022_09_22_02h42_KEP_GAT_PNA_CE"
STEP = "test"

if __name__ == "__main__":
    ## first, predicts the solutions on every instance of the test dataset,

    # select either CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # setup data
    dataset = get_dataset(dataset_name=DATASET_NAME, step=STEP)
    print_dataset_information(dataset=dataset)

    # load model
    model = load_model(trained_model_name=TRAINED_MODEL_NAME, dataset=dataset)

    print(f"\n\nPredicting on the {STEP} dataset")
    predictions_dir = get_predictions_folder_path(
        dataset_name=DATASET_NAME,
        step=STEP,
        trained_model_name=TRAINED_MODEL_NAME,
    )
    predict(
        model=model,
        device=device,
        dataset=dataset,
        output_dir=predictions_dir,
        batch_size=1,
        save_as_pt=True,
    )

    ## then, compute and save evaluation for the predictions done.
    kep_evaluation(
        step=STEP,
        trained_model_name=TRAINED_MODEL_NAME,
        dataset_name=DATASET_NAME,
    )