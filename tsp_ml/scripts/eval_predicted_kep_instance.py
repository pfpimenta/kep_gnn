# -*- coding: utf-8 -*-
# script to evaluate a solution for a KEP instance,
# predicted by the model specified by TRAINED_MODEL_NAME

import torch
from kep_evaluation import evaluate_kep_instance_prediction
from paths import get_predictions_folder_path

# TRAINED_MODEL_NAME = "2022_09_19_23h55_GreedyPathsModel"
TRAINED_MODEL_NAME = "2022_12_09_01h15_KEP_GAT_PNA_CE"
STEP = "test"

# get filepath
predictions_dir = get_predictions_folder_path(
    dataset_name="KEP",
    step=step,
    trained_model_name=TRAINED_MODEL_NAME,
)
predicted_instances_dir = predictions_dir / "predicted_instances"
# filename = listdir(predicted_instances_dir)[0]
instance_id = "c81db4ddd79ac29c2492bec7bb6727e1"
filename = instance_id + "_pred.pt"
filepath = predicted_instances_dir / filename

# load predicted instance
predicted_instance = torch.load(filepath)


# evaluate prediction
print(f"torch.sum(predicted_instance.pred): {torch.sum(predicted_instance.pred)}")
print(f"torch.unique(predicted_instance.pred): {torch.unique(predicted_instance.pred)}")
kep_prediction_evaluation = evaluate_kep_instance_prediction(predicted_instance)

breakpoint()
