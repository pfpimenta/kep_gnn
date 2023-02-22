# -*- coding: utf-8 -*-
# script to evaluate a solution found by the roth2014recursive solver
# for a test instance with 300 nodes
import torch
from kep_evaluation import evaluate_kep_instance_prediction
from paths import RESULTS_FOLDER_PATH, get_dataset_folder_path

# get KEP instance filepath
dataset_dir = get_dataset_folder_path(dataset_name="KEP", step="test")
filename = "kep_instance_c81db4ddd79ac29c2492bec7bb6727e1.pt"
filepath = dataset_dir / filename
# load KEP instance
kep_instance = torch.load(filepath)
print(f"Loaded KEP instance from {filepath}")
# load solution
solution_filepath = RESULTS_FOLDER_PATH / "2023_02_15_solution.txt"
# solution_filepath = RESULTS_FOLDER_PATH / "2023_02_19_solution.txt"
with open(solution_filepath) as file:
    solution = eval(file.read())
print(f"Loaded solution from {solution_filepath}")

# eval
kep_instance.pred = torch.Tensor(solution)
kep_prediction_evaluation = evaluate_kep_instance_prediction(kep_instance)
print(kep_prediction_evaluation)
breakpoint()
