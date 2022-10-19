# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)


from model_utils import save_model
from models.greedy_cycles_model import GreedyCyclesModel

# generate and save greedy model as a .PT file
greedy_model = GreedyCyclesModel()
training_report = {}
save_model(model=greedy_model, training_report=training_report)
