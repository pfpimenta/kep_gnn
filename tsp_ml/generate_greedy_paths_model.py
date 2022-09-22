# -*- coding: utf-8 -*-
from greedy_paths_model import GreedyPathsModel
from model_utils import save_model

# generate and save greedy model as a .PT file
greedy_model = GreedyPathsModel()
training_report = {}
save_model(model=greedy_model, training_report=training_report)
