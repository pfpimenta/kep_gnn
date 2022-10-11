# -*- coding: utf-8 -*-
from model_utils import save_model
from models.greedy_cycles_model import GreedyCyclesModel

# generate and save greedy model as a .PT file
greedy_model = GreedyCyclesModel()
training_report = {}
save_model(model=greedy_model, training_report=training_report)
