# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import torch
import torch_geometric
from python_tsp.exact import solve_tsp_dynamic_programming
from torch_geometric.data import DataLoader
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

print("...everything is fine! All imports worked!")
