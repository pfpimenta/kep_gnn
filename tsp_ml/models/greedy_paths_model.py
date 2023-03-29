# -*- coding: utf-8 -*-
from typing import Optional

import torch
from greedy import greedy
from torch import Tensor
from torch_geometric.data import Batch


class GreedyPathsModel(torch.nn.Module):
    """Greedy deterministic heuristic that only finds paths
    beggining in NDD (donor) nodes"""

    def __init__(
        self,
        cycle_path_size_limit: Optional[int] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.__device = device
        self.__path_size_limit = cycle_path_size_limit

    def forward(self, data: Batch) -> Tensor:
        edge_scores = data.edge_weights.view(-1, 1)
        # score for negative class = 1 - score
        edge_scores = edge_scores.repeat(1, 2)
        edge_scores[:, 1] = 1 - edge_scores[:, 1]
        return edge_scores

    def predict(self, data: Batch) -> Tensor:
        solution = greedy(
            edge_scores=data.scores,
            edge_index=data.edge_index,
            node_types=data.type[0],
            greedy_algorithm="greedy_paths",
            cycle_path_size_limit=self.__path_size_limit,
            device=self.__device,
        )
        return solution
