# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
from greedy import greedy
from paths import TRAINED_MODELS_FOLDER_PATH
from torch import Tensor
from torch_geometric.data import Batch


class KEP_GNN(torch.nn.Module):
    """Abstract class to generalize a couple of methods
    that all GNNs for the KEP problem should have
    """

    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.__device = device
        self.__training_report = None
        self.__trained_model_name = None
        self.__trained_model_dir = None

    @abstractmethod
    def forward(self, data: Batch) -> Tensor:
        raise NotImplementedError()

    def predict(self, data: Batch) -> Tensor:
        # TODO make simple edge classification (argmax) available
        solution = greedy(
            edge_scores=data.scores,
            edge_index=data.edge_index,
            node_types=data.type[0],
            greedy_algorithm=self.predict_method,
            device=self.__device,
        )
        return solution

    @property
    def training_report(self) -> Optional[Dict[str, Any]]:
        return self.__training_report

    @training_report.setter
    def training_report(self, training_report: Dict[str, Any]):
        if self.training_report is None:
            self.__training_report = training_report
        else:
            raise ValueError("Training report already set for this model!")

    @property
    def trained_model_name(self) -> str:
        """Generates a model name based on the name of the architecture and
        the current date and time"""
        if self.__trained_model_name is None:
            model_architecture_name = self.__class__.__name__
            training_timestamp = self.training_report["training_start_time"]
            self.__trained_model_name = (
                f"{training_timestamp}_{model_architecture_name}"
            )
            self.training_report["trained_model_name"] = self.__trained_model_name
        return self.__trained_model_name

    @property
    def trained_model_dir(self) -> str:
        if self.__trained_model_dir is None:
            self.__trained_model_dir = (
                TRAINED_MODELS_FOLDER_PATH / f"{self.trained_model_name}"
            )
        self.__trained_model_dir.mkdir(parents=True, exist_ok=True)
        return self.__trained_model_dir
