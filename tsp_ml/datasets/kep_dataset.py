# -*- coding: utf-8 -*-
from os import listdir
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree


class KEPDataset(Dataset):
    def __init__(self, dataset_folderpath: str, transform=None, pre_transform=None):
        super(KEPDataset, self).__init__(transform, pre_transform)
        self.dataset_folderpath = dataset_folderpath
        self.__num_edges = None
        self.__maximum_in_degree = None
        self.__in_degree_histogram = None

    @property
    def num_classes(self) -> int:
        """The Kidney Exchange Problem (KEP) is modelled here as an edge
        binary classification problem: an edge may either be or
        not be in the solution
        """
        return 2

    @property
    def num_edges(self) -> int:
        """Total number of edges in all graphs of dataset"""
        if self.__num_edges is None:
            self.__num_edges = 0
            for data in self:
                self.__num_edges += data.num_edges
        return self.__num_edges

    @property
    def get_class_weights(self) -> Tuple[float, float]:
        """Class weights to adjust the loss function
        based on the class distribution of the given dataset.
        WARNING: as there is no ground truth labels, it is not possible
        to compute the class weights.
        """
        raise ValueError(
            "As there is no ground truth labels, it is not possible"
            " to compute the class weights."
        )

    @property
    def processed_file_names(self) -> List[str]:
        processed_filenames = listdir(self.dataset_folderpath)
        return processed_filenames

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        filepath = Path(self.dataset_folderpath) / self.processed_file_names[idx]
        data = torch.load(filepath)
        return data

    @property
    def dataset_name(self) -> str:
        return "KEP"

    @property
    def maximum_in_degree(self) -> int:
        """Maximum in-degree of all nodes in the dataset"""
        print("Computing maximum in-degree of dataset...")
        if self.__maximum_in_degree is None:
            self.__maximum_in_degree = -1
            for data in self:
                d = degree(
                    data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
                )
                self.__maximum_in_degree = max(self.__maximum_in_degree, int(d.max()))
            print(f"Maximum in-degree of dataset: {self.__maximum_in_degree}")
        return self.__maximum_in_degree

    @property
    def in_degree_histogram(self) -> torch.Tensor:
        """Histogram of the quantity of in-degree values of the dataset nodes.
        This tensor is used by th PNA Convolution."""
        print("Computing in-degree histogram of dataset...")
        if self.__in_degree_histogram is None:
            deg = torch.zeros(self.maximum_in_degree + 1, dtype=torch.long)
            for data in self:
                d = degree(
                    data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
                )
                deg += torch.bincount(d, minlength=deg.numel())
            self.__in_degree_histogram = deg
        return self.__in_degree_histogram
