# -*- coding: utf-8 -*-
from os import listdir
from pathlib import Path
from typing import List

import torch
from torch_geometric.data import Data, Dataset


class DTSPDataset(Dataset):
    """TODO description"""

    def __init__(self, dataset_folderpath: str, transform=None, pre_transform=None):
        super(DTSPDataset, self).__init__(transform, pre_transform)
        self.dataset_folderpath = dataset_folderpath
        self.__num_edges = None

    @property
    def num_classes(self) -> int:
        """The DTSP problem is modelled here as an graph binary
        classification problem: a (graph, cost) tuple may either be
        valid or invalid.
        """
        return 2

    @property
    def num_edges(self) -> int:
        if self.__num_edges is None:
            self.__num_edges = 0
            for i in range(self.len()):
                filepath = Path(self.dataset_folderpath) / self.processed_file_names[i]
                data = torch.load(filepath)
                self.__num_edges += data.num_edges
        return self.__num_edges

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
        return "DTSP"
