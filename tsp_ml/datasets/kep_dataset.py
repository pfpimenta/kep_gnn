# -*- coding: utf-8 -*-
import json
from os import listdir
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree

DATASET_PROPERTIES_FILENAME = "dataset_properties.json"


class KEPDataset(Dataset):
    def __init__(self, dataset_folderpath: str, transform=None, pre_transform=None):
        super(KEPDataset, self).__init__(transform, pre_transform)
        self.dataset_folderpath = dataset_folderpath
        self.__num_edges = None
        self.__num_nodes = None
        self.__maximum_in_degree = None
        self.__in_degree_histogram = None

    @property
    def processed_file_names(self) -> List[str]:
        processed_filenames = listdir(self.dataset_folderpath)
        # filter out JSON file
        if DATASET_PROPERTIES_FILENAME in processed_filenames:
            processed_filenames.remove(DATASET_PROPERTIES_FILENAME)
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
    def properties_json_filepath(self):
        return self.dataset_folderpath / DATASET_PROPERTIES_FILENAME

    def save_properties(self) -> None:
        """Saves computed properties in a JSON file
        in the same folder as the dataset instances (self.dataset_folderpath)"""
        dataset_properties = {
            "maximum_in_degree": self.maximum_in_degree,
            "in_degree_histogram": self.in_degree_histogram.tolist(),
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }
        with open(self.properties_json_filepath, "w", encoding="utf-8") as f:
            json.dump(dataset_properties, f, indent=4)
        print(f"Saved properties at {self.properties_json_filepath}")

    def load_properties(self) -> None:
        with open(self.properties_json_filepath) as json_file:
            dataset_properties = json.load(json_file)
        self.__num_nodes = dataset_properties["num_nodes"]
        self.__num_edges = dataset_properties["num_edges"]
        self.__maximum_in_degree = dataset_properties["maximum_in_degree"]
        self.__in_degree_histogram = torch.Tensor(
            dataset_properties["in_degree_histogram"]
        ).to(int)
        print(f"Loaded properties: {dataset_properties}")

    def compute_properties(self) -> None:
        """Computes all the dataset properties
        and stores them in hidden class attributes.
        The dataset properties are:
        - self.num_edges: The total number of edges in the dataset
        - self.num_nodes:The total number of nodes in the dataset
        - self.maximum_in_degree: The maximum in-degree of all nodes of all instances of the dataset
        - self.in_degree_histogram: Histogram of the quantity of in-degree values of the dataset nodes
        """
        print("Computing maximum in-degree of dataset...")
        print("Computing total number of edges and nodes of dataset...")
        self.__num_edges = 0
        self.__num_nodes = 0
        self.__maximum_in_degree = 0
        for data in self:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            self.__maximum_in_degree = max(self.__maximum_in_degree, int(d.max()))
            self.__num_edges += data.num_edges
            self.__num_nodes += data.num_nodes
        self.__in_degree_histogram = torch.zeros(
            self.maximum_in_degree + 1, dtype=torch.long
        )
        print("Computing in-degree histogram of dataset...")
        for data in self:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            self.__in_degree_histogram += torch.bincount(
                d, minlength=self.__in_degree_histogram.numel()
            )

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
            if self.properties_json_filepath.exists():
                # load previously computed total number of edges of dataset
                self.load_properties()
            else:
                # compute total number of edges of dataset
                self.compute_properties()
        return self.__num_edges

    @property
    def num_nodes(self) -> int:
        """Total number of nodes in all graphs of dataset"""
        if self.__num_nodes is None:
            if self.properties_json_filepath.exists():
                # load previously computed total number of nodes of dataset
                self.load_properties()
            else:
                # compute total number of nodes of dataset
                self.compute_properties()
        print(f"self.__num_nodes: {self.__num_nodes}")
        return self.__num_nodes

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
    def maximum_in_degree(self) -> int:
        """Maximum in-degree of all nodes in the dataset"""
        if self.__maximum_in_degree is None:
            if self.properties_json_filepath.exists():
                # load previously computed maximum in-degree of dataset
                self.load_properties()
            else:
                self.compute_properties()
        print(f"Maximum in-degree of dataset: {self.__maximum_in_degree}")
        return self.__maximum_in_degree

    @property
    def in_degree_histogram(self) -> torch.Tensor:
        """Histogram of the quantity of in-degree values of the dataset nodes.
        This tensor is used by th PNA Convolution."""
        if self.__in_degree_histogram is None:
            # compute in-degree histogram of dataset
            self.compute_properties()
        return self.__in_degree_histogram
