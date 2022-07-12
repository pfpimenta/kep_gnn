from os import listdir
from typing import List
import torch
from torch_geometric.data import Dataset, Data

class TSPDataset(Dataset):
    def __init__(self, dataset_folderpath: str, transform=None, pre_transform=None):
        super(TSPDataset, self).__init__(transform, pre_transform)
        self.dataset_folderpath = dataset_folderpath

    @property
    def num_classes(self) -> int:
        ''' The TSP problem is modelled here as an edge binary
        classification problem: an edge may either be or
        not be in the solution route
        '''
        return 2

    @property
    def processed_file_names(self) -> List[str]:
        processed_filenames = listdir(self.dataset_folderpath)
        return processed_filenames

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        filepath = self.dataset_folderpath / self.processed_file_names[idx]
        data = torch.load(filepath)
        return data
