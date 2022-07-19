# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from paths import MODEL_PERFORMANCE_FOLDER_PATH


def confusion_matrix(
    pred: torch.Tensor, truth: torch.Tensor
) -> Tuple[int, int, int, int]:
    """Returns the confusion matrix for the values in the `pred` and `truth`
    tensors, i.e. the amount of positions where the values of `pred`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    confusion_vector = pred / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where pred and truth are 1 (True Positive)
    #   inf   where pred is 1 and truth is 0 (False Positive)
    #   nan   where pred and truth are 0 (True Negative)
    #   0     where pred is 0 and truth is 1 (False Negative)

    TP = torch.sum(confusion_vector == 1).item()
    FP = torch.sum(confusion_vector == float("inf")).item()
    TN = torch.sum(torch.isnan(confusion_vector)).item()
    FN = torch.sum(confusion_vector == 0).item()

    return TP, FP, TN, FN


@dataclass
class ModelPerformance:
    TP: int = 0  # true positives
    TN: int = 0  # true negatives
    FP: int = 0  # false positives
    FN: int = 0  # false negatives

    def update(self, pred: torch.Tensor, truth: torch.Tensor):
        # updates the performance report
        TP, FP, TN, FN = confusion_matrix(pred=pred, truth=truth)
        self.update_confusion_matrix(TP=TP, FP=FP, TN=TN, FN=FN)

    def update_confusion_matrix(self, TP: int, FP: int, TN: int, FN: int):
        self.TP += TP
        self.TN += TN
        self.FP += FP
        self.FN += FN

    @property
    def precision(self) -> float:
        precision = self.TP / (self.TP + self.FP)
        return precision

    @property
    def recall(self) -> float:
        recall = self.TP / (self.TP + self.FN)
        return recall

    @property
    def accuracy(self) -> float:
        total_corrects = self.TN + self.TP
        total_edges = total_corrects + self.FN + self.FP
        accuracy = total_corrects / total_edges
        return accuracy

    @property
    def report_dict(self) -> Dict[str, Any]:
        report_dict = {
            "TP": self.TP,
            "TN": self.TN,
            "FP": self.FP,
            "FN": self.FN,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
        }
        return report_dict

    def save(self, output_filename: str) -> None:
        # saves the performance report data in a JSON file
        filepath = MODEL_PERFORMANCE_FOLDER_PATH / f"{output_filename}.json"
        with open(filepath, "w") as outfile:
            json.dump(self.report_dict, outfile)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath) as json_file:
            data = json.load(json_file)
            print(data)
            # TODO
            # return cls(TP= , )

    def print(self):
        # Prints the report information in a "friendly and pretty" way
        # TODO
        print(f"TP, FP: {self.TP, self.FP}")
        print(f"TN, FN: {self.TN, self.FN}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"Accuracy: {self.accuracy:.4f}")
