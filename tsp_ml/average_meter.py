# -*- coding: utf-8 -*-
class AverageMeter(object):
    """Computes and stores the average"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val
        self.count += n

    @property
    def average(self) -> float:
        avg = self.sum / self.count
        return avg
