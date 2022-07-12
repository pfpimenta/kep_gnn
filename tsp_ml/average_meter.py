class AverageMeter(object):
    """Computes and stores the average"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: float) -> None:
        self.sum += val
        self.count += 1

    @property
    def average(self) -> float:
        avg = self.sum / self.count
        return avg
