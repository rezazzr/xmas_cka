import torch.nn
from metrics.base import PredictionBasedMetric
from torch.utils.data import Dataset
import numpy as np


class Loss(PredictionBasedMetric):
    def __init__(self, criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        self.device = "cpu"
        self.criterion = criterion
        self.summed_loss = 0.0
        self.batches_processed = 0

    def initialize_metric(self, dataset: Dataset, **kwargs):
        self.summed_loss = 0.0
        self.batches_processed = 0
        self.device = kwargs.get("device", "cpu")

    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray):
        loss = self.criterion(torch.from_numpy(logits).to(self.device), torch.from_numpy(targets).to(self.device))
        self.summed_loss += loss.item()
        self.batches_processed += 1

    def compute_metric(self) -> float:
        return self.summed_loss / self.batches_processed
