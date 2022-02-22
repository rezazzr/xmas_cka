import numpy as np
from torch.utils.data import Dataset

from metrics.base import PredictionBasedMetric


class Accuracy(PredictionBasedMetric):
    def __init__(self):
        self.correct_predictions = dict()
        self.total_predictions = dict()

    def initialize_metric(self, dataset: Dataset, **kwargs):
        self.correct_predictions = {i: 0 for i in range(kwargs["nb_classes"])}
        self.total_predictions = {i: 0 for i in range(kwargs["nb_classes"])}

    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray):
        predictions = self._logits_to_predictions(logits=logits)
        for target, prediction in zip(targets, predictions):
            if target == prediction:
                self.correct_predictions[target] += 1
            self.total_predictions[target] += 1

    def compute_metric(self) -> float:
        return (100.0 * sum(self.correct_predictions.values())) / sum(self.total_predictions.values())

    @staticmethod
    def _logits_to_predictions(logits: np.ndarray) -> np.ndarray:
        return np.argmax(logits, axis=1)
