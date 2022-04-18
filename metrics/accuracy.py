import numpy as np
from torch.utils.data import Dataset

from metrics.base import PredictionBasedMetric


class Accuracy(PredictionBasedMetric):
    def __init__(self) -> None:
        self.correct_predictions = 0.0
        self.total_predictions = 0.0

    def initialize_metric(self, dataset: Dataset, **kwargs) -> None:
        self.correct_predictions = 0.0
        self.total_predictions = 0.0

    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray) -> None:
        predictions = self._logits_to_predictions(logits=logits)
        self.correct_predictions += np.equal(targets, predictions).sum()
        self.total_predictions += len(predictions)

    def compute_metric(self) -> float:
        return (100.0 * self.correct_predictions) / self.total_predictions

    @staticmethod
    def _logits_to_predictions(logits: np.ndarray) -> np.ndarray:
        return np.argmax(logits, axis=1)
