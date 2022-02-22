from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset


class PredictionBasedMetric(ABC):
    @abstractmethod
    def initialize_metric(self, dataset: Dataset, **kwargs):
        pass

    @abstractmethod
    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray):
        pass

    @abstractmethod
    def compute_metric(self) -> float:
        pass
