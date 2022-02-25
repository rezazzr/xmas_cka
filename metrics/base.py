from abc import ABC, abstractmethod
from typing import List

import torch
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


class RepresentationBasedMetric(ABC):
    def __init__(self):
        self.representations_set_1: List[torch.Tensor] = []
        self.representations_set_2: List[torch.Tensor] = []

    def initialize_memory(self, representation_list: List[torch.Tensor], is_set_1: bool = True):
        if is_set_1:
            self.representations_set_1 = representation_list
        else:
            self.representations_set_2 = representation_list

    def aggregate_batches(self, representation_list: List[torch.Tensor], is_set_1: bool = True):
        if is_set_1:
            for i, tensor_val in enumerate(representation_list):
                self.representations_set_1[i] = torch.cat((self.representations_set_1[i], tensor_val), axis=0)
        else:
            for i, tensor_val in enumerate(representation_list):
                self.representations_set_2[i] = torch.cat((self.representations_set_2[i], tensor_val), axis=0)

    @abstractmethod
    def compute_metric(self) -> np.ndarray:
        pass
