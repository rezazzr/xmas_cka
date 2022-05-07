from abc import ABC, abstractmethod
from typing import List, Union, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor


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
        self.representations_set_1: List[Tensor] = []
        self.representations_set_2: List[Tensor] = []

    def initialize_memory(self, representation_list: List[Tensor], is_set_1: bool = True):
        if is_set_1:
            self.representations_set_1 = representation_list
        else:
            self.representations_set_2 = representation_list

    def aggregate_batches(self, representation_list: List[Tensor], is_set_1: bool = True):
        if is_set_1:
            for i, tensor_val in enumerate(representation_list):
                self.representations_set_1[i] = torch.cat((self.representations_set_1[i], tensor_val), axis=0)
        else:
            for i, tensor_val in enumerate(representation_list):
                self.representations_set_2[i] = torch.cat((self.representations_set_2[i], tensor_val), axis=0)

    @abstractmethod
    def compute_metric(self) -> np.ndarray:
        pass


class BatchRepresentationBasedMetric(ABC):
    def initialize_metric(self, activations_1: List[Tensor], activations_2: Optional[List[Tensor]], **kwargs):
        pass

    @abstractmethod
    def eval_one_batch(self, activations_1: List[Tensor], activations_2: Optional[List[Tensor]]):
        pass

    @abstractmethod
    def compute_metric(self) -> Union[float, np.ndarray]:
        pass
