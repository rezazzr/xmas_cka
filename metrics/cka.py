from typing import List, Optional

import numpy as np

from losses.cka_map import CKAMap
from metrics.base import RepresentationBasedMetric, BatchRepresentationBasedMetric
from utilities.utils import to_numpy
from torch import Tensor


class CKA(RepresentationBasedMetric):
    def __init__(self, comparison_mode: bool = False):
        super(CKA, self).__init__()
        self.comparison_mode = comparison_mode
        self.cka_map = CKAMap()

    def compute_metric(self) -> np.ndarray:
        if self.comparison_mode:
            return to_numpy(self.cka_map(self.representations_set_1, self.representations_set_2))
        return to_numpy(self.cka_map(self.representations_set_1))


class BatchCKA(BatchRepresentationBasedMetric):
    def __init__(self) -> None:
        self.cka_map = None
        self.cka_map_fn = CKAMap(evaluation_mode=True)
        self.processed_batches = 0.0

    def initialize_metric(self, activations_1: List[Tensor], activations_2: Optional[List[Tensor]], **kwargs) -> None:
        self.cka_map = self.cka_map_fn(activations=activations_1, activations_2=activations_2)
        self.processed_batches += 1

    def eval_one_batch(self, activations_1: List[Tensor], activations_2: Optional[List[Tensor]]) -> None:
        self.cka_map += self.cka_map_fn(activations=activations_1, activations_2=activations_2)
        self.processed_batches += 1

    def compute_metric(self) -> np.ndarray:
        self.cka_map /= self.processed_batches
        self.cka_map = to_numpy(self.cka_map)
        final_map = np.zeros([self.cka_map.shape[0], self.cka_map.shape[1]])
        for i in range(final_map.shape[0]):
            for j in range(final_map.shape[1]):
                if j < i:
                    final_map[i, j] = final_map[j, i]
                else:
                    final_map[i, j] = self.cka_map[i, j, 0] / np.sqrt(self.cka_map[i, j, 1] * self.cka_map[i, j, 2])
        return final_map
