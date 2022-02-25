import numpy as np

from losses.cka_map import CKAMap
from metrics.base import RepresentationBasedMetric
from utilities.utils import to_numpy


class CKA(RepresentationBasedMetric):
    def __init__(self, comparison_mode: bool = False):
        super(CKA, self).__init__()
        self.comparison_mode = comparison_mode
        self.cka_map = CKAMap()

    def compute_metric(self) -> np.ndarray:
        if self.comparison_mode:
            return to_numpy(self.cka_map(self.representations_set_1, self.representations_set_2))
        return to_numpy(self.cka_map(self.representations_set_1))
