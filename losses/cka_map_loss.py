from typing import List

import numpy as np
from torch import Tensor, from_numpy
from torch.nn.modules import CrossEntropyLoss, MSELoss
from torch.nn import Module
from losses.cka_map import CKAMap
from losses.distillation_loss import DistillationLoss
from losses.log_cosh_loss import LogCoshLoss


class CKAMapLossCE(Module):
    def __init__(self, alpha: float = 1.0, mse: bool = True):
        super(CKAMapLossCE, self).__init__()
        self.alpha = alpha
        self.mse = mse

    def forward(
        self,
        y_true: Tensor,
        y_prediction: Tensor,
        model_activations: List[Tensor],
        target_map: np.ndarray,
    ):
        cross_entropy_loss = CrossEntropyLoss()(input=y_prediction, target=y_true)
        model_cka_map = CKAMap()(activations=model_activations)
        map_loss = self.map_difference(model_map=model_cka_map, target_map=target_map)
        return cross_entropy_loss + self.alpha * map_loss, cross_entropy_loss, map_loss

    def map_difference(self, model_map: Tensor, target_map: np.ndarray):
        device = model_map.device
        target_map = from_numpy(target_map).to(device).float()
        if self.mse:
            mse_loss = MSELoss()
            return mse_loss(model_map, target_map)
        log_cosh_loss = LogCoshLoss()
        return log_cosh_loss(model_map, target_map)


class CKAMapLossDistill(Module):
    def __init__(self, teacher: Module, alpha: float = 1.0, mse: bool = True, temp: float = 2.0):
        super(CKAMapLossDistill, self).__init__()
        self.alpha = alpha
        self.mse = mse
        self.distillation_loss = DistillationLoss(teacher=teacher, temp=temp)

    def forward(
        self,
        features: Tensor,
        logits: Tensor,
        model_activations: List[Tensor],
        target_map: np.ndarray,
    ):
        dist_loss = self.distillation_loss(features=features, current_logits=logits)
        model_cka_map = CKAMap()(activations=model_activations)
        map_loss = self.map_difference(model_map=model_cka_map, target_map=target_map)
        return dist_loss + self.alpha * map_loss, dist_loss, map_loss

    def map_difference(self, model_map: Tensor, target_map: np.ndarray):
        device = model_map.device
        target_map = from_numpy(target_map).to(device).float()
        if self.mse:
            mse_loss = MSELoss()
            return mse_loss(model_map, target_map)
        log_cosh_loss = LogCoshLoss()
        return log_cosh_loss(model_map, target_map)
