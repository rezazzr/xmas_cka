from typing import List

import numpy as np
import torch
from torch.nn.modules import CrossEntropyLoss, MSELoss

from losses.cka_map import CKAMap
from losses.log_cosh_loss import LogCoshLoss


class CKAMapLossCE(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, mse: bool = True):
        super(CKAMapLossCE, self).__init__()
        self.alpha = alpha
        self.mse = mse

    def forward(
        self,
        y_true: torch.Tensor,
        y_prediction: torch.Tensor,
        model_activations: List[torch.Tensor],
        target_map: np.ndarray,
    ):
        cross_entropy_loss = CrossEntropyLoss()(input=y_prediction, target=y_true)
        model_cka_map = CKAMap()(activations=model_activations)
        map_loss = self.map_difference(model_map=model_cka_map, target_map=target_map)
        return cross_entropy_loss + self.alpha * map_loss, cross_entropy_loss, map_loss

    def map_difference(self, model_map: torch.Tensor, target_map: np.ndarray):
        device = model_map.device
        target_map = torch.from_numpy(target_map).to(device).float()
        if self.mse:
            mse_loss = MSELoss()
            return mse_loss(model_map, target_map)
        log_cosh_loss = LogCoshLoss()
        return log_cosh_loss(model_map, target_map)
