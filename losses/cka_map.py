from typing import List, Optional

from torch import Tensor, zeros, tensor
from torch.nn import Module

from losses.cka import BatchCKA, TorchCKA


class CKAMap(Module):
    def __init__(self, evaluation_mode: bool = False, rbf_sigma: float = -1):
        super(CKAMap, self).__init__()
        self.rbf_sigma = rbf_sigma
        self.evaluation_mode = evaluation_mode

    def _make_map(self, activations_1: List[Tensor], activations_2: Optional[List[Tensor]] = None) -> Tensor:
        device = activations_1[0].device
        batch = activations_1[0].shape[0]
        cka_function = BatchCKA(device=device) if self.rbf_sigma < 0 else TorchCKA(device=device, sigma=self.rbf_sigma)
        if activations_2 is None:
            activations_2 = activations_1
        cka_map = zeros(len(activations_1), len(activations_2), device=device)
        for i in range(len(activations_1)):
            for j in range(len(activations_2)):
                if j < i:
                    cka_map[i, j] = cka_map[j, i]
                else:
                    acts1 = activations_1[i].view(batch, -1)
                    acts2 = activations_2[j].view(batch, -1)
                    cka = cka_function(acts1, acts2)
                    cka_map[i, j] = cka
        return cka_map

    def _cka_components(self, activations_1: List[Tensor], activations_2: Optional[List[Tensor]] = None) -> Tensor:
        device = activations_1[0].device
        batch = activations_1[0].shape[0]
        if activations_2 is None:
            activations_2 = activations_1
        component_matrix = zeros(
            [len(activations_1), len(activations_2), 3], device=device
        )  # 3 for 3 components of cka
        for i in range(len(activations_1)):
            for j in range(len(activations_2)):
                if j < i:
                    component_matrix[i, j, :] = component_matrix[j, i, :]
                else:
                    acts1 = activations_1[i].view(batch, -1)
                    acts2 = activations_2[j].view(batch, -1)
                    component_matrix[i, j, :] = tensor(BatchCKA(device=device)(acts1, acts2, need_internals=True))
        return component_matrix

    def forward(self, activations: List[Tensor], activations_2: Optional[List[Tensor]] = None):
        if self.evaluation_mode:
            return self._cka_components(activations, activations_2)
        return self._make_map(activations, activations_2)
