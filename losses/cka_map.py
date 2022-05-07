from typing import List, Optional

from torch import Tensor, zeros
from torch.nn import Module

from losses.cka import TorchCKA


class CKAMap(Module):
    def __init__(self):
        super(CKAMap, self).__init__()

    def _make_map(self, activations_1: List[Tensor], activations_2: Optional[List[Tensor]] = None) -> Tensor:
        device = activations_1[0].device
        batch = activations_1[0].shape[0]
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
                    cka = TorchCKA(device=device)(acts1, acts2)
                    cka_map[i, j] = cka
        return cka_map

    def forward(self, activations: List[Tensor], activations_2: Optional[List[Tensor]] = None):
        return self._make_map(activations, activations_2)
