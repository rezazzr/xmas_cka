from typing import List, Optional

import torch

from losses.cka import TorchCKA


class CKAMap(torch.nn.Module):
    def __init__(self):
        super(CKAMap, self).__init__()

    def _make_self_map(self, activations: List[torch.Tensor]):
        device = activations[0].device
        batch = activations[0].shape[0]
        cka_map = torch.zeros(len(activations), len(activations), device=device)
        for i in range(len(activations)):
            for j in range(len(activations)):
                if j < i:
                    cka_map[i, j] = cka_map[j, i]
                else:
                    acts1 = activations[i].view(batch, -1)
                    acts2 = activations[j].view(batch, -1)
                    cka, _, _ = TorchCKA(device=device)(acts1, acts2)
                    cka_map[i, j] = cka
        return cka_map

    def _make_comparison_map(self, activations: List[torch.Tensor], activations_2: List[torch.Tensor]):
        device = activations[0].device
        batch = activations[0].shape[0]
        cka_map = torch.zeros(len(activations), device=device)
        for i in range(len(activations)):
            cka_map[i] = TorchCKA(device=device)(activations[i].view(batch, -1), activations_2[i].view(batch, -1))
        return cka_map

    def forward(self, activations: List[torch.Tensor], activations_2: Optional[List[torch.Tensor]] = None):
        if activations_2 is None:
            return self._make_self_map(activations)
        return self._make_comparison_map(activations, activations_2)
