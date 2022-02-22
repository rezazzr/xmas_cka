from typing import List

import torch

from losses.cka import TorchCKA


class CKAMap(torch.nn.Module):
    def __init__(self):
        super(CKAMap, self).__init__()

    def _make_map(self, activations: List[torch.Tensor]):
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

    def forward(self, activations):
        return self._make_map(activations)
