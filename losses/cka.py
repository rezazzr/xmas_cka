import math
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module


class TorchCKA(Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        identity = torch.eye(n, device=self.device)
        H = identity - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma: float):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        # get the median and place it to mdist, then multiply by the scaling factor sigma to get the final sigma
        mdist = torch.median(KX[KX != 0])
        mdist = math.sqrt(mdist)
        sigma = mdist * sigma
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma: float):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma: float):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

    def forward(self, X: Tensor, Y: Tensor, linear: bool = True):
        if linear:
            return self.linear_CKA(X=X, Y=Y)
        return self.kernel_CKA(X=X, Y=Y)


class BatchCKA(Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def _unbiased_HSIC(self, K: Tensor, L: Tensor) -> Tensor:
        n = K.shape[0]
        ones = torch.ones([n, 1], device=self.device)

        # set diagonal to 0
        K = K * (1 - torch.eye(n, n, device=self.device))
        L = L * (1 - torch.eye(n, n, device=self.device))

        trace = torch.trace(torch.matmul(K, L))

        one_t_k = torch.matmul(ones.T, K)
        l_one = torch.matmul(L, ones)

        numerator_1 = torch.matmul(one_t_k, ones)
        numerator_2 = torch.matmul(ones.T, l_one)
        denominator = (n - 1) * (n - 2)
        middle_argument = torch.matmul(numerator_1, numerator_2) / denominator

        multiplier_1 = 2 / (n - 2)
        multiplier_2 = torch.matmul(one_t_k, l_one)
        last_argument = multiplier_1 * multiplier_2

        unbiased_hsic = 1 / (n * (n - 3)) * (trace + middle_argument - last_argument)

        return unbiased_hsic

    def forward(
        self, X: Tensor, Y: Tensor, need_internals: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        gram_x = torch.matmul(X, X.T)
        gram_y = torch.matmul(Y, Y.T)
        numerator = self._unbiased_HSIC(gram_x, gram_y)
        denominator_1 = self._unbiased_HSIC(gram_x, gram_x)
        denominator_2 = self._unbiased_HSIC(gram_y, gram_y)
        if need_internals:
            return numerator, denominator_1, denominator_2
        cka = numerator / torch.sqrt(denominator_1 * denominator_2)
        return cka
