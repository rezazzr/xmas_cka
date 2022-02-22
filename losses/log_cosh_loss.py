import torch


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_predicted):
        diff = y_true - y_predicted
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))
