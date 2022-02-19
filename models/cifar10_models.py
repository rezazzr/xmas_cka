import torch
import torch.nn as nn


class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.layers = nn.ModuleList()

        self.layers += [
            nn.Sequential(nn.Conv2d(3, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(16, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(inplace=True)),
        ]
        self.layers += [
            nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        ]
        self.layers += [
            nn.Sequential(nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True)),
        ]
        self.layers += [
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        ]
        self.layers += [
            nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding="valid"), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        ]
        self.layers += [nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))]
        self.layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.fc = nn.Linear(64, 10)

    def forward(self, features: torch.Tensor, intermediate_activations_required: bool = False):
        activations = []
        for layer in self.layers:
            activations.append(features)
            features = layer(features)

        features = self.fc(features.view(-1, 64))

        if intermediate_activations_required:
            activations.pop(0)  # no need to have the original image
            return activations, features
        return features
