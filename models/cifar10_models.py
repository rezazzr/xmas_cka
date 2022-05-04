from torch import Tensor
import torch.nn as nn
import torch.nn.functional as nn_functions


class VGG(nn.Module):
    def __init__(self, width: int = 1):
        super(VGG, self).__init__()
        self.width = width
        self.layers = nn.ModuleList()

        self.layers += [
            nn.Sequential(
                nn.Conv2d(3, 16 * self.width, kernel_size=3),
                nn.BatchNorm2d(16 * self.width),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(16 * self.width, 16 * self.width, kernel_size=3),
                nn.BatchNorm2d(16 * self.width),
                nn.ReLU(inplace=True),
            ),
        ]
        self.layers += [
            nn.Sequential(
                nn.Conv2d(16 * self.width, 32 * self.width, kernel_size=3, stride=2),
                nn.BatchNorm2d(32 * self.width),
                nn.ReLU(inplace=True),
            )
        ]
        self.layers += [
            nn.Sequential(
                nn.Conv2d(32 * self.width, 32 * self.width, kernel_size=3),
                nn.BatchNorm2d(32 * self.width),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32 * self.width, 32 * self.width, kernel_size=3),
                nn.BatchNorm2d(32 * self.width),
                nn.ReLU(inplace=True),
            ),
        ]
        self.layers += [
            nn.Sequential(
                nn.Conv2d(32 * self.width, 64 * self.width, kernel_size=3, stride=2),
                nn.BatchNorm2d(64 * self.width),
                nn.ReLU(inplace=True),
            )
        ]
        self.layers += [
            nn.Sequential(
                nn.Conv2d(64 * self.width, 64 * self.width, kernel_size=3, padding="valid"),
                nn.BatchNorm2d(64 * self.width),
                nn.ReLU(inplace=True),
            )
        ]
        self.layers += [
            nn.Sequential(
                nn.Conv2d(64 * self.width, 64 * self.width, kernel_size=1),
                nn.BatchNorm2d(64 * self.width),
                nn.ReLU(inplace=True),
            )
        ]
        self.layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.fc = nn.Linear(64 * self.width, 10)

    def forward(self, features: Tensor, intermediate_activations_required: bool = False):
        activations = []
        for layer in self.layers:
            activations.append(features)
            features = layer(features)

        features = self.fc(features.view(-1, 64 * self.width))

        if intermediate_activations_required:
            activations.pop(0)  # no need to have the original image
            return activations, features
        return features


# This model has a kernel size of 7 for visualization purposes
class VGG7(nn.Module):
    def __init__(self):
        super(VGG7, self).__init__()
        self.layers = nn.ModuleList()

        self.layers += [
            nn.Sequential(nn.Conv2d(3, 16, kernel_size=7), nn.BatchNorm2d(16), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(16, 16, kernel_size=7), nn.BatchNorm2d(16), nn.ReLU(inplace=True)),
        ]
        self.layers += [
            nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        ]
        self.layers += [
            nn.Sequential(nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True)),
        ]
        self.layers += [
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        ]
        self.layers += [
            nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding="valid"), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        ]
        self.layers += [nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))]
        self.layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.fc = nn.Linear(64, 10)

    def forward(self, features: Tensor, intermediate_activations_required: bool = False):
        activations = []
        for layer in self.layers:
            activations.append(features)
            features = layer(features)

        features = self.fc(features.view(-1, 64))

        if intermediate_activations_required:
            activations.pop(0)  # no need to have the original image
            return activations, features
        return features


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, features: Tensor, intermediate_activations_required: bool = False):
        activations = []
        activation_append = activations.append
        out = nn_functions.relu(self.conv1(features))
        activation_append(out)
        out = nn_functions.max_pool2d(out, 2)
        out = nn_functions.relu(self.conv2(out))
        activation_append(out)
        out = nn_functions.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = nn_functions.relu(self.fc1(out))
        out = nn_functions.relu(self.fc2(out))
        out = self.fc3(out)
        if intermediate_activations_required:
            return activations, out
        return out
