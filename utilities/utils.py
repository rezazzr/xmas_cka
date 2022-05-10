import math
import os
import random
import typing
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable
from torch.nn import Module, Sequential, ModuleList, BatchNorm2d, AdaptiveAvgPool2d
from torch.optim.lr_scheduler import LambdaLR

from models.cifar_10_models.resnet import BasicBlock


def gpu_information_summary(show: bool = True) -> Tuple[int, torch.device]:
    """
    :param show: Controls whether or not to print the summary information
    :return: number of gpus and the device (CPU or GPU)
    """
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name() if n_gpu > 0 else "None"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = PrettyTable()
    table.field_names = ["Key", "Value"]
    table.add_row(["GPU", gpu_name])
    table.add_row(["Number of GPUs", n_gpu])
    if show:
        print(table)
    return n_gpu, device


def set_seed(seed_value: int, n_gpu: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_value)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def xavier_uniform_initialize(layer: Module):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)
    if type(layer) == nn.Conv2d:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)


def cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@dataclass
class CheckPointingConfig:
    model_name: str = "NN"
    history: int = 1
    verbose: bool = True
    directory: str = "progress_checkpoints"

    @property
    def address(self) -> str:
        return os.path.join(self.directory, self.model_name)


class CheckPointManager:
    def __init__(self, config: CheckPointingConfig = CheckPointingConfig(), trace_func=print):
        self.config = config
        self.saved_history = []
        self.trace_func = trace_func
        if not os.path.exists(config.address):
            os.makedirs(config.address)

    def __call__(self, model, step, optimizer):
        path = os.path.join(self.config.address, f"checkpoint_{step}.pt")

        if self.config.verbose:
            self.trace_func(f"Saving model at\n {path}")

        if len(self.saved_history) >= self.config.history:
            os.remove(self.saved_history.pop(0))

        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"{path}",
        )
        self.saved_history.append(path)


def safely_load_state_dict(checkpoint_path: str) -> typing.OrderedDict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path)["model_state_dict"]
    final_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            final_state_dict[key[7:]] = value
        else:
            final_state_dict[key] = value
    return final_state_dict


class AccumulateForLogging:
    def __init__(self, name: str, accumulation: int):
        self.name = name
        self.accumulation = accumulation
        self.value = 0.0
        self.accumulated = 0

    def __call__(self, value: float) -> typing.Optional[float]:
        self.value += value
        self.accumulated += 1
        if self.accumulated == self.accumulation:
            self.accumulated = 0
            normalized_value = self.value / self.accumulation
            self.value = 0.0
            return normalized_value
        return None


class MultiplicativeScalingFactorScheduler:
    """
    The idea here is that if a metric we are tracking falls bellow a certain threshold of tolerance, then
    we would want to scale the current_value parameter, which is usually used as a multiplier to enforce some kind of
    importance.
    """

    def __init__(self, initial_value: float, multiplier: float, original_metric_value: float, tolerance: float):
        self.original_metric_value = original_metric_value
        self.tolerance = tolerance
        self.multiplier = multiplier
        self.initial_value = initial_value
        self.current_value = initial_value

    def __call__(self, metric_value: float):
        difference = self.original_metric_value - metric_value
        if difference > self.tolerance:
            self.current_value *= self.multiplier
        else:
            self.current_value /= self.multiplier
        return self.current_value


def register_all_layers(model: Module, hook_fn, handles=None):
    if handles is None:
        handles = []
    for name, layer in model.named_children():
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        if isinstance(layer, Sequential):
            register_all_layers(layer, hook_fn, handles)
        elif isinstance(layer, ModuleList):
            register_all_layers(layer, hook_fn, handles)
        elif isinstance(layer, BasicBlock):
            register_all_layers(layer, hook_fn, handles)
        else:
            if not isinstance(layer, BatchNorm2d) and not isinstance(layer, AdaptiveAvgPool2d):
                handle = layer.register_forward_hook(hook_fn)
                handles.append(handle)
    return handles
