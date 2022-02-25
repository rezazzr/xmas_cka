from typing import Dict, Sequence, List

import numpy as np
import torch
from torch.utils.data import Dataset

from metrics.base import PredictionBasedMetric, RepresentationBasedMetric
from utilities.utils import gpu_information_summary, to_numpy


class PredictionBasedEvaluator:
    def __init__(self, metrics: Sequence[PredictionBasedMetric], batch_size: int = 32, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        n_gpu, self.device = gpu_information_summary(show=False)
        self.metrics = metrics

    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        nb_classes: int = -1,
    ) -> Dict[str, float]:
        self.before_eval_one_task(dataset=dataset, nb_classes=nb_classes)
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, evaluation_targets = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                evaluation_targets = evaluation_targets.numpy()
                logits = to_numpy(model(evaluation_features))
                self.eval_one_batch(logits=logits, targets=evaluation_targets)

        return self.compute()

    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray) -> None:
        for metric in self.metrics:
            metric.eval_one_batch(logits=logits, targets=targets)

    def compute(self) -> Dict[str, float]:
        metric_evaluation = dict()
        for metric in self.metrics:
            metric_evaluation[type(metric).__name__] = metric.compute_metric()

        return metric_evaluation

    def before_eval_one_task(self, dataset: Dataset, nb_classes: int = -1) -> None:
        for metric in self.metrics:
            metric.initialize_metric(dataset=dataset, nb_classes=nb_classes, device=self.device)


class RepresentationBasedEvaluator:
    def __init__(self, metrics: Sequence[RepresentationBasedMetric], batch_size: int = 32, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        _, self.device = gpu_information_summary(show=False)
        self.metrics = metrics

    def record_representations_set_1(self, model: torch.nn.Module, dataset: Dataset) -> None:
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, _ = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                activations, _ = model(evaluation_features, intermediate_activations_required=True)
                if batch_number == 0:
                    self.initialize_memory(representation_list=activations, is_set_1=True)
                else:
                    self.aggregate_batches(representation_list=activations, is_set_1=True)

    def record_representations_set_2(self, model: torch.nn.Module, dataset: Dataset) -> None:
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, _ = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                activations, _ = model(evaluation_features, intermediate_activations_required=True)
                if batch_number == 0:
                    self.initialize_memory(representation_list=activations, is_set_1=False)
                else:
                    self.aggregate_batches(representation_list=activations, is_set_1=False)

    def compute_metrics(self) -> Dict[str, np.ndarray]:
        metric_evaluation = dict()
        for metric in self.metrics:
            metric_evaluation[type(metric).__name__] = metric.compute_metric()
        return metric_evaluation

    def initialize_memory(self, representation_list: List[torch.Tensor], is_set_1: bool = True):
        for metric in self.metrics:
            metric.initialize_memory(representation_list=representation_list, is_set_1=is_set_1)

    def aggregate_batches(self, representation_list: List[torch.Tensor], is_set_1: bool = True):
        for metric in self.metrics:
            metric.aggregate_batches(representation_list=representation_list, is_set_1=is_set_1)
