from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset

from evaluators.base import RepresentationBasedEvaluator
from losses.cka_map_loss import CKAMapLossCE, CKAMapLossDistill
from metrics.cka import CKA
from trainers.base import TrainerBase, TrainerConfig
from utilities.utils import AccumulateForLogging


@dataclass
class MapTrainingConfig(TrainerConfig):
    cka_alpha: float = 1.0
    cka_difference_function: str = "MSE"
    target_cka: np.ndarray = np.zeros(1)
    hard_labels: bool = True
    teacher_model: Optional[Module] = None
    distillation_temp: float = 2

    @property
    def criterion(self) -> Module:
        if self.hard_labels:
            return CKAMapLossCE(alpha=self.cka_alpha, mse=True if self.cka_difference_function == "MSE" else False)
        return CKAMapLossDistill(
            teacher=self.teacher_model,
            alpha=self.cka_alpha,
            mse=True if self.cka_difference_function == "MSE" else False,
            temp=self.distillation_temp,
        )


class CEMapTrainer(TrainerBase):
    def __init__(
        self,
        model: Module,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset],
        config: MapTrainingConfig = MapTrainingConfig(),
    ):
        super(CEMapTrainer, self).__init__(
            model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, config=config
        )
        self.config = config
        self.training_loss_overall = AccumulateForLogging(name="overall", accumulation=10)
        self.training_loss_ce = AccumulateForLogging(name="ce", accumulation=10)
        self.training_loss_cka = AccumulateForLogging(name="cka", accumulation=10)
        self.hparam_training.pop("target_cka")

    def compute_loss(self, **kwargs) -> Tuple[float, ...]:
        training_features = kwargs["training_features"]
        training_targets = kwargs["training_targets"]
        activations, outputs = self.model(training_features, intermediate_activations_required=True)
        # CE loss
        loss, ce_loss, map_loss = self.config.criterion(
            y_true=training_targets,
            y_prediction=outputs,
            model_activations=activations,
            target_map=self.config.target_cka,
        )
        loss.backward()
        return loss.item(), ce_loss.item(), map_loss.item()

    def log_training_loss(self, loss: Tuple[float, ...]):
        tracked_loss_overall = self.training_loss_overall(value=loss[0])
        tracked_loss_ce = self.training_loss_ce(value=loss[1])
        tracked_loss_cka = self.training_loss_cka(value=loss[2])
        if tracked_loss_overall is not None:
            self.log(
                metric_name="Loss/Training",
                metric_value={"overall": tracked_loss_overall, "ce": tracked_loss_ce, "cka": tracked_loss_cka},
            )

    def after_training(self):
        representation_evaluator = RepresentationBasedEvaluator(
            metrics=[CKA()], batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )
        representation_evaluator.record_representations_set_1(model=self.model, dataset=self.valid_dataset)
        cka_results = representation_evaluator.compute_metrics()["CKA"]
        self.log(metric_name="CKA_Map", metric_value=cka_results)


class DistillMapTrainer(TrainerBase):
    def __init__(
        self,
        model: Module,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset],
        config: MapTrainingConfig = MapTrainingConfig(),
    ):
        super(DistillMapTrainer, self).__init__(
            model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, config=config
        )
        self.config = config
        self.training_loss_overall = AccumulateForLogging(name="overall", accumulation=10)
        self.training_loss_ce = AccumulateForLogging(name="distillation", accumulation=10)
        self.training_loss_cka = AccumulateForLogging(name="cka", accumulation=10)
        self.hparam_training.pop("target_cka")

    def compute_loss(self, **kwargs) -> Tuple[float, ...]:
        training_features = kwargs["training_features"]
        activations, outputs = self.model(training_features, intermediate_activations_required=True)
        # with distillation loss
        loss, distill_loss, map_loss = self.config.criterion(
            features=training_features,
            logits=outputs,
            model_activations=activations,
            target_map=self.config.target_cka,
        )
        loss.backward()
        return loss.item(), distill_loss.item(), map_loss.item()

    def log_training_loss(self, loss: Tuple[float, ...]):
        tracked_loss_overall = self.training_loss_overall(value=loss[0])
        tracked_loss_ce = self.training_loss_ce(value=loss[1])
        tracked_loss_cka = self.training_loss_cka(value=loss[2])
        if tracked_loss_overall is not None:
            self.log(
                metric_name="Loss/Training",
                metric_value={
                    "overall": tracked_loss_overall,
                    "distillation": tracked_loss_ce,
                    "cka": tracked_loss_cka,
                },
            )

    def after_training(self):
        representation_evaluator = RepresentationBasedEvaluator(
            metrics=[CKA()], batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )
        representation_evaluator.record_representations_set_1(model=self.model, dataset=self.valid_dataset)
        cka_results = representation_evaluator.compute_metrics()["CKA"]
        self.log(metric_name="CKA_Map", metric_value=cka_results)
