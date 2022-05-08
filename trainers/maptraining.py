from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset

from evaluators.base import BatchRepresentationBasedEvaluator
from losses.cka_map_loss import CKAMapLossCE, CKAMapLossDistill
from metrics.cka import BatchCKA
from trainers.base import TrainerBase, TrainerConfig
from utilities.utils import AccumulateForLogging, MultiplicativeScalingFactorScheduler, register_all_layers


@dataclass
class MapTrainingConfig(TrainerConfig):
    cka_alpha: float = 1.0
    cka_difference_function: str = "MSE"
    target_cka: np.ndarray = np.zeros(1)
    hard_labels: bool = True
    teacher_model: Optional[Module] = None
    distillation_temp: float = 2
    upper_bound_acc: Optional[float] = None
    acc_tolerance: float = 1.0
    reduction_factor: float = 0.5

    def __post_init__(self):
        self.dynamic_scheduler: Optional[MultiplicativeScalingFactorScheduler] = (
            MultiplicativeScalingFactorScheduler(
                initial_value=self.cka_alpha,
                multiplier=self.reduction_factor,
                original_metric_value=self.upper_bound_acc,
                tolerance=self.acc_tolerance,
            )
            if self.upper_bound_acc is not None
            else None
        )
        if self.hard_labels:
            self.criterion = CKAMapLossCE(
                alpha=self.cka_alpha,
                mse=True if self.cka_difference_function == "MSE" else False,
                dynamic_scheduler=self.dynamic_scheduler,
            )
        else:
            self.criterion = CKAMapLossDistill(
                teacher=self.teacher_model,
                alpha=self.cka_alpha,
                mse=True if self.cka_difference_function == "MSE" else False,
                temp=self.distillation_temp,
                dynamic_scheduler=self.dynamic_scheduler,
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
        self.activations = []

        def hook_fn(m, i, o):
            self.activations.append(o)

        self.handles = register_all_layers(self.model, hook_fn)

    def compute_loss(self, **kwargs) -> Tuple[float, ...]:
        training_features = kwargs["training_features"]
        training_targets = kwargs["training_targets"]
        outputs = self.model(training_features)
        # CE loss
        loss, ce_loss, map_loss = self.config.criterion(
            y_true=training_targets,
            y_prediction=outputs,
            model_activations=self.activations,
            target_map=self.config.target_cka,
        )
        loss.backward()
        self.activations = []
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
        if len(self.handles) > 0:
            for h in self.handles:
                h.remove()
        representation_evaluator = BatchRepresentationBasedEvaluator(
            metrics=[BatchCKA()], batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )

        cka_results = representation_evaluator.evaluate(model_1=self.model, dataset=self.valid_dataset)["BatchCKA"]
        self.log(metric_name="CKA_Map", metric_value=cka_results)

    def accuracy_got_updated_with(self, accuracy_value: float):
        if self.config.dynamic_scheduler is not None:
            current_multiplier_value = self.config.dynamic_scheduler(metric_value=accuracy_value)
            self.log("cka_multiplier", current_multiplier_value)


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
        self.activations = []

        def hook_fn(m, i, o):
            self.activations.append(o)

        self.handles = register_all_layers(self.model, hook_fn)

    def compute_loss(self, **kwargs) -> Tuple[float, ...]:
        training_features = kwargs["training_features"]
        outputs = self.model(training_features)
        # with distillation loss
        loss, distill_loss, map_loss = self.config.criterion(
            features=training_features,
            logits=outputs,
            model_activations=self.activations,
            target_map=self.config.target_cka,
        )
        loss.backward()
        self.activations = []
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
        if len(self.handles) > 0:
            for h in self.handles:
                h.remove()
        representation_evaluator = BatchRepresentationBasedEvaluator(
            metrics=[BatchCKA()], batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )

        cka_results = representation_evaluator.evaluate(model_1=self.model, dataset=self.valid_dataset)["BatchCKA"]
        self.log(metric_name="CKA_Map", metric_value=cka_results)

    def accuracy_got_updated_with(self, accuracy_value: float):
        if self.config.dynamic_scheduler is not None:
            current_multiplier_value = self.config.dynamic_scheduler(metric_value=accuracy_value)
            self.log("cka_multiplier", current_multiplier_value)
