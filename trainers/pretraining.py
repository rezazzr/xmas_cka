from typing import Optional

import torch
from torch.utils.data import Dataset

from evaluators.base import BatchRepresentationBasedEvaluator
from metrics.cka import BatchCKA
from trainers.base import TrainerBase, TrainerConfig
from utilities.utils import AccumulateForLogging


class Trainer(TrainerBase):
    def __init__(
        self, model: torch.nn.Module, train_dataset: Dataset, valid_dataset: Optional[Dataset], config: TrainerConfig
    ):
        super(Trainer, self).__init__(
            model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, config=config
        )
        self.training_loss_tracker = AccumulateForLogging(name="Loss/Training", accumulation=10)

    def compute_loss(self, **kwargs) -> float:
        training_features = kwargs["training_features"]
        training_targets = kwargs["training_targets"]
        outputs = self.model(training_features)
        # CE loss
        loss = self.config.criterion(outputs, training_targets)
        loss.backward()
        return loss.item()

    def log_training_loss(self, loss: float):
        tracked_loss = self.training_loss_tracker(value=loss)
        if tracked_loss is not None:
            self.log(metric_name=self.training_loss_tracker.name, metric_value=tracked_loss)

    def after_training(self):
        representation_evaluator = BatchRepresentationBasedEvaluator(
            metrics=[BatchCKA()], batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )

        cka_results = representation_evaluator.evaluate(model_1=self.model, dataset=self.valid_dataset)["BatchCKA"]
        self.log(metric_name="CKA_Map", metric_value=cka_results)
