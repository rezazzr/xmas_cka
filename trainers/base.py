import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from typing import Optional, List, Any, Tuple, Union, Dict

import torch.nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from evaluators.base import PredictionBasedEvaluator
from loggers.base import Loggers
from loggers.display import IOLogger
from loggers.tensorboard import TensorboardLogger
from utilities.utils import (
    gpu_information_summary,
    cosine_with_hard_restarts_schedule_with_warmup,
    CheckPointManager,
    CheckPointingConfig,
)
import time
from torch.nn import Module


@dataclass
class TrainerConfig:
    prediction_evaluator: Optional[PredictionBasedEvaluator] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    seed_value: int = 1609
    nb_epochs: int = 2
    num_workers: int = 0
    batch_size: int = 32
    logging_step: int = 1
    loggers: Optional[List[Loggers]] = None
    log_dir: Optional[str] = None
    verbose: bool = True
    save_progress: bool = False
    saving_dir: Optional[str] = "model_zoo"
    use_scheduler: bool = True
    nb_warmup_steps: int = -1
    learning_rate: float = 1e-5
    experiment_name: Optional[str] = None
    nb_classes: int = 1
    max_grad_norm: float = 1.0
    progress_history: int = 1

    @property
    def criterion(self) -> Module:
        return torch.nn.CrossEntropyLoss()


class TrainerBase(ABC):
    @abstractmethod
    def __init__(
        self, model: torch.nn.Module, train_dataset: Dataset, valid_dataset: Optional[Dataset], config: TrainerConfig
    ):
        self.config = config
        self.valid_dataset = valid_dataset
        self.train_dataset = train_dataset
        self.model = model

        self.hparam_training = asdict(self.config)
        keys_to_remove = ["prediction_evaluator", "criterion", "optimizer", "loggers"]
        for key in keys_to_remove:
            self.hparam_training.pop(key)

        if self.config.save_progress:
            if not os.path.exists(self.config.saving_dir):
                os.makedirs(self.config.saving_dir)

        self.n_gpu, self.device = gpu_information_summary(show=self.config.verbose)
        self.model.to(self.device)

        if self.config.optimizer is None:
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
            self.config.optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=5e-4,
                amsgrad=True,
            )

        self.global_step = 0
        if self.config.loggers is None:
            self.init_loggers()

        self.scheduler = None
        self.hparam_metrics = None
        self.checkpoint = CheckPointManager(
            config=CheckPointingConfig(
                model_name=type(self.model).__name__,
                history=self.config.progress_history,
                verbose=True,
                directory=self.config.saving_dir,
            )
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        self.nb_training_steps = len(self.train_loader) * self.config.nb_epochs

        if self.config.use_scheduler:
            self.scheduler = cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.config.optimizer,
                num_warmup_steps=self.config.nb_warmup_steps,
                num_training_steps=self.nb_training_steps - self.config.nb_warmup_steps,
                num_cycles=1,
            )

    def train(self):
        for epoch in tqdm(range(self.config.nb_epochs), desc="Epoch progress"):
            self.train_epoch(train_loader=self.train_loader)
            if self.config.save_progress:
                self.checkpoint(model=self.model, step=epoch, optimizer=self.config.optimizer)

        if self.hparam_metrics is not None:
            hparam_logging = (self.hparam_training, self.hparam_metrics)
            self.log(metric_name="hparams", metric_value=hparam_logging)
        self.after_training()
        self.terminate_logging()

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> None:
        # for each epoch call train_iter
        for batch_number, training_instance in enumerate(train_loader):
            loss = self.train_iter(training_instance=training_instance)
            self.log_training_loss(loss)

            if self.global_step % 10 == 9:
                self.log(metric_name="LR", metric_value=self.config.optimizer.param_groups[0]["lr"])

            if (
                self.config.logging_step > 0
                and self.global_step % self.config.logging_step == (self.config.logging_step - 1)
            ) and self.config.prediction_evaluator is not None:
                start_time_eval = time.time()
                evaluator_results = self.config.prediction_evaluator.evaluate(
                    model=self.model, dataset=self.valid_dataset, nb_classes=self.config.nb_classes
                )
                self.hparam_metrics = dict()
                for metric_name, metric_value in evaluator_results.items():
                    self.hparam_metrics[f"hparams/{metric_name}"] = metric_value
                    metric_name = f"{metric_name}/Evaluation"
                    self.log(metric_name=metric_name, metric_value=metric_value)
                end_time_eval = time.time()
                print(f"Evaluation took: {end_time_eval - start_time_eval}s.")

    def train_iter(self, training_instance: Tuple[torch.Tensor, torch.Tensor]) -> Union[float, Tuple[float, ...]]:
        # for each iteration of training call this function
        # get the inputs; data is a list of [inputs, labels]
        self.model.train()
        training_features, training_targets = tuple([tensor.to(self.device) for tensor in training_instance])

        # zero the parameter gradients
        self.config.optimizer.zero_grad()

        loss = self.compute_loss(training_features=training_features, training_targets=training_targets)
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.config.optimizer.step()
        if self.config.use_scheduler:
            self.scheduler.step()
        self.global_step += 1

        return loss

    @abstractmethod
    def compute_loss(self, **kwargs) -> Union[float, Tuple[float, ...]]:
        pass

    @abstractmethod
    def log_training_loss(self, loss: Union[float, Tuple[float, ...]]):
        pass

    @abstractmethod
    def after_training(self):
        pass

    def terminate_logging(self):
        for logger in self.config.loggers:
            logger.terminate()

    def log(self, metric_name: str, metric_value: Any, global_step: Optional[int] = None):
        if global_step is None:
            global_step = self.global_step
        for logger in self.config.loggers:
            logger.log_metric(metric_name=metric_name, metric_value=metric_value, global_step=global_step)

    def init_loggers(self):
        # this is for the case of linear probes since they have intended block and we want to avoid over writing
        experiment_name = f"/{self.config.experiment_name}" if self.config.experiment_name is not None else ""
        log_dir = self.config.log_dir if self.config.log_dir is not None else "tb_logs"
        tb_logger = TensorboardLogger(
            log_dir=f"{log_dir}" + experiment_name,
            model_name=f"{type(self.model).__name__}",
            seed_value=self.config.seed_value,
        )
        io_logger = IOLogger(
            log_dir="",
            model_name=f"{type(self.model).__name__}",
            seed_value=self.config.seed_value,
        )

        self.config.loggers = [io_logger, tb_logger]
