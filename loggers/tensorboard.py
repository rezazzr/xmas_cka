import datetime
import os
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from loggers.base import Loggers


class TensorboardLogger(Loggers):
    def __init__(
        self,
        log_dir: str,
        model_name: str,
        seed_value: int,
    ):
        super(TensorboardLogger, self).__init__(log_dir, model_name, seed_value)
        experiment_path = os.path.join(
            log_dir, model_name, f"seed_{seed_value}", str(datetime.datetime.now()).replace(" ", "_")
        )
        self.writer = SummaryWriter(experiment_path)

    def log_metric(self, metric_name: str, metric_value: Any, global_step: int):
        if isinstance(metric_value, dict):
            self.writer.add_scalars(metric_name, metric_value, global_step)
        elif isinstance(metric_value, float):
            self.writer.add_scalar(metric_name, metric_value, global_step)
        elif isinstance(metric_value, tuple):
            self.writer.add_hparams(hparam_dict=metric_value[0], metric_dict=metric_value[1])
        elif isinstance(metric_value, np.ndarray):
            self.writer.add_image(tag=metric_name, img_tensor=metric_value, global_step=global_step, dataformats="HW")
        else:
            raise TypeError(f"metric_value is of type: {type(metric_value).__name__} which is not supported")

    def terminate(self):
        self.writer.close()
