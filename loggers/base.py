from abc import abstractmethod, ABC
from typing import Any


class Loggers(ABC):
    @abstractmethod
    def __init__(
        self,
        log_dir: str,
        model_name: str,
        seed_value: int,
    ):
        self.log_dir = log_dir
        self.model_name = model_name
        self.seed_value = seed_value

    @abstractmethod
    def log_metric(self, metric_name: str, metric_value: Any, global_step: int):
        pass

    @abstractmethod
    def terminate(self):
        pass
