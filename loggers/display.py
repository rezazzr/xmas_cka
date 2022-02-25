import pprint
from typing import Any

import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from loggers.base import Loggers


class IOLogger(Loggers):
    def __init__(
        self,
        log_dir: str,
        model_name: str,
        seed_value: int,
    ):
        super(IOLogger, self).__init__(log_dir, model_name, seed_value)

    def log_metric(self, metric_name: str, metric_value: Any, global_step: int):
        print("~" * 35)
        print(f"  Model:{self.model_name}, seed:{self.seed_value}, {metric_name} @ step: {global_step}")
        if isinstance(metric_value, dict):
            pprint.pprint(metric_value, indent=2)
        elif isinstance(metric_value, float):
            print(metric_value)
        elif isinstance(metric_value, tuple):
            table = PrettyTable(["KEY", "VALUE"])
            for key, val in metric_value[1].items():
                table.add_row([key, val])
            for key, val in metric_value[0].items():
                table.add_row([key, val])
            print(table)

        elif isinstance(metric_value, np.ndarray):
            plt.figure()
            plt.imshow(metric_value, origin="upper")
            plt.colorbar()
            plt.title(metric_name)
            plt.show()
        else:
            raise TypeError(f"metric_value is of type: {type(metric_value).__name__} which is not supported")
        print("~" * 35)

    def terminate(self):
        print("-" * 35)
        print(f"  End of Training.")
        print("-" * 35)
