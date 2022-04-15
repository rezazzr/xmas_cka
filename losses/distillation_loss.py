from torch import Tensor, log_softmax, softmax, inference_mode
from torch.nn import Module, KLDivLoss


class DistillationLoss(Module):
    def __init__(self, teacher: Module, temp: float = 2) -> None:
        """
        :param teacher: Teacher model to get the logits for distillation
        :type teacher: Module
        :param temp: The temperature of the softmax function, defaults to 2
        :type temp: float (optional)
        """
        super().__init__()
        self.temp = temp
        self.teacher = teacher

    def _distillation_loss(self, student_out: Tensor, teacher_out: Tensor) -> Tensor:
        log_p = log_softmax(student_out / self.temp, dim=1)
        q = softmax(teacher_out / self.temp, dim=1)
        result = KLDivLoss(reduction="batchmean")(log_p, q)
        return result

    def forward(self, features: Tensor, current_logits: Tensor) -> Tensor:
        with inference_mode():
            teacher_logits = self.teacher(features=features)

        dist_loss = self._distillation_loss(
            student_out=current_logits,
            teacher_out=teacher_logits.clone(),
        )
        return dist_loss
