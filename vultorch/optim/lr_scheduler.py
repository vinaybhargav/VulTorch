"""
Learning Rate Schedulers (vultorch.optim.lr_scheduler)
"""

import math
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Optimizer

class LRScheduler:
    def __init__(self, optimizer: Any, last_epoch: int = -1, verbose: bool = False):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose

        # Save base learning rates
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]

        self.step()

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_lr(self) -> List[float]:
        raise NotImplementedError

    def step(self, epoch: int = None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class MultiStepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1):
        self.milestones = set(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class LinearLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, start_factor: float = 1.0 / 3, end_factor: float = 1.0, total_iters: int = 5, last_epoch: int = -1):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        if self.last_epoch > self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        ratio = self.last_epoch / self.total_iters
        factor = self.start_factor + (self.end_factor - self.start_factor) * ratio
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = max(1, T_max)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        return [
            self.eta_min + (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2.0
            for base_lr in self.base_lrs
        ]


class OneCycleLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, max_lr: float, total_steps: int, pct_start: float = 0.3, div_factor: float = 25.0, final_div_factor: float = 1e4, last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = max(1, total_steps)
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.step_up = int(self.total_steps * pct_start)
        self.step_down = self.total_steps - self.step_up
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step <= self.step_up:
            pct = step / max(1, self.step_up)
            lr = self.initial_lr + pct * (self.max_lr - self.initial_lr)
        else:
            pct = (step - self.step_up) / max(1, self.step_down)
            cos_out = math.cos(math.pi * pct) + 1.0
            lr = self.min_lr + (self.max_lr - self.min_lr) * cos_out / 2.0
        return [lr for _ in self.base_lrs]
