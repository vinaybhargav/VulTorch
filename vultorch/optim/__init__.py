"""
VulTorch Optimizers (vultorch.optim)
"""

import math
from typing import List, Dict, Any, Iterable
import vultorch
from ..tensor import Tensor
from ..autograd import no_grad
from . import lr_scheduler

class Optimizer:
    def __init__(self, params: Iterable[Tensor], defaults: Dict[str, Any]):
        self.defaults = defaults
        self.param_groups: List[Dict[str, Any]] = []

        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("optimizer got an empty parameter list")

        self.param_groups.append({"params": param_list, **defaults})
        self.state: Dict[Tensor, Dict[str, Any]] = {}

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None

    def state_dict(self) -> Dict[str, Any]:
        param_mappings = {}
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                param_mappings[p] = (i, j)

        state_out = {}
        for p, s in self.state.items():
            if p in param_mappings:
                state_out[param_mappings[p]] = {
                    k: v.clone() if isinstance(v, Tensor) else v
                    for k, v in s.items()
                }

        groups_out = []
        for group in self.param_groups:
            g = {k: v for k, v in group.items() if k != "params"}
            groups_out.append(g)

        return {"state": state_out, "param_groups": groups_out}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for group, g_saved in zip(self.param_groups, state_dict.get("param_groups", [])):
            group.update(g_saved)

        param_lookup = {}
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                param_lookup[(i, j)] = p

        for key, s in state_dict.get("state", {}).items():
            k_tuple = tuple(key) if isinstance(key, (list, tuple)) else key
            if k_tuple in param_lookup:
                p = param_lookup[k_tuple]
                self.state[p] = {
                    k: v.clone() if isinstance(v, Tensor) else v
                    for k, v in s.items()
                }

    def step(self):
        # Parameter updates are not differentiable and must not extend the graph.
        with no_grad():
            self._step_impl()

    def _step_impl(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3,
                 momentum: float = 0.0, dampening: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False,
                 fused: bool = True):
        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov
        )
        super().__init__(params, defaults)
        self.fused = fused

    def _step_impl(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group.get("dampening", 0.0)
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if self.fused and p._native.is_contiguous() and p.grad._native.is_contiguous():
                    param_state = self.state.setdefault(p, {})
                    first_step = False
                    mom_native = None
                    if momentum != 0.0:
                        if "momentum_buffer" not in param_state:
                            param_state["momentum_buffer"] = vultorch.empty(p.shape, device=p.device)
                            first_step = True
                        mom_native = param_state["momentum_buffer"]._native

                    p._native.sgd_step_(
                        p.grad._native, mom_native,
                        float(lr), float(momentum), float(dampening),
                        float(weight_decay), bool(nesterov), first_step
                    )
                    continue

                d_p = p.grad
                if weight_decay != 0.0:
                    d_p = d_p + p * weight_decay

                if momentum != 0.0:
                    param_state = self.state.setdefault(p, {})
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = d_p.clone()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf = buf * momentum + d_p * (1.0 - dampening)
                        param_state["momentum_buffer"] = buf

                    if nesterov:
                        d_p = d_p + buf * momentum
                    else:
                        d_p = buf

                # In-place parameter update: p = p - lr * d_p
                p.copy_(p - d_p * lr)


def _fused_adam_update(p, state, group, decoupled):
    """One-kernel Adam/AdamW update. Returns False if the fused path can't be used."""
    grad = p.grad
    if grad is None or grad._native is None or not p._native.is_contiguous() \
            or not grad._native.is_contiguous():
        return False
    if len(state) == 0:
        state["step"] = 0
        state["exp_avg"] = vultorch.zeros_like(p)
        state["exp_avg_sq"] = vultorch.zeros_like(p)
    state["step"] += 1
    beta1, beta2 = group["betas"]
    p._native.adam_step_(
        grad._native, state["exp_avg"]._native, state["exp_avg_sq"]._native,
        group["lr"], beta1, beta2, group["eps"], group["weight_decay"],
        decoupled, state["step"])
    return True


class Adam(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 fused: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.fused = fused

    def _step_impl(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state.setdefault(p, {})

                if self.fused and _fused_adam_update(p, state, group, decoupled=False):
                    continue

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = vultorch.zeros_like(p)
                    state["exp_avg_sq"] = vultorch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    grad = grad + p * weight_decay

                # Decay the first and second moment running average coefficient
                exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
                exp_avg_sq = exp_avg_sq * beta2 + (grad * grad) * (1.0 - beta2)
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq / bias_correction2).sqrt() + eps

                update = exp_avg / denom * step_size
                p.copy_(p - update)

class AdamW(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2,
                 fused: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.fused = fused

    def _step_impl(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state.setdefault(p, {})
                if self.fused and _fused_adam_update(p, state, group, decoupled=True):
                    continue

                # Decoupled weight decay
                p_decayed = p * (1.0 - lr * weight_decay)

                grad = p.grad

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = vultorch.zeros_like(p)
                    state["exp_avg_sq"] = vultorch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
                exp_avg_sq = exp_avg_sq * beta2 + (grad * grad) * (1.0 - beta2)
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq / bias_correction2).sqrt() + eps

                update = exp_avg / denom * step_size
                p.copy_(p_decayed - update)
