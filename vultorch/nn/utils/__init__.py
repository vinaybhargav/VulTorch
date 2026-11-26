"""
Neural Network Utilities (vultorch.nn.utils)
"""

from typing import Iterable, Union
import math
import vultorch
from ...tensor import Tensor
from ...autograd import no_grad

def clip_grad_norm_(
    parameters: Union[Iterable[Tensor], Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tensor:
    """
    Clips gradient norm of an iterable of parameters.
    Executes reduction across parameters on the GPU directly without per-parameter host roundtrips.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(grads) == 0:
        return vultorch.tensor(0.0)

    with no_grad():
        first_dev = grads[0].device
        if norm_type == math.inf:
            norms = [g.abs().max() for g in grads]
            total_norm = max(float(n.item()) for n in norms)
            total_norm_t = vultorch.tensor(total_norm, device=first_dev)
        else:
            # GPU sum of power of norms
            sum_norms = vultorch.tensor(0.0, device=first_dev)
            for g in grads:
                g_dev = g.to(first_dev)
                if norm_type == 2.0:
                    sum_norms = sum_norms + (g_dev * g_dev).sum()
                else:
                    sum_norms = sum_norms + (g_dev.abs() ** norm_type).sum()
            total_norm_t = sum_norms ** (1.0 / norm_type)
            total_norm = float(total_norm_t.item())

        if error_if_nonfinite and (math.isnan(total_norm) or math.isinf(total_norm)):
            raise RuntimeError(f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, so it cannot be clipped.")

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for g in grads:
                g.copy_(g * clip_coef)

        return total_norm_t


def clip_grad_value_(parameters: Union[Iterable[Tensor], Tensor], clip_value: float):
    """
    Clips gradient of an iterable of parameters at specified value.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    with no_grad():
        for p in parameters:
            if p.grad is not None:
                p.grad.clamp_(-clip_value, clip_value)
