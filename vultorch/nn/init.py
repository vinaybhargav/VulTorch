"""
Parameter Initializations (vultorch.nn.init)
"""

import math
from typing import Tuple
import vultorch
from ..tensor import Tensor
from ..autograd import no_grad

def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def calculate_gain(nonlinearity: str, param: float = None) -> float:
    linear_gains = {
        "linear": 1.0,
        "conv1d": 1.0,
        "conv2d": 1.0,
        "conv3d": 1.0,
        "conv_transpose1d": 1.0,
        "conv_transpose2d": 1.0,
        "conv_transpose3d": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3,
        "relu": math.sqrt(2.0),
        "leaky_relu": math.sqrt(2.0 / (1.0 + (param if param is not None else 0.01) ** 2)),
        "selu": 3.0 / 4,
    }
    if nonlinearity in linear_gains:
        return linear_gains[nonlinearity]
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    with no_grad():
        r = vultorch.rand(tensor.shape, device=tensor.device, dtype=tensor.dtype)
        val = r * (b - a) + a
        tensor.copy_(val)
        return tensor

def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    with no_grad():
        r = vultorch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype)
        val = r * std + mean
        tensor.copy_(val)
        return tensor

def constant_(tensor: Tensor, val: float) -> Tensor:
    with no_grad():
        c = vultorch.full(tensor.shape, val, device=tensor.device, dtype=tensor.dtype)
        tensor.copy_(c)
        return tensor

def zeros_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 0.0)

def ones_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 1.0)

def eye_(tensor: Tensor) -> Tensor:
    with no_grad():
        if tensor.dim() != 2:
            raise ValueError("Only 2D tensors supported for eye_")
        e = vultorch.eye(tensor.shape[0], tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
        tensor.copy_(e)
        return tensor

def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a)

def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std)

def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)

def kaiming_normal_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)
