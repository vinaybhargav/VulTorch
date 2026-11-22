from typing import Sequence, Union
import vultorch
from ..tensor import Tensor
from ..autograd import Node, is_grad_enabled


def pad(
    input: Tensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    """Pad tensor according to PyTorch convention (pairs starting from last dimension)."""
    if len(pad) % 2 != 0 or len(pad) // 2 > input.ndim:
        raise ValueError(f"Invalid pad length {len(pad)} for {input.ndim}D tensor")
    mode_map = {"constant": 0, "reflect": 1, "replicate": 2}
    if mode not in mode_map:
        raise ValueError(f"Unsupported pad mode: {mode!r}")
    mode_id = mode_map[mode]

    ndim = input.ndim
    pad_before = [0] * ndim
    pad_after = [0] * ndim
    for i in range(len(pad) // 2):
        dim = ndim - 1 - i
        pad_before[dim] = int(pad[2 * i])
        pad_after[dim] = int(pad[2 * i + 1])

    native_out = input._native.pad_nd(pad_before, pad_after, mode_id, float(value))
    req_grad = input.requires_grad and is_grad_enabled()
    out = Tensor._from_native(native_out, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        in_shape = list(input.shape)
        dev = input.device

        def pad_backward(g: Tensor):
            gin_nat = g._native.pad_nd_backward(in_shape, pad_before, mode_id)
            return Tensor._from_native(gin_nat, dev, False)

        out.grad_fn = Node("PadBackward", pad_backward, [input])
    return out


def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.softmax(dim=dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.log_softmax(dim=dim)


def linear(x: Tensor, weight: Tensor, bias: Union[Tensor, None] = None) -> Tensor:
    out = x.matmul(weight, trans_b=True, bias=bias)
    return out
