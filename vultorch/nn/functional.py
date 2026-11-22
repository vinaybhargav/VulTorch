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


def snake(x: Tensor, a: float = 1.0, eps: float = 1e-9) -> Tensor:
    return x.snake(a=a, eps=eps)


def snake_gamma(x: Tensor, a: float = 1.0, gamma: float = 0.0, eps: float = 1e-9) -> Tensor:
    return x.snake_gamma(a=a, gamma=gamma, eps=eps)


def snake_cos(x: Tensor, a: float = 1.0, eps: float = 1e-9) -> Tensor:
    return x.snake_cos(a=a, eps=eps)


def snake_rational(x: Tensor, a: float = 1.0, eps: float = 1e-9) -> Tensor:
    return x.snake_rational(a=a, eps=eps)


def snake_sinc(x: Tensor, a: float = 1.0, eps: float = 1e-9) -> Tensor:
    return x.snake_sinc(a=a, eps=eps)


def softplus(x: Tensor) -> Tensor:
    return (1.0 + x.exp()).log()


def selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Union[Tensor, None] = None,
) -> Tensor:
    """
    Mamba (Gu & Dao 2023) Selective State Space Model (S6) parallel scan:
        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
        y_t = sum_n (C_{n,t} * h_{n,t}) + D * u_t
    Shapes:
        u, delta: [B, D, L] (delta > 0, typically post-softplus)
        A:        [D, N]    (N <= 32)
        B, C:     [B, N, L]
        D:        [D] or None
    """
    d_nat = D._native if D is not None else None
    y_nat = u._native.selective_scan(delta._native, A._native, B._native, C._native, d_nat)
    inputs = [u, delta, A, B, C] + ([D] if D is not None else [])
    req_grad = any(t.requires_grad for t in inputs) and is_grad_enabled()
    out = Tensor._from_native(y_nat, u.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = u.device
        has_D = D is not None

        def ssm_backward(g: Tensor):
            du_n, ddt_n, dA_n, dB_n, dC_n, dD_n = u._native.selective_scan_backward(
                g.contiguous()._native, delta._native, A._native, B._native, C._native, d_nat
            )
            du = Tensor._from_native(du_n, dev, False) if u.requires_grad else None
            ddt = Tensor._from_native(ddt_n, dev, False) if delta.requires_grad else None
            dA = Tensor._from_native(dA_n, dev, False) if A.requires_grad else None
            dB = Tensor._from_native(dB_n, dev, False) if B.requires_grad else None
            dC = Tensor._from_native(dC_n, dev, False) if C.requires_grad else None
            if has_D:
                dD = Tensor._from_native(dD_n, dev, False) if D.requires_grad else None
                return du, ddt, dA, dB, dC, dD
            return du, ddt, dA, dB, dC

        out.grad_fn = Node("SelectiveScanBackward", ssm_backward, inputs)
    return out


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
