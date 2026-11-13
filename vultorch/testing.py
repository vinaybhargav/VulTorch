"""Numerical gradient checking utilities.

Mirrors the intent of ``torch.autograd.gradcheck``: compare the analytic
gradients produced by ``Tensor.backward`` against central finite differences of
the same function.

The checker is deliberately simple and slow (two forward passes per element).
Use it on small tensors in tests, not on models.

Tolerance: because the whole stack is float32, central differences with
``eps=1e-3`` are only accurate to roughly 1e-3 relative. The defaults below are
tuned so a *correct* implementation passes and a wrong one is caught; they are
not a precision benchmark.
"""

from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

from .autograd import no_grad
from .tensor import Tensor


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _build(arrays: Sequence[np.ndarray], device: Any, requires_grad: bool) -> List[Tensor]:
    import vultorch

    return [vultorch.tensor(a, device=device, requires_grad=requires_grad) for a in arrays]


def _scalar(fn: Callable, arrays: Sequence[np.ndarray], device: Any) -> float:
    with no_grad():
        out = fn(*_build(arrays, device, requires_grad=False))
    if not isinstance(out, Tensor):
        raise TypeError(f"gradcheck expects fn to return a Tensor, got {type(out)}")
    if out.numel != 1:
        raise ValueError(f"gradcheck requires a scalar output, got shape {out.shape}")
    return float(np.asarray(out.numpy()).reshape(-1)[0])


def gradcheck(
    fn: Callable[..., Tensor],
    *inputs: Any,
    eps: float = 1e-3,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    device: Any = "cpu",
    raise_exception: bool = True,
) -> bool:
    """Verify analytic gradients of ``fn`` against finite differences.

    Args:
        fn: callable taking Tensors and returning a single-element Tensor.
        *inputs: numpy arrays or Tensors; treated as leaves with requires_grad.
        eps: finite-difference step.
        atol, rtol: tolerances passed to ``np.allclose``.
        device: device to run the check on ("cpu" or "vk:N").
        raise_exception: raise AssertionError on mismatch instead of returning False.

    Returns:
        True if every input gradient matches.
    """
    if not inputs:
        raise ValueError("gradcheck requires at least one input")

    arrays = [_to_numpy(x) for x in inputs]
    tensors = _build(arrays, device, requires_grad=True)

    out = fn(*tensors)
    if not isinstance(out, Tensor):
        raise TypeError(f"gradcheck expects fn to return a Tensor, got {type(out)}")
    if out.numel != 1:
        raise ValueError(f"gradcheck requires a scalar output, got shape {out.shape}")
    out.backward()

    analytic = []
    for tensor, array in zip(tensors, arrays):
        if tensor.grad is None:
            analytic.append(np.zeros_like(array))
        else:
            analytic.append(np.asarray(tensor.grad.numpy(), dtype=np.float32))

    numeric = []
    for array in arrays:
        grad = np.zeros_like(array)
        flat = array.reshape(-1)
        grad_flat = grad.reshape(-1)
        for j in range(flat.size):
            original = flat[j]
            flat[j] = original + eps
            plus = _scalar(fn, arrays, device)
            flat[j] = original - eps
            minus = _scalar(fn, arrays, device)
            flat[j] = original
            grad_flat[j] = (plus - minus) / (2.0 * eps)
        numeric.append(grad)

    failures = []
    for i, (analytic_i, numeric_i) in enumerate(zip(analytic, numeric)):
        if not np.allclose(analytic_i, numeric_i, atol=atol, rtol=rtol):
            denom = np.maximum(np.abs(analytic_i) + np.abs(numeric_i), 1e-8)
            worst = float(np.max(np.abs(analytic_i - numeric_i) / denom))
            failures.append(
                f"input[{i}] shape={analytic_i.shape} max_rel_err={worst:.3e}\n"
                f"    analytic={analytic_i.ravel()[:8]}\n"
                f"    numeric ={numeric_i.ravel()[:8]}"
            )

    if failures:
        message = "gradcheck failed:\n" + "\n".join(failures)
        if raise_exception:
            raise AssertionError(message)
        return False

    return True


__all__ = ["gradcheck"]
