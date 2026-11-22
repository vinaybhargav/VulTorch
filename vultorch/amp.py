"""
VulTorch Automatic Mixed Precision (vultorch.amp)
- `autocast`: Thread-local context manager and decorator for mixed-precision forward passes.
- `GradScaler`: Dynamic loss scaling with GPU-side inf/nan gradient detection and unscaling.
"""

import threading
from typing import Optional, Union, Iterable
from .tensor import Tensor, DType, parse_dtype, float16, float32
from . import _vultorch_backend as _backend

_amp_state = threading.local()


def is_autocast_enabled() -> bool:
    return getattr(_amp_state, "enabled", False)


def get_autocast_dtype() -> DType:
    return getattr(_amp_state, "dtype", float16)


class autocast:
    """
    Context manager and decorator that enables automatic mixed precision (`float16` / `bfloat16`)
    for compute-intensive operations (`Linear`, `matmul`, `Conv2d`, `scaled_dot_product_attention`)
    while keeping reductions and normalizations (`LayerNorm`, `RMSNorm`, `Softmax`, `CrossEntropyLoss`)
    in `float32`.
    """

    def __init__(
        self,
        device_type: str = "vk",
        dtype: Union[DType, str, None] = None,
        enabled: bool = True,
    ):
        # Support calling `autocast(dtype=...)` or `autocast("vk", dtype=...)`
        if isinstance(device_type, DType):
            dtype = device_type
            device_type = "vk"
        self.device_type = device_type
        self.dtype = parse_dtype(dtype) if dtype is not None else float16
        self.enabled = bool(enabled)
        self._prev_enabled = False
        self._prev_dtype = float16

    def __enter__(self):
        self._prev_enabled = is_autocast_enabled()
        self._prev_dtype = get_autocast_dtype()
        _amp_state.enabled = self.enabled
        _amp_state.dtype = self.dtype
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _amp_state.enabled = self._prev_enabled
        _amp_state.dtype = self._prev_dtype
        return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class GradScaler:
    """
    Dynamic loss scaler for mixed-precision training.
    Scales loss before `backward()`, unscales gradients and checks for `inf`/`nan`
    on the GPU via `amp_unscale_check` in `shaders/gemm_quant.slang`, and skips
    `optimizer.step()` if overflow is detected.
    """

    def __init__(
        self,
        init_scale: float = 1024.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self._scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.enabled = bool(enabled)
        self._growth_tracker = 0
        self._found_inf_per_opt = {}

    def get_scale(self) -> float:
        return self._scale if self.enabled else 1.0

    def is_enabled(self) -> bool:
        return self.enabled

    def scale(self, outputs: Union[Tensor, Iterable[Tensor]]):
        if not self.enabled:
            return outputs
        if isinstance(outputs, Tensor):
            return outputs * self._scale
        return type(outputs)(self.scale(o) for o in outputs)

    def unscale_(self, optimizer) -> bool:
        if not self.enabled:
            return False
        opt_id = id(optimizer)
        if opt_id in self._found_inf_per_opt:
            return self._found_inf_per_opt[opt_id]

        grad_natives = []
        for group in getattr(optimizer, "param_groups", []):
            for p in group.get("params", []):
                if p is not None and p.grad is not None:
                    grad_natives.append(p.grad._native)
        for p in getattr(optimizer, "params", []):
            if p is not None and p.grad is not None:
                grad_natives.append(p.grad._native)

        inv_scale = 1.0 / self._scale
        found_inf = _backend.NativeTensor.amp_unscale_and_check(grad_natives, float(inv_scale))
        self._found_inf_per_opt[opt_id] = bool(found_inf)
        return bool(found_inf)

    def step(self, optimizer, *args, **kwargs):
        if not self.enabled:
            return optimizer.step(*args, **kwargs)
        opt_id = id(optimizer)
        if opt_id not in self._found_inf_per_opt:
            self.unscale_(optimizer)
        found_inf = self._found_inf_per_opt[opt_id]
        if not found_inf:
            return optimizer.step(*args, **kwargs)
        return None

    def update(self, new_scale: Optional[float] = None):
        if not self.enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
        else:
            any_inf = any(self._found_inf_per_opt.values())
            if any_inf:
                self._scale = max(1.0, self._scale * self.backoff_factor)
                self._growth_tracker = 0
            else:
                self._growth_tracker += 1
                if self._growth_tracker >= self.growth_interval:
                    self._scale *= self.growth_factor
                    self._growth_tracker = 0
        self._found_inf_per_opt.clear()
