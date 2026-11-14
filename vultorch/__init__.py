"""
VulTorch: A High-Performance Pure Vulkan Deep Learning Library
"""

import numpy as np
from typing import Tuple, List, Union, Optional

from .tensor import Tensor, Device
from .autograd import no_grad, is_grad_enabled, backward
from .testing import gradcheck
from . import nn
from . import optim
from . import utils
from . import _vultorch_backend as _backend

import threading

__version__ = "0.1.0"

_rng_lock = threading.Lock()
_rng_seed: int = 0x5DEECE66D
_rng_offset: int = 0

def manual_seed(seed: int):
    """Sets the global Philox PRNG seed and resets the counter offset."""
    global _rng_seed, _rng_offset
    with _rng_lock:
        _rng_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        _rng_offset = 0
    np.random.seed(int(seed) & 0xFFFFFFFF)

def _next_philox_state() -> Tuple[int, int]:
    global _rng_offset
    with _rng_lock:
        seed = _rng_seed
        off = _rng_offset
        _rng_offset = (off + 1) & 0xFFFFFFFF
        return seed, off

def device(d: Union[str, Device]) -> Device:
    return Device(d)

def tensor(data, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    return Tensor(data, device=device, requires_grad=requires_grad)

def empty(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    dev = Device(device)
    dev_id = -1 if dev.type == "cpu" else dev.index
    native = _backend.NativeTensor([int(d) for d in shape], dev_id)
    return Tensor(native, device=dev, requires_grad=requires_grad)

def zeros(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    t = empty(*shape, device=device, requires_grad=requires_grad)
    t._native.zero_()
    return t

def ones(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    t = empty(*shape, device=device, requires_grad=requires_grad)
    t._native.fill_(1.0)
    return t

def zeros_like(t: Tensor, device: Optional[Union[str, Device]] = None, requires_grad: bool = False) -> Tensor:
    dev = device if device is not None else t.device
    return zeros(t.shape, device=dev, requires_grad=requires_grad)

def ones_like(t: Tensor, device: Optional[Union[str, Device]] = None, requires_grad: bool = False) -> Tensor:
    dev = device if device is not None else t.device
    return ones(t.shape, device=dev, requires_grad=requires_grad)

def randn(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    t = empty(*shape, device=device, requires_grad=requires_grad)
    seed, offset = _next_philox_state()
    t._native.random_philox_(1, 0.0, 1.0, seed, offset, None)
    return t

def rand(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    t = empty(*shape, device=device, requires_grad=requires_grad)
    seed, offset = _next_philox_state()
    t._native.random_philox_(0, 0.0, 1.0, seed, offset, None)
    return t

def arange(start: float, end: Optional[float] = None, step: float = 1.0, device: Union[str, Device] = "vk") -> Tensor:
    if end is None:
        arr = np.arange(start, dtype=np.float32)
    else:
        arr = np.arange(start, end, step, dtype=np.float32)
    return Tensor(arr, device=device)

def linspace(start: float, end: float, steps: int, device: Union[str, Device] = "vk") -> Tensor:
    arr = np.linspace(start, end, steps, dtype=np.float32)
    return Tensor(arr, device=device)

def eye(n: int, m: Optional[int] = None, device: Union[str, Device] = "vk") -> Tensor:
    arr = np.eye(n, m, dtype=np.float32)
    return Tensor(arr, device=device)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b

def cat(tensors: List[Tensor], dim: int = 0, device: Optional[Union[str, Device]] = None) -> Tensor:
    if not tensors:
        raise ValueError("cat expects a non-empty list of tensors")
    dev = Device(device) if device is not None else tensors[0].device
    ndim = tensors[0].ndim
    norm_dim = dim if dim >= 0 else dim + ndim
    req_grad = any(t.requires_grad for t in tensors) and is_grad_enabled()

    if norm_dim == 0 and ndim >= 1 and all(t._native.is_contiguous() for t in tensors):
        target_dev_id = -1 if dev.type == "cpu" else dev.index
        native_out = _backend.NativeTensor.cat_dim0([t._native for t in tensors], target_dev_id)
        out = Tensor(native_out, device=dev, requires_grad=req_grad)
    else:
        arrs = [t.numpy() for t in tensors]
        res = np.concatenate(arrs, axis=dim)
        out = Tensor(res, device=dev, requires_grad=req_grad)

    if req_grad:
        out.is_leaf = False
        split_sizes = [t.shape[dim] for t in tensors]
        src_devices = [t.device for t in tensors]
        def cat_backward(grad_out: Tensor):
            grads = []
            curr = 0
            for sz, sdev in zip(split_sizes, src_devices):
                idx = [slice(None)] * grad_out.ndim
                idx[dim] = slice(curr, curr + sz)
                g_chunk = grad_out[tuple(idx)]
                if g_chunk.device != sdev:
                    g_chunk = g_chunk.to(sdev)
                grads.append(g_chunk)
                curr += sz
            return tuple(grads)
        from .autograd import Node
        out.grad_fn = Node("CatBackward", cat_backward, tensors)
    return out

def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    expanded = [t.unsqueeze(dim) for t in tensors]
    return cat(expanded, dim=dim)

def save(obj, f):
    import pickle
    with open(f, 'wb') as fp:
        pickle.dump(obj, fp)

def load(f):
    import pickle
    with open(f, 'rb') as fp:
        return pickle.load(fp)

# Vulkan System and Diagnostics
class _VulkanInfo:
    _tflops_cache: dict = {}

    @staticmethod
    def is_available() -> bool:
        return _backend.get_device_count() > 0

    @staticmethod
    def device_count() -> int:
        return _backend.get_device_count()

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return _backend.get_device_info(device_id)["name"]

    @staticmethod
    def get_device_properties(device_id: int = 0) -> dict:
        return _backend.get_device_info(device_id)

    @staticmethod
    def get_all_devices() -> list:
        """Properties (including capabilities) for every detected device."""
        return _backend.get_all_devices()

    @staticmethod
    def synchronize(device_id: Union[int, str, Device] = 0):
        if isinstance(device_id, (str, Device)):
            dev = Device(device_id)
            if dev.type == "cpu":
                return
            device_id = dev.index
        _backend.synchronize(int(device_id))

    @staticmethod
    def synchronize_all():
        _backend.synchronize_all()

    @classmethod
    def get_device_tflops(cls, device_id: int = 0, n: int = 1024, force_recalibrate: bool = False) -> float:
        """Measures and caches effective FP32 GEMM throughput (TFLOPS) for `vk:<device_id>`."""
        key = (int(device_id), int(n))
        if not force_recalibrate and key in cls._tflops_cache:
            return cls._tflops_cache[key]

        import time
        dev_str = f"vk:{device_id}"
        with no_grad():
            a = ones((n, n), device=dev_str)
            b = ones((n, n), device=dev_str)
            # Warmup (includes kernel autotuning on first run)
            for _ in range(2):
                _ = a @ b
            cls.synchronize(device_id)

            reps = 3
            t0 = time.perf_counter()
            for _ in range(reps):
                _ = a @ b
            cls.synchronize(device_id)
            elapsed = max(time.perf_counter() - t0, 1e-9)

        flops = 2.0 * (n ** 3) * reps
        tflops = (flops / elapsed) / 1e12
        cls._tflops_cache[key] = tflops
        return tflops

    @classmethod
    def compute_split_sizes(
        cls,
        batch_size: int,
        device_ids: Optional[List[Union[str, Device]]] = None,
        weights: Optional[List[float]] = None,
        min_chunk: int = 1,
    ) -> List[int]:
        """Partitions `batch_size` across `device_ids` proportional to measured TFLOPS."""
        if device_ids is None:
            device_ids = [f"vk:{i}" for i in range(cls.device_count())]
        n_devs = len(device_ids)
        if n_devs == 0:
            return []
        if n_devs == 1 or batch_size <= 0:
            return [batch_size] + [0] * (n_devs - 1)

        if weights is None:
            raw_weights = []
            for d in device_ids:
                dev_obj = d if isinstance(d, Device) else Device(d)
                if dev_obj.type == "vk":
                    raw_weights.append(max(cls.get_device_tflops(dev_obj.index), 1e-3))
                else:
                    raw_weights.append(0.05)
        else:
            raw_weights = [max(float(w), 0.0) for w in weights]

        total_w = sum(raw_weights)
        if total_w <= 0.0:
            raw_weights = [1.0] * n_devs
            total_w = float(n_devs)

        norm_w = [w / total_w for w in raw_weights]
        effective_min = min_chunk if batch_size >= n_devs * min_chunk else 0

        sizes = [max(effective_min, int(batch_size * w)) for w in norm_w]
        # Adjust remainder or excess according to fractional parts
        diff = batch_size - sum(sizes)
        if diff > 0:
            fracs = sorted(
                range(n_devs),
                key=lambda i: (batch_size * norm_w[i]) - sizes[i],
                reverse=True,
            )
            for i in range(diff):
                sizes[fracs[i % n_devs]] += 1
        elif diff < 0:
            fracs = sorted(
                range(n_devs),
                key=lambda i: sizes[i] - effective_min,
                reverse=True,
            )
            for idx in fracs:
                while diff < 0 and sizes[idx] > effective_min:
                    sizes[idx] -= 1
                    diff += 1
        return sizes

vulkan = _VulkanInfo()

# Friendly reminder for users looking for cuda
class _NoCudaNotice:
    def __getattr__(self, name):
        raise AttributeError(
            "VulTorch is built entirely on open Vulkan! "
            "No CUDA or ROCm is used. Use 'vultorch.vulkan' or device='vk' / 'vk:0' instead."
        )

cuda = _NoCudaNotice()
