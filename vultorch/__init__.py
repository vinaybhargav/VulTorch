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

__version__ = "0.1.0"

def device(d: Union[str, Device]) -> Device:
    return Device(d)

def tensor(data, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    return Tensor(data, device=device, requires_grad=requires_grad)

def zeros(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    arr = np.zeros(shape, dtype=np.float32)
    return Tensor(arr, device=device, requires_grad=requires_grad)

def ones(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    arr = np.ones(shape, dtype=np.float32)
    return Tensor(arr, device=device, requires_grad=requires_grad)

def zeros_like(t: Tensor, device: Optional[Union[str, Device]] = None, requires_grad: bool = False) -> Tensor:
    dev = device if device is not None else t.device
    return zeros(t.shape, device=dev, requires_grad=requires_grad)

def ones_like(t: Tensor, device: Optional[Union[str, Device]] = None, requires_grad: bool = False) -> Tensor:
    dev = device if device is not None else t.device
    return ones(t.shape, device=dev, requires_grad=requires_grad)

def randn(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    arr = np.random.randn(*shape).astype(np.float32)
    return Tensor(arr, device=device, requires_grad=requires_grad)

def rand(*shape, device: Union[str, Device] = "vk", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    arr = np.random.rand(*shape).astype(np.float32)
    return Tensor(arr, device=device, requires_grad=requires_grad)

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

def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("cat expects a non-empty list of tensors")
    dev = tensors[0].device
    arrs = [t.numpy() for t in tensors]
    res = np.concatenate(arrs, axis=dim)
    req_grad = any(t.requires_grad for t in tensors)
    out = Tensor(res, device=dev, requires_grad=req_grad)

    if is_grad_enabled() and req_grad:
        out.is_leaf = False
        split_sizes = [t.shape[dim] for t in tensors]
        def cat_backward(grad_out: Tensor):
            grads = []
            curr = 0
            for sz in split_sizes:
                idx = [slice(None)] * grad_out.ndim
                idx[dim] = slice(curr, curr + sz)
                grads.append(grad_out[tuple(idx)])
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
    def synchronize(device_id: int = 0):
        _backend.synchronize(device_id)

    @staticmethod
    def synchronize_all():
        _backend.synchronize_all()

vulkan = _VulkanInfo()

# Friendly reminder for users looking for cuda
class _NoCudaNotice:
    def __getattr__(self, name):
        raise AttributeError(
            "VulTorch is built entirely on open Vulkan! "
            "No CUDA or ROCm is used. Use 'vultorch.vulkan' or device='vk' / 'vk:0' instead."
        )

cuda = _NoCudaNotice()
