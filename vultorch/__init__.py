"""
VulTorch: A High-Performance Pure Vulkan Deep Learning Library
"""

import numpy as np
from typing import Tuple, List, Union, Optional

from .tensor import (
    Tensor,
    Device,
    DType,
    parse_dtype,
    get_default_device,
    set_default_device,
    resolve_device,
    float32,
    float16,
    half,
    bfloat16,
    int32,
    int64,
    long,
    bool_dtype as _bool_dtype,
    uint8,
    byte,
)
from .autograd import no_grad, enable_grad, is_grad_enabled, backward
from .graph import VulkanGraph, compile
from .testing import gradcheck
from . import amp
from . import nn
from . import optim
from . import utils
from . import linalg
from . import fft
from .serialization import save_safetensors, load_safetensors
from . import _vultorch_backend as _backend

import builtins as _builtins
import threading

__version__ = "0.1.0"

_rng_lock = threading.Lock()
_rng_seed: _builtins.int = 0x5DEECE66D
_rng_offset: _builtins.int = 0


def manual_seed(seed: _builtins.int):
    """Sets the global Philox PRNG seed and resets the counter offset."""
    global _rng_seed, _rng_offset
    with _rng_lock:
        _rng_seed = _builtins.int(seed) & 0xFFFFFFFFFFFFFFFF
        _rng_offset = 0
    np.random.seed(_builtins.int(seed) & 0xFFFFFFFF)


def _next_philox_state() -> Tuple[_builtins.int, _builtins.int]:
    global _rng_offset
    with _rng_lock:
        seed = _rng_seed
        off = _rng_offset
        _rng_offset = (off + 1) & 0xFFFFFFFF
        return seed, off


def device(d: Union[str, Device]) -> Device:
    return Device(d)


def tensor(
    data,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    return Tensor(data, device=device, requires_grad=requires_grad, dtype=dtype)


def from_numpy(
    ndarray: np.ndarray, device: Optional[Union[str, Device]] = None, dtype=None
) -> Tensor:
    if not isinstance(ndarray, np.ndarray):
        ndarray = np.asarray(ndarray)
    return Tensor(ndarray, device=device, dtype=dtype)


def empty(
    *shape,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    dev = resolve_device(device)
    dev_id = -1 if dev.type == "cpu" else dev.index
    dt = parse_dtype(dtype) if dtype is not None else float32
    if requires_grad and not dt.is_floating_point:
        raise RuntimeError("Only Tensors of floating point dtype can require gradients")
    native = _backend.NativeTensor([_builtins.int(d) for d in shape], dev_id, dt.name)
    return Tensor(native, device=dev, requires_grad=requires_grad)


def zeros(
    *shape,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    t = empty(*shape, device=device, requires_grad=requires_grad, dtype=dtype)
    t._native.zero_()
    return t


def ones(
    *shape,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    t = empty(*shape, device=device, requires_grad=requires_grad, dtype=dtype)
    t._native.fill_(1.0)
    return t


def full(
    shape,
    fill_value,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    if isinstance(shape, _builtins.int):
        shape = (shape,)
    if dtype is None:
        if isinstance(fill_value, _builtins.bool):
            dtype = _bool_dtype
        elif isinstance(fill_value, _builtins.int):
            dtype = int64
        else:
            dtype = float32
    t = empty(shape, device=device, requires_grad=requires_grad, dtype=dtype)
    t._native.fill_(_builtins.float(fill_value))
    return t


def zeros_like(
    t: Tensor,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    dev = device if device is not None else t.device
    dt = dtype if dtype is not None else t.dtype
    return zeros(t.shape, device=dev, requires_grad=requires_grad, dtype=dt)


def ones_like(
    t: Tensor,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    dev = device if device is not None else t.device
    dt = dtype if dtype is not None else t.dtype
    return ones(t.shape, device=dev, requires_grad=requires_grad, dtype=dt)


def empty_like(
    t: Tensor,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    dev = device if device is not None else t.device
    dt = dtype if dtype is not None else t.dtype
    return empty(t.shape, device=dev, requires_grad=requires_grad, dtype=dt)


def full_like(
    t: Tensor,
    fill_value,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    dev = device if device is not None else t.device
    dt = dtype if dtype is not None else t.dtype
    return full(t.shape, fill_value, device=dev, requires_grad=requires_grad, dtype=dt)


def rand_like(
    t: Tensor,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    dev = device if device is not None else t.device
    dt = dtype if dtype is not None else t.dtype
    return rand(t.shape, device=dev, requires_grad=requires_grad, dtype=dt)


def randn_like(
    t: Tensor,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    dev = device if device is not None else t.device
    dt = dtype if dtype is not None else t.dtype
    return randn(t.shape, device=dev, requires_grad=requires_grad, dtype=dt)


def randint_like(
    t: Tensor,
    low: _builtins.int,
    high: Optional[_builtins.int] = None,
    device: Optional[Union[str, Device]] = None,
    dtype=None,
) -> Tensor:
    dev = device if device is not None else t.device
    dt = dtype if dtype is not None else t.dtype
    return randint(low, high, size=t.shape, device=dev, dtype=dt)


def randperm(n: _builtins.int, device: Optional[Union[str, Device]] = None, dtype=None) -> Tensor:
    dt = parse_dtype(dtype) if dtype is not None else int64
    r = rand(_builtins.int(n), device=device, dtype=float32)
    _, idx = r.sort(dim=0, descending=False)
    return idx if idx.dtype == dt else idx.cast(dt)


def randn(
    *shape,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    t = empty(*shape, device=device, requires_grad=requires_grad, dtype=dtype)
    seed, offset = _next_philox_state()
    t._native.random_philox_(1, 0.0, 1.0, seed, offset, None)
    return t


def rand(
    *shape,
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    t = empty(*shape, device=device, requires_grad=requires_grad, dtype=dtype)
    seed, offset = _next_philox_state()
    t._native.random_philox_(0, 0.0, 1.0, seed, offset, None)
    return t


def randint(
    low: _builtins.int,
    high: Optional[_builtins.int] = None,
    size: Union[_builtins.int, Sequence[_builtins.int]] = (),
    device: Optional[Union[str, Device]] = None,
    requires_grad: _builtins.bool = False,
    dtype=None,
) -> Tensor:
    if high is None:
        high = low
        low = 0
    if isinstance(size, _builtins.int):
        size = (size,)
    dt = parse_dtype(dtype) if dtype is not None else int64
    dev = resolve_device(device)
    if dev.type == "vk" and len(size) > 0:
        # Pure on-GPU Philox integer sampling (zero PCIe upload)
        u = empty(*size, device=dev, dtype=float32)
        seed, offset = _next_philox_state()
        u._native.random_philox_(0, _builtins.float(low), _builtins.float(high), seed, offset, None)
        floored = u.floor().clamp(min=_builtins.float(low), max=_builtins.float(high - 1))
        out = floored.cast(dt)
        out.requires_grad = requires_grad
        return out
    np_dt = dt.numpy_dtype if dt.numpy_dtype is not None else np.int64
    arr = np.random.randint(low, high, size=size, dtype=np_dt)
    return tensor(arr, device=device, requires_grad=requires_grad, dtype=dt)


def arange(
    start, end=None, step=1, device: Optional[Union[str, Device]] = None, dtype=None
) -> Tensor:
    dt = parse_dtype(dtype) if dtype is not None else float32
    np_dt = dt.numpy_dtype if dt.numpy_dtype is not None else np.float32
    if end is None:
        arr = np.arange(start, dtype=np_dt)
    else:
        arr = np.arange(start, end, step, dtype=np_dt)
    return Tensor(arr, device=device, dtype=dt)


def linspace(
    start: _builtins.float,
    end: _builtins.float,
    steps: _builtins.int,
    device: Optional[Union[str, Device]] = None,
    dtype=None,
) -> Tensor:
    dt = parse_dtype(dtype) if dtype is not None else float32
    np_dt = dt.numpy_dtype if dt.numpy_dtype is not None else np.float32
    arr = np.linspace(start, end, steps, dtype=np_dt)
    return Tensor(arr, device=device, dtype=dt)


def eye(
    n: _builtins.int,
    m: Optional[_builtins.int] = None,
    device: Optional[Union[str, Device]] = None,
    dtype=None,
) -> Tensor:
    dt = parse_dtype(dtype) if dtype is not None else float32
    np_dt = dt.numpy_dtype if dt.numpy_dtype is not None else np.float32
    arr = np.eye(n, m, dtype=np_dt)
    return Tensor(arr, device=device, dtype=dt)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b


def abs(x: Tensor) -> Tensor:
    return x.abs()


def sign(x: Tensor) -> Tensor:
    return x.sign()


def exp(x: Tensor) -> Tensor:
    return x.exp()


def log(x: Tensor) -> Tensor:
    return x.log()


def sqrt(x: Tensor) -> Tensor:
    return x.sqrt()


def rsqrt(x: Tensor) -> Tensor:
    return x.rsqrt()


def sin(x: Tensor) -> Tensor:
    return x.sin()


def cos(x: Tensor) -> Tensor:
    return x.cos()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def round(x: Tensor) -> Tensor:
    return x.round()


def ceil(x: Tensor) -> Tensor:
    return x.ceil()


def floor(x: Tensor) -> Tensor:
    return x.floor()


def trunc(x: Tensor) -> Tensor:
    return x.trunc()


def frac(x: Tensor) -> Tensor:
    return x.frac()


def square(x: Tensor) -> Tensor:
    return x.square()


def reciprocal(x: Tensor) -> Tensor:
    return x.reciprocal()


def neg(x: Tensor) -> Tensor:
    return -x


def pow(x: Tensor, exponent) -> Tensor:
    return x**exponent


def exp2(x: Tensor) -> Tensor:
    return x.exp2()


def expm1(x: Tensor) -> Tensor:
    return x.expm1()


def log2(x: Tensor) -> Tensor:
    return x.log2()


def log10(x: Tensor) -> Tensor:
    return x.log10()


def log1p(x: Tensor) -> Tensor:
    return x.log1p()


def tan(x: Tensor) -> Tensor:
    return x.tan()


def asin(x: Tensor) -> Tensor:
    return x.asin()


def acos(x: Tensor) -> Tensor:
    return x.acos()


def atan(x: Tensor) -> Tensor:
    return x.atan()


def sinh(x: Tensor) -> Tensor:
    return x.sinh()


def cosh(x: Tensor) -> Tensor:
    return x.cosh()


def asinh(x: Tensor) -> Tensor:
    return x.asinh()


def acosh(x: Tensor) -> Tensor:
    return x.acosh()


def atanh(x: Tensor) -> Tensor:
    return x.atanh()


def erf(x: Tensor) -> Tensor:
    return x.erf()


def erfc(x: Tensor) -> Tensor:
    return x.erfc()


def isnan(x: Tensor) -> Tensor:
    return x.isnan()


def isinf(x: Tensor) -> Tensor:
    return x.isinf()


def isfinite(x: Tensor) -> Tensor:
    return x.isfinite()


def nan_to_num(
    x: Tensor,
    nan: _builtins.float = 0.0,
    posinf: Optional[_builtins.float] = None,
    neginf: Optional[_builtins.float] = None,
) -> Tensor:
    return x.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)


def maximum(a: Tensor, b) -> Tensor:
    return a.maximum(b)


def minimum(a: Tensor, b) -> Tensor:
    return a.minimum(b)


def lerp(input: Tensor, end: Tensor, weight) -> Tensor:
    return input.lerp(end, weight)


def addcmul(
    input: Tensor, tensor1: Tensor, tensor2: Tensor, value: _builtins.float = 1.0
) -> Tensor:
    return input + value * (tensor1 * tensor2)


def addcdiv(
    input: Tensor, tensor1: Tensor, tensor2: Tensor, value: _builtins.float = 1.0
) -> Tensor:
    return input + value * (tensor1 / tensor2)


def eq(a: Tensor, b) -> Tensor:
    return a == b


def ne(a: Tensor, b) -> Tensor:
    return a != b


def lt(a: Tensor, b) -> Tensor:
    return a < b


def le(a: Tensor, b) -> Tensor:
    return a <= b


def gt(a: Tensor, b) -> Tensor:
    return a > b


def ge(a: Tensor, b) -> Tensor:
    return a >= b


def isclose(
    input: Tensor,
    other: Tensor,
    rtol: _builtins.float = 1e-5,
    atol: _builtins.float = 1e-8,
    equal_nan: _builtins.bool = False,
) -> Tensor:
    diff = (input - other).abs()
    tol = atol + rtol * other.abs()
    res = diff <= tol
    if equal_nan:
        res = res | (input.isnan() & other.isnan())
    return res


def allclose(
    input: Tensor,
    other: Tensor,
    rtol: _builtins.float = 1e-5,
    atol: _builtins.float = 1e-8,
    equal_nan: _builtins.bool = False,
) -> _builtins.bool:
    return _builtins.bool(
        isclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan).all().item()
    )


def equal(input: Tensor, other: Tensor) -> _builtins.bool:
    if input.shape != other.shape or input.dtype != other.dtype:
        return False
    return _builtins.bool((input == other).all().item())


def logical_and(a: Tensor, b: Tensor) -> Tensor:
    return a.logical_and(b)


def logical_or(a: Tensor, b: Tensor) -> Tensor:
    return a.logical_or(b)


def logical_not(a: Tensor) -> Tensor:
    return a.logical_not()


def logical_xor(a: Tensor, b: Tensor) -> Tensor:
    return a.logical_xor(b)


def min(x: Tensor, dim: Optional[_builtins.int] = None, keepdim: _builtins.bool = False):
    if not isinstance(x, Tensor):
        return _builtins.min(x, dim) if dim is not None else _builtins.min(x)
    return x.min(dim=dim, keepdim=keepdim)


def amin(x: Tensor, dim: Optional[_builtins.int] = None, keepdim: _builtins.bool = False) -> Tensor:
    return x.amin(dim=dim, keepdim=keepdim)


def amax(x: Tensor, dim: Optional[_builtins.int] = None, keepdim: _builtins.bool = False) -> Tensor:
    return x.amax(dim=dim, keepdim=keepdim)


def all(x: Tensor, dim: Optional[_builtins.int] = None, keepdim: _builtins.bool = False):
    if not isinstance(x, Tensor):
        return _builtins.all(x)
    return x.all(dim=dim, keepdim=keepdim)


def any(x: Tensor, dim: Optional[_builtins.int] = None, keepdim: _builtins.bool = False):
    if not isinstance(x, Tensor):
        return _builtins.any(x)
    return x.any(dim=dim, keepdim=keepdim)


def argsort(x: Tensor, dim: _builtins.int = -1, descending: _builtins.bool = False) -> Tensor:
    return x.argsort(dim=dim, descending=descending)


def squeeze(x: Tensor, dim: Optional[_builtins.int] = None) -> Tensor:
    return x.squeeze(dim) if dim is not None else x.squeeze()


def unsqueeze(x: Tensor, dim: _builtins.int) -> Tensor:
    return x.unsqueeze(dim)


def flatten(x: Tensor, start_dim: _builtins.int = 0, end_dim: _builtins.int = -1) -> Tensor:
    return x.flatten(start_dim=start_dim, end_dim=end_dim)


def reshape(x: Tensor, shape) -> Tensor:
    return x.reshape(shape)


def clone(x: Tensor) -> Tensor:
    return x.clone()


def detach(x: Tensor) -> Tensor:
    return x.detach()


def masked_fill(x: Tensor, mask: Tensor, value: _builtins.float) -> Tensor:
    return x.masked_fill(mask, value)


def concat(tensors: List[Tensor], dim: _builtins.int = 0) -> Tensor:
    return cat(tensors, dim=dim)


def concatenate(tensors: List[Tensor], dim: _builtins.int = 0) -> Tensor:
    return cat(tensors, dim=dim)


def hstack(tensors: List[Tensor]) -> Tensor:
    if tensors[0].ndim == 1:
        return cat(tensors, dim=0)
    return cat(tensors, dim=1)


def vstack(tensors: List[Tensor]) -> Tensor:
    t_list = [t.unsqueeze(0) if t.ndim == 1 else t for t in tensors]
    return cat(t_list, dim=0)


def tile(x: Tensor, dims) -> Tensor:
    if isinstance(dims, _builtins.int):
        dims = (dims,)
    return x.repeat(*dims)


def movedim(x: Tensor, source: _builtins.int, destination: _builtins.int) -> Tensor:
    ndim = x.ndim
    src = source if source >= 0 else source + ndim
    dst = destination if destination >= 0 else destination + ndim
    order = [i for i in range(ndim) if i != src]
    order.insert(dst, src)
    return x.permute(order)


moveaxis = movedim


def swapaxes(x: Tensor, axis0: _builtins.int, axis1: _builtins.int) -> Tensor:
    return x.transpose(axis0, axis1)


swapdims = swapaxes


def broadcast_to(x: Tensor, shape) -> Tensor:
    return x.expand(*shape)


def unbind(x: Tensor, dim: _builtins.int = 0):
    norm_dim = dim if dim >= 0 else dim + x.ndim
    return tuple(t.squeeze(norm_dim) for t in x.split(1, dim=norm_dim))


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.softmax(dim=dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.log_softmax(dim=dim)


def clamp(x: Tensor, min: Optional[float] = None, max: Optional[float] = None) -> Tensor:
    return x.clamp(min=min, max=max)


def clip(x: Tensor, min: Optional[float] = None, max: Optional[float] = None) -> Tensor:
    return x.clamp(min=min, max=max)


def sum(
    x: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Tensor:
    if not isinstance(x, Tensor):
        return _builtins.sum(x)
    return x.sum(dim=dim, keepdim=keepdim)


def mean(
    x: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Tensor:
    return x.mean(dim=dim, keepdim=keepdim)


def max(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    if not isinstance(x, Tensor):
        return _builtins.max(x, dim) if dim is not None else _builtins.max(x)
    return x.max(dim=dim, keepdim=keepdim)


def as_strided(x: Tensor, size, stride, storage_offset=None) -> Tensor:
    return x.as_strided(size, stride, storage_offset=storage_offset)


def transpose(x: Tensor, dim0: _builtins.int, dim1: _builtins.int) -> Tensor:
    return x.transpose(dim0, dim1)


def permute(x: Tensor, dims) -> Tensor:
    return x.permute(dims)


def narrow(x: Tensor, dim: _builtins.int, start: _builtins.int, length: _builtins.int) -> Tensor:
    return x.narrow(dim, start, length)


def where(condition: Tensor, x, y) -> Tensor:
    dev = condition.device
    xt = x if isinstance(x, Tensor) else Tensor(x, device=dev)
    yt = y if isinstance(y, Tensor) else Tensor(y, device=dev, dtype=xt.dtype)
    if xt.device != dev:
        xt = xt.to(dev)
    if yt.device != dev:
        yt = yt.to(dev)
    native_out = _backend.NativeTensor.where_op(condition._native, xt._native, yt._native)
    req_grad = (xt.requires_grad or yt.requires_grad) and is_grad_enabled()
    out = Tensor._from_native(native_out, dev, req_grad)
    if req_grad:
        out.is_leaf = False
        cond_ref = condition.detach()

        def where_backward(g: Tensor):
            z = zeros_like(g)
            gx = where(cond_ref, g, z) if xt.requires_grad else None
            gy = where(cond_ref, z, g) if yt.requires_grad else None
            return gx, gy

        from .autograd import Node

        out.grad_fn = Node("WhereBackward", where_backward, [xt, yt])
    return out


def gather(x: Tensor, dim: _builtins.int, index: Tensor) -> Tensor:
    return x.gather(dim, index)


def index_select(x: Tensor, dim: _builtins.int, index: Tensor) -> Tensor:
    return x.index_select(dim, index)


def split(x: Tensor, split_size_or_sections, dim: _builtins.int = 0):
    return x.split(split_size_or_sections, dim=dim)


def chunk(x: Tensor, chunks: _builtins.int, dim: _builtins.int = 0):
    return x.chunk(chunks, dim=dim)


def tril(x: Tensor, diagonal: _builtins.int = 0) -> Tensor:
    return x.tril(diagonal=diagonal)


def triu(x: Tensor, diagonal: _builtins.int = 0) -> Tensor:
    return x.triu(diagonal=diagonal)


def flip(x: Tensor, dims) -> Tensor:
    return x.flip(dims)


def roll(x: Tensor, shifts, dims=None) -> Tensor:
    return x.roll(shifts, dims=dims)


def repeat_interleave(
    x: Tensor, repeats: _builtins.int, dim: Optional[_builtins.int] = None
) -> Tensor:
    return x.repeat_interleave(repeats, dim=dim)


def argmax(
    x: Tensor, dim: Optional[_builtins.int] = None, keepdim: _builtins.bool = False
) -> Tensor:
    return x.argmax(dim=dim, keepdim=keepdim)


def argmin(
    x: Tensor, dim: Optional[_builtins.int] = None, keepdim: _builtins.bool = False
) -> Tensor:
    return x.argmin(dim=dim, keepdim=keepdim)


def prod(x: Tensor, dim: Optional[_builtins.int] = None, keepdim: _builtins.bool = False) -> Tensor:
    return x.prod(dim=dim, keepdim=keepdim)


def var(
    x: Tensor,
    dim: Optional[Union[_builtins.int, Tuple[_builtins.int, ...]]] = None,
    unbiased: _builtins.bool = True,
    keepdim: _builtins.bool = False,
) -> Tensor:
    return x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)


def std(
    x: Tensor,
    dim: Optional[Union[_builtins.int, Tuple[_builtins.int, ...]]] = None,
    unbiased: _builtins.bool = True,
    keepdim: _builtins.bool = False,
) -> Tensor:
    return x.std(dim=dim, unbiased=unbiased, keepdim=keepdim)


def norm(
    x: Tensor,
    p: _builtins.float = 2.0,
    dim: Optional[Union[_builtins.int, Tuple[_builtins.int, ...]]] = None,
    keepdim: _builtins.bool = False,
) -> Tensor:
    return x.norm(p=p, dim=dim, keepdim=keepdim)


def cumsum(x: Tensor, dim: _builtins.int) -> Tensor:
    return x.cumsum(dim)


def cumprod(x: Tensor, dim: _builtins.int) -> Tensor:
    return x.cumprod(dim)


def sort(x: Tensor, dim: _builtins.int = -1, descending: _builtins.bool = False):
    return x.sort(dim=dim, descending=descending)


def topk(
    x: Tensor,
    k: _builtins.int,
    dim: _builtins.int = -1,
    largest: _builtins.bool = True,
    sorted: _builtins.bool = True,
):
    return x.topk(k, dim=dim, largest=largest, sorted=sorted)


def snake(x: Tensor, a: _builtins.float = 1.0, eps: _builtins.float = 1e-9) -> Tensor:
    return x.snake(a=a, eps=eps)


def snake_gamma(
    x: Tensor, a: _builtins.float = 1.0, gamma: _builtins.float = 0.0, eps: _builtins.float = 1e-9
) -> Tensor:
    return x.snake_gamma(a=a, gamma=gamma, eps=eps)


def snake_cos(x: Tensor, a: _builtins.float = 1.0, eps: _builtins.float = 1e-9) -> Tensor:
    return x.snake_cos(a=a, eps=eps)


def snake_rational(x: Tensor, a: _builtins.float = 1.0, eps: _builtins.float = 1e-9) -> Tensor:
    return x.snake_rational(a=a, eps=eps)


def snake_sinc(x: Tensor, a: _builtins.float = 1.0, eps: _builtins.float = 1e-9) -> Tensor:
    return x.snake_sinc(a=a, eps=eps)


def selective_scan(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Optional[Tensor] = None
) -> Tensor:
    from .nn import functional as _F

    return _F.selective_scan(u, delta, A, B, C, D)


def quantize(
    model: "nn.Module",
    bits: _builtins.int = 4,
    group_size: _builtins.int = 64,
    inplace: _builtins.bool = False,
) -> "nn.Module":
    """
    One-line weight quantization helper: recursively replaces every `nn.Linear` layer
    (whose `in_features` is divisible by `group_size` or `8`) in `model` with a
    packed `nn.QuantizedLinear` layer (`bits=4` or `bits=8`).
    """
    import copy

    target = model if inplace else copy.deepcopy(model)

    def _replace(mod: "nn.Module"):
        for name, child in list(mod._modules.items()):
            if child is None:
                continue
            if isinstance(child, nn.Linear):
                pack_div = 8 if bits == 4 else 4
                if child.in_features % pack_div == 0:
                    eff_gs = min(group_size, child.in_features)
                    while child.in_features % eff_gs != 0 and eff_gs > pack_div:
                        eff_gs //= 2
                    q_layer = nn.QuantizedLinear.from_float(child, bits=bits, group_size=eff_gs)
                    setattr(mod, name, q_layer)
            else:
                _replace(child)

    if isinstance(target, nn.Linear):
        pack_div = 8 if bits == 4 else 4
        eff_gs = min(group_size, target.in_features)
        while target.in_features % eff_gs != 0 and eff_gs > pack_div:
            eff_gs //= 2
        return nn.QuantizedLinear.from_float(target, bits=bits, group_size=eff_gs)

    _replace(target)
    return target


def cat(
    tensors: List[Tensor], dim: _builtins.int = 0, device: Optional[Union[str, Device]] = None
) -> Tensor:
    if not tensors:
        raise ValueError("cat expects a non-empty list of tensors")
    dev = Device(device) if device is not None else tensors[0].device
    ndim = tensors[0].ndim
    norm_dim = dim if dim >= 0 else dim + ndim
    req_grad = any(t.requires_grad for t in tensors) and is_grad_enabled()

    target_dev_id = -1 if dev.type == "cpu" else dev.index
    native_out = _backend.NativeTensor.cat([t._native for t in tensors], norm_dim, target_dev_id)
    out = Tensor._from_native(native_out, dev, req_grad)

    if req_grad:
        out.is_leaf = False
        split_sizes = [t.shape[norm_dim] for t in tensors]
        src_devices = [t.device for t in tensors]

        def cat_backward(grad_out: Tensor):
            grads = []
            curr = 0
            for sz, sdev in zip(split_sizes, src_devices):
                g_chunk = grad_out.narrow(norm_dim, curr, sz)
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

    with open(f, "wb") as fp:
        pickle.dump(obj, fp)


def load(f):
    import pickle

    with open(f, "rb") as fp:
        return pickle.load(fp)


# ----------------------------------------------------------------------------
# Contractions & Matrix Operations
# ----------------------------------------------------------------------------
def mm(input: Tensor, mat2: Tensor) -> Tensor:
    return input.matmul(mat2)


def bmm(input: Tensor, mat2: Tensor) -> Tensor:
    return input.bmm(mat2)


def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    beta: _builtins.float = 1.0,
    alpha: _builtins.float = 1.0,
) -> Tensor:
    return input * beta + (mat1 @ mat2) * alpha


def baddbmm(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    beta: _builtins.float = 1.0,
    alpha: _builtins.float = 1.0,
) -> Tensor:
    return input * beta + batch1.bmm(batch2) * alpha


def dot(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    if tensor1.dim() != 1 or tensor2.dim() != 1:
        raise RuntimeError("1D tensors expected for dot")
    return (tensor1 * tensor2).sum()


def mv(mat: Tensor, vec: Tensor) -> Tensor:
    if mat.dim() != 2 or vec.dim() != 1:
        raise RuntimeError("2D matrix and 1D vector expected for mv")
    return (mat @ vec.unsqueeze(1)).squeeze(1)


def outer(vec1: Tensor, vec2: Tensor) -> Tensor:
    if vec1.dim() != 1 or vec2.dim() != 1:
        raise RuntimeError("1D vectors expected for outer")
    return vec1.unsqueeze(1) @ vec2.unsqueeze(0)


def einsum(equation: str, *operands: Tensor) -> Tensor:
    """Evaluates Einstein summation convention on the operands."""
    import numpy as np

    eq = equation.replace(" ", "")
    ops_np = [op.cpu().numpy() for op in operands]
    res_np = np.einsum(eq, *ops_np)
    first_op = operands[0]
    return tensor(res_np, device=first_op.device, dtype=first_op.dtype)


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
    def synchronize(device_id: Union[_builtins.int, str, Device] = 0):
        if isinstance(device_id, (str, Device)):
            dev = Device(device_id)
            if dev.type == "cpu":
                return
            device_id = dev.index
        _backend.synchronize(_builtins.int(device_id))

    @staticmethod
    def synchronize_all():
        _backend.synchronize_all()

    @staticmethod
    def recorded_dispatches(device_id: _builtins.int = 0) -> _builtins.int:
        return _backend.recorded_dispatches(_builtins.int(device_id))

    @staticmethod
    def submitted_batches(device_id: _builtins.int = 0) -> _builtins.int:
        return _backend.submitted_batches(_builtins.int(device_id))

    @staticmethod
    def driver_allocation_count(device_id: _builtins.int = -1) -> _builtins.int:
        return _backend.driver_allocation_count(_builtins.int(device_id))

    @staticmethod
    def active_arena_bytes(device_id: _builtins.int = -1) -> _builtins.int:
        return _backend.active_arena_bytes(_builtins.int(device_id))

    @staticmethod
    def reserved_arena_bytes(device_id: _builtins.int = -1) -> _builtins.int:
        return _backend.reserved_arena_bytes(_builtins.int(device_id))

    @staticmethod
    def pooled_buffer_count() -> _builtins.int:
        return _backend.pooled_buffer_count()

    @staticmethod
    def live_allocation_count() -> _builtins.int:
        return _backend.live_allocation_count()

    @classmethod
    def get_device_tflops(
        cls,
        device_id: _builtins.int = 0,
        n: _builtins.int = 1024,
        force_recalibrate: _builtins.bool = False,
    ) -> _builtins.float:
        """Measures and caches effective FP32 GEMM throughput (TFLOPS) for `vk:<device_id>`."""
        key = (_builtins.int(device_id), _builtins.int(n))
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
            elapsed = _builtins.max(time.perf_counter() - t0, 1e-9)

        flops = 2.0 * (n**3) * reps
        tflops = (flops / elapsed) / 1e12
        cls._tflops_cache[key] = tflops
        return tflops

    @classmethod
    def compute_split_sizes(
        cls,
        batch_size: _builtins.int,
        device_ids: Optional[List[Union[str, Device]]] = None,
        weights: Optional[List[_builtins.float]] = None,
        min_chunk: _builtins.int = 1,
    ) -> List[_builtins.int]:
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
                    raw_weights.append(_builtins.max(cls.get_device_tflops(dev_obj.index), 1e-3))
                else:
                    raw_weights.append(0.05)
        else:
            raw_weights = [_builtins.max(_builtins.float(w), 0.0) for w in weights]

        total_w = _builtins.sum(raw_weights)
        if total_w <= 0.0:
            raw_weights = [1.0] * n_devs
            total_w = _builtins.float(n_devs)

        norm_w = [w / total_w for w in raw_weights]
        effective_min = min_chunk if batch_size >= n_devs * min_chunk else 0

        sizes = [_builtins.max(effective_min, _builtins.int(batch_size * w)) for w in norm_w]
        # Adjust remainder or excess according to fractional parts
        diff = batch_size - _builtins.sum(sizes)
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

# Export PyTorch-compatible DType singletons at package level
float = float32
int = int32
bool = _bool_dtype


# Friendly reminder for users looking for cuda
class _NoCudaNotice:
    def __getattr__(self, name):
        raise AttributeError(
            "VulTorch is built entirely on open Vulkan! "
            "No CUDA or ROCm is used. Use 'vultorch.vulkan' or device='vk' / 'vk:0' instead."
        )


cuda = _NoCudaNotice()
