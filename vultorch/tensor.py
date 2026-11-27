"""
VulTorch Tensor Implementation
Hardware-accelerated tensor abstraction backed by Vulkan SPIR-V compute.
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Any
import os
import threading

from . import _vultorch_backend as backend
from .autograd import is_grad_enabled, Node, backward

_device_state = threading.local()
_global_default_device: str = "vk"


def get_default_device() -> "Device":
    override = getattr(_device_state, "default_device", None)
    return Device(override if override is not None else _global_default_device)


def set_default_device(device: Union[str, "Device"]):
    global _global_default_device
    _global_default_device = str(Device(device))
    _device_state.default_device = _global_default_device


def resolve_device(device: Optional[Union[str, "Device"]] = None) -> "Device":
    if device is None:
        return get_default_device()
    return Device(device)


def _check_same_device(a: "Tensor", b: "Tensor", op_name: str = "operation"):
    if a.device != b.device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device for {op_name}, "
            f"but found '{a.device}' and '{b.device}'. "
            f"Call tensor.to('{a.device}') or model.to('{a.device}') to move them to the same device."
        )


# Shader lookup.
#
# Installed layout:   <site-packages>/vultorch/shaders/*.spv
# In-place checkout:  vultorch/shaders/*.spv (copied by the CMake dev target),
#                     falling back to <repo>/shaders for an unbuilt tree.
def _resolve_shader_directory() -> str:
    override = os.environ.get("VULTORCH_SHADER_DIR")
    if override:
        return override

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "shaders"),
        os.path.join(os.path.dirname(here), "shaders"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


backend.set_shader_directory(_resolve_shader_directory())


class Device:
    def __init__(self, type_or_str: Union[str, "Device"]):
        if isinstance(type_or_str, Device):
            self.type = type_or_str.type
            self.index = type_or_str.index
            return

        s = str(type_or_str).lower().strip()
        if s == "cpu":
            self.type = "cpu"
            self.index = -1
        elif s in ("vk", "vulkan"):
            self.type = "vk"
            self.index = 0
        elif s.startswith("vk:") or s.startswith("vulkan:"):
            parts = s.split(":")
            self.type = "vk"
            self.index = int(parts[1])
        else:
            raise ValueError(
                f"Unknown device: {type_or_str}. Valid options: 'cpu', 'vk', 'vk:0', 'vk:1', etc."
            )

    @property
    def device_id(self) -> int:
        return self.index if self.type == "vk" else -1

    def __repr__(self):
        if self.type == "cpu":
            return "device(type='cpu')"
        return f"device(type='vk', index={self.index})"

    def __str__(self):
        if self.type == "cpu":
            return "cpu"
        return f"vk:{self.index}"

    def __eq__(self, other):
        if isinstance(other, str):
            other = Device(other)
        if not isinstance(other, Device):
            return NotImplemented
        return self.type == other.type and self.index == other.index

    def __hash__(self):
        return hash((self.type, self.index))

    def __enter__(self):
        self._prev_dev = getattr(_device_state, "default_device", None)
        _device_state.default_device = str(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _device_state.default_device = self._prev_dev
        return False


def _canonical_dtype_name(name: str) -> str:
    s = name.strip().lower()
    aliases = {
        "float32": "float32",
        "float": "float32",
        "f32": "float32",
        "float16": "float16",
        "half": "float16",
        "f16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "int32": "int32",
        "int": "int32",
        "i32": "int32",
        "int64": "int64",
        "long": "int64",
        "i64": "int64",
        "bool": "bool",
        "bool_": "bool",
        "uint8": "uint8",
        "byte": "uint8",
        "u8": "uint8",
    }
    if s not in aliases:
        raise ValueError(f"Unsupported VulTorch dtype: '{name}'")
    return aliases[s]


class DType:
    """Descriptor for VulTorch tensor element data types."""

    __slots__ = ("name", "itemsize", "is_floating_point", "numpy_dtype")

    def __init__(self, name: str, itemsize: int, is_floating_point: bool, numpy_dtype: Any):
        self.name = name
        self.itemsize = itemsize
        self.is_floating_point = is_floating_point
        self.numpy_dtype = numpy_dtype

    def __repr__(self) -> str:
        return f"vultorch.{self.name}"

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        if isinstance(other, str):
            try:
                return self.name == _canonical_dtype_name(other)
            except ValueError:
                return False
        if isinstance(other, np.dtype) or (
            isinstance(other, type) and issubclass(other, np.generic)
        ):
            return self.numpy_dtype is not None and np.dtype(other) == np.dtype(self.numpy_dtype)
        return False


float32 = DType("float32", 4, True, np.float32)
float16 = DType("float16", 2, True, np.float16)
bfloat16 = DType("bfloat16", 2, True, None)
int32 = DType("int32", 4, False, np.int32)
int64 = DType("int64", 8, False, np.int64)
bool_dtype = DType("bool", 1, False, np.bool_)
uint8 = DType("uint8", 1, False, np.uint8)

half = float16
long = int64
byte = uint8

_DTYPE_REGISTRY = {
    "float32": float32,
    "float16": float16,
    "bfloat16": bfloat16,
    "int32": int32,
    "int64": int64,
    "bool": bool_dtype,
    "uint8": uint8,
}


def parse_dtype(dt: Union[str, DType, np.dtype, type, None]) -> DType:
    if dt is None:
        return float32
    if isinstance(dt, DType):
        return dt
    if isinstance(dt, str):
        return _DTYPE_REGISTRY[_canonical_dtype_name(dt)]
    if dt is float or dt is np.float32 or dt is np.float64:
        return float32
    if dt is int or dt is np.int64:
        return int64
    if dt is bool or dt is np.bool_:
        return bool_dtype
    if dt is np.float16:
        return float16
    if dt is np.int32:
        return int32
    if dt is np.uint8:
        return uint8
    if isinstance(dt, np.dtype):
        name = dt.name
        if name == "float64":
            return float32
        if name in _DTYPE_REGISTRY:
            return _DTYPE_REGISTRY[name]
    raise ValueError(f"Unsupported dtype: {dt!r}")


def _infer_list_bool(obj: Any) -> bool:
    if isinstance(obj, bool):
        return True
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return False
        return all(_infer_list_bool(x) for x in obj)
    return False


class Tensor:
    def __init__(
        self,
        data,
        device: Optional[Union[str, Device]] = None,
        requires_grad: bool = False,
        dtype: Union[str, DType, np.dtype, type, None] = None,
    ):
        dev = resolve_device(device)
        self.device = dev
        self.requires_grad = requires_grad
        self.grad: Optional["Tensor"] = None
        self.grad_fn: Optional[Node] = None
        self.is_leaf = True

        target_dt = parse_dtype(dtype) if dtype is not None else None

        if isinstance(data, backend.NativeTensor):
            native = data
            if target_dt is not None and native.dtype != target_dt.name:
                native = native.cast(target_dt.name)
            self._native = native
        elif isinstance(data, np.ndarray):
            if target_dt is not None:
                self._native = backend.NativeTensor.from_numpy(data, dev.device_id, target_dt.name)
            else:
                if data.dtype == np.float64:
                    data = data.astype(np.float32, copy=False)
                elif data.dtype not in (
                    np.float32,
                    np.float16,
                    np.int32,
                    np.int64,
                    np.bool_,
                    np.uint8,
                ):
                    data = data.astype(np.float32, copy=False)
                self._native = backend.NativeTensor.from_numpy(data, dev.device_id, "")
        elif isinstance(data, (list, tuple, float, int, bool, np.number, np.generic)):
            if target_dt is not None:
                np_dt = target_dt.numpy_dtype if target_dt.numpy_dtype is not None else np.float32
                np_data = np.array(data, dtype=np_dt)
                self._native = backend.NativeTensor.from_numpy(
                    np_data, dev.device_id, target_dt.name
                )
            else:
                if isinstance(data, (bool, np.bool_)) or (
                    isinstance(data, (list, tuple)) and _infer_list_bool(data)
                ):
                    np_data = np.array(data, dtype=np.bool_)
                elif isinstance(data, np.generic) and type(data) in (
                    np.float16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.bool_,
                ):
                    np_data = np.array(data, dtype=type(data))
                else:
                    np_data = np.array(data, dtype=np.float32)
                self._native = backend.NativeTensor.from_numpy(np_data, dev.device_id, "")
        elif isinstance(data, Tensor):
            native = data._native.clone()
            if data.device != dev:
                native = native.to_device(dev.device_id)
            if target_dt is not None and native.dtype != target_dt.name:
                native = native.cast(target_dt.name)
            self._native = native
        else:
            raise TypeError(f"Unsupported data type for Tensor: {type(data)}")

        if requires_grad and not self.dtype.is_floating_point:
            raise RuntimeError("Only Tensors of floating point dtype can require gradients")

    @classmethod
    def _from_native(cls, native, device_obj: Device, requires_grad: bool = False) -> "Tensor":
        out = cls.__new__(cls)
        out.device = device_obj
        out.requires_grad = requires_grad
        out.grad = None
        out.grad_fn = None
        out.is_leaf = True
        out._native = native
        return out

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._native.shape)

    @property
    def strides(self) -> Tuple[int, ...]:
        return tuple(self._native.strides)

    @property
    def ndim(self) -> int:
        return len(self._native.shape)

    @property
    def numel(self) -> int:
        return self._native.numel

    @property
    def dtype(self) -> DType:
        return _DTYPE_REGISTRY[self._native.dtype]

    @property
    def itemsize(self) -> int:
        return self._native.itemsize()

    def element_size(self) -> int:
        return self._native.itemsize()

    @property
    def nbytes(self) -> int:
        return self._native.nbytes()

    def storage_offset(self) -> int:
        return self._native.storage_offset

    def storage_id(self) -> int:
        return self._native.storage_id()

    def data_ptr(self) -> int:
        return self._native.data_ptr()

    def is_contiguous(self) -> bool:
        return self._native.is_contiguous()

    def numpy(self) -> np.ndarray:
        """Copies tensor data to host NumPy array."""
        return self._native.to_numpy()

    def _cast_dtype(self, dt: Union[str, DType, np.dtype, type]) -> "Tensor":
        target_dt = parse_dtype(dt)
        if target_dt == self.dtype:
            return self
        new_native = self._native.cast(target_dt.name)
        req_grad = self.requires_grad and target_dt.is_floating_point and is_grad_enabled()
        out = Tensor(new_native, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            src_dt = self.dtype
            out.grad_fn = Node("CastBackward", lambda g: g._cast_dtype(src_dt), [self])
        return out

    def to(self, *args, **kwargs) -> "Tensor":
        target_dev = kwargs.get("device", None)
        target_dtype = kwargs.get("dtype", None)
        for arg in args:
            if isinstance(arg, (Device, str)):
                if isinstance(arg, str):
                    try:
                        target_dtype = parse_dtype(arg)
                        continue
                    except ValueError:
                        pass
                target_dev = Device(arg)
            elif isinstance(arg, (DType, np.dtype)) or (
                isinstance(arg, type) and issubclass(arg, np.generic)
            ):
                target_dtype = parse_dtype(arg)
            else:
                raise TypeError(f"Unsupported argument to Tensor.to(): {arg!r}")

        res = self
        if target_dev is not None:
            dev = Device(target_dev)
            if dev != res.device:
                new_native = res._native.to_device(dev.device_id)
                req = res.requires_grad and is_grad_enabled()
                moved = Tensor(new_native, device=dev, requires_grad=req)
                if req:
                    moved.is_leaf = False
                    src_dev = res.device
                    moved.grad_fn = Node("ToDeviceBackward", lambda g: g.to(src_dev), [res])
                res = moved
        if target_dtype is not None:
            res = res._cast_dtype(target_dtype)
        return res

    def float(self) -> "Tensor":
        return self._cast_dtype(float32)

    def half(self) -> "Tensor":
        return self._cast_dtype(float16)

    def bfloat16(self) -> "Tensor":
        return self._cast_dtype(bfloat16)

    def int(self) -> "Tensor":
        return self._cast_dtype(int32)

    def long(self) -> "Tensor":
        return self._cast_dtype(int64)

    def bool(self) -> "Tensor":
        return self._cast_dtype(bool_dtype)

    def byte(self) -> "Tensor":
        return self._cast_dtype(uint8)

    def type(self, new_type: Optional[Union[str, DType]] = None):
        if new_type is None:
            return str(self.dtype)
        return self._cast_dtype(new_type)

    def vk(self, device_id: int = 0) -> "Tensor":
        return self.to(f"vk:{device_id}")

    def cpu(self) -> "Tensor":
        return self.to("cpu")

    def detach(self) -> "Tensor":
        # Zero-copy view sharing storage without tracking autograd history.
        native = self._native.as_strided(
            list(self.shape), list(self.strides), self._native.storage_offset
        )
        out = Tensor(native, device=self.device, requires_grad=False)
        out.is_leaf = True
        out.grad_fn = None
        return out

    def __deepcopy__(self, memo):
        return Tensor(self._native.clone(), device=self.device, requires_grad=self.requires_grad)

    def fill_(self, val: float) -> "Tensor":
        self._native.fill_(float(val))
        return self

    def zero_(self) -> "Tensor":
        self._native.zero_()
        return self

    def uniform_(self, low: float = 0.0, high: float = 1.0) -> "Tensor":
        import vultorch

        seed, offset = vultorch._next_philox_state()
        self._native.random_philox_(0, float(low), float(high), seed, offset, None)
        return self

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> "Tensor":
        import vultorch

        seed, offset = vultorch._next_philox_state()
        self._native.random_philox_(1, float(mean), float(std), seed, offset, None)
        return self

    def clone(self) -> "Tensor":
        out = Tensor(self._native.clone(), device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("CloneBackward", lambda g: g.clone(), [self])
        return out

    def copy_(self, src: Union["Tensor", np.ndarray]):
        if isinstance(src, Tensor):
            if src.device == self.device and src.dtype == self.dtype:
                # Device-to-device: no host round trip.
                self._native.copy_from_native(src._native)
            elif src.device == self.device:
                self._native.copy_from_native(src._native.cast(self.dtype.name))
            else:
                self._native.copy_from_numpy(src.numpy())
        elif isinstance(src, np.ndarray):
            self._native.copy_from_numpy(src)
        else:
            self._native.copy_from_numpy(np.asarray(src))
        return self

    def item(self):
        if self.numel != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return self.numpy().item()

    def backward(self, gradient: Optional["Tensor"] = None):
        backward(self, gradient)

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------------------
    # Arithmetic & Operators
    # ------------------------------------------------------------------------

    def __add__(self, other) -> "Tensor":
        if isinstance(other, (int, float)):
            native_out = self._native.binary_op_scalar(0, float(other), False)
            req_grad = self.requires_grad and is_grad_enabled()
            out = Tensor(native_out, device=self.device, requires_grad=req_grad)
            if req_grad:
                out.is_leaf = False
                out.grad_fn = Node("AddScalarBackward", lambda g: g, [self])
            return out

        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        _check_same_device(self, other, "add")
        native_out = self._native.binary_op(0, other._native)
        req_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("AddBackward", lambda g: (g, g), [self, other])
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other) -> "Tensor":
        if isinstance(other, (int, float)):
            native_out = self._native.binary_op_scalar(1, float(other), False)
            req_grad = self.requires_grad and is_grad_enabled()
            out = Tensor(native_out, device=self.device, requires_grad=req_grad)
            if req_grad:
                out.is_leaf = False
                out.grad_fn = Node("SubScalarBackward", lambda g: g, [self])
            return out

        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        _check_same_device(self, other, "sub")
        native_out = self._native.binary_op(1, other._native)
        req_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("SubBackward", lambda g: (g, -g), [self, other])
        return out

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            native_out = self._native.binary_op_scalar(1, float(other), True)
            req_grad = self.requires_grad and is_grad_enabled()
            out = Tensor(native_out, device=self.device, requires_grad=req_grad)
            if req_grad:
                out.is_leaf = False
                out.grad_fn = Node("RSubScalarBackward", lambda g: -g, [self])
            return out
        other = Tensor(other, device=self.device)
        return other.__sub__(self)

    def __mul__(self, other) -> "Tensor":
        if isinstance(other, (int, float)):
            val = float(other)
            native_out = self._native.binary_op_scalar(2, val, False)
            req_grad = self.requires_grad and is_grad_enabled()
            out = Tensor(native_out, device=self.device, requires_grad=req_grad)
            if req_grad:
                out.is_leaf = False
                out.grad_fn = Node("MulScalarBackward", lambda g: g * val, [self])
            return out

        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        _check_same_device(self, other, "mul")
        native_out = self._native.binary_op(2, other._native)
        req_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("MulBackward", lambda g: (g * other, g * self), [self, other])
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        if isinstance(other, (int, float)):
            val = float(other)
            return self.__mul__(1.0 / val)

        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        _check_same_device(self, other, "div")
        native_out = self._native.binary_op(3, other._native)
        req_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node(
                "DivBackward", lambda g: (g / other, -g * self / (other * other)), [self, other]
            )
        return out

    def __rtruediv__(self, other):
        other = Tensor(other, device=self.device)
        return other.__truediv__(self)

    def __neg__(self) -> "Tensor":
        native_out = self._native.unary_op(0)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("NegBackward", lambda g: -g, [self])
        return out

    def matmul(
        self,
        other: "Tensor",
        trans_a: bool = False,
        trans_b: bool = False,
        bias: Optional["Tensor"] = None,
        epilogue: int = 0,
    ) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        _check_same_device(self, other, "matmul")
        if bias is not None:
            _check_same_device(self, bias, "matmul bias")

        from .amp import is_autocast_enabled, get_autocast_dtype

        a_in = self
        b_in = other
        bias_in = bias
        if is_autocast_enabled():
            amp_dt = get_autocast_dtype()
            if a_in.dtype.is_floating_point and a_in.dtype != amp_dt:
                a_in = a_in.to(amp_dt)
            if b_in.dtype.is_floating_point and b_in.dtype != amp_dt:
                b_in = b_in.to(amp_dt)
            if bias_in is not None and bias_in.dtype.is_floating_point and bias_in.dtype != amp_dt:
                bias_in = bias_in.to(amp_dt)

        b_nat = bias_in._native if bias_in is not None else None
        native_out = a_in._native.matmul(
            b_in._native, bool(trans_a), bool(trans_b), b_nat, int(epilogue)
        )
        req_grad = (
            self.requires_grad or other.requires_grad or (bias is not None and bias.requires_grad)
        ) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            inputs = [self, other] + ([bias] if bias is not None else [])

            def matmul_backward(grad_out: "Tensor"):
                g_work = grad_out
                if g_work.dtype != a_in.dtype:
                    g_work = g_work.to(a_in.dtype)
                g_self = None
                g_other = None
                g_bias = None
                if self.requires_grad:
                    if not trans_a and not trans_b:
                        g_self = Tensor(
                            g_work._native.matmul(b_in._native, False, True, None),
                            device=self.device,
                        )
                    elif not trans_a and trans_b:
                        g_self = Tensor(
                            g_work._native.matmul(b_in._native, False, False, None),
                            device=self.device,
                        )
                    elif trans_a and not trans_b:
                        g_self = Tensor(
                            b_in._native.matmul(g_work._native, False, True, None),
                            device=self.device,
                        )
                    else:
                        g_self = Tensor(
                            b_in._native.matmul(g_work._native, True, True, None),
                            device=self.device,
                        )
                    if g_self.dtype != self.dtype:
                        g_self = g_self.to(self.dtype)
                if other.requires_grad:
                    if not trans_a and not trans_b:
                        if a_in.ndim > 2 and b_in.ndim == 2:
                            k = a_in.shape[-1]
                            n = g_work.shape[-1]
                            m_total = a_in.numel // k
                            x_2d = a_in.contiguous()._native.reshape_view([m_total, k])
                            g_2d = g_work.contiguous()._native.reshape_view([m_total, n])
                            g_other = Tensor(
                                x_2d.matmul(g_2d, True, False, None), device=self.device
                            )
                        else:
                            g_other = Tensor(
                                a_in._native.matmul(g_work._native, True, False, None),
                                device=self.device,
                            )
                    elif not trans_a and trans_b:
                        if a_in.ndim > 2 and b_in.ndim == 2:
                            k = a_in.shape[-1]
                            n = g_work.shape[-1]
                            m_total = a_in.numel // k
                            x_2d = a_in.contiguous()._native.reshape_view([m_total, k])
                            g_2d = g_work.contiguous()._native.reshape_view([m_total, n])
                            g_other = Tensor(
                                g_2d.matmul(x_2d, True, False, None), device=self.device
                            )
                        else:
                            g_other = Tensor(
                                g_work._native.matmul(a_in._native, True, False, None),
                                device=self.device,
                            )
                    elif trans_a and not trans_b:
                        g_other = Tensor(
                            a_in._native.matmul(g_work._native, False, False, None),
                            device=self.device,
                        )
                    else:
                        g_other = Tensor(
                            g_work._native.matmul(a_in._native, True, True, None),
                            device=self.device,
                        )
                    if g_other.dtype != other.dtype:
                        g_other = g_other.to(other.dtype)
                if bias is not None and bias.requires_grad:
                    n = g_work.shape[-1]
                    m_total = g_work.numel // n
                    g_2d = g_work.contiguous()._native.reshape_view([m_total, n])
                    g_bias = Tensor(g_2d.reduce_sum(0, False), device=self.device)
                    if g_bias.dtype != bias.dtype:
                        g_bias = g_bias.to(bias.dtype)
                if bias is not None:
                    return g_self, g_other, g_bias
                return g_self, g_other

            out.grad_fn = Node("MatmulBackward", matmul_backward, inputs)
        return out

    def __matmul__(self, other) -> "Tensor":
        return self.matmul(other)

    def __pow__(self, exponent: Union[int, float]) -> "Tensor":
        p = float(exponent)
        native_out = self._native.binary_op_scalar(4, p, False)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("PowBackward", lambda g: g * p * (self ** (p - 1)), [self])
        return out

    # ------------------------------------------------------------------------
    # Activations & Math Functions
    # ------------------------------------------------------------------------

    def exp(self) -> "Tensor":
        native_out = self._native.unary_op(1)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("ExpBackward", lambda g: g * out, [self])
        return out

    def log(self) -> "Tensor":
        native_out = self._native.unary_op(2)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("LogBackward", lambda g: g / self, [self])
        return out

    def sqrt(self) -> "Tensor":
        native_out = self._native.unary_op(3)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("SqrtBackward", lambda g: g / (2.0 * out), [self])
        return out

    def rsqrt(self) -> "Tensor":
        native_out = self._native.unary_op(4)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("RsqrtBackward", lambda g: -0.5 * g * out * out * out, [self])
        return out

    def abs(self) -> "Tensor":
        native_out = self._native.unary_op(5)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node(
                "AbsBackward",
                lambda g: Tensor(g._native.binary_op(13, self._native), device=self.device),
                [self],
            )
        return out

    def __abs__(self) -> "Tensor":
        return self.abs()

    def sign(self) -> "Tensor":
        native_out = self._native.unary_op(14)
        return Tensor(native_out, device=self.device, requires_grad=False)

    def _unary_simple(self, op_code: int, name: str, bwd_fn=None) -> "Tensor":
        native_out = self._native.unary_op(op_code)
        req_grad = (bwd_fn is not None) and self.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node(name, lambda g: bwd_fn(g, out), [self])
        return out

    def round(self) -> "Tensor":
        return self._unary_simple(16, "RoundBackward")

    def ceil(self) -> "Tensor":
        return self._unary_simple(17, "CeilBackward")

    def floor(self) -> "Tensor":
        return self._unary_simple(18, "FloorBackward")

    def trunc(self) -> "Tensor":
        return self._unary_simple(19, "TruncBackward")

    def frac(self) -> "Tensor":
        return self._unary_simple(20, "FracBackward", lambda g, _o: g)

    def square(self) -> "Tensor":
        return self._unary_simple(21, "SquareBackward", lambda g, _o: g * (2.0 * self))

    def reciprocal(self) -> "Tensor":
        return self._unary_simple(22, "ReciprocalBackward", lambda g, o: -g * o * o)

    def neg(self) -> "Tensor":
        return -self

    def pow(self, exponent) -> "Tensor":
        return self**exponent

    def exp2(self) -> "Tensor":
        ln2 = 0.6931471805599453
        return self._unary_simple(23, "Exp2Backward", lambda g, o: g * o * ln2)

    def expm1(self) -> "Tensor":
        return self._unary_simple(24, "Expm1Backward", lambda g, o: g * (o + 1.0))

    def log2(self) -> "Tensor":
        inv_ln2 = 1.4426950408889634
        return self._unary_simple(25, "Log2Backward", lambda g, _o: g * inv_ln2 / self)

    def log10(self) -> "Tensor":
        inv_ln10 = 0.4342944819032518
        return self._unary_simple(26, "Log10Backward", lambda g, _o: g * inv_ln10 / self)

    def log1p(self) -> "Tensor":
        return self._unary_simple(27, "Log1pBackward", lambda g, _o: g / (self + 1.0))

    def tan(self) -> "Tensor":
        return self._unary_simple(28, "TanBackward", lambda g, o: g * (1.0 + o * o))

    def asin(self) -> "Tensor":
        return self._unary_simple(29, "AsinBackward", lambda g, _o: g * (1.0 - self * self).rsqrt())

    def acos(self) -> "Tensor":
        return self._unary_simple(
            30, "AcosBackward", lambda g, _o: -g * (1.0 - self * self).rsqrt()
        )

    def atan(self) -> "Tensor":
        return self._unary_simple(31, "AtanBackward", lambda g, _o: g / (1.0 + self * self))

    def sinh(self) -> "Tensor":
        return self._unary_simple(32, "SinhBackward", lambda g, _o: g * self.cosh())

    def cosh(self) -> "Tensor":
        return self._unary_simple(33, "CoshBackward", lambda g, _o: g * self.sinh())

    def asinh(self) -> "Tensor":
        return self._unary_simple(
            34, "AsinhBackward", lambda g, _o: g * (self * self + 1.0).rsqrt()
        )

    def acosh(self) -> "Tensor":
        return self._unary_simple(
            35, "AcoshBackward", lambda g, _o: g * (self * self - 1.0).rsqrt()
        )

    def atanh(self) -> "Tensor":
        return self._unary_simple(36, "AtanhBackward", lambda g, _o: g / (1.0 - self * self))

    def erf(self) -> "Tensor":
        two_over_sqrt_pi = 1.1283791670955126
        return self._unary_simple(
            37, "ErfBackward", lambda g, _o: g * two_over_sqrt_pi * (-(self * self)).exp()
        )

    def erfc(self) -> "Tensor":
        two_over_sqrt_pi = 1.1283791670955126
        return self._unary_simple(
            38, "ErfcBackward", lambda g, _o: -g * two_over_sqrt_pi * (-(self * self)).exp()
        )

    def isnan(self) -> "Tensor":
        return Tensor(
            self._native.unary_op(39).cast("bool"), device=self.device, requires_grad=False
        )

    def isinf(self) -> "Tensor":
        return Tensor(
            self._native.unary_op(40).cast("bool"), device=self.device, requires_grad=False
        )

    def isfinite(self) -> "Tensor":
        return Tensor(
            self._native.unary_op(41).cast("bool"), device=self.device, requires_grad=False
        )

    def nan_to_num(
        self, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None
    ) -> "Tensor":
        import vultorch

        hi = 3.4028235e38 if posinf is None else float(posinf)
        lo = -3.4028235e38 if neginf is None else float(neginf)
        out = vultorch.where(self.isnan(), float(nan), self)
        pos_inf_mask = self.isinf() & (self > 0.0)
        neg_inf_mask = self.isinf() & (self < 0.0)
        out = vultorch.where(pos_inf_mask, hi, out)
        out = vultorch.where(neg_inf_mask, lo, out)
        return out

    def cast(self, dtype) -> "Tensor":
        dt = parse_dtype(dtype)
        if dt == self.dtype:
            return self
        return Tensor(
            self._native.cast(dt.name),
            device=self.device,
            requires_grad=self.requires_grad and dt.is_floating_point,
        )

    def __and__(self, other: "Tensor") -> "Tensor":
        return self.logical_and(other)

    def __or__(self, other: "Tensor") -> "Tensor":
        return self.logical_or(other)

    def __invert__(self) -> "Tensor":
        return self.logical_not()

    def maximum(self, other) -> "Tensor":
        other_t = (
            other
            if isinstance(other, Tensor)
            else Tensor(other, device=self.device, dtype=self.dtype)
        )
        if other_t.device != self.device:
            other_t = other_t.to(self.device)
        native_out = self._native.binary_op(5, other_t._native)
        req_grad = (self.requires_grad or other_t.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            import vultorch

            mask_a = self >= other_t

            def _max_bwd(g: "Tensor"):
                ga = vultorch.where(mask_a, g, 0.0) if self.requires_grad else None
                gb = vultorch.where(mask_a, 0.0, g) if other_t.requires_grad else None
                return ga, gb

            out.grad_fn = Node("MaximumBackward", _max_bwd, [self, other_t])
        return out

    def minimum(self, other) -> "Tensor":
        other_t = (
            other
            if isinstance(other, Tensor)
            else Tensor(other, device=self.device, dtype=self.dtype)
        )
        if other_t.device != self.device:
            other_t = other_t.to(self.device)
        native_out = self._native.binary_op(6, other_t._native)
        req_grad = (self.requires_grad or other_t.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            import vultorch

            mask_a = self <= other_t

            def _min_bwd(g: "Tensor"):
                ga = vultorch.where(mask_a, g, 0.0) if self.requires_grad else None
                gb = vultorch.where(mask_a, 0.0, g) if other_t.requires_grad else None
                return ga, gb

            out.grad_fn = Node("MinimumBackward", _min_bwd, [self, other_t])
        return out

    def lerp(self, end: "Tensor", weight) -> "Tensor":
        return self + weight * (end - self)

    def eq(self, other) -> "Tensor":
        return self == other

    def ne(self, other) -> "Tensor":
        return self != other

    def lt(self, other) -> "Tensor":
        return self < other

    def le(self, other) -> "Tensor":
        return self <= other

    def gt(self, other) -> "Tensor":
        return self > other

    def ge(self, other) -> "Tensor":
        return self >= other

    def logical_and(self, other: "Tensor") -> "Tensor":
        b = other if other.device == self.device else other.to(self.device)
        return ((self != 0.0).cast("float32") * (b != 0.0).cast("float32")) != 0.0

    def logical_or(self, other: "Tensor") -> "Tensor":
        b = other if other.device == self.device else other.to(self.device)
        return ((self != 0.0).cast("float32") + (b != 0.0).cast("float32")) != 0.0

    def logical_not(self) -> "Tensor":
        return self == 0.0

    def logical_xor(self, other: "Tensor") -> "Tensor":
        return (self.logical_or(other)).logical_and(self.logical_and(other).logical_not())

    def tolist(self):
        return self.numpy().tolist()

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def dim(self) -> int:
        return self.ndim

    def is_floating_point(self) -> bool:
        return self.dtype.is_floating_point

    def requires_grad_(self, mode: bool = True) -> "Tensor":
        if mode and not self.dtype.is_floating_point:
            raise RuntimeError("only Tensors of floating point dtype can require gradients")
        self.requires_grad = bool(mode)
        return self

    def retain_grad(self):
        pass

    def type_as(self, other: "Tensor") -> "Tensor":
        return self.to(other.device).cast(other.dtype)

    def add_(self, other) -> "Tensor":
        res = self + other
        self._native.copy_from_native(res._native)
        return self

    def sub_(self, other) -> "Tensor":
        res = self - other
        self._native.copy_from_native(res._native)
        return self

    def mul_(self, other) -> "Tensor":
        res = self * other
        self._native.copy_from_native(res._native)
        return self

    def div_(self, other) -> "Tensor":
        res = self / other
        self._native.copy_from_native(res._native)
        return self

    def min(self, dim: Optional[int] = None, keepdim: bool = False):
        if dim is None:
            return -Tensor(
                (-self)._native.reduce_max(-1, False), device=self.device, requires_grad=False
            )
        norm_dim = dim if dim >= 0 else dim + self.ndim
        return -Tensor(
            (-self)._native.reduce_max(norm_dim, keepdim), device=self.device, requires_grad=False
        )

    def amin(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        return self.min(dim=dim, keepdim=keepdim)

    def amax(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            return Tensor(
                self._native.reduce_max(-1, False), device=self.device, requires_grad=False
            )
        norm_dim = dim if dim >= 0 else dim + self.ndim
        return Tensor(
            self._native.reduce_max(norm_dim, keepdim), device=self.device, requires_grad=False
        )

    def argsort(self, dim: int = -1, descending: bool = False) -> "Tensor":
        _, idx = self.sort(dim=dim, descending=descending)
        return idx

    def sin(self) -> "Tensor":
        native_out = self._native.unary_op(6)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("SinBackward", lambda g: g * self.cos(), [self])
        return out

    def cos(self) -> "Tensor":
        native_out = self._native.unary_op(7)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("CosBackward", lambda g: -g * self.sin(), [self])
        return out

    def clamp(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> "Tensor":
        if min is None and max is None:
            raise ValueError("clamp requires at least one of min or max to be specified")
        lo = -3.402823466e38 if min is None else float(min)
        hi = 3.402823466e38 if max is None else float(max)
        native_out = self._native.unary_op(13, lo, hi)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            dev = self.device
            out_native = native_out
            in_native = self._native

            def clamp_backward(g: "Tensor"):
                # Mask is 1.0 where clamped output equals input (i.e. inside [min, max]), 0.0 where clamped
                mask_native = in_native.binary_op(14, out_native)
                if g._native.dtype != "float32":
                    mask_native = mask_native.cast(g._native.dtype)
                return Tensor(g._native.binary_op(2, mask_native), device=dev)

            out.grad_fn = Node("ClampBackward", clamp_backward, [self])
        return out

    def clip(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> "Tensor":
        return self.clamp(min=min, max=max)

    def clamp_(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> "Tensor":
        clamped = self.clamp(min=min, max=max)
        self._native.copy_from_native(clamped._native)
        return self

    __hash__ = object.__hash__

    def _coerce_other(self, other) -> Union[float, "Tensor"]:
        if isinstance(other, bool):
            return 1.0 if other else 0.0
        if isinstance(other, (int, float, np.number)):
            return float(other)
        if isinstance(other, Tensor):
            return other
        return Tensor(other, device=self.device)

    def __gt__(self, other) -> "Tensor":
        coerced = self._coerce_other(other)
        if isinstance(coerced, float):
            res = self._native.binary_op_scalar(7, coerced, False)
        else:
            res = self._native.binary_op(7, coerced._native)
        return Tensor(res.cast("bool"), device=self.device)

    def __lt__(self, other) -> "Tensor":
        coerced = self._coerce_other(other)
        if isinstance(coerced, float):
            res = self._native.binary_op_scalar(8, coerced, False)
        else:
            res = self._native.binary_op(8, coerced._native)
        return Tensor(res.cast("bool"), device=self.device)

    def __ge__(self, other) -> "Tensor":
        coerced = self._coerce_other(other)
        if isinstance(coerced, float):
            res = self._native.binary_op_scalar(15, coerced, False)
        else:
            res = self._native.binary_op(15, coerced._native)
        return Tensor(res.cast("bool"), device=self.device)

    def __le__(self, other) -> "Tensor":
        coerced = self._coerce_other(other)
        if isinstance(coerced, float):
            res = self._native.binary_op_scalar(16, coerced, False)
        else:
            res = self._native.binary_op(16, coerced._native)
        return Tensor(res.cast("bool"), device=self.device)

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, (Tensor, int, float, bool, np.ndarray, np.number, list, tuple)):
            return False
        coerced = self._coerce_other(other)
        if isinstance(coerced, float):
            res = self._native.binary_op_scalar(9, coerced, False)
        else:
            res = self._native.binary_op(9, coerced._native)
        return Tensor(res.cast("bool"), device=self.device)

    def __ne__(self, other):
        if other is None:
            return True
        if not isinstance(other, (Tensor, int, float, bool, np.ndarray, np.number, list, tuple)):
            return True
        coerced = self._coerce_other(other)
        if isinstance(coerced, float):
            eq_res = self._native.binary_op_scalar(9, coerced, False)
        else:
            eq_res = self._native.binary_op(9, coerced._native)
        ne_res = eq_res.binary_op_scalar(1, 1.0, True)
        return Tensor(ne_res.cast("bool"), device=self.device)

    def __bool__(self) -> bool:
        if self.numel == 1:
            return bool(self.numpy().item())
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")

    def relu(self) -> "Tensor":
        native_out = self._native.unary_op(10)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node(
                "ReluBackward",
                lambda g: Tensor(g._native.binary_op(10, self._native), device=self.device),
                [self],
            )
        return out

    def gelu(self) -> "Tensor":
        native_out = self._native.unary_op(11)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node(
                "GeluBackward",
                lambda g: Tensor(g._native.binary_op(11, self._native), device=self.device),
                [self],
            )
        return out

    def silu(self) -> "Tensor":
        native_out = self._native.unary_op(12)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node(
                "SiluBackward",
                lambda g: Tensor(g._native.binary_op(12, self._native), device=self.device),
                [self],
            )
        return out

    def sigmoid(self) -> "Tensor":
        native_out = self._native.unary_op(9)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("SigmoidBackward", lambda g: g * out * (1.0 - out), [self])
        return out

    def tanh(self) -> "Tensor":
        native_out = self._native.unary_op(8)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("TanhBackward", lambda g: g * (1.0 - out * out), [self])
        return out

    def softmax(self, dim: int = -1) -> "Tensor":
        native_out = self._native.softmax(dim, False)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False

            def softmax_backward(g: "Tensor"):
                # grad = s * (g - sum(g * s, dim, keepdim=True))
                sum_gs = (g * out).sum(dim=dim, keepdim=True)
                return out * (g - sum_gs)

            out.grad_fn = Node("SoftmaxBackward", softmax_backward, [self])
        return out

    def log_softmax(self, dim: int = -1) -> "Tensor":
        native_out = self._native.softmax(dim, True)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False

            def log_softmax_backward(g: "Tensor"):
                # grad = g - exp(out) * sum(g, dim, keepdim=True)
                sum_g = g.sum(dim=dim, keepdim=True)
                return g - out.exp() * sum_g

            out.grad_fn = Node("LogSoftmaxBackward", log_softmax_backward, [self])
        return out

    # ------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------

    def sum(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> "Tensor":
        if dim is None:
            native_out = self._native.reduce_sum(-999999, keepdim)
            req_grad = self.requires_grad and is_grad_enabled()
            out = Tensor(native_out, device=self.device, requires_grad=req_grad)
            if req_grad:
                out.is_leaf = False
                orig_shape = self.shape
                out.grad_fn = Node("SumAllBackward", lambda g: g.expand(orig_shape), [self])
            return out

        if isinstance(dim, (list, tuple)):
            res = self
            for d in sorted(dim, reverse=True):
                res = res.sum(dim=d, keepdim=keepdim)
            return res

        native_out = self._native.reduce_sum(dim, keepdim)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            orig_shape = self.shape

            def sum_dim_backward(g: "Tensor"):
                if not keepdim:
                    g = g.unsqueeze(dim)
                return g.expand(orig_shape)

            out.grad_fn = Node("SumDimBackward", sum_dim_backward, [self])
        return out

    def mean(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> "Tensor":
        s = self.sum(dim=dim, keepdim=keepdim)
        if not s.dtype.is_floating_point:
            s = s.float()
        count = (
            self.numel
            if dim is None
            else (
                self.shape[dim] if isinstance(dim, int) else np.prod([self.shape[d] for d in dim])
            )
        )
        return s * (1.0 / count)

    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        native_dim = -999999 if dim is None else dim
        native_out = self._native.reduce_max(native_dim, keepdim)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            norm_dim = None if dim is None else (dim if dim >= 0 else dim + self.ndim)

            def max_backward(g: "Tensor"):
                # Route the gradient to the first maximal element of each group,
                # matching torch.max(dim=...) semantics. Zeros elsewhere.
                x = self.numpy()
                if norm_dim is None:
                    sel = np.array([int(np.argmax(x))], dtype=np.int64)
                else:
                    flat_map = np.arange(int(self.numel), dtype=np.int64).reshape(src_shape)
                    am = np.argmax(x, axis=norm_dim)
                    sel = np.take_along_axis(
                        flat_map, np.expand_dims(am, norm_dim), axis=norm_dim
                    ).reshape(-1)

                base_dt = (
                    self.dtype.numpy_dtype if self.dtype.numpy_dtype is not None else np.float32
                )
                base = np.zeros(int(self.numel), dtype=base_dt)
                np.add.at(base, sel.reshape(-1), np.asarray(g.numpy(), dtype=base_dt).reshape(-1))
                return Tensor(base.reshape(src_shape), device=self.device, dtype=self.dtype)

            out.grad_fn = Node("MaxBackward", max_backward, [self])
        return out

    # ------------------------------------------------------------------------
    # Shape Transformations (Zero-Copy Views)
    # ------------------------------------------------------------------------

    def as_strided(
        self,
        size: Union[Tuple[int, ...], List[int]],
        stride: Union[Tuple[int, ...], List[int]],
        storage_offset: Optional[int] = None,
    ) -> "Tensor":
        off = self._native.storage_offset if storage_offset is None else int(storage_offset)
        native = self._native.as_strided([int(d) for d in size], [int(s) for s in stride], off)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            src_strides = self.strides
            src_numel = int(self.numel)
            out_shape = out.shape
            out_strides = out.strides
            src_dtype = self.dtype
            dev = self.device

            def as_strided_backward(g: "Tensor"):
                base_dt = src_dtype.numpy_dtype if src_dtype.numpy_dtype is not None else np.float32
                grad_buf = np.zeros(src_numel, dtype=base_dt)
                grid = (
                    np.meshgrid(*[np.arange(s, dtype=np.int64) for s in out_shape], indexing="ij")
                    if out_shape
                    else []
                )
                dst_idx = np.zeros(out_shape, dtype=np.int64)
                for coords, st in zip(grid, out_strides):
                    dst_idx = dst_idx + coords * st
                rel_idx = (off - self._native.storage_offset) + dst_idx
                # Map linear storage offsets back onto dense src indices when contiguous;
                # for arbitrary non-contiguous src views, scatter via element coordinates.
                if self.is_contiguous():
                    np.add.at(
                        grad_buf,
                        rel_idx.reshape(-1),
                        np.asarray(g.numpy(), dtype=base_dt).reshape(-1),
                    )
                else:
                    src_grid = np.meshgrid(
                        *[np.arange(s, dtype=np.int64) for s in src_shape], indexing="ij"
                    )
                    src_off = np.zeros(src_shape, dtype=np.int64) + self._native.storage_offset
                    for coords, st in zip(src_grid, src_strides):
                        src_off = src_off + coords * st
                    out_off = dst_idx + off
                    g_flat = np.asarray(g.numpy(), dtype=base_dt).reshape(-1)
                    out_flat = out_off.reshape(-1)
                    src_flat = src_off.reshape(-1)
                    for idx_val, g_val in zip(out_flat, g_flat):
                        grad_buf[src_flat == idx_val] += g_val
                return Tensor(grad_buf.reshape(src_shape), device=dev, dtype=src_dtype)

            out.grad_fn = Node("AsStridedBackward", as_strided_backward, [self])
        return out

    def reshape(self, *new_shape) -> "Tensor":
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])

        # Infer -1 dimension
        if -1 in new_shape:
            known_prod = 1
            infer_idx = -1
            for i, d in enumerate(new_shape):
                if d == -1:
                    infer_idx = i
                else:
                    known_prod *= d
            resolved_shape = list(new_shape)
            resolved_shape[infer_idx] = self.numel // known_prod
            new_shape = tuple(resolved_shape)

        # Shares storage without GPU dispatch whenever stride-compatible.
        native = self._native.reshape_view(list(new_shape))
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            orig_shape = self.shape
            out.grad_fn = Node("ReshapeBackward", lambda g: g.reshape(orig_shape), [self])
        return out

    def view(self, *new_shape) -> "Tensor":
        return self.reshape(*new_shape)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        ndim = self.ndim
        d0 = dim0 if dim0 >= 0 else dim0 + ndim
        d1 = dim1 if dim1 >= 0 else dim1 + ndim
        if not (0 <= d0 < ndim and 0 <= d1 < ndim):
            raise IndexError(f"transpose dims ({dim0}, {dim1}) out of range for ndim={ndim}")

        native = self._native.transpose_view(d0, d1)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("TransposeBackward", lambda g: g.transpose(d0, d1), [self])
        return out

    def t(self) -> "Tensor":
        if self.ndim <= 1:
            return self.as_strided(self.shape, self.strides)
        if self.ndim != 2:
            raise RuntimeError("t() expects a tensor with <= 2 dimensions")
        return self.transpose(0, 1)

    def permute(self, *dims) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        if len(dims) != self.ndim:
            raise ValueError(f"permute expected {self.ndim} dims, got {len(dims)}")
        norm_dims = [d if d >= 0 else d + self.ndim for d in dims]
        native = self._native.permute_view(norm_dims)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            inv_dims = [0] * self.ndim
            for i, d in enumerate(norm_dims):
                inv_dims[d] = i
            out.grad_fn = Node("PermuteBackward", lambda g: g.permute(inv_dims), [self])
        return out

    def narrow(self, dim: int, start: int, length: int) -> "Tensor":
        ndim = self.ndim
        d = dim if dim >= 0 else dim + ndim
        if not (0 <= d < ndim):
            raise IndexError(f"narrow dim {dim} out of range for ndim={ndim}")
        if length < 0 or start < 0 or start + length > self.shape[d]:
            raise IndexError(
                f"narrow(dim={dim}, start={start}, length={length}) invalid for shape {self.shape}"
            )

        native = self._native.slice_view(d, start, start + length, 1)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            dev = self.device
            stop = start + length

            def narrow_backward(g: "Tensor"):
                if d == 0:
                    parts = []
                    if start > 0:
                        z = backend.NativeTensor(
                            [start, *src_shape[1:]], g._native.device_id, g._native.dtype
                        )
                        z.zero_()
                        parts.append(z)
                    parts.append(g._native)
                    if stop < src_shape[0]:
                        z = backend.NativeTensor(
                            [src_shape[0] - stop, *src_shape[1:]],
                            g._native.device_id,
                            g._native.dtype,
                        )
                        z.zero_()
                        parts.append(z)
                    if len(parts) == 1:
                        return Tensor(parts[0], device=dev)
                    return Tensor(
                        backend.NativeTensor.cat_dim0(parts, g._native.device_id), device=dev
                    )

                # Swap dim `d` to 0 so concatenation stays on-device.
                g_t = g.transpose(0, d).contiguous()
                parts = []
                if start > 0:
                    pre_shape = list(g_t.shape)
                    pre_shape[0] = start
                    z = backend.NativeTensor(pre_shape, g._native.device_id, g._native.dtype)
                    z.zero_()
                    parts.append(z)
                parts.append(g_t._native)
                if stop < src_shape[d]:
                    post_shape = list(g_t.shape)
                    post_shape[0] = src_shape[d] - stop
                    z = backend.NativeTensor(post_shape, g._native.device_id, g._native.dtype)
                    z.zero_()
                    parts.append(z)
                combined_t = Tensor(
                    backend.NativeTensor.cat_dim0(parts, g._native.device_id), device=dev
                )
                return combined_t.transpose(0, d).contiguous()

            out.grad_fn = Node("NarrowBackward", narrow_backward, [self])
        return out

    def unsqueeze(self, dim: int) -> "Tensor":
        ndim = self.ndim + 1
        d = dim if dim >= 0 else dim + ndim
        if not (0 <= d < ndim):
            raise IndexError(f"unsqueeze dim {dim} out of range for ndim={ndim}")
        new_shape = list(self.shape)
        new_shape.insert(d, 1)
        new_strides = list(self.strides)
        stride_val = new_strides[d] if d < len(new_strides) else 1
        new_strides.insert(d, stride_val)

        native = self._native.as_strided(new_shape, new_strides, self._native.storage_offset)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            orig_shape = self.shape
            out.grad_fn = Node(
                "UnsqueezeBackward",
                lambda g: g.squeeze(d) if g.ndim == ndim else g.reshape(orig_shape),
                [self],
            )
        return out

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        if dim is None:
            keep = [(s, st) for s, st in zip(self.shape, self.strides) if s != 1]
            if not keep:
                keep = [(1, 1)]
        else:
            d = dim if dim >= 0 else dim + self.ndim
            if not (0 <= d < self.ndim):
                raise IndexError(f"squeeze dim {dim} out of range for ndim={self.ndim}")
            keep = [
                (s, st)
                for i, (s, st) in enumerate(zip(self.shape, self.strides))
                if not (i == d and s == 1)
            ]
            if not keep:
                keep = [(1, 1)]

        new_shape = [p[0] for p in keep]
        new_strides = [p[1] for p in keep]
        native = self._native.as_strided(new_shape, new_strides, self._native.storage_offset)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            orig_shape = self.shape
            out.grad_fn = Node("SqueezeBackward", lambda g: g.reshape(orig_shape), [self])
        return out

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        if end_dim < 0:
            end_dim += self.ndim
        new_shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim + 1 :]
        return self.reshape(new_shape)

    def expand(self, *target_shape) -> "Tensor":
        if len(target_shape) == 1 and isinstance(target_shape[0], (list, tuple)):
            target_shape = tuple(target_shape[0])
        try:
            native = self._native.expand_view([int(d) for d in target_shape])
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            # The engine reduces the broadcast gradient back to self.shape.
            out.grad_fn = Node("ExpandBackward", lambda g: g, [self])
        return out

    def contiguous(self) -> "Tensor":
        if self._native.is_contiguous():
            return self
        native = self._native.contiguous()
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("ContiguousBackward", lambda g: g, [self])
        return out

    # ------------------------------------------------------------------------
    # Indexing & Slicing (Zero-Copy View Fast Path)
    # ------------------------------------------------------------------------

    def _try_view_getitem(self, idx) -> Optional["Tensor"]:
        """Executes slice/integer/None/Ellipsis indexing as a zero-copy NativeTensor view."""
        if not isinstance(idx, tuple):
            items = [idx]
        else:
            items = list(idx)

        # Reject advanced indexing (lists, arrays, bool masks, Tensors)
        for item in items:
            if not (isinstance(item, (slice, int, np.integer)) or item is None or item is Ellipsis):
                return None
            if isinstance(item, slice) and item.step is not None and item.step <= 0:
                return None

        # Expand Ellipsis
        num_real = sum(1 for it in items if it is not None and it is not Ellipsis)
        ell_count = sum(1 for it in items if it is Ellipsis)
        if ell_count > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if ell_count == 1:
            ell_pos = items.index(Ellipsis)
            missing = max(0, self.ndim - num_real)
            items = items[:ell_pos] + [slice(None)] * missing + items[ell_pos + 1 :]

        # Pad trailing dimensions with `:`
        real_after = sum(1 for it in items if it is not None)
        if real_after > self.ndim:
            return None
        items.extend([slice(None)] * (self.ndim - real_after))

        # Apply slicing / integer indexing / None unsqueezing purely on metadata
        cur = self._native
        axis = 0
        for item in items:
            if item is None:
                cur_shape = list(cur.shape)
                cur_strides = list(cur.strides)
                st = cur_strides[axis] if axis < len(cur_strides) else 1
                cur_shape.insert(axis, 1)
                cur_strides.insert(axis, st)
                cur = cur.as_strided(cur_shape, cur_strides, cur.storage_offset)
                axis += 1
            elif isinstance(item, (int, np.integer)):
                size = cur.shape[axis]
                k = int(item)
                if k < 0:
                    k += size
                if k < 0 or k >= size:
                    raise IndexError(
                        f"index {item} is out of bounds for axis {axis} with size {size}"
                    )
                new_off = cur.storage_offset + k * cur.strides[axis]
                cur_shape = list(cur.shape)
                cur_strides = list(cur.strides)
                del cur_shape[axis]
                del cur_strides[axis]
                cur = cur.as_strided(cur_shape, cur_strides, new_off)
            else:
                start, stop, step = item.indices(cur.shape[axis])
                if stop <= start:
                    return None  # empty slice handled cleanly via fallback
                if not (start == 0 and stop == cur.shape[axis] and step == 1):
                    cur = cur.slice_view(axis, start, stop, step)
                axis += 1

        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(cur, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            src_dtype = self.dtype
            dev = self.device

            # Fast GPU SliceBackward for simple dim-0 contiguous slice[start:stop]
            if isinstance(idx, slice) and (idx.step is None or idx.step == 1) and self.ndim >= 1:
                start, stop, _ = idx.indices(src_shape[0])

                def slice_backward(g: "Tensor"):
                    parts = []
                    if start > 0:
                        z = backend.NativeTensor(
                            [start, *src_shape[1:]], g._native.device_id, g._native.dtype
                        )
                        z.zero_()
                        parts.append(z)
                    parts.append(g._native)
                    if stop < src_shape[0]:
                        z = backend.NativeTensor(
                            [src_shape[0] - stop, *src_shape[1:]],
                            g._native.device_id,
                            g._native.dtype,
                        )
                        z.zero_()
                        parts.append(z)
                    if len(parts) == 1:
                        return Tensor(parts[0], device=dev)
                    return Tensor(
                        backend.NativeTensor.cat_dim0(parts, g._native.device_id), device=dev
                    )

                out.grad_fn = Node("SliceBackward", slice_backward, [self])
            else:

                def view_getitem_backward(g: "Tensor"):
                    base_dt = (
                        src_dtype.numpy_dtype if src_dtype.numpy_dtype is not None else np.float32
                    )
                    grad_arr = np.zeros(src_shape, dtype=base_dt)
                    grad_arr[idx] = np.asarray(g.numpy(), dtype=base_dt)
                    return Tensor(grad_arr, device=dev, dtype=src_dtype)

                out.grad_fn = Node("GetItemBackward", view_getitem_backward, [self])

        return out

    def __getitem__(self, idx) -> "Tensor":
        if isinstance(idx, slice) and (idx.step is None or idx.step > 0):
            nshape = self._native.shape
            if len(nshape) >= 1:
                start, stop, step = idx.indices(nshape[0])
                if stop > start:
                    native = self._native.slice_view(0, start, stop, step)
                    req_grad = self.requires_grad and is_grad_enabled()
                    out = Tensor._from_native(native, self.device, req_grad)
                    if req_grad:
                        out.is_leaf = False
                        src_shape = tuple(nshape)
                        dev = self.device
                        if step == 1:

                            def slice_backward(g: "Tensor"):
                                parts = []
                                if start > 0:
                                    z = backend.NativeTensor(
                                        [start, *src_shape[1:]],
                                        g._native.device_id,
                                        g._native.dtype,
                                    )
                                    z.zero_()
                                    parts.append(z)
                                parts.append(g._native)
                                if stop < src_shape[0]:
                                    z = backend.NativeTensor(
                                        [src_shape[0] - stop, *src_shape[1:]],
                                        g._native.device_id,
                                        g._native.dtype,
                                    )
                                    z.zero_()
                                    parts.append(z)
                                if len(parts) == 1:
                                    return Tensor(parts[0], device=dev)
                                return Tensor(
                                    backend.NativeTensor.cat_dim0(parts, g._native.device_id),
                                    device=dev,
                                )

                            out.grad_fn = Node("SliceBackward", slice_backward, [self])
                        else:
                            src_dtype = self.dtype

                            def strided_slice_backward(g: "Tensor"):
                                base_dt = (
                                    src_dtype.numpy_dtype
                                    if src_dtype.numpy_dtype is not None
                                    else np.float32
                                )
                                grad_arr = np.zeros(src_shape, dtype=base_dt)
                                grad_arr[idx] = np.asarray(g.numpy(), dtype=base_dt)
                                return Tensor(grad_arr, device=dev, dtype=src_dtype)

                            out.grad_fn = Node("GetItemBackward", strided_slice_backward, [self])
                    return out

        view_out = self._try_view_getitem(idx)
        if view_out is not None:
            return view_out

        arr = self.numpy()[idx]
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(arr, device=self.device, dtype=self.dtype, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            numel = int(self.numel)
            src_dtype = self.dtype

            def getitem_backward(g: "Tensor"):
                base_dt = src_dtype.numpy_dtype if src_dtype.numpy_dtype is not None else np.float32
                base = np.zeros(numel, dtype=base_dt)
                flat_map = np.arange(numel, dtype=np.int64).reshape(src_shape)
                sel = flat_map[idx]
                np.add.at(base, sel.reshape(-1), np.asarray(g.numpy(), dtype=base_dt).reshape(-1))
                return Tensor(base.reshape(src_shape), device=self.device, dtype=src_dtype)

            out.grad_fn = Node("GetItemBackward", getitem_backward, [self])
        return out

    # ------------------------------------------------------------------------
    # Phase 3.1 & 3.2: Indexing, Conditional, Pad, Tril/Triu, Flip, Roll, Repeat
    # ------------------------------------------------------------------------

    def where(self, condition: "Tensor", other) -> "Tensor":
        import vultorch

        return vultorch.where(condition, self, other)

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        if self.device != mask.device:
            mask = mask.to(self.device)
        native_out = self._native.masked_fill(mask._native, float(value))
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            m_ref = mask.detach()

            def masked_fill_backward(g: "Tensor"):
                return g.masked_fill(m_ref, 0.0)

            out.grad_fn = Node("MaskedFillBackward", masked_fill_backward, [self])
        return out

    def masked_fill_(self, mask: "Tensor", value: float) -> "Tensor":
        if self.device != mask.device:
            mask = mask.to(self.device)
        native_out = self._native.masked_fill(mask._native, float(value))
        self._native.copy_from_native(native_out)
        return self

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        if self.device != index.device:
            index = index.to(self.device)
        norm_dim = dim if dim >= 0 else dim + self.ndim
        native_out = self._native.gather(norm_dim, index._native)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            src_dtype = self.dtype
            dev = self.device
            idx_ref = index.detach()

            def gather_backward(g: "Tensor"):
                z = backend.NativeTensor(list(src_shape), g._native.device_id, g._native.dtype)
                z.zero_()
                grad_nat = z.scatter(norm_dim, idx_ref._native, g._native, 0.0, False, True)
                return Tensor._from_native(grad_nat, dev, False)

            out.grad_fn = Node("GatherBackward", gather_backward, [self])
        return out

    def scatter(
        self, dim: int, index: "Tensor", src_or_val, reduce: Optional[str] = None
    ) -> "Tensor":
        if self.device != index.device:
            index = index.to(self.device)
        norm_dim = dim if dim >= 0 else dim + self.ndim
        acc = reduce == "add"
        if isinstance(src_or_val, Tensor):
            src_t = src_or_val if src_or_val.device == self.device else src_or_val.to(self.device)
            native_out = self._native.scatter(
                norm_dim, index._native, src_t._native, 0.0, False, acc
            )
        else:
            native_out = self._native.scatter(
                norm_dim, index._native, None, float(src_or_val), True, acc
            )
        return Tensor._from_native(native_out, self.device, False)

    def scatter_(
        self, dim: int, index: "Tensor", src_or_val, reduce: Optional[str] = None
    ) -> "Tensor":
        res = self.scatter(dim, index, src_or_val, reduce=reduce)
        self._native.copy_from_native(res._native)
        return self

    def scatter_add(self, dim: int, index: "Tensor", src: "Tensor") -> "Tensor":
        return self.scatter(dim, index, src, reduce="add")

    def scatter_add_(self, dim: int, index: "Tensor", src: "Tensor") -> "Tensor":
        return self.scatter_(dim, index, src, reduce="add")

    def index_select(self, dim: int, index: "Tensor") -> "Tensor":
        if self.device != index.device:
            index = index.to(self.device)
        norm_dim = dim if dim >= 0 else dim + self.ndim
        native_out = self._native.index_select(norm_dim, index._native)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            dev = self.device
            idx_ref = index.detach()

            def index_select_backward(g: "Tensor"):
                z = backend.NativeTensor(list(src_shape), g._native.device_id, g._native.dtype)
                z.zero_()
                z.index_add_(norm_dim, idx_ref._native, g._native, 1.0)
                return Tensor._from_native(z, dev, False)

            out.grad_fn = Node("IndexSelectBackward", index_select_backward, [self])
        return out

    def index_add_(
        self, dim: int, index: "Tensor", source: "Tensor", alpha: float = 1.0
    ) -> "Tensor":
        if self.device != index.device:
            index = index.to(self.device)
        if self.device != source.device:
            source = source.to(self.device)
        norm_dim = dim if dim >= 0 else dim + self.ndim
        self._native.index_add_(norm_dim, index._native, source._native, float(alpha))
        return self

    def tril(self, diagonal: int = 0) -> "Tensor":
        native_out = self._native.tril_triu(False, int(diagonal))
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            diag = int(diagonal)
            out.grad_fn = Node("TrilBackward", lambda g: g.tril(diag), [self])
        return out

    def triu(self, diagonal: int = 0) -> "Tensor":
        native_out = self._native.tril_triu(True, int(diagonal))
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            diag = int(diagonal)
            out.grad_fn = Node("TriuBackward", lambda g: g.triu(diag), [self])
        return out

    def flip(self, dims: Union[int, Tuple[int, ...], List[int]]) -> "Tensor":
        dim_list = [int(dims)] if isinstance(dims, int) else [int(d) for d in dims]
        native_out = self._native.flip(dim_list)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("FlipBackward", lambda g: g.flip(dim_list), [self])
        return out

    def roll(
        self,
        shifts: Union[int, Tuple[int, ...], List[int]],
        dims: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
    ) -> "Tensor":
        if dims is None:
            flat = self.reshape(-1)
            sh = int(shifts) if isinstance(shifts, int) else int(shifts[0])
            rolled = Tensor._from_native(flat._native.roll([sh], [0]), self.device, False)
            out = rolled.reshape(self.shape)
            if self.requires_grad and is_grad_enabled():
                out.requires_grad = True
                out.is_leaf = False
                out.grad_fn = Node("RollBackward", lambda g: g.roll(-sh), [self])
            return out
        sh_list = [int(shifts)] if isinstance(shifts, int) else [int(s) for s in shifts]
        d_list = [int(dims)] if isinstance(dims, int) else [int(d) for d in dims]
        native_out = self._native.roll(sh_list, d_list)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            neg_sh = [-s for s in sh_list]
            out.grad_fn = Node("RollBackward", lambda g: g.roll(neg_sh, d_list), [self])
        return out

    def split(
        self, split_size_or_sections: Union[int, List[int], Tuple[int, ...]], dim: int = 0
    ) -> Tuple["Tensor", ...]:
        norm_dim = dim if dim >= 0 else dim + self.ndim
        dim_sz = self.shape[norm_dim]
        if isinstance(split_size_or_sections, int):
            step = split_size_or_sections
            if step <= 0:
                raise ValueError("split_size must be > 0")
            sizes = [step] * (dim_sz // step)
            rem = dim_sz % step
            if rem > 0:
                sizes.append(rem)
        else:
            sizes = [int(s) for s in split_size_or_sections]
            if sum(sizes) != dim_sz:
                raise ValueError(f"split_with_sizes expects sum(sizes)=={dim_sz}, got {sum(sizes)}")
        res = []
        offset = 0
        for sz in sizes:
            res.append(self.narrow(norm_dim, offset, sz))
            offset += sz
        return tuple(res)

    def chunk(self, chunks: int, dim: int = 0) -> Tuple["Tensor", ...]:
        if chunks <= 0:
            raise ValueError("chunks must be > 0")
        norm_dim = dim if dim >= 0 else dim + self.ndim
        dim_sz = self.shape[norm_dim]
        chunk_sz = (dim_sz + chunks - 1) // chunks
        if chunk_sz == 0:
            return ()
        return self.split(chunk_sz, dim=norm_dim)

    def repeat(self, *sizes: int) -> "Tensor":
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        if len(sizes) < self.ndim:
            raise RuntimeError(
                "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor"
            )
        diff = len(sizes) - self.ndim
        x = self
        for _ in range(diff):
            x = x.unsqueeze(0)
        # Interleave unsqueeze + expand + reshape for each repeated dimension
        cur_shape = list(x.shape)
        inter_shape = []
        exp_shape = []
        final_shape = []
        for d, rep in enumerate(sizes):
            inter_shape.extend([1, cur_shape[d]])
            exp_shape.extend([int(rep), cur_shape[d]])
            final_shape.append(int(rep) * cur_shape[d])
        return x.reshape(*inter_shape).expand(*exp_shape).contiguous().reshape(*final_shape)

    def repeat_interleave(self, repeats: int, dim: Optional[int] = None) -> "Tensor":
        if dim is None:
            return self.reshape(-1).repeat_interleave(repeats, dim=0)
        norm_dim = dim if dim >= 0 else dim + self.ndim
        x = self.unsqueeze(norm_dim + 1)
        exp_shape = list(x.shape)
        exp_shape[norm_dim + 1] = int(repeats)
        out_shape = list(self.shape)
        out_shape[norm_dim] *= int(repeats)
        return x.expand(*exp_shape).contiguous().reshape(*out_shape)

    # ------------------------------------------------------------------------
    # Phase 3.3: Extended Reductions, Prefix Scans, Sort / Top-K
    # ------------------------------------------------------------------------

    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        native_dim = -999999 if dim is None else int(dim)
        return Tensor._from_native(
            self._native.argmax_argmin(native_dim, keepdim, False), self.device, False
        )

    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        native_dim = -999999 if dim is None else int(dim)
        return Tensor._from_native(
            self._native.argmax_argmin(native_dim, keepdim, True), self.device, False
        )

    def prod(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        native_dim = -999999 if dim is None else int(dim)
        native_out = self._native.reduce_prod_all_any(native_dim, keepdim, 0)
        return Tensor._from_native(native_out, self.device, False)

    def all(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        native_dim = -999999 if dim is None else int(dim)
        return Tensor._from_native(
            self._native.reduce_prod_all_any(native_dim, keepdim, 1), self.device, False
        )

    def any(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        native_dim = -999999 if dim is None else int(dim)
        return Tensor._from_native(
            self._native.reduce_prod_all_any(native_dim, keepdim, 2), self.device, False
        )

    def var(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        unbiased: bool = True,
        keepdim: bool = False,
    ) -> "Tensor":
        m = self.mean(dim=dim, keepdim=True)
        diff = self - m
        sq_sum = (diff * diff).sum(dim=dim, keepdim=keepdim)
        if dim is None:
            n = int(self.numel)
        elif isinstance(dim, int):
            n = int(self.shape[dim])
        else:
            n = int(np.prod([self.shape[d] for d in dim]))
        denom = max(n - 1, 1) if unbiased else max(n, 1)
        return sq_sum * (1.0 / denom)

    def std(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        unbiased: bool = True,
        keepdim: bool = False,
    ) -> "Tensor":
        return self.var(dim=dim, unbiased=unbiased, keepdim=keepdim).sqrt()

    def norm(
        self,
        p: float = 2.0,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
    ) -> "Tensor":
        return (self.abs() ** p).sum(dim=dim, keepdim=keepdim) ** (1.0 / p)

    def cumsum(self, dim: int) -> "Tensor":
        norm_dim = dim if dim >= 0 else dim + self.ndim
        native_out = self._native.scan_op(norm_dim, 0)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node(
                "CumsumBackward",
                lambda g: g.flip([norm_dim]).cumsum(norm_dim).flip([norm_dim]),
                [self],
            )
        return out

    def cumprod(self, dim: int) -> "Tensor":
        norm_dim = dim if dim >= 0 else dim + self.ndim
        native_out = self._native.scan_op(norm_dim, 1)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            out_ref = out.detach()
            x_ref = self.detach()

            def cumprod_backward(g: "Tensor"):
                rev = (g * out_ref).flip([norm_dim]).cumsum(norm_dim).flip([norm_dim])
                return rev / x_ref

            out.grad_fn = Node("CumprodBackward", cumprod_backward, [self])
        return out

    def sort(self, dim: int = -1, descending: bool = False):
        from collections import namedtuple

        SortResult = namedtuple("SortResult", ["values", "indices"])
        norm_dim = dim if dim >= 0 else dim + self.ndim
        val_nat, idx_nat = self._native.sort_topk(norm_dim, -1, bool(descending))
        req_grad = self.requires_grad and is_grad_enabled()
        vals = Tensor._from_native(val_nat, self.device, req_grad)
        inds = Tensor._from_native(idx_nat, self.device, False)
        if req_grad:
            vals.is_leaf = False
            src_shape = self.shape
            dev = self.device
            idx_ref = inds

            def sort_backward(g: "Tensor"):
                z = backend.NativeTensor(list(src_shape), g._native.device_id, g._native.dtype)
                z.zero_()
                grad_nat = z.scatter(norm_dim, idx_ref._native, g._native, 0.0, False, True)
                return Tensor._from_native(grad_nat, dev, False)

            vals.grad_fn = Node("SortBackward", sort_backward, [self])
        return SortResult(vals, inds)

    def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        from collections import namedtuple

        TopkResult = namedtuple("TopkResult", ["values", "indices"])
        norm_dim = dim if dim >= 0 else dim + self.ndim
        val_nat, idx_nat = self._native.sort_topk(norm_dim, int(k), bool(largest))
        req_grad = self.requires_grad and is_grad_enabled()
        vals = Tensor._from_native(val_nat, self.device, req_grad)
        inds = Tensor._from_native(idx_nat, self.device, False)
        if req_grad:
            vals.is_leaf = False
            src_shape = self.shape
            dev = self.device
            idx_ref = inds

            def topk_backward(g: "Tensor"):
                z = backend.NativeTensor(list(src_shape), g._native.device_id, g._native.dtype)
                z.zero_()
                grad_nat = z.scatter(norm_dim, idx_ref._native, g._native, 0.0, False, True)
                return Tensor._from_native(grad_nat, dev, False)

            vals.grad_fn = Node("TopkBackward", topk_backward, [self])
        return TopkResult(vals, inds)

    def _snake_mode(
        self, mode: int, a: float = 1.0, gamma: float = 0.0, eps: float = 1e-9
    ) -> "Tensor":
        native_out = self._native.snake(
            None, None, 1, False, float(a), float(eps), int(mode), None, float(gamma)
        )
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor._from_native(native_out, self.device, req_grad)
        if req_grad:
            out.is_leaf = False
            a_val = float(a)
            g_val = float(gamma)
            eps_val = float(eps)
            mode_val = int(mode)
            dev = self.device

            def snake_scalar_backward(g: "Tensor"):
                dx_nat, _, _, _ = self._native.snake_backward(
                    g.contiguous()._native,
                    None,
                    None,
                    1,
                    False,
                    a_val,
                    eps_val,
                    True,
                    False,
                    False,
                    mode_val,
                    None,
                    g_val,
                    False,
                )
                return Tensor._from_native(dx_nat, dev, False)

            out.grad_fn = Node("SnakeBackward", snake_scalar_backward, [self])
        return out

    def snake(self, a: float = 1.0, eps: float = 1e-9) -> "Tensor":
        return self._snake_mode(0, a=a, eps=eps)

    def snake_gamma(self, a: float = 1.0, gamma: float = 0.0, eps: float = 1e-9) -> "Tensor":
        return self._snake_mode(1, a=a, gamma=gamma, eps=eps)

    def snake_cos(self, a: float = 1.0, eps: float = 1e-9) -> "Tensor":
        return self._snake_mode(2, a=a, eps=eps)

    def snake_rational(self, a: float = 1.0, eps: float = 1e-9) -> "Tensor":
        return self._snake_mode(3, a=a, eps=eps)

    def snake_sinc(self, a: float = 1.0, eps: float = 1e-9) -> "Tensor":
        return self._snake_mode(4, a=a, eps=eps)

    def __setitem__(self, idx, val):
        arr = self.numpy()
        arr[idx] = val.numpy() if isinstance(val, Tensor) else val
        self._native.copy_from_numpy(arr)

    def __repr__(self):
        arr_str = np.array2string(self.numpy(), separator=", ", precision=4, suppress_small=True)
        req = ", requires_grad=True" if self.requires_grad else ""
        dt = f", dtype={self.dtype.name}" if self.dtype != float32 else ""
        dev = f", device='{self.device}'"
        return f"tensor({arr_str}{dev}{dt}{req})"
