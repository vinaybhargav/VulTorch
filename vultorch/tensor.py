"""
VulTorch Tensor Implementation
Hardware-accelerated tensor abstraction backed by Vulkan SPIR-V compute.
"""

import numpy as np
from typing import Union, Tuple, List, Optional
import os

from . import _vultorch_backend as backend
from .autograd import is_grad_enabled, Node, backward

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
    def __init__(self, type_or_str: Union[str, 'Device']):
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
            raise ValueError(f"Unknown device: {type_or_str}. Valid options: 'cpu', 'vk', 'vk:0', 'vk:1', etc.")

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

class Tensor:
    def __init__(self, data, device: Union[str, Device] = "vk", requires_grad: bool = False):
        dev = Device(device)
        self.device = dev
        self.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None
        self.grad_fn: Optional[Node] = None
        self.is_leaf = True

        if isinstance(data, backend.NativeTensor):
            self._native = data
        elif isinstance(data, np.ndarray):
            data_f32 = data.astype(np.float32, copy=False)
            self._native = backend.NativeTensor.from_numpy(data_f32, dev.device_id)
        elif isinstance(data, (list, tuple, float, int, np.number, np.generic)):
            np_data = np.array(data, dtype=np.float32)
            self._native = backend.NativeTensor.from_numpy(np_data, dev.device_id)
        elif isinstance(data, Tensor):
            self._native = data._native.clone()
            if data.device != dev:
                self._native = self._native.to_device(dev.device_id)
        else:
            raise TypeError(f"Unsupported data type for Tensor: {type(data)}")

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
    def dtype(self):
        return "float32"

    def numpy(self) -> np.ndarray:
        """Copies tensor data to host NumPy array."""
        return self._native.to_numpy()

    def to(self, device: Union[str, Device]) -> 'Tensor':
        target_dev = Device(device)
        if target_dev == self.device:
            return self
        new_native = self._native.to_device(target_dev.device_id)
        out = Tensor(new_native, device=target_dev, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            src_dev = self.device
            out.grad_fn = Node("ToDeviceBackward", lambda g: g.to(src_dev), [self])
        return out

    def vk(self, device_id: int = 0) -> 'Tensor':
        return self.to(f"vk:{device_id}")

    def cpu(self) -> 'Tensor':
        return self.to("cpu")

    def detach(self) -> 'Tensor':
        # Shares storage: no copy, no host round trip.
        out = Tensor(self._native.reshape_view(list(self.shape)), device=self.device, requires_grad=False)
        out.is_leaf = True
        out.grad_fn = None
        return out

    def __deepcopy__(self, memo):
        return Tensor(self.numpy(), device=self.device, requires_grad=self.requires_grad)

    def clone(self) -> 'Tensor':
        out = Tensor(self._native.clone(), device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("CloneBackward", lambda g: g.clone(), [self])
        return out

    def copy_(self, src: Union['Tensor', np.ndarray]):
        if isinstance(src, Tensor):
            if src.device == self.device:
                # Device-to-device: no host round trip.
                self._native.copy_from_native(src._native)
            else:
                self._native.copy_from_numpy(src.numpy())
        elif isinstance(src, np.ndarray):
            self._native.copy_from_numpy(src)
        else:
            self._native.copy_from_numpy(np.array(src, dtype=np.float32))
        return self

    def item(self) -> float:
        if self.numel != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return float(self.numpy().item())

    def backward(self, gradient: Optional['Tensor'] = None):
        backward(self, gradient)

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------------------
    # Arithmetic & Operators
    # ------------------------------------------------------------------------

    def __add__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            native_out = self._native.binary_op_scalar(0, float(other), False)
            req_grad = self.requires_grad and is_grad_enabled()
            out = Tensor(native_out, device=self.device, requires_grad=req_grad)
            if req_grad:
                out.is_leaf = False
                out.grad_fn = Node("AddScalarBackward", lambda g: g, [self])
            return out

        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        native_out = self._native.binary_op(0, other._native)
        req_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("AddBackward", lambda g: (g, g), [self, other])
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            native_out = self._native.binary_op_scalar(1, float(other), False)
            req_grad = self.requires_grad and is_grad_enabled()
            out = Tensor(native_out, device=self.device, requires_grad=req_grad)
            if req_grad:
                out.is_leaf = False
                out.grad_fn = Node("SubScalarBackward", lambda g: g, [self])
            return out

        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
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

    def __mul__(self, other) -> 'Tensor':
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
        native_out = self._native.binary_op(2, other._native)
        req_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("MulBackward", lambda g: (g * other, g * self), [self, other])
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            val = float(other)
            return self.__mul__(1.0 / val)

        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        native_out = self._native.binary_op(3, other._native)
        req_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("DivBackward", lambda g: (g / other, -g * self / (other * other)), [self, other])
        return out

    def __rtruediv__(self, other):
        other = Tensor(other, device=self.device)
        return other.__truediv__(self)

    def __neg__(self) -> 'Tensor':
        native_out = self._native.unary_op(0)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("NegBackward", lambda g: -g, [self])
        return out

    def __matmul__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        native_out = self._native.matmul(other._native, False, False, None)
        req_grad = (self.requires_grad or other.requires_grad) and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            def matmul_backward(grad_out: 'Tensor'):
                g_self = None
                g_other = None
                if self.requires_grad:
                    g_self = Tensor(grad_out._native.matmul(other._native, False, True, None), device=self.device)
                if other.requires_grad:
                    if self.ndim > 2 and other.ndim == 2:
                        k = self.shape[-1]
                        n = grad_out.shape[-1]
                        m_total = self.numel // k
                        x_2d = self.contiguous()._native.reshape_view([m_total, k])
                        g_2d = grad_out.contiguous()._native.reshape_view([m_total, n])
                        g_other = Tensor(x_2d.matmul(g_2d, True, False, None), device=self.device)
                    else:
                        g_other = Tensor(self._native.matmul(grad_out._native, True, False, None), device=self.device)
                return g_self, g_other
            out.grad_fn = Node("MatmulBackward", matmul_backward, [self, other])
        return out

    def __pow__(self, exponent: Union[int, float]) -> 'Tensor':
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

    def exp(self) -> 'Tensor':
        native_out = self._native.unary_op(1)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("ExpBackward", lambda g: g * out, [self])
        return out

    def log(self) -> 'Tensor':
        native_out = self._native.unary_op(2)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("LogBackward", lambda g: g / self, [self])
        return out

    def sqrt(self) -> 'Tensor':
        native_out = self._native.unary_op(3)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("SqrtBackward", lambda g: g / (2.0 * out), [self])
        return out

    def relu(self) -> 'Tensor':
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

    def gelu(self) -> 'Tensor':
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

    def silu(self) -> 'Tensor':
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

    def sigmoid(self) -> 'Tensor':
        native_out = self._native.unary_op(9)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("SigmoidBackward", lambda g: g * out * (1.0 - out), [self])
        return out

    def tanh(self) -> 'Tensor':
        native_out = self._native.unary_op(8)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            out.grad_fn = Node("TanhBackward", lambda g: g * (1.0 - out * out), [self])
        return out

    def softmax(self, dim: int = -1) -> 'Tensor':
        native_out = self._native.softmax(dim, False)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            def softmax_backward(g: 'Tensor'):
                # grad = s * (g - sum(g * s, dim, keepdim=True))
                sum_gs = (g * out).sum(dim=dim, keepdim=True)
                return out * (g - sum_gs)
            out.grad_fn = Node("SoftmaxBackward", softmax_backward, [self])
        return out

    def log_softmax(self, dim: int = -1) -> 'Tensor':
        native_out = self._native.softmax(dim, True)
        out = Tensor(native_out, device=self.device, requires_grad=self.requires_grad)
        if is_grad_enabled() and self.requires_grad:
            out.is_leaf = False
            def log_softmax_backward(g: 'Tensor'):
                # grad = g - exp(out) * sum(g, dim, keepdim=True)
                sum_g = g.sum(dim=dim, keepdim=True)
                return g - out.exp() * sum_g
            out.grad_fn = Node("LogSoftmaxBackward", log_softmax_backward, [self])
        return out

    # ------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------

    def sum(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> 'Tensor':
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
            def sum_dim_backward(g: 'Tensor'):
                if not keepdim:
                    g = g.unsqueeze(dim)
                return g.expand(orig_shape)
            out.grad_fn = Node("SumDimBackward", sum_dim_backward, [self])
        return out

    def mean(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> 'Tensor':
        s = self.sum(dim=dim, keepdim=keepdim)
        count = self.numel if dim is None else (self.shape[dim] if isinstance(dim, int) else np.prod([self.shape[d] for d in dim]))
        return s * (1.0 / count)

    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        native_dim = -999999 if dim is None else dim
        native_out = self._native.reduce_max(native_dim, keepdim)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            norm_dim = None if dim is None else (dim if dim >= 0 else dim + self.ndim)

            def max_backward(g: 'Tensor'):
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

                base = np.zeros(int(self.numel), dtype=np.float32)
                np.add.at(base, sel.reshape(-1), np.asarray(g.numpy()).reshape(-1))
                return Tensor(base.reshape(src_shape), device=self.device)

            out.grad_fn = Node("MaxBackward", max_backward, [self])
        return out

    # ------------------------------------------------------------------------
    # Shape Transformations
    # ------------------------------------------------------------------------

    def reshape(self, *new_shape) -> 'Tensor':
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

        # Shares storage: no copy and no host round trip.
        native = self._native.reshape_view(list(new_shape))
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            orig_shape = self.shape
            out.grad_fn = Node("ReshapeBackward", lambda g: g.reshape(orig_shape), [self])
        return out

    def view(self, *new_shape) -> 'Tensor':
        return self.reshape(*new_shape)

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        ndim = self.ndim
        d0 = dim0 if dim0 >= 0 else dim0 + ndim
        d1 = dim1 if dim1 >= 0 else dim1 + ndim
        if not (0 <= d0 < ndim and 0 <= d1 < ndim):
            raise IndexError(f"transpose dims ({dim0}, {dim1}) out of range for ndim={ndim}")

        perm = list(range(ndim))
        perm[d0], perm[d1] = perm[d1], perm[d0]

        shape, strides = self.shape, self.strides
        new_shape = [shape[p] for p in perm]
        src_strides = [strides[p] for p in perm]

        # Materialised on the device. A lazy strided view would avoid the copy,
        # but every kernel currently assumes contiguous inputs.
        native = self._native.strided_copy(new_shape, src_strides)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            out.grad_fn = Node("TransposeBackward", lambda g: g.transpose(d0, d1), [self])
        return out

    def t(self) -> 'Tensor':
        if self.ndim != 2:
            raise RuntimeError("t() expects a tensor with <= 2 dimensions")
        return self.transpose(0, 1)

    def unsqueeze(self, dim: int) -> 'Tensor':
        ndim = self.ndim + 1
        d = dim if dim >= 0 else dim + ndim
        if not (0 <= d < ndim):
            raise IndexError(f"unsqueeze dim {dim} out of range for ndim={ndim}")
        new_shape = list(self.shape)
        new_shape.insert(d, 1)

        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(self._native.reshape_view(new_shape), device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            orig_shape = self.shape
            out.grad_fn = Node("UnsqueezeBackward", lambda g: g.reshape(orig_shape), [self])
        return out

    def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
        if dim is None:
            new_shape = tuple(s for s in self.shape if s != 1)
        else:
            d = dim if dim >= 0 else dim + self.ndim
            if not (0 <= d < self.ndim):
                raise IndexError(f"squeeze dim {dim} out of range for ndim={self.ndim}")
            new_shape = tuple(s for i, s in enumerate(self.shape) if not (i == d and s == 1))
        if not new_shape:
            new_shape = (1,)

        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(self._native.reshape_view(list(new_shape)), device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            orig_shape = self.shape
            out.grad_fn = Node("SqueezeBackward", lambda g: g.reshape(orig_shape), [self])
        return out

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'Tensor':
        if end_dim < 0:
            end_dim += self.ndim
        new_shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim + 1:]
        return self.reshape(new_shape)

    def expand(self, *target_shape) -> 'Tensor':
        if len(target_shape) == 1 and isinstance(target_shape[0], (list, tuple)):
            target_shape = tuple(target_shape[0])
        if len(target_shape) < self.ndim:
            raise ValueError(
                f"expand: {len(target_shape)} dims given for a {self.ndim}-dim tensor"
            )

        # Right-align the source against the target.
        pad = len(target_shape) - self.ndim
        src_shape = (1,) * pad + self.shape
        src_strides = (0,) * pad + self.strides

        new_strides = []
        for size, want, stride in zip(src_shape, target_shape, src_strides):
            if size == want:
                new_strides.append(stride)
            elif size == 1:
                new_strides.append(0)  # broadcast axis
            else:
                raise ValueError(
                    f"expand: cannot expand dim of size {size} to {want}"
                )

        native = self._native.strided_copy(list(target_shape), new_strides)
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(native, device=self.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            # The engine reduces the broadcast gradient back to self.shape.
            out.grad_fn = Node("ExpandBackward", lambda g: g, [self])
        return out

    def contiguous(self) -> 'Tensor':
        return self

    # ------------------------------------------------------------------------
    # Indexing & Slicing
    # ------------------------------------------------------------------------

    def __getitem__(self, idx) -> 'Tensor':
        # Fast GPU path for contiguous outer-dimension slicing `x[start:stop]`
        if (
            isinstance(idx, slice)
            and (idx.step is None or idx.step == 1)
            and self.ndim >= 1
            and self._native.is_contiguous()
        ):
            start, stop, _ = idx.indices(self.shape[0])
            if stop > start:
                native_out = self._native.slice_dim0(start, stop)
                req_grad = self.requires_grad and is_grad_enabled()
                out = Tensor(native_out, device=self.device, requires_grad=req_grad)
                if req_grad:
                    out.is_leaf = False
                    src_shape = self.shape
                    dev = self.device

                    def slice_backward(g: 'Tensor'):
                        parts = []
                        if start > 0:
                            parts.append(Tensor(np.zeros((start,) + src_shape[1:], dtype=np.float32), device=dev))
                        parts.append(g)
                        if stop < src_shape[0]:
                            parts.append(Tensor(np.zeros((src_shape[0] - stop,) + src_shape[1:], dtype=np.float32), device=dev))
                        if len(parts) == 1:
                            return parts[0]
                        return Tensor(backend.NativeTensor.cat_dim0([p._native for p in parts], g._native.device_id), device=dev)

                    out.grad_fn = Node("SliceBackward", slice_backward, [self])
                return out

        arr = self.numpy()[idx]
        req_grad = self.requires_grad and is_grad_enabled()
        out = Tensor(arr, device=self.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            src_shape = self.shape
            numel = int(self.numel)

            def getitem_backward(g: 'Tensor'):
                # Scatter the upstream gradient back into a dense buffer. The
                # flat index map reproduces exactly the selection made in the
                # forward pass, so advanced indexing (array/mask) and repeated
                # indices are handled without special-casing.
                base = np.zeros(numel, dtype=np.float32)
                flat_map = np.arange(numel, dtype=np.int64).reshape(src_shape)
                sel = flat_map[idx]
                np.add.at(base, sel.reshape(-1), np.asarray(g.numpy()).reshape(-1))
                return Tensor(base.reshape(src_shape), device=self.device)

            out.grad_fn = Node("GetItemBackward", getitem_backward, [self])
        return out

    def __setitem__(self, idx, val):
        arr = self.numpy()
        arr[idx] = val.numpy() if isinstance(val, Tensor) else val
        self._native.copy_from_numpy(arr)

    def __repr__(self):
        arr_str = np.array2string(self.numpy(), separator=', ', precision=4, suppress_small=True)
        req = f", requires_grad=True" if self.requires_grad else ""
        dev = f", device='{self.device}'"
        return f"tensor({arr_str}{dev}{req})"
