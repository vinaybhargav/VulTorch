"""
VulTorch Neural Network Library (vultorch.nn)
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Iterator
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import vultorch
from ..tensor import Tensor, Device, resolve_device, _check_same_device
from ..autograd import is_grad_enabled, no_grad, Node

class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""
    def __init__(self, data: Union[Tensor, np.ndarray], requires_grad: bool = True):
        if isinstance(data, Tensor):
            super().__init__(data._native, device=data.device, requires_grad=requires_grad)
        else:
            super().__init__(data, device=resolve_device(None), requires_grad=requires_grad)

class Module:
    def __init__(self):
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self.training = True

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def register_parameter(self, name: str, param: Optional[Parameter]):
        self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Optional[Tensor]):
        self._buffers[name] = tensor
        super().__setattr__(name, tensor)

    def register_module(self, name: str, module: Optional['Module']):
        self._modules[name] = module

    def __setattr__(self, name: str, value):
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        super().__setattr__(name, value)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for param in self._parameters.values():
            if param is not None:
                yield param
        if recurse:
            for module in self._modules.values():
                if module is not None:
                    yield from module.parameters(recurse=True)

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            if param is not None:
                yield f"{prefix}.{name}" if prefix else name, param
        if recurse:
            for mod_name, module in self._modules.items():
                if module is not None:
                    sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                    yield from module.named_parameters(prefix=sub_prefix, recurse=True)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        for name, buf in getattr(self, "_buffers", {}).items():
            if buf is not None:
                yield f"{prefix}.{name}" if prefix else name, buf
        if recurse:
            for mod_name, module in self._modules.items():
                if module is not None:
                    sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                    yield from module.named_buffers(prefix=sub_prefix, recurse=True)

    def modules(self) -> Iterator['Module']:
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True) -> Iterator[Tuple[str, 'Module']]:
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is not None:
                    sub_prefix = f"{prefix}.{name}" if prefix else name
                    yield from module.named_modules(memo, sub_prefix, remove_duplicate)

    def to(self, device: Union[str, Device]) -> 'Module':
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = Parameter(param.to(device), requires_grad=param.requires_grad)
                super().__setattr__(name, self._parameters[name])
        for name, buf in getattr(self, "_buffers", {}).items():
            if buf is not None:
                self._buffers[name] = buf.to(device)
                super().__setattr__(name, self._buffers[name])
        for module in self._modules.values():
            if module is not None:
                module.to(device)
        return self

    def train(self, mode: bool = True) -> 'Module':
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self

    def eval(self) -> 'Module':
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> OrderedDict:
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param.detach().clone()
        return state

    def load_state_dict(self, state_dict: dict):
        for name, param in self.named_parameters():
            if name in state_dict:
                src = state_dict[name]
                param.copy_(src)

    def multi_gpu(
        self,
        device_ids: Optional[Union[str, List[str]]] = None,
        balance: Union[bool, str] = "tflops",
        device_weights: Optional[List[float]] = None,
    ) -> 'MultiGPU':
        """Wraps this module in `nn.MultiGPU` across `device_ids` (defaulting to `['all']`)."""
        return MultiGPU(self, device_ids=device_ids, balance=balance, device_weights=device_weights)

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.register_module(str(idx), module)

    def __getitem__(self, idx: int) -> Module:
        if idx < 0:
            idx += len(self._modules)
        return self._modules[str(idx)]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

class ModuleList(Module):
    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        if modules is not None:
            for idx, m in enumerate(modules):
                self.register_module(str(idx), m)

    def __getitem__(self, idx: int) -> Module:
        return self._modules[str(idx)]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, epilogue: Optional[str] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epilogue = epilogue
        dev = resolve_device(device)

        # Kaiming uniform initialization
        bound = 1.0 / math.sqrt(in_features)
        w = vultorch.empty((out_features, in_features), device=dev).uniform_(-bound, bound)
        self.weight = Parameter(w, requires_grad=True)

        if bias:
            b = vultorch.empty((out_features,), device=dev).uniform_(-bound, bound)
            self.bias = Parameter(b, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        _check_same_device(x, self.weight, "Linear")
        if self.epilogue is not None:
            return functional.linear(x, self.weight, self.bias, epilogue=self.epilogue)

        from ..amp import is_autocast_enabled, get_autocast_dtype
        x_in = x
        w_in = self.weight
        b_in = self.bias
        if is_autocast_enabled():
            amp_dt = get_autocast_dtype()
            if x_in.dtype != amp_dt:
                x_in = x_in.to(amp_dt)
            if w_in.dtype != amp_dt:
                w_in = w_in.to(amp_dt)
            if b_in is not None and b_in.dtype != amp_dt:
                b_in = b_in.to(amp_dt)

        # Fused y = x @ W^T + b
        b_native = b_in._native if b_in is not None else None
        orig_shape = x_in.shape
        if x_in.ndim > 2:
            m_total = x_in.numel // self.in_features
            x_2d = x_in.contiguous()._native.reshape_view([m_total, self.in_features])
            out_2d = x_2d.matmul(w_in._native, False, True, b_native)
            native_out = out_2d.reshape_view(list(orig_shape[:-1]) + [self.out_features])
        else:
            x_2d = x_in._native
            native_out = x_2d.matmul(w_in._native, False, True, b_native)

        req_grad = (
            x.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        ) and is_grad_enabled()

        out = Tensor(native_out, device=x.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            weight = self.weight
            bias = self.bias
            in_feat = self.in_features
            out_feat = self.out_features

            def linear_backward(grad_out: Tensor):
                g_work = grad_out.to(w_in.dtype) if grad_out.dtype != w_in.dtype else grad_out
                if g_work.ndim > 2:
                    m_tot = g_work.numel // out_feat
                    g_2d = g_work.contiguous()._native.reshape_view([m_tot, out_feat])
                    x_mat = x_in.contiguous()._native.reshape_view([m_tot, in_feat])
                else:
                    g_2d = g_work._native
                    x_mat = x_in._native

                g_x = None
                g_w = None
                g_b = None

                if x.requires_grad:
                    gx_2d = g_2d.matmul(w_in._native, False, False, None)
                    if len(orig_shape) > 2:
                        gx_2d = gx_2d.reshape_view(list(orig_shape))
                    g_x = Tensor(gx_2d, device=x.device)
                    if g_x.dtype != x.dtype:
                        g_x = g_x.to(x.dtype)

                if weight.requires_grad:
                    g_w = Tensor(g_2d.matmul(x_mat, True, False, None), device=x.device)
                    if g_w.dtype != weight.dtype:
                        g_w = g_w.to(weight.dtype)

                if bias is not None and bias.requires_grad:
                    g_b = Tensor(g_2d.reduce_sum(0, False), device=x.device)
                    if g_b.dtype != bias.dtype:
                        g_b = g_b.to(bias.dtype)
                    return g_x, g_w, g_b

                if bias is not None:
                    return g_x, g_w, None
                return g_x, g_w

            inputs = [x, self.weight]
            if self.bias is not None:
                inputs.append(self.bias)
            out.grad_fn = Node("LinearBackward", linear_backward, inputs)
        return out

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 groups: int = 1,
                 bias: bool = True, device=None):
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride_h, self.stride_w = (stride, stride) if isinstance(stride, int) else stride
        self.pad_h, self.pad_w = (padding, padding) if isinstance(padding, int) else padding
        self.groups = int(groups)
        dev = resolve_device(device)

        # Kaiming init
        c_in_per_g = in_channels // self.groups
        k = 1.0 / math.sqrt(c_in_per_g * self.kH * self.kW)
        w = vultorch.empty((out_channels, c_in_per_g, self.kH, self.kW), device=dev).uniform_(-k, k)
        self.weight = Parameter(w, requires_grad=True)

        if bias:
            b = vultorch.empty((out_channels,), device=dev).uniform_(-k, k)
            self.bias = Parameter(b, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        _check_same_device(x, self.weight, "Conv2d")
        from ..amp import is_autocast_enabled, get_autocast_dtype
        x_in = x
        w_in = self.weight
        b_in = self.bias
        if is_autocast_enabled():
            amp_dt = get_autocast_dtype()
            if x_in.dtype != amp_dt:
                x_in = x_in.to(amp_dt)
            if w_in.dtype != amp_dt:
                w_in = w_in.to(amp_dt)
            if b_in is not None and b_in.dtype != amp_dt:
                b_in = b_in.to(amp_dt)

        b_native = b_in._native if b_in is not None else None
        native_out = x_in._native.conv2d(w_in._native, b_native,
                                         self.stride_h, self.stride_w,
                                         self.pad_h, self.pad_w,
                                         self.groups, True)
        req_grad = x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)
        out = Tensor(native_out, device=x.device, requires_grad=req_grad)

        if is_grad_enabled() and req_grad:
            out.is_leaf = False
            sh, sw, ph, pw, grps = self.stride_h, self.stride_w, self.pad_h, self.pad_w, self.groups
            def conv_backward(grad_out: Tensor):
                g_in = None
                g_w = None
                g_b = None

                if x.requires_grad:
                    gin_native = grad_out._native.conv2d_backward_input(
                        self.weight._native, x.shape, sh, sw, ph, pw, grps, True)
                    g_in = Tensor(gin_native, device=x.device)

                if self.weight.requires_grad:
                    gw_native = grad_out._native.conv2d_backward_weight(
                        x._native, self.weight.shape, sh, sw, ph, pw, grps, True)
                    g_w = Tensor(gw_native, device=x.device)

                if self.bias is not None and self.bias.requires_grad:
                    # Sum grad_out over N, H, W
                    g_b = grad_out.sum(dim=(0, 2, 3))

                return g_in, g_w, g_b

            inputs = [x, self.weight]
            if self.bias is not None:
                inputs.append(self.bias)
            out.grad_fn = Node("Conv2dBackward", conv_backward, inputs)
        return out

class Conv1d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1,
                 bias: bool = True, device=None):
        super().__init__()
        self.conv2d = Conv2d(in_channels, out_channels, (kernel_size, 1),
                             stride=(stride, 1), padding=(padding, 0),
                             groups=groups, bias=bias, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2d(x.unsqueeze(-1)).squeeze(-1)

class ConvTranspose2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride_h, self.stride_w = (stride, stride) if isinstance(stride, int) else stride
        self.pad_h, self.pad_w = (padding, padding) if isinstance(padding, int) else padding
        dev = resolve_device(device)

        k = 1.0 / math.sqrt(out_channels * self.kH * self.kW)
        w = vultorch.empty((in_channels, out_channels, self.kH, self.kW), device=dev).uniform_(-k, k)
        self.weight = Parameter(w, requires_grad=True)

        if bias:
            b = vultorch.empty((out_channels,), device=dev).uniform_(-k, k)
            self.bias = Parameter(b, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, _, H_in, W_in = x.shape
        H_out = (H_in - 1) * self.stride_h - 2 * self.pad_h + self.kH
        W_out = (W_in - 1) * self.stride_w - 2 * self.pad_w + self.kW
        out_shape = (N, self.out_channels, H_out, W_out)

        native_out = x._native.conv2d_backward_input(
            self.weight._native, list(out_shape),
            self.stride_h, self.stride_w, self.pad_h, self.pad_w, 1, True
        )
        req_grad = (x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)) and is_grad_enabled()
        out = Tensor(native_out, device=x.device, requires_grad=req_grad)

        if req_grad:
            out.is_leaf = False
            sh, sw, ph, pw = self.stride_h, self.stride_w, self.pad_h, self.pad_w
            w_ref = self.weight
            b_ref = self.bias
            dev = x.device

            def conv_t_backward(grad_out: Tensor):
                g_in = None
                g_w = None
                g_b = None

                if x.requires_grad:
                    gin_nat = grad_out._native.conv2d(
                        w_ref._native, None, sh, sw, ph, pw, 1, True
                    )
                    g_in = Tensor(gin_nat, device=dev)

                if w_ref.requires_grad:
                    # For ConvTranspose2d(x, w): weight is [C_in, C_out, kH, kW].
                    # In forward, x acts like grad_out in conv2d_backward_input, and grad_out acts like input!
                    gw_nat = x._native.conv2d_backward_weight(
                        grad_out._native, list(w_ref.shape), sh, sw, ph, pw, 1, True
                    )
                    g_w = Tensor(gw_nat, device=dev)

                if b_ref is not None and b_ref.requires_grad:
                    g_b = grad_out.sum(dim=(0, 2, 3))

                if b_ref is not None:
                    return g_in, g_w, g_b
                return g_in, g_w

            inputs = [x, self.weight]
            if self.bias is not None:
                inputs.append(self.bias)
            out.grad_fn = Node("ConvTranspose2dBackward", conv_t_backward, inputs)

        if self.bias is not None:
            out = out + self.bias.reshape(1, self.out_channels, 1, 1)
        return out

class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.kH, self.kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = stride if stride is not None else kernel_size
        self.stride_h, self.stride_w = (stride, stride) if isinstance(stride, int) else stride
        self.pad_h, self.pad_w = (padding, padding) if isinstance(padding, int) else padding

    def forward(self, x: Tensor) -> Tensor:
        out_native, idx_native = x._native.maxpool2d(
            self.kH, self.kW, self.stride_h, self.stride_w, self.pad_h, self.pad_w
        )
        req_grad = x.requires_grad and is_grad_enabled()
        out = Tensor(out_native, device=x.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            in_shape = list(x.shape)
            kH, kW = self.kH, self.kW
            sh, sw = self.stride_h, self.stride_w
            ph, pw = self.pad_h, self.pad_w
            dev = x.device

            def pool_backward(grad_out: Tensor):
                gin_native = grad_out.contiguous()._native.maxpool2d_backward(
                    idx_native, in_shape, kH, kW, sh, sw, ph, pw
                )
                return Tensor(gin_native, device=dev)

            out.grad_fn = Node("MaxPool2dBackward", pool_backward, [x])
        return out

class AvgPool2d(Module):
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 count_include_pad: bool = True):
        super().__init__()
        self.kH, self.kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = stride if stride is not None else kernel_size
        self.stride_h, self.stride_w = (stride, stride) if isinstance(stride, int) else stride
        self.pad_h, self.pad_w = (padding, padding) if isinstance(padding, int) else padding
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        out_native = x._native.avgpool2d(
            self.kH, self.kW, self.stride_h, self.stride_w,
            self.pad_h, self.pad_w, self.count_include_pad, False, 0, 0
        )
        req_grad = x.requires_grad and is_grad_enabled()
        out = Tensor(out_native, device=x.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            in_shape = list(x.shape)
            kH, kW = self.kH, self.kW
            sh, sw = self.stride_h, self.stride_w
            ph, pw = self.pad_h, self.pad_w
            cip = self.count_include_pad
            dev = x.device

            def avgpool_backward(grad_out: Tensor):
                gin = grad_out.contiguous()._native.avgpool2d_backward(
                    in_shape, kH, kW, sh, sw, ph, pw, cip, False
                )
                return Tensor(gin, device=dev)

            out.grad_fn = Node("AvgPool2dBackward", avgpool_backward, [x])
        return out

class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.out_h, self.out_w = (output_size, output_size) if isinstance(output_size, int) else output_size

    def forward(self, x: Tensor) -> Tensor:
        out_native = x._native.avgpool2d(
            1, 1, 1, 1, 0, 0, False, True, self.out_h, self.out_w
        )
        req_grad = x.requires_grad and is_grad_enabled()
        out = Tensor(out_native, device=x.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            in_shape = list(x.shape)
            dev = x.device

            def adaptive_avgpool_backward(grad_out: Tensor):
                gin = grad_out.contiguous()._native.avgpool2d_backward(
                    in_shape, 1, 1, 1, 1, 0, 0, False, True
                )
                return Tensor(gin, device=dev)

            out.grad_fn = Node("AdaptiveAvgPool2dBackward", adaptive_avgpool_backward, [x])
        return out

class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)

class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, elementwise_affine: bool = True, device=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        dev = resolve_device(device)

        if elementwise_affine:
            self.weight = Parameter(vultorch.ones(self.normalized_shape, device=dev), requires_grad=True)
            self.bias = Parameter(vultorch.zeros(self.normalized_shape, device=dev), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        ndim_norm = len(self.normalized_shape)
        if x.ndim < ndim_norm:
            raise ValueError(
                f"LayerNorm expected at least {ndim_norm} dims, got {x.ndim}"
            )

        g_native = self.weight._native if self.weight is not None else None
        b_native = self.bias._native if self.bias is not None else None
        native_out = x._native.layernorm(g_native, b_native, self.eps, ndim_norm)

        req_grad = (
            x.requires_grad
            or (self.weight is not None and self.weight.requires_grad)
            or (self.bias is not None and self.bias.requires_grad)
        ) and is_grad_enabled()

        out = Tensor(native_out, device=x.device, requires_grad=req_grad)
        if not req_grad:
            return out

        out.is_leaf = False
        eps = self.eps
        weight, bias = self.weight, self.bias
        dev = x.device

        def layernorm_backward(g: Tensor):
            need_dx = x.requires_grad
            need_dg = weight is not None and weight.requires_grad
            need_db = bias is not None and bias.requires_grad
            dx_nat, dg_nat, db_nat = x._native.layernorm_backward(
                g.contiguous()._native, g_native, eps, ndim_norm, need_dx, need_dg, need_db
            )
            grads = [Tensor(dx_nat, device=dev) if dx_nat is not None else None]
            if weight is not None:
                grads.append(Tensor(dg_nat, device=dev) if dg_nat is not None else None)
            if bias is not None:
                grads.append(Tensor(db_nat, device=dev) if db_nat is not None else None)
            return tuple(grads)

        inputs = [x]
        if self.weight is not None:
            inputs.append(self.weight)
        if self.bias is not None:
            inputs.append(self.bias)
        out.grad_fn = Node("LayerNormBackward", layernorm_backward, inputs)
        return out

class RMSNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-6, elementwise_affine: bool = True, device=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        dev = resolve_device(device)

        if elementwise_affine:
            self.weight = Parameter(vultorch.ones(self.normalized_shape, device=dev), requires_grad=True)
        else:
            self.weight = None

    def forward(self, x: Tensor) -> Tensor:
        ndim_norm = len(self.normalized_shape)
        if x.ndim < ndim_norm:
            raise ValueError(
                f"RMSNorm expected at least {ndim_norm} dims, got {x.ndim}"
            )

        g_native = self.weight._native if self.weight is not None else None
        native_out = x._native.rmsnorm(g_native, self.eps, ndim_norm)

        req_grad = (
            x.requires_grad or (self.weight is not None and self.weight.requires_grad)
        ) and is_grad_enabled()

        out = Tensor(native_out, device=x.device, requires_grad=req_grad)
        if not req_grad:
            return out

        out.is_leaf = False
        eps = self.eps
        weight = self.weight
        dev = x.device

        def rmsnorm_backward(g: Tensor):
            need_dx = x.requires_grad
            need_dg = weight is not None and weight.requires_grad
            dx_nat, dg_nat = x._native.rmsnorm_backward(
                g.contiguous()._native, g_native, eps, ndim_norm, need_dx, need_dg
            )
            grads = [Tensor(dx_nat, device=dev) if dx_nat is not None else None]
            if weight is not None:
                grads.append(Tensor(dg_nat, device=dev) if dg_nat is not None else None)
            return tuple(grads)

        inputs = [x]
        if self.weight is not None:
            inputs.append(self.weight)
        out.grad_fn = Node("RMSNormBackward", rmsnorm_backward, inputs)
        return out

class _BatchNormBase(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        dev = resolve_device(device)

        if affine:
            self.weight = Parameter(vultorch.ones((num_features,), device=dev), requires_grad=True)
            self.bias = Parameter(vultorch.zeros((num_features,), device=dev), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.register_buffer("running_mean", vultorch.zeros((num_features,), device=dev))
            self.register_buffer("running_var", vultorch.ones((num_features,), device=dev))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x: Tensor) -> Tensor:
        rm_nat = self.running_mean._native if self.running_mean is not None else None
        rv_nat = self.running_var._native if self.running_var is not None else None
        g_nat = self.weight._native if self.weight is not None else None
        b_nat = self.bias._native if self.bias is not None else None

        y_nat, save_mean_nat, save_invstd_nat = x._native.batchnorm(
            rm_nat, rv_nat, g_nat, b_nat, self.training, self.momentum, self.eps
        )
        req_grad = (
            x.requires_grad
            or (self.weight is not None and self.weight.requires_grad)
            or (self.bias is not None and self.bias.requires_grad)
        ) and is_grad_enabled()

        out = Tensor(y_nat, device=x.device, requires_grad=req_grad)
        if not req_grad:
            return out

        out.is_leaf = False
        eps = self.eps
        weight, bias = self.weight, self.bias
        dev = x.device

        def batchnorm_backward(g: Tensor):
            need_dx = x.requires_grad
            need_dg = weight is not None and weight.requires_grad
            need_db = bias is not None and bias.requires_grad
            dx_nat, dg_nat, db_nat = x._native.batchnorm_backward(
                g.contiguous()._native, save_mean_nat, save_invstd_nat,
                g_nat, eps, need_dx, need_dg, need_db
            )
            grads = [Tensor(dx_nat, device=dev) if dx_nat is not None else None]
            if weight is not None:
                grads.append(Tensor(dg_nat, device=dev) if dg_nat is not None else None)
            if bias is not None:
                grads.append(Tensor(db_nat, device=dev) if db_nat is not None else None)
            return tuple(grads)

        inputs = [x]
        if self.weight is not None:
            inputs.append(self.weight)
        if self.bias is not None:
            inputs.append(self.bias)
        out.grad_fn = Node("BatchNormBackward", batchnorm_backward, inputs)
        return out


class BatchNorm1d(_BatchNormBase):
    pass


class BatchNorm2d(_BatchNormBase):
    pass


class GroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                 affine: bool = True, device=None):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        dev = resolve_device(device)

        if affine:
            self.weight = Parameter(vultorch.ones((num_channels,), device=dev), requires_grad=True)
            self.bias = Parameter(vultorch.zeros((num_channels,), device=dev), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        g_nat = self.weight._native if self.weight is not None else None
        b_nat = self.bias._native if self.bias is not None else None
        y_nat, mean_invstd_nat = x._native.groupnorm(
            self.num_groups, g_nat, b_nat, self.eps
        )
        req_grad = (
            x.requires_grad
            or (self.weight is not None and self.weight.requires_grad)
            or (self.bias is not None and self.bias.requires_grad)
        ) and is_grad_enabled()

        out = Tensor(y_nat, device=x.device, requires_grad=req_grad)
        if not req_grad:
            return out

        out.is_leaf = False
        ng = self.num_groups
        weight, bias = self.weight, self.bias
        dev = x.device

        def groupnorm_backward(g: Tensor):
            need_dx = x.requires_grad
            need_dg = weight is not None and weight.requires_grad
            need_db = bias is not None and bias.requires_grad
            dx_nat, dg_nat, db_nat = x._native.groupnorm_backward(
                g.contiguous()._native, mean_invstd_nat, ng, g_nat,
                need_dx, need_dg, need_db
            )
            grads = [Tensor(dx_nat, device=dev) if dx_nat is not None else None]
            if weight is not None:
                grads.append(Tensor(dg_nat, device=dev) if dg_nat is not None else None)
            if bias is not None:
                grads.append(Tensor(db_nat, device=dev) if db_nat is not None else None)
            return tuple(grads)

        inputs = [x]
        if self.weight is not None:
            inputs.append(self.weight)
        if self.bias is not None:
            inputs.append(self.bias)
        out.grad_fn = Node("GroupNormBackward", groupnorm_backward, inputs)
        return out


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        seed, offset = vultorch._next_philox_state()
        inv_keep = 1.0 / (1.0 - self.p)
        req_grad = x.requires_grad and is_grad_enabled()

        out = vultorch.empty(x.shape, device=x.device, requires_grad=req_grad)
        out._native.random_philox_(2, self.p, inv_keep, seed, offset, x.contiguous()._native)

        if req_grad:
            out.is_leaf = False
            p_val = self.p
            dev = x.device

            def dropout_backward(g: Tensor):
                gin = vultorch.empty(g.shape, device=dev)
                gin._native.random_philox_(2, p_val, inv_keep, seed, offset, g.contiguous()._native)
                return gin

            out.grad_fn = Node("DropoutBackward", dropout_backward, [x])
        return out

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = -1 if padding_idx is None else int(padding_idx)
        dev = resolve_device(device)
        self.weight = Parameter(
            vultorch.randn(num_embeddings, embedding_dim, device=dev),
            requires_grad=True,
        )
        if padding_idx is not None and 0 <= self.padding_idx < num_embeddings:
            with no_grad():
                p_idx = self.padding_idx
                zero_row = vultorch.zeros((1, embedding_dim), device=dev)
                parts = []
                if p_idx > 0:
                    parts.append(self.weight[:p_idx])
                parts.append(zero_row)
                if p_idx + 1 < num_embeddings:
                    parts.append(self.weight[p_idx + 1:])
                self.weight.copy_(vultorch.cat(parts, dim=0, device=dev))

    def forward(self, indices: Tensor) -> Tensor:
        if indices.device != self.weight.device:
            indices = indices.to(self.weight.device)
        native_out = indices.contiguous()._native.embedding(self.weight._native, self.padding_idx)
        req_grad = self.weight.requires_grad and is_grad_enabled()
        out = Tensor(native_out, device=self.weight.device, requires_grad=req_grad)
        if req_grad:
            out.is_leaf = False
            idx_native = indices.contiguous()._native
            num_emb = self.num_embeddings
            pad_idx = self.padding_idx
            dev = self.weight.device

            def emb_backward(g: Tensor):
                gw_native = g.contiguous()._native.embedding_backward(idx_native, num_emb, pad_idx)
                return Tensor(gw_native, device=dev)

            out.grad_fn = Node("EmbeddingBackward", emb_backward, [self.weight])
        return out

# Activations
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()

class Mish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.mish(x)

class Hardswish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.hardswish(x)

class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)

class ELU(Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: Tensor) -> Tensor:
        return F.elu(x, alpha=self.alpha)


class _SnakeBase(Module):
    """
    Unified base class for the Snake family of periodic non-linear activations:
      - mode 0 (Snake / SnakeBeta):   x + (1 / b) * sin^2(a * x)
      - mode 1 (SnakeGamma):          x + (1 / b) * sin^2(a * x + gamma)
      - mode 2 (SnakeCos):            x + (1 / b) * cos^2(a * x)
      - mode 3 (SnakeRational):       x + (1 / b) * sin^2(a * x) / (1 + (a * x)^2)
      - mode 4 (SnakeSinc):           x + (1 / b) * (sin(a * x) / (a * x))^2
    """
    def __init__(self, in_features: int, a: float = 1.0, trainable: bool = True,
                 alpha_logscale: bool = False, channel_dim: int = 1,
                 eps: float = 1e-9, mode: int = 0, has_beta: bool = False,
                 has_gamma: bool = False, init_gamma: float = 0.0, device=None):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        self.channel_dim = channel_dim
        self.eps = eps
        self.mode = int(mode)
        self.has_beta = bool(has_beta)
        self.has_gamma = bool(has_gamma)
        dev = resolve_device(device)

        init_val = math.log(a) if alpha_logscale else float(a)
        alpha_t = vultorch.full((in_features,), init_val, device=dev)
        if trainable:
            self.alpha = Parameter(alpha_t, requires_grad=True)
        else:
            self.register_buffer("alpha", alpha_t)

        if self.has_beta:
            beta_t = vultorch.full((in_features,), init_val, device=dev)
            if trainable:
                self.beta = Parameter(beta_t, requires_grad=True)
            else:
                self.register_buffer("beta", beta_t)
        else:
            self.beta = None

        if self.has_gamma:
            gamma_t = vultorch.full((in_features,), float(init_gamma), device=dev)
            if trainable:
                self.gamma = Parameter(gamma_t, requires_grad=True)
            else:
                self.register_buffer("gamma", gamma_t)
        else:
            self.gamma = None

    def forward(self, x: Tensor) -> Tensor:
        cd = self.channel_dim if x.ndim > 1 else 0
        b_nat = self.beta._native if self.beta is not None else None
        g_nat = self.gamma._native if self.gamma is not None else None
        y_nat = x._native.snake(
            self.alpha._native, b_nat, cd, self.alpha_logscale, 1.0, self.eps,
            self.mode, g_nat, 0.0
        )
        req_grad = (
            x.requires_grad
            or self.alpha.requires_grad
            or (self.beta is not None and self.beta.requires_grad)
            or (self.gamma is not None and self.gamma.requires_grad)
        ) and is_grad_enabled()
        out = Tensor(y_nat, device=x.device, requires_grad=req_grad)
        if not req_grad:
            return out

        out.is_leaf = False
        alpha_ref = self.alpha
        beta_ref = self.beta
        gamma_ref = self.gamma
        log_sc = self.alpha_logscale
        eps_val = self.eps
        mode_val = self.mode
        dev = x.device

        inputs = [x]
        if isinstance(alpha_ref, Parameter):
            inputs.append(alpha_ref)
        if isinstance(beta_ref, Parameter):
            inputs.append(beta_ref)
        if isinstance(gamma_ref, Parameter):
            inputs.append(gamma_ref)

        def snake_family_backward(g: Tensor):
            need_dx = x.requires_grad
            need_da = isinstance(alpha_ref, Parameter) and alpha_ref.requires_grad
            need_db = isinstance(beta_ref, Parameter) and beta_ref.requires_grad
            need_dg = isinstance(gamma_ref, Parameter) and gamma_ref.requires_grad
            dx_nat, da_nat, db_nat, dg_nat = x._native.snake_backward(
                g.contiguous()._native, alpha_ref._native, b_nat,
                cd, log_sc, 1.0, eps_val, need_dx, need_da, need_db,
                mode_val, g_nat, 0.0, need_dg
            )
            grads = [Tensor(dx_nat, device=dev) if dx_nat is not None else None]
            if isinstance(alpha_ref, Parameter):
                grads.append(Tensor(da_nat, device=dev) if da_nat is not None else None)
            if isinstance(beta_ref, Parameter):
                grads.append(Tensor(db_nat, device=dev) if db_nat is not None else None)
            if isinstance(gamma_ref, Parameter):
                grads.append(Tensor(dg_nat, device=dev) if dg_nat is not None else None)
            return grads[0] if len(grads) == 1 else tuple(grads)

        out.grad_fn = Node("SnakeFamilyBackward", snake_family_backward, inputs)
        return out


class Snake(_SnakeBase):
    """
    Snake periodic non-linear activation (Ziyin et al. 2020):
        Snake_a(x) = x + (1 / a) * sin^2(a * x)
    """
    def __init__(self, in_features: int, a: float = 1.0, trainable: bool = True,
                 alpha_logscale: bool = False, channel_dim: int = 1,
                 eps: float = 1e-9, device=None):
        super().__init__(
            in_features, a=a, trainable=trainable, alpha_logscale=alpha_logscale,
            channel_dim=channel_dim, eps=eps, mode=0, has_beta=False, has_gamma=False,
            device=device
        )


class SnakeBeta(_SnakeBase):
    """
    SnakeBeta periodic activation (BigVGAN / Descript Audio Codec) with decoupled
    frequency `alpha` and inverse-magnitude `beta`:
        SnakeBeta_{a, b}(x) = x + (1 / b) * sin^2(a * x)
    """
    def __init__(self, in_features: int, a: float = 1.0, trainable: bool = True,
                 alpha_logscale: bool = False, channel_dim: int = 1,
                 eps: float = 1e-9, device=None):
        super().__init__(
            in_features, a=a, trainable=trainable, alpha_logscale=alpha_logscale,
            channel_dim=channel_dim, eps=eps, mode=0, has_beta=True, has_gamma=False,
            device=device
        )


class SnakeGamma(_SnakeBase):
    """
    SnakeGamma phase-shifted periodic activation with trainable frequency `alpha`,
    inverse-magnitude `beta`, and phase shift `gamma`:
        SnakeGamma_{a, b, gamma}(x) = x + (1 / b) * sin^2(a * x + gamma)
    """
    def __init__(self, in_features: int, a: float = 1.0, gamma: float = 0.0,
                 trainable: bool = True, alpha_logscale: bool = False,
                 channel_dim: int = 1, eps: float = 1e-9, device=None):
        super().__init__(
            in_features, a=a, trainable=trainable, alpha_logscale=alpha_logscale,
            channel_dim=channel_dim, eps=eps, mode=1, has_beta=True, has_gamma=True,
            init_gamma=gamma, device=device
        )


class SnakeCos(_SnakeBase):
    """
    SnakeCos cosine-squared periodic activation:
        SnakeCos_{a, b}(x) = x + (1 / b) * cos^2(a * x)
    """
    def __init__(self, in_features: int, a: float = 1.0, trainable: bool = True,
                 alpha_logscale: bool = False, has_beta: bool = True,
                 channel_dim: int = 1, eps: float = 1e-9, device=None):
        super().__init__(
            in_features, a=a, trainable=trainable, alpha_logscale=alpha_logscale,
            channel_dim=channel_dim, eps=eps, mode=2, has_beta=has_beta, has_gamma=False,
            device=device
        )


class SnakeRational(_SnakeBase):
    """
    SnakeRational damped periodic activation:
        SnakeRational_{a, b}(x) = x + (1 / b) * sin^2(a * x) / (1 + (a * x)^2)
    """
    def __init__(self, in_features: int, a: float = 1.0, trainable: bool = True,
                 alpha_logscale: bool = False, has_beta: bool = True,
                 channel_dim: int = 1, eps: float = 1e-9, device=None):
        super().__init__(
            in_features, a=a, trainable=trainable, alpha_logscale=alpha_logscale,
            channel_dim=channel_dim, eps=eps, mode=3, has_beta=has_beta, has_gamma=False,
            device=device
        )


class SnakeSinc(_SnakeBase):
    """
    SnakeSinc wavepacket periodic activation:
        SnakeSinc_{a, b}(x) = x + (1 / b) * (sin(a * x) / (a * x))^2
    """
    def __init__(self, in_features: int, a: float = 1.0, trainable: bool = True,
                 alpha_logscale: bool = False, has_beta: bool = True,
                 channel_dim: int = 1, eps: float = 1e-9, device=None):
        super().__init__(
            in_features, a=a, trainable=trainable, alpha_logscale=alpha_logscale,
            channel_dim=channel_dim, eps=eps, mode=4, has_beta=has_beta, has_gamma=False,
            device=device
        )


class Mamba(Module):
    """
    Mamba Selective State Space Model (S6) block (Gu & Dao 2023):
      - Input projection `in_proj`: d_model -> 2 * d_inner (splits into x and gate z)
      - Causal depthwise 1D convolution `conv1d` (groups = d_inner) + SiLU
      - Selective state-space projection `x_proj`: d_inner -> (dt_rank + 2 * d_state)
      - Time-step projection `dt_proj`: dt_rank -> d_inner + softplus
      - Fused Slang selective scan (`ssm_scan_forward` / `ssm_scan_backward`)
      - Gated output `(y * SiLU(z))` + `out_proj`: d_inner -> d_model
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        bias: bool = False,
        conv_bias: bool = True,
        use_snake: bool = False,
        device=None,
    ):
        super().__init__()
        if d_state > 32:
            raise ValueError("Mamba d_state up to 32 is supported by the register scan kernel")
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank is not None else max(1, math.ceil(d_model / 16))
        dev = resolve_device(device)

        self.in_proj = Linear(d_model, self.d_inner * 2, bias=bias, device=dev)
        self.conv1d = Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
            device=dev,
        )
        self.act = SnakeBeta(self.d_inner, alpha_logscale=True, device=dev) if use_snake else SiLU()
        self.x_proj = Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, device=dev)
        self.dt_proj = Linear(self.dt_rank, self.d_inner, bias=True, device=dev)

        # Initialize A_log = log(1..d_state) repeated across d_inner
        a_init = np.tile(np.arange(1, d_state + 1, dtype=np.float32), (self.d_inner, 1))
        self.A_log = Parameter(vultorch.tensor(np.log(a_init), device=dev), requires_grad=True)
        self.D = Parameter(vultorch.ones((self.d_inner,), device=dev), requires_grad=True)
        self.out_proj = Linear(self.d_inner, d_model, bias=bias, device=dev)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: [B, L, d_model]
        Returns:
            out: [B, L, d_model]
        """
        B_sz, L_sz, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)  # [B, L, 2 * d_inner]
        x, z = xz.chunk(2, dim=-1)        # each [B, L, d_inner]

        # Depthwise causal Conv1d on [B, d_inner, L]
        x_conv = self.conv1d(x.transpose(1, 2))
        if self.d_conv > 1:
            x_conv = x_conv.narrow(2, 0, L_sz)
        x_act = self.act(x_conv)          # [B, d_inner, L]

        # Selective state space parameters from x_act^T: [B, L, d_inner]
        x_Flat = x_act.transpose(1, 2)    # [B, L, d_inner]
        x_dbl = self.x_proj(x_Flat)       # [B, L, dt_rank + 2 * d_state]
        dt_raw, B_sel, C_sel = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Project delta: [B, L, dt_rank] -> [B, L, d_inner] -> [B, d_inner, L]
        delta = functional.softplus(self.dt_proj(dt_raw).transpose(1, 2))
        A = -(self.A_log.exp())           # [d_inner, d_state]
        B_mat = B_sel.transpose(1, 2)     # [B, d_state, L]
        C_mat = C_sel.transpose(1, 2)     # [B, d_state, L]

        y = functional.selective_scan(x_act, delta, A, B_mat, C_mat, self.D)  # [B, d_inner, L]
        y_gated = y.transpose(1, 2) * z.silu()                                # [B, L, d_inner]
        return self.out_proj(y_gated)


MambaBlock = Mamba


class RotaryEmbedding(Module):
    """
    Rotary Position Embedding (RoPE — Su et al. 2021) precomputing (cos, sin) frequency
    tables and applying fused complex rotation to 4D query/key tensors `[B, H, S, D]`
    (when `seq_dim=2`) or `[B, S, H, D]` (when `seq_dim=1`).
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        seq_dim: int = 2,
        device=None,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding dim must be even, got {dim}")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)
        self.seq_dim = int(seq_dim)
        dev = resolve_device(device)

        inv_freq = 1.0 / (self.base ** (np.arange(0, dim, 2, dtype=np.float32) / float(dim)))
        t = np.arange(max_position_embeddings, dtype=np.float32)
        freqs = np.outer(t, inv_freq).astype(np.float32)  # [max_pos, dim // 2]
        self.register_buffer("cos_cached", vultorch.tensor(np.cos(freqs), device=dev))
        self.register_buffer("sin_cached", vultorch.tensor(np.sin(freqs), device=dev))

    def get_cos_sin(self, seq_len: int, device=None) -> Tuple[Tensor, Tensor]:
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if device is not None and cos.device != Device(device):
            cos = cos.to(device)
            sin = sin.to(device)
        return cos, sin

    def forward(self, x: Tensor, seq_dim: Optional[int] = None) -> Tensor:
        sd = self.seq_dim if seq_dim is None else int(seq_dim)
        norm_sd = sd if sd >= 0 else sd + x.ndim
        seq_len = x.shape[norm_sd]
        cos, sin = self.get_cos_sin(seq_len, device=x.device)
        return functional.apply_rotary_pos_emb(x, cos, sin, seq_dim=norm_sd)


class SwiGLU(Module):
    """Fused SwiGLU activation (`SiLU(a) * b`). Splits input along last dim when a single tensor is given."""
    def forward(self, a: Tensor, b: Optional[Tensor] = None) -> Tensor:
        return functional.swiglu(a, b)


class GeGLU(Module):
    """Fused GeGLU activation (`GELU(a) * b`). Splits input along last dim when a single tensor is given."""
    def forward(self, a: Tensor, b: Optional[Tensor] = None) -> Tensor:
        return functional.geglu(a, b)


class MultiheadAttention(Module):
    """
    Multi-Head & Grouped-Query Attention (GQA) backed by fused Slang FlashAttention-2
    (`F.scaled_dot_product_attention`) with optional built-in Rotary Position Embeddings (`use_rope=True`).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 4096,
        device=None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.dropout = dropout
        self.batch_first = batch_first
        dev = resolve_device(device)

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, device=dev)
        self.k_proj = Linear(embed_dim, self.kv_dim, bias=bias, device=dev)
        self.v_proj = Linear(embed_dim, self.kv_dim, bias=bias, device=dev)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, device=dev)
        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len, seq_dim=2, device=dev) if use_rope else None

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ):
        if key is None:
            key = query
        if value is None:
            value = key

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B_sz, S_q, _ = query.shape
        S_k = key.shape[1]

        q = self.q_proj(query).reshape(B_sz, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B_sz, S_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B_sz, S_k, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        attn_out = functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )
        merged = attn_out.transpose(1, 2).contiguous().reshape(B_sz, S_q, self.embed_dim)
        out = self.out_proj(merged)
        if not self.batch_first:
            out = out.transpose(0, 1)
        if need_weights:
            return out, None
        return out


class QuantizedLinear(Module):
    """
    Weight-Only `int4` / `int8` Group-Wise Quantized Linear layer (`shaders/gemm_quant.slang`).
    Stores packed 4-bit (8 weights per uint32) or 8-bit (4 weights per uint32) weights
    alongside per-group scales `[out_features, in_features // group_size]` and dequantizes
    in registers during the Slang matrix multiplication dispatch.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 64,
        bias: bool = True,
        device=None,
    ):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"QuantizedLinear supports bits=4 or bits=8, got {bits}")
        pack_factor = 8 if bits == 4 else 4
        if in_features % pack_factor != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by {pack_factor} for int{bits}")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = int(bits)
        self.group_size = min(int(group_size), in_features)
        self.num_groups = (in_features + self.group_size - 1) // self.group_size
        dev = resolve_device(device)

        words_per_row = in_features // pack_factor
        self.register_buffer("qweight", vultorch.zeros((out_features, words_per_row), device=dev, dtype=vultorch.int32))
        self.register_buffer("scales", vultorch.ones((out_features, self.num_groups), device=dev, dtype=vultorch.float32))
        if bias:
            self.register_buffer("bias", vultorch.zeros((out_features,), device=dev, dtype=vultorch.float32))
        else:
            self.bias = None

    @property
    def weight_nbytes(self) -> int:
        return self.qweight.nbytes + self.scales.nbytes

    @classmethod
    def from_float(
        cls,
        linear: Linear,
        bits: int = 4,
        group_size: int = 64,
    ) -> 'QuantizedLinear':
        dev = linear.weight.device
        in_f = linear.in_features
        out_f = linear.out_features
        eff_gs = min(int(group_size), in_f)
        q_mod = cls(in_f, out_f, bits=bits, group_size=eff_gs, bias=(linear.bias is not None), device=dev)

        qw_nat, sc_nat = linear.weight._native.quantize_weights(eff_gs, int(bits))
        q_mod.qweight.copy_(Tensor._from_native(qw_nat, dev, False))
        q_mod.scales.copy_(Tensor._from_native(sc_nat, dev, False))
        if linear.bias is not None and q_mod.bias is not None:
            q_mod.bias.copy_(linear.bias.detach().to(vultorch.float32))
        return q_mod

    def forward(self, x: Tensor) -> Tensor:
        _check_same_device(x, self.qweight, "QuantizedLinear")
        b_nat = self.bias._native if self.bias is not None else None
        out_nat = x._native.gemm_quant(
            self.qweight._native, self.scales._native, b_nat, self.group_size, self.bits
        )
        return Tensor._from_native(out_nat, x.device, False)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(self.dim)

class LogSoftmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.log_softmax(self.dim)

# Loss Functions
class MSELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        sq = diff * diff
        if self.reduction == "mean":
            return sq.mean()
        elif self.reduction == "sum":
            return sq.sum()
        return sq

class SmoothL1Loss(Module):
    def __init__(self, reduction: str = "mean", beta: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = float(beta)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)

class CosineSimilarity(Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)

class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x

class L1Loss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)

class BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)

class Upsample(Module):
    def __init__(self, size=None, scale_factor=None, mode: str = "nearest", align_corners: Optional[bool] = None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class ParameterList(Module):
    def __init__(self, parameters=None):
        super().__init__()
        if parameters is not None:
            for i, p in enumerate(parameters):
                self.register_parameter(str(i), p)

    def __getitem__(self, idx: int) -> Parameter:
        keys = list(self._parameters.keys())
        return self._parameters[keys[idx]]

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def append(self, p: Parameter):
        self.register_parameter(str(len(self._parameters)), p)
        return self

class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for k, m in modules.items():
                self.register_module(str(k), m)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module):
        self.register_module(str(key), module)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.keys())

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

class CrossEntropyLoss(Module):
    """
    Numerically stable Cross Entropy Loss with LogSoftmax + NLL.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Integer class indices take the fused path; anything else is treated as
        # a (soft) target distribution.
        if target.ndim == 1 or (target.ndim == 2 and target.shape[-1] == 1):
            return self._index_target_loss(pred, target)

        nll = -(target * pred.log_softmax(dim=-1)).sum(dim=-1)
        return nll.mean() if self.reduction == "mean" else nll.sum()

    def _index_target_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.ndim != 2:
            raise ValueError(f"CrossEntropyLoss expects 2-D logits, got shape {pred.shape}")

        rows = pred.shape[0]
        indices = target.reshape((rows,))

        req_grad = pred.requires_grad and is_grad_enabled()
        per_row_native = pred._native.cross_entropy_per_row(indices._native)

        if self.reduction == "mean":
            out_native = per_row_native.reduce_mean(-999999, False)
            scale = 1.0 / float(rows) if rows > 0 else 1.0
        elif self.reduction == "sum":
            out_native = per_row_native.reduce_sum(-999999, False)
            scale = 1.0
        else:
            out_native = per_row_native
            scale = 1.0

        out = Tensor(
            out_native,
            device=pred.device,
            requires_grad=req_grad,
        )

        if req_grad:
            out.is_leaf = False
            pred_native = pred._native
            indices_native = indices._native
            device = pred.device
            reduction = self.reduction

            def ce_backward(g: Tensor):
                if reduction in ("mean", "sum"):
                    g_native = g._native if g.numel == 1 else None
                    grad_native = pred_native.cross_entropy_grad(indices_native, g_native, scale)
                    grad = Tensor(grad_native, device=device)
                    return grad if g.numel == 1 else grad * g.reshape((rows, 1))
                else:
                    grad = Tensor(pred_native.cross_entropy_grad(indices_native, None, 1.0), device=device)
                    return grad * g.reshape((rows, 1))

            out.grad_fn = Node("CrossEntropyBackward", ce_backward, [pred])

        return out

from . import functional
from . import functional as F
from . import utils
from . import init

# ----------------------------------------------------------------------------
# Recurrent Layers (RNN, LSTM, GRU)
# ----------------------------------------------------------------------------
class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = "tanh", bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(vultorch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(vultorch.empty(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(vultorch.zeros(hidden_size))
            self.bias_hh = Parameter(vultorch.zeros(hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.1
        init.uniform_(self.weight_ih, -std, std)
        init.uniform_(self.weight_hh, -std, std)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = vultorch.zeros(input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype)
        gate = input @ self.weight_ih.t() + hx @ self.weight_hh.t()
        if self.bias_ih is not None:
            gate = gate + self.bias_ih + self.bias_hh
        return gate.tanh() if self.nonlinearity == "tanh" else gate.relu()


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(vultorch.empty(4 * hidden_size, input_size))
        self.weight_hh = Parameter(vultorch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(vultorch.zeros(4 * hidden_size))
            self.bias_hh = Parameter(vultorch.zeros(4 * hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.1
        init.uniform_(self.weight_ih, -std, std)
        init.uniform_(self.weight_hh, -std, std)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        if hx is None:
            h = vultorch.zeros(input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype)
            c = vultorch.zeros(input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype)
        else:
            h, c = hx
        gates = input @ self.weight_ih.t() + h @ self.weight_hh.t()
        if self.bias_ih is not None:
            gates = gates + self.bias_ih + self.bias_hh
        H = self.hidden_size
        i_gate = gates[:, 0:H].sigmoid()
        f_gate = gates[:, H:2*H].sigmoid()
        g_gate = gates[:, 2*H:3*H].tanh()
        o_gate = gates[:, 3*H:4*H].sigmoid()
        c_next = f_gate * c + i_gate * g_gate
        h_next = o_gate * c_next.tanh()
        return h_next, c_next


class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(vultorch.empty(3 * hidden_size, input_size))
        self.weight_hh = Parameter(vultorch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(vultorch.zeros(3 * hidden_size))
            self.bias_hh = Parameter(vultorch.zeros(3 * hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.1
        init.uniform_(self.weight_ih, -std, std)
        init.uniform_(self.weight_hh, -std, std)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = vultorch.zeros(input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype)
        H = self.hidden_size
        gi = input @ self.weight_ih.t()
        gh = hx @ self.weight_hh.t()
        if self.bias_ih is not None:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh
        r = (gi[:, 0:H] + gh[:, 0:H]).sigmoid()
        z = (gi[:, H:2*H] + gh[:, H:2*H]).sigmoid()
        n = (gi[:, 2*H:3*H] + r * gh[:, 2*H:3*H]).tanh()
        h_next = (1.0 - z) * n + z * hx
        return h_next


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layers = ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ])

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            # (B, T, D) -> (T, B, D)
            input = input.transpose(0, 1)
        seq_len, batch_size, _ = input.shape
        if hx is None:
            h_0 = vultorch.zeros(self.num_layers, batch_size, self.hidden_size, device=input.device, dtype=input.dtype)
            c_0 = vultorch.zeros(self.num_layers, batch_size, self.hidden_size, device=input.device, dtype=input.dtype)
        else:
            h_0, c_0 = hx

        h_out_layers = []
        c_out_layers = []
        cur_input = input

        for layer_idx, cell in enumerate(self.layers):
            layer_h = h_0[layer_idx]
            layer_c = c_0[layer_idx]
            step_outputs = []
            for t in range(seq_len):
                layer_h, layer_c = cell(cur_input[t], (layer_h, layer_c))
                step_outputs.append(layer_h.unsqueeze(0))
            cur_input = vultorch.cat(step_outputs, dim=0)
            h_out_layers.append(layer_h.unsqueeze(0))
            c_out_layers.append(layer_c.unsqueeze(0))

        out = cur_input
        if self.batch_first:
            out = out.transpose(0, 1)
        h_n = vultorch.cat(h_out_layers, dim=0)
        c_n = vultorch.cat(c_out_layers, dim=0)
        return out, (h_n, c_n)


class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layers = ModuleList([
            GRUCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ])

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            input = input.transpose(0, 1)
        seq_len, batch_size, _ = input.shape
        if hx is None:
            h_0 = vultorch.zeros(self.num_layers, batch_size, self.hidden_size, device=input.device, dtype=input.dtype)
        else:
            h_0 = hx

        h_out_layers = []
        cur_input = input
        for layer_idx, cell in enumerate(self.layers):
            layer_h = h_0[layer_idx]
            step_outputs = []
            for t in range(seq_len):
                layer_h = cell(cur_input[t], layer_h)
                step_outputs.append(layer_h.unsqueeze(0))
            cur_input = vultorch.cat(step_outputs, dim=0)
            h_out_layers.append(layer_h.unsqueeze(0))

        out = cur_input
        if self.batch_first:
            out = out.transpose(0, 1)
        h_n = vultorch.cat(h_out_layers, dim=0)
        return out, h_n


# ----------------------------------------------------------------------------
# Transformer Containers
# ----------------------------------------------------------------------------
class TransformerEncoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = ReLU() if activation == "relu" else GELU()

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Pre-LN or Post-LN standard transformer layer
        attn_out, _ = self.self_attn(src, src, src, attn_mask=mask)
        x = self.norm1(src + self.dropout(attn_out))
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerEncoder(Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        import copy
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, mask=mask)
        return out

# ----------------------------------------------------------------------------
class MultiGPU(Module):
    """
    Heterogeneous Multi-GPU container with TFLOPS-aware load balancing and
    automatic backward gradient reduction.
    - `device_ids` accepts `['all']` (or `'all'` / `None`) to use all available Vulkan GPUs,
      or an explicit list like `['vk:0', 'vk:1']`.
    - `loss.backward()` automatically reduces gradients from all GPU replicas into the
      master module's parameters, and the next `forward()` automatically syncs updated
      parameters and clears stale replica gradients—requiring zero extra calls in the
      training loop.
    """
    def __init__(
        self,
        module: Module,
        device_ids: Optional[Union[str, List[str]]] = None,
        balance: Union[bool, str] = "tflops",
        device_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if (
            device_ids is None
            or device_ids == "all"
            or (isinstance(device_ids, (list, tuple)) and len(device_ids) == 1 and str(device_ids[0]).lower() == "all")
        ):
            count = vultorch.vulkan.device_count()
            resolved_ids = [f"vk:{i}" for i in range(count)] if count > 0 else ["cpu"]
        elif isinstance(device_ids, str):
            resolved_ids = [device_ids]
        else:
            resolved_ids = list(device_ids)
        self.module = module
        self.device_ids = [str(Device(d)) for d in resolved_ids]
        self.balance = balance
        self.device_weights = list(device_weights) if device_weights is not None else None
        self.pool: Optional[ThreadPoolExecutor] = None
        self.replicas: Dict[str, Module] = {}
        self.last_split_sizes: List[int] = []
        self._replicas_synced: bool = False
        self.sync_every_forward: bool = True
        self._grad_reduced_this_step: bool = False
        # Closed-loop dynamic load balancer state
        self._measured_throughputs: Dict[str, float] = {}
        self._step_count: int = 0

    def _get_pool(self) -> ThreadPoolExecutor:
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=max(1, len(self.device_ids)))
        return self.pool

    def __del__(self):
        if getattr(self, "pool", None) is not None:
            try:
                self.pool.shutdown(wait=False)
            except Exception:
                pass

    def get_split_sizes(self, batch_size: int) -> List[int]:
        n_devs = len(self.device_ids)
        if n_devs <= 1:
            return [batch_size]

        # Automatic small batch bypass: if batch_size is too small (< 64), scatter/gather overhead
        # across PCIe exceeds secondary GPU speedup. Route entire batch to primary GPU (device 0).
        if self.balance in (True, "auto", "tflops") and batch_size < 64:
            splits = [0] * n_devs
            splits[0] = batch_size
            return splits

        if self.device_weights is not None:
            return vultorch.vulkan.compute_split_sizes(
                batch_size, self.device_ids, weights=self.device_weights
            )

        # Closed-loop adaptive balance: combine static GEMM TFLOPS calibration with measured per-step latency
        if self.balance == "auto" and len(self._measured_throughputs) == n_devs and self._step_count > 2:
            dynamic_weights = [self._measured_throughputs[d] for d in self.device_ids]
            total_w = sum(dynamic_weights)
            if total_w > 0:
                norm_w = [w / total_w for w in dynamic_weights]
                return vultorch.vulkan.compute_split_sizes(batch_size, self.device_ids, weights=norm_w)

        if self.balance in (True, "tflops", "auto"):
            return vultorch.vulkan.compute_split_sizes(batch_size, self.device_ids)
        # Equal split fallback (balance=False or balance="equal")
        equal_w = [1.0] * n_devs
        return vultorch.vulkan.compute_split_sizes(batch_size, self.device_ids, weights=equal_w)

    def sync_parameters(self):
        """Copies master module parameters and buffers to all created replicas."""
        master_params = list(self.module.named_parameters())
        master_buffers = list(self.module.named_buffers())
        if not master_params and not master_buffers:
            return
        master_dev = (master_params[0][1] if master_params else master_buffers[0][1]).device
        with no_grad():
            for dev_id, replica in self.replicas.items():
                if Device(dev_id) == master_dev or replica is self.module:
                    continue
                for (_, p_src), (_, p_dst) in zip(master_params, replica.named_parameters()):
                    p_dst.copy_(p_src.to(dev_id))
                for (_, b_src), (_, b_dst) in zip(master_buffers, replica.named_buffers()):
                    b_dst.copy_(b_src.to(dev_id))
        self._replicas_synced = True

    def _get_replica(self, device: str) -> Module:
        target_dev = Device(device)
        master_params = list(self.module.named_parameters())
        master_buffers = list(self.module.named_buffers())
        if master_params and master_params[0][1].device == target_dev:
            self.replicas[device] = self.module
            return self.module

        if device not in self.replicas:
            import copy
            replica = copy.deepcopy(self.module)
            replica.to(device)
            self.replicas[device] = replica
        elif self.sync_every_forward or not self._replicas_synced:
            replica = self.replicas[device]
            with no_grad():
                for (_, p_src), (_, p_dst) in zip(master_params, replica.named_parameters()):
                    p_dst.copy_(p_src.to(device))
                for (_, b_src), (_, b_dst) in zip(master_buffers, replica.named_buffers()):
                    b_dst.copy_(b_src.to(device))
        return self.replicas[device]

    @staticmethod
    def _move_tree(obj, device):
        if isinstance(obj, Tensor):
            return obj.to(device)
        if isinstance(obj, tuple):
            if hasattr(obj, "_fields"):  # namedtuple
                return type(obj)(*(DataParallel._move_tree(v, device) for v in obj))
            return tuple(DataParallel._move_tree(v, device) for v in obj)
        if isinstance(obj, list):
            return [DataParallel._move_tree(v, device) for v in obj]
        if isinstance(obj, dict):
            return type(obj)((k, DataParallel._move_tree(v, device)) for k, v in obj.items())
        return obj

    @staticmethod
    def _slice_tree(obj, start: int, end: int, dev_id: str):
        if isinstance(obj, Tensor):
            return obj[start:end].to(dev_id)
        if isinstance(obj, tuple):
            if hasattr(obj, "_fields"):
                return type(obj)(*(DataParallel._slice_tree(v, start, end, dev_id) for v in obj))
            return tuple(DataParallel._slice_tree(v, start, end, dev_id) for v in obj)
        if isinstance(obj, list):
            return [DataParallel._slice_tree(v, start, end, dev_id) for v in obj]
        if isinstance(obj, dict):
            return type(obj)((k, DataParallel._slice_tree(v, start, end, dev_id)) for k, v in obj.items())
        return obj

    @staticmethod
    def _gather_outputs(outputs: List, target_device):
        first = outputs[0]
        if isinstance(first, Tensor):
            return vultorch.cat(outputs, dim=0, device=target_device)
        if isinstance(first, tuple):
            gathered = tuple(
                DataParallel._gather_outputs([o[i] for o in outputs], target_device)
                for i in range(len(first))
            )
            if hasattr(first, "_fields"):  # namedtuple
                return type(first)(*gathered)
            return gathered
        if isinstance(first, list):
            return [
                DataParallel._gather_outputs([o[i] for o in outputs], target_device)
                for i in range(len(first))
            ]
        if isinstance(first, dict):
            return type(first)(
                (k, DataParallel._gather_outputs([o[k] for o in outputs], target_device))
                for k in first.keys()
            )
        return first

    @staticmethod
    def _find_first_tensor(obj) -> Optional[Tensor]:
        if isinstance(obj, Tensor):
            return obj
        if isinstance(obj, (tuple, list)):
            for item in obj:
                found = DataParallel._find_first_tensor(item)
                if found is not None:
                    return found
        elif isinstance(obj, dict):
            for item in obj.values():
                found = DataParallel._find_first_tensor(item)
                if found is not None:
                    return found
        return None

    def forward(self, *inputs, **kwargs):
        if len(self.device_ids) <= 1:
            return self.module(*inputs, **kwargs)

        first_tensor = self._find_first_tensor((inputs, kwargs))
        if first_tensor is None:
            return self.module(*inputs, **kwargs)

        batch_size = first_tensor.shape[0]
        target_device = first_tensor.device
        split_sizes = self.get_split_sizes(batch_size)
        self.last_split_sizes = split_sizes

        chunks = []
        offset = 0
        for dev_id, sz in zip(self.device_ids, split_sizes):
            if sz > 0 and offset < batch_size:
                end = min(offset + sz, batch_size)
                chunks.append((dev_id, offset, end))
                offset = end

        if len(chunks) == 1:
            dev_id, start, end = chunks[0]
            rep = self._get_replica(dev_id)
            dev_inputs = self._move_tree(inputs, dev_id)
            dev_kwargs = self._move_tree(kwargs, dev_id)
            return self._move_tree(rep(*dev_inputs, **dev_kwargs), target_device)

        # Ensure replicas are initialized, synced with latest master weights, and replica grads cleared
        for dev_id, _, _ in chunks:
            rep = self._get_replica(dev_id)
            if rep is not self.module:
                rep.zero_grad()
        self._replicas_synced = True
        self._grad_reduced_this_step = False

        import time

        def run_device(dev_id, start, end):
            t_start = time.perf_counter()
            rep = self.replicas[dev_id]
            chunk_inputs = self._slice_tree(inputs, start, end, dev_id)
            chunk_kwargs = self._slice_tree(kwargs, start, end, dev_id)
            out = rep(*chunk_inputs, **chunk_kwargs)
            # Sync device to measure true completion time
            if dev_id.startswith("vk"):
                vultorch.vulkan.synchronize(dev_id)
            t_elapsed = max(time.perf_counter() - t_start, 1e-6)
            return out, (dev_id, end - start, t_elapsed)

        pool = self._get_pool()
        futures = [pool.submit(run_device, dev, s, e) for dev, s, e in chunks]
        results_and_times = [f.result() for f in futures]
        results = [rt[0] for rt in results_and_times]
        gathered = self._gather_outputs(results, target_device)

        # Update closed-loop throughput measurements
        self._step_count += 1
        alpha = 0.25
        for _, (dev_id, chunk_sz, elapsed) in enumerate(rt[1] for rt in results_and_times):
            throughput = float(chunk_sz) / elapsed
            if dev_id in self._measured_throughputs:
                self._measured_throughputs[dev_id] = (
                    (1.0 - alpha) * self._measured_throughputs[dev_id] + alpha * throughput
                )
            else:
                self._measured_throughputs[dev_id] = throughput

        # Attach automatic post-backward hook (via weakref so no reference cycle keeps MultiGPU alive)
        first_out = self._find_first_tensor(gathered)
        if first_out is not None and first_out.requires_grad and first_out.grad_fn is not None:
            import weakref
            weak_self = weakref.ref(self)
            def _hook():
                inst = weak_self()
                if inst is not None:
                    inst._auto_reduce_after_backward()
            first_out.grad_fn.post_hook = _hook

        return gathered

    def _auto_reduce_after_backward(self):
        if not self._grad_reduced_this_step:
            self.reduce_gradients()

    def zero_grad(self):
        self.module.zero_grad()
        for rep in self.replicas.values():
            rep.zero_grad()
        self._replicas_synced = False
        self._grad_reduced_this_step = False

    def reduce_gradients(self):
        """
        All-reduces / accumulates gradients from all GPU replicas into master module parameters.
        Safe and idempotent if called both automatically during `loss.backward()` and manually.
        """
        if self._grad_reduced_this_step:
            return
        master_params = dict(self.module.named_parameters())
        if not master_params:
            return
        master_dev = list(master_params.values())[0].device

        for name, p_master in master_params.items():
            accum = None
            for dev_id, rep in self.replicas.items():
                p_rep = dict(rep.named_parameters()).get(name)
                if p_rep is not None and p_rep.grad is not None:
                    g = p_rep.grad.to(master_dev)
                    if accum is None:
                        accum = g.clone()
                    else:
                        accum = accum + g
            if accum is not None:
                p_master.grad = accum
        self._grad_reduced_this_step = True


DataParallel = MultiGPU

