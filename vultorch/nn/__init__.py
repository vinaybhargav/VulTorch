"""
VulTorch Neural Network Library (vultorch.nn)
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Iterator
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import vultorch
from ..tensor import Tensor, Device
from ..autograd import is_grad_enabled, no_grad, Node

class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""
    def __init__(self, data: Union[Tensor, np.ndarray], requires_grad: bool = True):
        if isinstance(data, Tensor):
            super().__init__(data._native, device=data.device, requires_grad=requires_grad)
        else:
            super().__init__(data, device="vk", requires_grad=requires_grad)

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

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.register_module(str(idx), module)

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
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device="vk"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform initialization
        bound = 1.0 / math.sqrt(in_features)
        w = vultorch.empty((out_features, in_features), device=device).uniform_(-bound, bound)
        self.weight = Parameter(w, requires_grad=True)

        if bias:
            b = vultorch.empty((out_features,), device=device).uniform_(-bound, bound)
            self.bias = Parameter(b, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Fused y = x @ W^T + b
        b_native = self.bias._native if self.bias is not None else None
        orig_shape = x.shape
        if x.ndim > 2:
            m_total = x.numel // self.in_features
            x_2d = x.contiguous()._native.reshape_view([m_total, self.in_features])
            out_2d = x_2d.matmul(self.weight._native, False, True, b_native)
            native_out = out_2d.reshape_view(list(orig_shape[:-1]) + [self.out_features])
        else:
            x_2d = x._native
            native_out = x_2d.matmul(self.weight._native, False, True, b_native)

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
                if grad_out.ndim > 2:
                    m_tot = grad_out.numel // out_feat
                    g_2d = grad_out.contiguous()._native.reshape_view([m_tot, out_feat])
                    x_mat = x.contiguous()._native.reshape_view([m_tot, in_feat])
                else:
                    g_2d = grad_out._native
                    x_mat = x._native

                g_x = None
                g_w = None
                g_b = None

                if x.requires_grad:
                    gx_2d = g_2d.matmul(weight._native, False, False, None)
                    if len(orig_shape) > 2:
                        gx_2d = gx_2d.reshape_view(list(orig_shape))
                    g_x = Tensor(gx_2d, device=x.device)

                if weight.requires_grad:
                    g_w = Tensor(g_2d.matmul(x_mat, True, False, None), device=x.device)

                if bias is not None and bias.requires_grad:
                    g_b = Tensor(g_2d.reduce_sum(0, False), device=x.device)
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
                 bias: bool = True, device="vk"):
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride_h, self.stride_w = (stride, stride) if isinstance(stride, int) else stride
        self.pad_h, self.pad_w = (padding, padding) if isinstance(padding, int) else padding
        self.groups = int(groups)

        # Kaiming init
        c_in_per_g = in_channels // self.groups
        k = 1.0 / math.sqrt(c_in_per_g * self.kH * self.kW)
        w = vultorch.empty((out_channels, c_in_per_g, self.kH, self.kW), device=device).uniform_(-k, k)
        self.weight = Parameter(w, requires_grad=True)

        if bias:
            b = vultorch.empty((out_channels,), device=device).uniform_(-k, k)
            self.bias = Parameter(b, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        b_native = self.bias._native if self.bias is not None else None
        native_out = x._native.conv2d(self.weight._native, b_native,
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
                 bias: bool = True, device="vk"):
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
                 bias: bool = True, device="vk"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride_h, self.stride_w = (stride, stride) if isinstance(stride, int) else stride
        self.pad_h, self.pad_w = (padding, padding) if isinstance(padding, int) else padding

        k = 1.0 / math.sqrt(out_channels * self.kH * self.kW)
        w = vultorch.empty((in_channels, out_channels, self.kH, self.kW), device=device).uniform_(-k, k)
        self.weight = Parameter(w, requires_grad=True)

        if bias:
            b = vultorch.empty((out_channels,), device=device).uniform_(-k, k)
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
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, elementwise_affine: bool = True, device="vk"):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(vultorch.ones(self.normalized_shape, device=device), requires_grad=True)
            self.bias = Parameter(vultorch.zeros(self.normalized_shape, device=device), requires_grad=True)
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
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-6, elementwise_affine: bool = True, device="vk"):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(vultorch.ones(self.normalized_shape, device=device), requires_grad=True)
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
                 affine: bool = True, track_running_stats: bool = True, device="vk"):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = Parameter(vultorch.ones((num_features,), device=device), requires_grad=True)
            self.bias = Parameter(vultorch.zeros((num_features,), device=device), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.register_buffer("running_mean", vultorch.zeros((num_features,), device=device))
            self.register_buffer("running_var", vultorch.ones((num_features,), device=device))
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
                 affine: bool = True, device="vk"):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Parameter(vultorch.ones((num_channels,), device=device), requires_grad=True)
            self.bias = Parameter(vultorch.zeros((num_channels,), device=device), requires_grad=True)
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
                 padding_idx: Optional[int] = None, device="vk"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = -1 if padding_idx is None else int(padding_idx)
        self.weight = Parameter(
            vultorch.randn(num_embeddings, embedding_dim, device=device),
            requires_grad=True,
        )
        if padding_idx is not None and 0 <= self.padding_idx < num_embeddings:
            with no_grad():
                p_idx = self.padding_idx
                zero_row = vultorch.zeros((1, embedding_dim), device=device)
                parts = []
                if p_idx > 0:
                    parts.append(self.weight[:p_idx])
                parts.append(zero_row)
                if p_idx + 1 < num_embeddings:
                    parts.append(self.weight[p_idx + 1:])
                self.weight.copy_(vultorch.cat(parts, dim=0, device=device))

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

# ----------------------------------------------------------------------------
# Multi-GPU DataParallel
# ----------------------------------------------------------------------------
class DataParallel(Module):
    """
    Heterogeneous Multi-GPU DataParallel container with TFLOPS-aware load balancing.
    Splits mini-batch chunks across multiple Vulkan devices proportionally to each
    GPU's measured FP32 GEMM throughput (or custom weights), computes concurrently
    across all GPUs with the GIL released, and merges outputs and gradients.
    """
    def __init__(
        self,
        module: Module,
        device_ids: Optional[List[str]] = None,
        balance: Union[bool, str] = "tflops",
        device_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if device_ids is None:
            count = vultorch.vulkan.device_count()
            device_ids = [f"vk:{i}" for i in range(count)]
        self.module = module
        self.device_ids = [str(Device(d)) for d in device_ids]
        self.balance = balance
        self.device_weights = list(device_weights) if device_weights is not None else None
        self.pool: Optional[ThreadPoolExecutor] = None
        self.replicas: Dict[str, Module] = {}
        self.last_split_sizes: List[int] = []
        self._replicas_synced: bool = False
        self.sync_every_forward: bool = False

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
        if self.device_weights is not None:
            return vultorch.vulkan.compute_split_sizes(
                batch_size, self.device_ids, weights=self.device_weights
            )
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

        # Ensure replicas are initialized (and synced if needed) before parallel dispatch
        for dev_id, _, _ in chunks:
            self._get_replica(dev_id)
        self._replicas_synced = True

        def run_device(dev_id, start, end):
            rep = self.replicas[dev_id]
            chunk_inputs = self._slice_tree(inputs, start, end, dev_id)
            chunk_kwargs = self._slice_tree(kwargs, start, end, dev_id)
            return rep(*chunk_inputs, **chunk_kwargs)

        pool = self._get_pool()
        futures = [pool.submit(run_device, dev, s, e) for dev, s, e in chunks]
        results = [f.result() for f in futures]
        return self._gather_outputs(results, target_device)

    def zero_grad(self):
        self.module.zero_grad()
        for rep in self.replicas.values():
            rep.zero_grad()
        self._replicas_synced = False

    def reduce_gradients(self):
        """
        All-reduces / accumulates gradients from all GPU replicas into master module parameters.
        """
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

