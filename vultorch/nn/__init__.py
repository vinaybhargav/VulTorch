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
from ..autograd import is_grad_enabled, Node

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
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self.training = True

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def register_parameter(self, name: str, param: Optional[Parameter]):
        self._parameters[name] = param

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
        w_data = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32)
        self.weight = Parameter(vultorch.tensor(w_data, device=device), requires_grad=True)

        if bias:
            b_data = np.random.uniform(-bound, bound, (out_features,)).astype(np.float32)
            self.bias = Parameter(vultorch.tensor(b_data, device=device), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # y = x @ W^T + b
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out

class Conv2d(Module):
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

        # Kaiming init
        k = 1.0 / math.sqrt(in_channels * self.kH * self.kW)
        w_data = np.random.uniform(-k, k, (out_channels, in_channels, self.kH, self.kW)).astype(np.float32)
        self.weight = Parameter(vultorch.tensor(w_data, device=device), requires_grad=True)

        if bias:
            b_data = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(vultorch.tensor(b_data, device=device), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        b_native = self.bias._native if self.bias is not None else None
        native_out = x._native.conv2d(self.weight._native, b_native,
                                      self.stride_h, self.stride_w,
                                      self.pad_h, self.pad_w)
        req_grad = x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)
        out = Tensor(native_out, device=x.device, requires_grad=req_grad)

        if is_grad_enabled() and req_grad:
            out.is_leaf = False
            sh, sw, ph, pw = self.stride_h, self.stride_w, self.pad_h, self.pad_w
            def conv_backward(grad_out: Tensor):
                g_in = None
                g_w = None
                g_b = None

                if x.requires_grad:
                    gin_native = grad_out._native.conv2d_backward_input(
                        self.weight._native, x.shape, sh, sw, ph, pw)
                    g_in = Tensor(gin_native, device=x.device)

                if self.weight.requires_grad:
                    gw_native = grad_out._native.conv2d_backward_weight(
                        x._native, self.weight.shape, sh, sw, ph, pw)
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
        # Spatial pooling forward
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad_h - self.kH) // self.stride_h + 1
        out_w = (W + 2 * self.pad_w - self.kW) // self.stride_w + 1

        x_np = x.numpy()
        out_np = np.zeros((N, C, out_h, out_w), dtype=np.float32)

        for oh in range(out_h):
            for ow in range(out_w):
                ih_start = oh * self.stride_h - self.pad_h
                iw_start = ow * self.stride_w - self.pad_w
                ih_end = min(ih_start + self.kH, H)
                iw_end = min(iw_start + self.kW, W)
                ih_start = max(ih_start, 0)
                iw_start = max(iw_start, 0)
                window = x_np[:, :, ih_start:ih_end, iw_start:iw_end]
                out_np[:, :, oh, ow] = np.max(window, axis=(2, 3))

        out = Tensor(out_np, device=x.device, requires_grad=x.requires_grad)
        if is_grad_enabled() and x.requires_grad:
            out.is_leaf = False
            def pool_backward(grad_out: Tensor):
                g_in = np.zeros_like(x_np)
                g_out_np = grad_out.numpy()
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih_start = oh * self.stride_h - self.pad_h
                        iw_start = ow * self.stride_w - self.pad_w
                        ih_end = min(ih_start + self.kH, H)
                        iw_end = min(iw_start + self.kW, W)
                        ih_start = max(ih_start, 0)
                        iw_start = max(iw_start, 0)
                        window = x_np[:, :, ih_start:ih_end, iw_start:iw_end]
                        max_mask = (window == np.max(window, axis=(2, 3), keepdims=True))
                        g_in[:, :, ih_start:ih_end, iw_start:iw_end] += (
                            max_mask * g_out_np[:, :, oh:oh+1, ow:ow+1]
                        )
                return Tensor(g_in, device=x.device)
            out.grad_fn = Node("MaxPool2dBackward", pool_backward, [x])
        return out

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
        if x.ndim < len(self.normalized_shape):
            raise ValueError(
                f"LayerNorm expected at least {len(self.normalized_shape)} dims, got {x.ndim}"
            )

        g_native = self.weight._native if self.weight is not None else None
        b_native = self.bias._native if self.bias is not None else None
        native_out = x._native.layernorm(g_native, b_native, self.eps)

        req_grad = (
            x.requires_grad
            or (self.weight is not None and self.weight.requires_grad)
            or (self.bias is not None and self.bias.requires_grad)
        ) and is_grad_enabled()

        out = Tensor(native_out, device=x.device, requires_grad=req_grad)
        if not req_grad:
            return out

        out.is_leaf = False
        ndim_norm = len(self.normalized_shape)
        dims = tuple(range(x.ndim - ndim_norm, x.ndim))
        outer_dims = tuple(range(0, x.ndim - ndim_norm))
        D = 1
        for d in self.normalized_shape:
            D *= d
        eps = self.eps
        weight, bias = self.weight, self.bias

        def layernorm_backward(g: Tensor):
            # Analytic backward. Statistics are recomputed here rather than
            # cached, so the forward pass stays a single dispatch.
            mean = x.mean(dim=dims, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
            std = (var + eps).sqrt()
            xhat = (x - mean) / std

            gy = g * weight if weight is not None else g
            sum_gy = gy.sum(dim=dims, keepdim=True)
            sum_gy_xhat = (gy * xhat).sum(dim=dims, keepdim=True)

            inv_d = 1.0 / D
            dx = (gy - sum_gy * inv_d - xhat * sum_gy_xhat * inv_d) / std

            grads = [dx if x.requires_grad else None]
            if weight is not None:
                grads.append(((g * xhat).sum(dim=outer_dims)) if weight.requires_grad else None)
            if bias is not None:
                grads.append((g.sum(dim=outer_dims)) if bias.requires_grad else None)
            return tuple(grads)

        inputs = [x]
        if self.weight is not None:
            inputs.append(self.weight)
        if self.bias is not None:
            inputs.append(self.bias)
        out.grad_fn = Node("LayerNormBackward", layernorm_backward, inputs)
        return out

class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask_np = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        mask = Tensor(mask_np, device=x.device)
        return x * mask

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device="vk"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self.weight = Parameter(vultorch.tensor(w_data, device=device), requires_grad=True)

    def forward(self, indices: Tensor) -> Tensor:
        idx_np = indices.numpy().astype(int)
        out_np = self.weight.numpy()[idx_np]
        out = Tensor(out_np, device=indices.device, requires_grad=self.weight.requires_grad)
        if is_grad_enabled() and self.weight.requires_grad:
            out.is_leaf = False
            def emb_backward(g: Tensor):
                grad_w = np.zeros_like(self.weight.numpy())
                np.add.at(grad_w, idx_np, g.numpy())
                return Tensor(grad_w, device=indices.device)
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
        log_probs = pred.log_softmax(dim=-1)
        # target can be integer class indices or one-hot probabilities
        if target.ndim == 1 or (target.ndim == 2 and target.shape[-1] == 1):
            indices = target.numpy().astype(int).flatten()
            N = pred.shape[0]
            log_p_np = log_probs.numpy()
            nll = -log_p_np[np.arange(N), indices]

            loss_val = np.mean(nll) if self.reduction == "mean" else np.sum(nll)
            out = Tensor(np.array([loss_val], dtype=np.float32), device=pred.device, requires_grad=pred.requires_grad)

            if is_grad_enabled() and pred.requires_grad:
                out.is_leaf = False
                probs_np = pred.softmax(dim=-1).numpy()
                probs_np[np.arange(N), indices] -= 1.0
                if self.reduction == "mean":
                    probs_np /= N
                grad_pred = Tensor(probs_np, device=pred.device)
                out.grad_fn = Node("CrossEntropyBackward", lambda g: g * grad_pred, [pred])
            return out
        else:
            # One-hot target
            nll = -(target * log_probs).sum(dim=-1)
            return nll.mean() if self.reduction == "mean" else nll.sum()

# ----------------------------------------------------------------------------
# Multi-GPU DataParallel
# ----------------------------------------------------------------------------
class DataParallel(Module):
    """
    Heterogeneous Multi-GPU DataParallel container.
    Splits mini-batch chunks across multiple Vulkan devices (e.g. AMD 7900 XTX 'vk:0' + NVIDIA RTX 3080 'vk:1'),
    computes concurrently on both GPUs, and merges outputs!
    """
    def __init__(self, module: Module, device_ids: Optional[List[str]] = None):
        super().__init__()
        if device_ids is None:
            count = vultorch.vulkan.device_count()
            device_ids = [f"vk:{i}" for i in range(count)]
        self.module = module
        self.device_ids = device_ids
        self.pool = ThreadPoolExecutor(max_workers=len(device_ids))
        self.replicas: Dict[str, Module] = {}

    def _get_replica(self, device: str) -> Module:
        if device not in self.replicas:
            import copy
            replica = copy.deepcopy(self.module)
            replica.to(device)
            self.replicas[device] = replica
        else:
            # Sync parameters
            replica = self.replicas[device]
            for (name, p_src), (_, p_dst) in zip(self.module.named_parameters(), replica.named_parameters()):
                p_dst.copy_(p_src.to(device))
        return self.replicas[device]

    def forward(self, x: Tensor) -> Tensor:
        if len(self.device_ids) <= 1:
            return self.module(x)

        # Split batch dimension
        batch_size = x.shape[0]
        n_devs = len(self.device_ids)
        chunk_size = (batch_size + n_devs - 1) // n_devs

        chunks = []
        for i in range(n_devs):
            start = i * chunk_size
            end = min(start + chunk_size, batch_size)
            if start < batch_size:
                chunk = x[start:end].to(self.device_ids[i])
                chunks.append((self.device_ids[i], chunk))

        def run_device(dev_id, chunk_x):
            rep = self._get_replica(dev_id)
            return rep(chunk_x).to(x.device)

        futures = [self.pool.submit(run_device, dev, chunk) for dev, chunk in chunks]
        results = [f.result() for f in futures]
        return vultorch.cat(results, dim=0)

    def zero_grad(self):
        self.module.zero_grad()
        for rep in self.replicas.values():
            rep.zero_grad()

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

