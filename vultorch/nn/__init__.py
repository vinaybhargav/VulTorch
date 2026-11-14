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
        """Copies master module parameters to all created replicas."""
        master_params = list(self.module.named_parameters())
        if not master_params:
            return
        master_dev = master_params[0][1].device
        with no_grad():
            for dev_id, replica in self.replicas.items():
                if Device(dev_id) == master_dev or replica is self.module:
                    continue
                for (_, p_src), (_, p_dst) in zip(master_params, replica.named_parameters()):
                    p_dst.copy_(p_src.to(dev_id))
        self._replicas_synced = True

    def _get_replica(self, device: str) -> Module:
        target_dev = Device(device)
        master_params = list(self.module.named_parameters())
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
        return self.replicas[device]

    def forward(self, x: Tensor) -> Tensor:
        if len(self.device_ids) <= 1:
            return self.module(x)

        batch_size = x.shape[0]
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
            return rep(x.to(dev_id)).to(x.device)

        # Ensure replicas are initialized (and synced if needed) before parallel dispatch
        for dev_id, _, _ in chunks:
            self._get_replica(dev_id)
        self._replicas_synced = True

        def run_device(dev_id, start, end):
            rep = self.replicas[dev_id]
            chunk_x = x[start:end].to(dev_id)
            return rep(chunk_x)

        pool = self._get_pool()
        futures = [pool.submit(run_device, dev, s, e) for dev, s, e in chunks]
        results = [f.result() for f in futures]
        return vultorch.cat(results, dim=0, device=x.device)

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

