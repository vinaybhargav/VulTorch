"""
VulTorch Command-Buffer Graph Capture & Replay (`VulkanGraph` and `@vultorch.compile`).
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import functools

from .tensor import Tensor, Device, resolve_device
from . import _vultorch_backend as _backend


class VulkanGraph:
    """
    Captures a sequence of Vulkan compute dispatches and device copies into a single
    reusable `VkCommandBuffer`, replaying the entire graph in one `vkQueueSubmit`.
    """

    def __init__(self, device: Optional[Union[str, Device]] = None):
        dev = resolve_device(device)
        if dev.type != "vk":
            raise RuntimeError(
                f"VulkanGraph requires a Vulkan device ('vk:0', 'vk:1', ...), got '{dev}'"
            )
        self.device = dev
        self._native = _backend.VulkanGraph(dev.index)

    def capture_begin(self):
        self._native.capture_begin()

    def capture_end(self):
        self._native.capture_end()

    def replay(self):
        self._native.replay()

    def release(self):
        self._native.release()

    @property
    def num_dispatches(self) -> int:
        return self._native.num_dispatches

    @property
    def is_captured(self) -> bool:
        return self._native.is_captured

    def __enter__(self):
        self.capture_begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._native.release()
            return False
        self.capture_end()
        return False


def _clone_tree_static(obj):
    if isinstance(obj, Tensor):
        t = obj.detach().clone()
        t.requires_grad = obj.requires_grad
        return t
    if isinstance(obj, tuple):
        return tuple(_clone_tree_static(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_tree_static(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clone_tree_static(v) for k, v in obj.items()}
    return obj


def _copy_tree_into_static(src, dst):
    if isinstance(src, Tensor) and isinstance(dst, Tensor):
        if src._native.storage_id() != dst._native.storage_id():
            dst._native.copy_from_native(src._native)
        return
    if isinstance(src, (tuple, list)) and isinstance(dst, (tuple, list)):
        for s, d in zip(src, dst):
            _copy_tree_into_static(s, d)
        return
    if isinstance(src, dict) and isinstance(dst, dict):
        for k in dst:
            _copy_tree_into_static(src[k], dst[k])


def _shape_dtype_key(obj):
    if isinstance(obj, Tensor):
        return (str(obj.device), obj.shape, obj.dtype.name)
    if isinstance(obj, tuple):
        return ("tuple", tuple(_shape_dtype_key(x) for x in obj))
    if isinstance(obj, list):
        return ("list", tuple(_shape_dtype_key(x) for x in obj))
    if isinstance(obj, dict):
        return ("dict", tuple((k, _shape_dtype_key(v)) for k, v in sorted(obj.items())))
    return ("val", obj)


def _find_vk_device(args, kwargs) -> Optional[Device]:
    stack = list(args) + list(kwargs.values())
    while stack:
        item = stack.pop()
        if isinstance(item, Tensor):
            if item.device.type == "vk":
                return item.device
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
    return None


def compile(fn: Optional[Callable] = None, *, device: Optional[Union[str, Device]] = None):
    """
    Transparently compiles a function or `nn.Module` using `VulkanGraph` capture/replay.
    Caches captured graphs keyed by input shapes, dtypes, and devices; falls back to
    eager execution on CPU tensors or when no Vulkan dispatches are recorded.
    """
    from .nn import Module

    if fn is None:
        return lambda f: compile(f, device=device)

    if isinstance(fn, Module):
        mod = fn
        orig_forward = mod.forward
        # Infer device from module parameters if not explicitly given
        inferred_dev = device
        if inferred_dev is None:
            for p in mod.parameters():
                if p.device.type == "vk":
                    inferred_dev = p.device
                    break
        compiled_fwd = compile(orig_forward, device=inferred_dev)
        mod.forward = compiled_fwd
        return mod

    cache: Dict[Any, Tuple[Optional[VulkanGraph], Any, Any]] = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Only replay graphs when not inside an active outer graph capture and
        # when not building a Python autograd graph over dynamic intermediates.
        target_dev = _find_vk_device(args, kwargs)
        if target_dev is None and device is not None:
            d = resolve_device(device)
            if d.type == "vk":
                target_dev = d

        if target_dev is None:
            return fn(*args, **kwargs)

        key = (_shape_dtype_key(args), _shape_dtype_key(kwargs))
        if key in cache:
            graph, static_args, static_out = cache[key]
            if graph is None:
                return fn(*args, **kwargs)
            _copy_tree_into_static(args, static_args)
            graph.replay()
            return static_out

        # Warmup once before capture (ensures pipelines & autotune caches are warm)
        _ = fn(*args, **kwargs)

        static_args = _clone_tree_static(args)
        graph = VulkanGraph(target_dev)
        try:
            with graph:
                static_out = fn(*static_args, **kwargs)
        except Exception:
            graph.release()
            cache[key] = (None, None, None)
            return fn(*args, **kwargs)

        if graph.num_dispatches == 0:
            graph.release()
            cache[key] = (None, None, None)
            return static_out

        cache[key] = (graph, static_args, static_out)
        graph.replay()
        return static_out

    wrapper._vultorch_graph_cache = cache
    return wrapper
