"""
VulTorch Autograd Engine
Dynamic reverse-mode automatic differentiation DAG, custom Function,
higher-order grad(), and in-place version counter verification.
"""

from typing import List, Tuple, Callable, Optional, Set, Sequence, Union
import threading

# Thread-local rather than global: a no_grad() entered on one thread (for
# example the optimizer step) must not disable tracking in a DataParallel
# worker computing its forward pass on another thread.
_state = threading.local()


def is_grad_enabled() -> bool:
    return getattr(_state, "enabled", True)


class _set_grad_enabled:
    def __init__(self, mode: bool):
        self.mode = bool(mode)
        self.prev = True

    def __enter__(self):
        self.prev = is_grad_enabled()
        _state.enabled = self.mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _state.enabled = self.prev
        return False


class no_grad:
    """Context manager that disables gradient calculation in this thread."""

    def __enter__(self):
        self.prev = is_grad_enabled()
        _state.enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _state.enabled = self.prev
        return False


class enable_grad:
    """Context manager that enables gradient calculation in this thread."""

    def __enter__(self):
        self.prev = is_grad_enabled()
        _state.enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _state.enabled = self.prev
        return False


class Context:
    """Context for saving forward tensors and metadata for the backward pass."""

    def __init__(self):
        self._saved_tensors: Tuple = ()
        self._saved_versions: Tuple[Optional[int], ...] = ()
        self._non_differentiable: Set[int] = set()

    def save_for_backward(self, *tensors):
        self._saved_tensors = tensors
        self._saved_versions = tuple(
            t._native.version() if hasattr(t, "_native") else None for t in tensors
        )

    def mark_non_differentiable(self, *tensors):
        for t in tensors:
            self._non_differentiable.add(id(t))

    @property
    def saved_tensors(self) -> Tuple:
        for t, saved_ver in zip(self._saved_tensors, self._saved_versions):
            if saved_ver is not None and t._native.version() != saved_ver:
                raise RuntimeError(
                    "one of the variables needed for gradient computation "
                    "has been modified by an inplace operation"
                )
        return self._saved_tensors


class Node:
    """A node in the computation graph that computes backward gradients."""

    def __init__(
        self,
        name: str,
        backward_fn: Callable,
        inputs: List["Tensor"],
        outputs: Optional[List["Tensor"]] = None,
    ):
        self.name = name
        self.backward_fn = backward_fn
        self.inputs = inputs
        self.outputs = outputs  # non-None for multi-output autograd.Function
        self.post_hook: Optional[Callable] = None

    def apply(self, *grad_outputs):
        res = self.backward_fn(*grad_outputs)
        if self.post_hook is not None:
            self.post_hook()
        return res


class Function:
    """
    Base class for custom autograd operations.

    Subclasses must implement static `forward(ctx, *args)` and `backward(ctx, *grad_outputs)`.
    """

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, *grad_outputs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        from .tensor import Tensor

        ctx = Context()
        tensor_inputs: List[Tensor] = [a for a in args if isinstance(a, Tensor)]
        input_is_tensor: List[bool] = [isinstance(a, Tensor) for a in args]
        any_req_grad = any(t.requires_grad for t in tensor_inputs) and is_grad_enabled()

        with no_grad():
            raw_out = cls.forward(ctx, *args, **kwargs)

        single_out = not isinstance(raw_out, tuple)
        outs = (raw_out,) if single_out else raw_out

        diff_outs: List[Tensor] = []
        for o in outs:
            if isinstance(o, Tensor):
                req = (
                    any_req_grad
                    and (id(o) not in ctx._non_differentiable)
                    and o.dtype.is_floating_point
                )
                o.requires_grad = req
                if req:
                    o.is_leaf = False
                    diff_outs.append(o)

        if any_req_grad and diff_outs:

            def _fn_backward(*grad_outputs):
                # Ensure saved_tensors version check runs even if user didn't access
                # ctx.saved_tensors in a constant-output backward, when tensors were saved.
                _ = ctx.saved_tensors
                g_all = cls.backward(ctx, *grad_outputs)
                if not isinstance(g_all, tuple):
                    g_all = (g_all,)
                if len(g_all) != len(args):
                    raise RuntimeError(
                        f"{cls.__name__}.backward returned {len(g_all)} gradients, "
                        f"expected {len(args)} (one per forward argument)"
                    )
                return tuple(g for g, is_t in zip(g_all, input_is_tensor) if is_t)

            node = Node(
                f"{cls.__name__}Backward",
                _fn_backward,
                tensor_inputs,
                outputs=list(outs),
            )
            for o in diff_outs:
                o.grad_fn = node

        return outs[0] if single_out else outs


def _run_backward(
    root_tensors: Sequence["Tensor"],
    grad_outputs: Optional[Sequence[Optional["Tensor"]]],
    target_inputs: Optional[Sequence["Tensor"]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
    accumulate_untargeted_leaves: bool = False,
):
    import vultorch

    with _set_grad_enabled(create_graph):
        grads = {}
        used_leaf_natives = set()

        if grad_outputs is None:
            grad_outputs = [None] * len(root_tensors)

        for root, g in zip(root_tensors, grad_outputs):
            if not root.requires_grad:
                continue
            if g is None:
                if root.numel != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                g = vultorch.ones_like(root)
            if root in grads:
                grads[root] = grads[root] + g
            else:
                grads[root] = g
            used_leaf_natives.add(id(g._native))

        # Iterative post-order topological sort over computation graph
        topo_nodes: List[Tuple["Tensor", Node]] = []
        visited_tensors: Set["Tensor"] = set()
        visited_nodes: Set[int] = set()
        stack: List[Tuple["Tensor", bool]] = [(r, False) for r in reversed(root_tensors)]

        while stack:
            node_t, expanded = stack.pop()
            if expanded:
                if node_t.grad_fn is not None and id(node_t.grad_fn) not in visited_nodes:
                    visited_nodes.add(id(node_t.grad_fn))
                    topo_nodes.append((node_t, node_t.grad_fn))
                continue
            if node_t in visited_tensors:
                continue
            visited_tensors.add(node_t)
            if node_t.grad_fn is not None:
                stack.append((node_t, True))
                for inp in reversed(node_t.grad_fn.inputs):
                    if inp not in visited_tensors:
                        stack.append((inp, False))

        post_hooks: List[Callable] = []
        accumulate_leaves = target_inputs is None
        target_set = set(target_inputs) if target_inputs is not None else set()

        for owner_tensor, node in reversed(topo_nodes):
            if node.outputs is not None:
                if not any(o in grads for o in node.outputs):
                    continue
                g_outs = [
                    grads[o]
                    if o in grads
                    else (vultorch.zeros_like(o) if o.requires_grad else None)
                    for o in node.outputs
                ]
                input_grads = node.backward_fn(*g_outs)
            else:
                if owner_tensor not in grads:
                    continue
                g_out = grads[owner_tensor]
                input_grads = node.backward_fn(g_out)

            if getattr(node, "post_hook", None) is not None:
                post_hooks.append(node.post_hook)

            if not isinstance(input_grads, (list, tuple)):
                input_grads = [input_grads]

            for inp, g_in in zip(node.inputs, input_grads):
                if g_in is None or not inp.requires_grad:
                    continue

                if g_in.shape != inp.shape:
                    g_in = _reduce_broadcast_gradient(g_in, inp.shape)

                if inp in grads:
                    grads[inp] = grads[inp] + g_in
                else:
                    grads[inp] = g_in

                if inp.is_leaf and (
                    accumulate_leaves or (accumulate_untargeted_leaves and inp not in target_set)
                ):
                    if inp.grad is None:
                        nid = id(g_in._native)
                        if nid in used_leaf_natives or not g_in._native.is_dense_zero_offset():
                            inp.grad = g_in.clone()
                        else:
                            used_leaf_natives.add(nid)
                            inp.grad = g_in
                    else:
                        inp.grad = inp.grad + g_in

        for hook in post_hooks:
            hook()
        if not retain_graph:
            for _, node in topo_nodes:
                node.post_hook = None

        result = None
        if target_inputs is not None:
            out_list = []
            for inp in target_inputs:
                g = grads.get(inp, None)
                if g is None and not allow_unused:
                    raise RuntimeError(
                        "One of the differentiated Tensors appears to not have been used in the graph."
                    )
                out_list.append(g)
            result = tuple(out_list)

        grads.clear()
        visited_tensors.clear()
        visited_nodes.clear()
        topo_nodes.clear()
        return result


def backward(
    root_tensor: "Tensor",
    grad: Optional["Tensor"] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
):
    """
    Computes the gradient of the root tensor with respect to graph leaves.
    """
    if not root_tensor.requires_grad:
        return
    _run_backward(
        [root_tensor],
        [grad],
        target_inputs=None,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )


def grad(
    outputs: Union["Tensor", Sequence["Tensor"]],
    inputs: Union["Tensor", Sequence["Tensor"]],
    grad_outputs: Optional[Union["Tensor", Sequence[Optional["Tensor"]]]] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> Tuple[Optional["Tensor"], ...]:
    """
    Computes and returns the sum of gradients of outputs with respect to the inputs.
    """
    from .tensor import Tensor

    out_seq = [outputs] if isinstance(outputs, Tensor) else list(outputs)
    inp_seq = [inputs] if isinstance(inputs, Tensor) else list(inputs)

    if grad_outputs is None:
        g_seq = None
    elif isinstance(grad_outputs, Tensor):
        g_seq = [grad_outputs]
    else:
        g_seq = list(grad_outputs)

    if retain_graph is None:
        retain_graph = create_graph

    return _run_backward(
        out_seq,
        g_seq,
        target_inputs=inp_seq,
        retain_graph=bool(retain_graph),
        create_graph=bool(create_graph),
        allow_unused=bool(allow_unused),
    )


def _reduce_broadcast_gradient(grad: "Tensor", target_shape: Tuple[int, ...]) -> "Tensor":
    """Sums gradient over dimensions that were broadcasted during forward pass."""
    grad_shape = grad.shape
    ndim_diff = len(grad_shape) - len(target_shape)

    # Sum along prepended broadcast dimensions
    for _ in range(ndim_diff):
        grad = grad.sum(dim=0, keepdim=False)

    # Sum along dimensions where target_shape is 1
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(dim=i, keepdim=True)

    return grad.reshape(target_shape)
