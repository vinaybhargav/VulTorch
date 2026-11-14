"""
VulTorch Autograd Engine
Dynamic reverse-mode automatic differentiation DAG.
"""

from typing import List, Tuple, Callable, Optional, Set
import threading

# Thread-local rather than global: a no_grad() entered on one thread (for
# example the optimizer step) must not disable tracking in a DataParallel
# worker computing its forward pass on another thread.
_state = threading.local()


def is_grad_enabled() -> bool:
    return getattr(_state, "enabled", True)


class no_grad:
    """Context manager that disables gradient calculation in this thread."""
    def __enter__(self):
        self.prev = is_grad_enabled()
        _state.enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _state.enabled = self.prev
        return False

class Context:
    """Context for saving forward tensors for the backward pass."""
    def __init__(self):
        self.saved_tensors: Tuple = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

class Node:
    """A node in the computation graph that computes backward gradients."""
    def __init__(self, name: str, backward_fn: Callable, inputs: List['Tensor']):
        self.name = name
        self.backward_fn = backward_fn
        self.inputs = inputs

    def apply(self, grad_output):
        return self.backward_fn(grad_output)

def backward(root_tensor: 'Tensor', grad: Optional['Tensor'] = None):
    """
    Computes the gradient of the root tensor with respect to graph leaves.
    """
    if not root_tensor.requires_grad:
        return

    import vultorch

    with no_grad():
        if grad is None:
            if root_tensor.numel != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad = vultorch.ones_like(root_tensor)

        # Topological sort of the computation DAG
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()

        def build_topo(tensor: 'Tensor'):
            if tensor not in visited:
                visited.add(tensor)
                if tensor.grad_fn is not None:
                    for inp in tensor.grad_fn.inputs:
                        build_topo(inp)
                    topo.append(tensor)

        build_topo(root_tensor)

        # Dictionary of accumulated gradients: Tensor -> Tensor
        grads = {root_tensor: grad}
        used_leaf_natives = {id(grad._native)}

        for tensor in reversed(topo):
            if tensor not in grads or tensor.grad_fn is None:
                continue

            g_out = grads[tensor]
            input_grads = tensor.grad_fn.apply(g_out)

            if not isinstance(input_grads, (list, tuple)):
                input_grads = [input_grads]

            for inp, g_in in zip(tensor.grad_fn.inputs, input_grads):
                if g_in is None or not inp.requires_grad:
                    continue

                # Handle shape broadcasting reduction if gradient shape doesn't match input shape
                if g_in.shape != inp.shape:
                    g_in = _reduce_broadcast_gradient(g_in, inp.shape)

                if inp in grads:
                    grads[inp] = grads[inp] + g_in
                else:
                    grads[inp] = g_in

                if inp.is_leaf:
                    if inp.grad is None:
                        nid = id(g_in._native)
                        if nid in used_leaf_natives or not g_in._native.is_contiguous():
                            inp.grad = g_in.clone()
                        else:
                            used_leaf_natives.add(nid)
                            inp.grad = g_in
                    else:
                        inp.grad = inp.grad + g_in

def _reduce_broadcast_gradient(grad: 'Tensor', target_shape: Tuple[int, ...]) -> 'Tensor':
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
