"""
Activation Checkpointing (vultorch.utils.checkpoint)
Trades extra recomputation during backward for大幅 reduced peak activation VRAM.
"""

from typing import Callable, Tuple, Any
import vultorch
from ..autograd import Function, no_grad, enable_grad, _run_backward


class _CheckpointFunction(Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, _param_sentinel, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.had_autocast = vultorch.amp.is_autocast_enabled()
        ctx.autocast_dtype = vultorch.amp.get_autocast_dtype()

        tensor_indices = []
        tensor_inputs = []
        non_tensor_inputs = {}
        for idx, arg in enumerate(args):
            if isinstance(arg, vultorch.Tensor):
                tensor_indices.append(idx)
                tensor_inputs.append(arg)
            else:
                non_tensor_inputs[idx] = arg

        ctx.tensor_indices = tensor_indices
        ctx.non_tensor_inputs = non_tensor_inputs
        ctx.num_args = len(args)
        ctx.save_for_backward(*tensor_inputs)

        with no_grad():
            return run_function(*args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved = ctx.saved_tensors
        detached_inputs = []
        rebuilt_args = [None] * ctx.num_args

        for idx, t in zip(ctx.tensor_indices, saved):
            t_det = t.detach()
            t_det.requires_grad = t.requires_grad
            if t.requires_grad:
                detached_inputs.append(t_det)
            rebuilt_args[idx] = t_det

        for idx, val in ctx.non_tensor_inputs.items():
            rebuilt_args[idx] = val

        with enable_grad():
            with vultorch.amp.autocast(
                dtype=ctx.autocast_dtype, enabled=ctx.had_autocast
            ):
                outputs = ctx.run_function(*rebuilt_args)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        diff_outputs = []
        diff_grads = []
        for out, g in zip(outputs, grad_outputs):
            if isinstance(out, vultorch.Tensor) and out.requires_grad:
                diff_outputs.append(out)
                diff_grads.append(g)

        if diff_outputs:
            input_grads = _run_backward(
                diff_outputs,
                diff_grads,
                target_inputs=detached_inputs,
                allow_unused=True,
                accumulate_untargeted_leaves=True,
            ) or ()
        else:
            input_grads = tuple(None for _ in detached_inputs)

        grad_iter = iter(input_grads)
        arg_grads = [None] * ctx.num_args
        for idx, t in zip(ctx.tensor_indices, saved):
            if t.requires_grad:
                arg_grads[idx] = next(grad_iter)

        return (None, None, None, *arg_grads)


def checkpoint(
    function: Callable,
    *args: Any,
    use_reentrant: bool = True,
    preserve_rng_state: bool = False,
):
    """
    Checkpoints `function(*args)` by running forward under `no_grad()` and
    recomputing intermediate activations during backward.
    """
    _ = use_reentrant
    sentinel = None
    if vultorch.is_grad_enabled() and not any(
        isinstance(a, vultorch.Tensor) and a.requires_grad for a in args
    ):
        if isinstance(function, vultorch.nn.Module) and any(
            p.requires_grad for p in function.parameters()
        ):
            sentinel = vultorch.zeros(1, device="cpu", requires_grad=True)
    return _CheckpointFunction.apply(function, preserve_rng_state, sentinel, *args)
