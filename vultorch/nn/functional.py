from typing import Sequence, Union, Optional, Tuple
import math
import vultorch
from ..tensor import Tensor, _check_same_device
from ..autograd import Node, is_grad_enabled


def pad(
    input: Tensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    """Pad tensor according to PyTorch convention (pairs starting from last dimension)."""
    if len(pad) % 2 != 0 or len(pad) // 2 > input.ndim:
        raise ValueError(f"Invalid pad length {len(pad)} for {input.ndim}D tensor")
    mode_map = {"constant": 0, "reflect": 1, "replicate": 2}
    if mode not in mode_map:
        raise ValueError(f"Unsupported pad mode: {mode!r}")
    mode_id = mode_map[mode]

    ndim = input.ndim
    pad_before = [0] * ndim
    pad_after = [0] * ndim
    for i in range(len(pad) // 2):
        dim = ndim - 1 - i
        pad_before[dim] = int(pad[2 * i])
        pad_after[dim] = int(pad[2 * i + 1])

    native_out = input._native.pad_nd(pad_before, pad_after, mode_id, float(value))
    req_grad = input.requires_grad and is_grad_enabled()
    out = Tensor._from_native(native_out, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        in_shape = list(input.shape)
        dev = input.device

        def pad_backward(g: Tensor):
            gin_nat = g._native.pad_nd_backward(in_shape, pad_before, mode_id)
            return Tensor._from_native(gin_nat, dev, False)

        out.grad_fn = Node("PadBackward", pad_backward, [input])
    return out


def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def snake(x: Tensor, a: float = 1.0, eps: float = 1e-9) -> Tensor:
    return x.snake(a=a, eps=eps)


def snake_gamma(x: Tensor, a: float = 1.0, gamma: float = 0.0, eps: float = 1e-9) -> Tensor:
    return x.snake_gamma(a=a, gamma=gamma, eps=eps)


def snake_cos(x: Tensor, a: float = 1.0, eps: float = 1e-9) -> Tensor:
    return x.snake_cos(a=a, eps=eps)


def snake_rational(x: Tensor, a: float = 1.0, eps: float = 1e-9) -> Tensor:
    return x.snake_rational(a=a, eps=eps)


def snake_sinc(x: Tensor, a: float = 1.0, eps: float = 1e-9) -> Tensor:
    return x.snake_sinc(a=a, eps=eps)


def softplus(x: Tensor) -> Tensor:
    return (1.0 + x.exp()).log()


def selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Union[Tensor, None] = None,
) -> Tensor:
    """
    Mamba (Gu & Dao 2023) Selective State Space Model (S6) parallel scan:
        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
        y_t = sum_n (C_{n,t} * h_{n,t}) + D * u_t
    Shapes:
        u, delta: [B, D, L] (delta > 0, typically post-softplus)
        A:        [D, N]    (N <= 32)
        B, C:     [B, N, L]
        D:        [D] or None
    """
    d_nat = D._native if D is not None else None
    y_nat = u._native.selective_scan(delta._native, A._native, B._native, C._native, d_nat)
    inputs = [u, delta, A, B, C] + ([D] if D is not None else [])
    req_grad = any(t.requires_grad for t in inputs) and is_grad_enabled()
    out = Tensor._from_native(y_nat, u.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = u.device
        has_D = D is not None

        def ssm_backward(g: Tensor):
            du_n, ddt_n, dA_n, dB_n, dC_n, dD_n = u._native.selective_scan_backward(
                g.contiguous()._native, delta._native, A._native, B._native, C._native, d_nat
            )
            du = Tensor._from_native(du_n, dev, False) if u.requires_grad else None
            ddt = Tensor._from_native(ddt_n, dev, False) if delta.requires_grad else None
            dA = Tensor._from_native(dA_n, dev, False) if A.requires_grad else None
            dB = Tensor._from_native(dB_n, dev, False) if B.requires_grad else None
            dC = Tensor._from_native(dC_n, dev, False) if C.requires_grad else None
            if has_D:
                dD = Tensor._from_native(dD_n, dev, False) if D.requires_grad else None
                return du, ddt, dA, dB, dC, dD
            return du, ddt, dA, dB, dC

        out.grad_fn = Node("SelectiveScanBackward", ssm_backward, inputs)
    return out


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.softmax(dim=dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.log_softmax(dim=dim)


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    epilogue: Optional[str] = None,
) -> Tensor:
    ep_map = {None: 0, "none": 0, "relu": 1, "gelu": 2, "silu": 3}
    if epilogue not in ep_map:
        raise ValueError(f"Unsupported linear epilogue: {epilogue!r}")
    ep_code = ep_map[epilogue]
    if ep_code != 0 and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)) and is_grad_enabled():
        # When training with gradients enabled, decompose activation so standard autograd tracks it
        base = linear(x, weight, bias, epilogue=None)
        if ep_code == 1:
            return base.relu()
        if ep_code == 2:
            return base.gelu()
        return base.silu()

    orig_shape = x.shape
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    if x.ndim > 2:
        m_total = x.numel // in_features
        x_2d = x.contiguous().reshape(m_total, in_features)
        out_2d = x_2d.matmul(weight, trans_b=True, bias=bias, epilogue=ep_code)
        return out_2d.reshape(*orig_shape[:-1], out_features)
    return x.matmul(weight, trans_b=True, bias=bias, epilogue=ep_code)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Fused Slang FlashAttention-2 (Dao et al. 2023) Scaled Dot-Product Attention:
      - Zero O(S^2) global memory allocation when attn_mask is None.
      - Supports causal masking (`is_causal=True`), custom `scale`, and Grouped-Query
        Attention (GQA: `query` has `H_q` heads, `key`/`value` have `H_kv` heads where `H_q % H_kv == 0`).
      - Supports automatic mixed precision (`vultorch.amp.autocast`).
    """
    _check_same_device(query, key, "scaled_dot_product_attention")
    _check_same_device(query, value, "scaled_dot_product_attention")

    from ..amp import is_autocast_enabled, get_autocast_dtype
    q_in, k_in, v_in = query, key, value
    if is_autocast_enabled():
        amp_dt = get_autocast_dtype()
        if q_in.dtype != amp_dt:
            q_in = q_in.to(amp_dt)
        if k_in.dtype != amp_dt:
            k_in = k_in.to(amp_dt)
        if v_in.dtype != amp_dt:
            v_in = v_in.to(amp_dt)

    d_head = q_in.shape[-1]
    eff_scale = float(scale) if scale is not None else (1.0 / math.sqrt(d_head))

    # Fallback path only if an explicit arbitrary additive/boolean attn_mask tensor is provided
    if attn_mask is not None:
        h_q = q_in.shape[1]
        h_kv = k_in.shape[1]
        if h_q != h_kv:
            rep = h_q // h_kv
            k_exp = k_in.repeat_interleave(rep, dim=1)
            v_exp = v_in.repeat_interleave(rep, dim=1)
        else:
            k_exp, v_exp = k_in, v_in
        scores = (q_in @ k_exp.transpose(-2, -1)) * eff_scale
        if attn_mask.dtype.name == "bool":
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        else:
            scores = scores + attn_mask
        probs = scores.softmax(dim=-1)
        return probs @ v_exp

    out_nat, lse_nat = q_in._native.flash_attention(
        k_in._native, v_in._native, bool(is_causal), float(eff_scale)
    )
    req_grad = (query.requires_grad or key.requires_grad or value.requires_grad) and is_grad_enabled()
    out = Tensor._from_native(out_nat, query.device, req_grad)

    if req_grad:
        out.is_leaf = False
        dev = query.device
        causal_flag = bool(is_causal)

        def flash_attn_backward(g: Tensor):
            dq_nat, dk_nat, dv_nat = q_in._native.flash_attention_backward(
                g.contiguous()._native, k_in._native, v_in._native,
                out_nat, lse_nat, causal_flag, float(eff_scale)
            )
            dq = Tensor._from_native(dq_nat, dev, False) if query.requires_grad else None
            dk = Tensor._from_native(dk_nat, dev, False) if key.requires_grad else None
            dv = Tensor._from_native(dv_nat, dev, False) if value.requires_grad else None
            if dq is not None and dq.dtype != query.dtype:
                dq = dq.to(query.dtype)
            if dk is not None and dk.dtype != key.dtype:
                dk = dk.to(key.dtype)
            if dv is not None and dv.dtype != value.dtype:
                dv = dv.to(value.dtype)
            return dq, dk, dv

        out.grad_fn = Node("FlashAttentionBackward", flash_attn_backward, [query, key, value])
    return out


def apply_rotary_pos_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seq_dim: int = 2,
) -> Tensor:
    """
    Applies fused Rotary Position Embedding (RoPE) to 4D tensor `x` (`[B, H, S, D]` when `seq_dim=2`
    or `[B, S, H, D]` when `seq_dim=1`) using precomputed `cos` and `sin` tables (`[S, D/2]` or `[S, D]`).
    """
    _check_same_device(x, cos, "apply_rotary_pos_emb")
    _check_same_device(x, sin, "apply_rotary_pos_emb")
    norm_seq_dim = seq_dim if seq_dim >= 0 else seq_dim + x.ndim
    out_nat = x._native.rope(cos._native, sin._native, int(norm_seq_dim), False)
    req_grad = x.requires_grad and is_grad_enabled()
    out = Tensor._from_native(out_nat, x.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = x.device
        c_nat = cos._native
        s_nat = sin._native

        def rope_backward(g: Tensor):
            dx_nat = g.contiguous()._native.rope(c_nat, s_nat, int(norm_seq_dim), True)
            return Tensor._from_native(dx_nat, dev, False)

        out.grad_fn = Node("RoPEBackward", rope_backward, [x])
    return out


def swiglu(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """
    Fused SwiGLU activation: `SiLU(a) * b`.
    If `b` is None, splits `a` in half along the last dimension (`a, b = a.chunk(2, dim=-1)`).
    """
    if b is None:
        a, b = a.chunk(2, dim=-1)
    _check_same_device(a, b, "swiglu")
    out_nat = a._native.glu(b._native, 0)
    req_grad = (a.requires_grad or b.requires_grad) and is_grad_enabled()
    out = Tensor._from_native(out_nat, a.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = a.device

        def swiglu_backward(g: Tensor):
            da_nat, db_nat = a._native.glu_backward(g.contiguous()._native, b._native, 0)
            da = Tensor._from_native(da_nat, dev, False) if a.requires_grad else None
            db = Tensor._from_native(db_nat, dev, False) if b.requires_grad else None
            return da, db

        out.grad_fn = Node("SwiGLUBackward", swiglu_backward, [a, b])
    return out


def geglu(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """
    Fused GeGLU activation: `GELU(a) * b`.
    If `b` is None, splits `a` in half along the last dimension (`a, b = a.chunk(2, dim=-1)`).
    """
    if b is None:
        a, b = a.chunk(2, dim=-1)
    _check_same_device(a, b, "geglu")
    out_nat = a._native.glu(b._native, 1)
    req_grad = (a.requires_grad or b.requires_grad) and is_grad_enabled()
    out = Tensor._from_native(out_nat, a.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = a.device

        def geglu_backward(g: Tensor):
            da_nat, db_nat = a._native.glu_backward(g.contiguous()._native, b._native, 1)
            da = Tensor._from_native(da_nat, dev, False) if a.requires_grad else None
            db = Tensor._from_native(db_nat, dev, False) if b.requires_grad else None
            return da, db

        out.grad_fn = Node("GeGLUBackward", geglu_backward, [a, b])
    return out


def _autodiff_unary(x: Tensor, op_type: int, name: str, param: float = 0.0) -> Tensor:
    out_nat = x._native.autodiff_unary_fwd(op_type, float(param))
    req_grad = x.requires_grad and is_grad_enabled()
    out = Tensor._from_native(out_nat, x.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = x.device
        x_ref = x.detach()

        def _bwd(g: Tensor):
            dx_nat = x_ref._native.autodiff_unary_bwd(
                op_type, g.contiguous()._native, float(param)
            )
            return Tensor._from_native(dx_nat, dev, False)

        out.grad_fn = Node(f"{name}Backward", _bwd, [x])
    return out


def mish(x: Tensor) -> Tensor:
    """Mish activation backed by Slang `[Differentiable]` bwd_diff kernel."""
    return _autodiff_unary(x, 0, "Mish")


def hardswish(x: Tensor) -> Tensor:
    """HardSwish activation backed by Slang `[Differentiable]` bwd_diff kernel."""
    return _autodiff_unary(x, 1, "HardSwish")


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """LeakyReLU activation backed by Slang `[Differentiable]` bwd_diff kernel."""
    return _autodiff_unary(x, 2, "LeakyReLU", float(negative_slope))


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """ELU activation backed by Slang `[Differentiable]` bwd_diff kernel."""
    return _autodiff_unary(x, 3, "ELU", float(alpha))


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    """Smooth L1 (Huber) loss backed by Slang `[Differentiable]` bwd_diff kernel."""
    _check_same_device(input, target, "smooth_l1_loss")
    out_nat = input._native.autodiff_binary_fwd(0, target._native, float(beta))
    req_grad = (input.requires_grad or target.requires_grad) and is_grad_enabled()
    out = Tensor._from_native(out_nat, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = input.device
        a_ref = input.detach()
        b_ref = target.detach()

        def _bwd(g: Tensor):
            da_nat, db_nat = a_ref._native.autodiff_binary_bwd(
                0, b_ref._native, g.contiguous()._native, float(beta)
            )
            da = Tensor._from_native(da_nat, dev, False) if input.requires_grad else None
            db = Tensor._from_native(db_nat, dev, False) if target.requires_grad else None
            return da, db

        out.grad_fn = Node("SmoothL1Backward", _bwd, [input, target])

    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    return out


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Computes cosine similarity between `x1` and `x2` along `dim`."""
    _check_same_device(x1, x2, "cosine_similarity")
    num = (x1 * x2).sum(dim=dim)
    n1 = (x1 * x1).sum(dim=dim).sqrt().clamp(min=eps)
    n2 = (x2 * x2).sum(dim=dim).sqrt().clamp(min=eps)
    return num / (n1 * n2)


def selu(x: Tensor) -> Tensor:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * elu(x, alpha=alpha)


def hardsigmoid(x: Tensor) -> Tensor:
    return ((x + 3.0) / 6.0).clamp(min=0.0, max=1.0)


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    denom = input.norm(p=p, dim=dim, keepdim=True).clamp(min=eps)
    return input / denom


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p <= 0.0:
        return input
    if p >= 1.0:
        return vultorch.zeros_like(input)
    seed, offset = vultorch._next_philox_state()
    out_nat = input._native.dropout_philox(float(p), seed, offset)
    req_grad = input.requires_grad and is_grad_enabled()
    out = Tensor._from_native(out_nat, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = input.device
        def _bwd(g: Tensor):
            dx_nat = g._native.dropout_philox(float(p), seed, offset)
            return Tensor._from_native(dx_nat, dev, False)
        out.grad_fn = Node("DropoutBackward", _bwd, [input])
    return out


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = input - target
    sq = diff * diff
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    return sq


def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = (input - target).abs()
    if reduction == "mean":
        return diff.mean()
    if reduction == "sum":
        return diff.sum()
    return diff


def binary_cross_entropy_with_logits(
    input: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    # Numerically stable: max(x, 0) - x * target + log(1 + exp(-|x|))
    loss = input.relu() - input * target + ((-input.abs()).exp() + 1.0).log()
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def binary_cross_entropy(
    input: Tensor, target: Tensor, reduction: str = "mean", eps: float = 1e-7
) -> Tensor:
    p = input.clamp(min=eps, max=1.0 - eps)
    loss = -(target * p.log() + (1.0 - target) * (1.0 - p).log())
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    from . import CrossEntropyLoss
    return CrossEntropyLoss(reduction=reduction)(input, target)


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    one_hot_t = one_hot(target, num_classes=input.shape[-1]).cast(input.dtype)
    loss = -(one_hot_t * input).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def kl_div(input: Tensor, target: Tensor, reduction: str = "mean", log_target: bool = False) -> Tensor:
    if log_target:
        loss = target.exp() * (target - input)
    else:
        safe_t = target.clamp(min=1e-12)
        loss = target * (safe_t.log() - input)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    if num_classes <= 0:
        num_classes = int(tensor.amax().item()) + 1
    z = vultorch.zeros(*tensor.shape, num_classes, device=tensor.device, dtype=vultorch.int64)
    idx = tensor.unsqueeze(-1).cast(vultorch.int64)
    return z.scatter(-1, idx, 1)


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride=1,
    padding=0,
    groups: int = 1,
) -> Tensor:
    sh, sw = (stride, stride) if isinstance(stride, int) else tuple(stride)
    ph, pw = (padding, padding) if isinstance(padding, int) else tuple(padding)
    out_nat = input._native.conv2d(
        weight._native,
        bias._native if bias is not None else None,
        sh, sw, ph, pw, groups, True,
    )
    req_grad = (
        input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    ) and is_grad_enabled()
    out = Tensor._from_native(out_nat, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = input.device
        in_shape = list(input.shape)
        w_shape = list(weight.shape)
        inputs = [input, weight] + ([bias] if bias is not None else [])
        def _bwd(g: Tensor):
            gc = g.contiguous()
            dx = (
                Tensor._from_native(
                    gc._native.conv2d_backward_input(weight._native, in_shape, sh, sw, ph, pw, groups, True),
                    dev, False,
                )
                if input.requires_grad else None
            )
            dw = (
                Tensor._from_native(
                    gc._native.conv2d_backward_weight(input._native, w_shape, sh, sw, ph, pw, groups, True),
                    dev, False,
                )
                if weight.requires_grad else None
            )
            res = [dx, dw]
            if bias is not None:
                db = gc.sum(dim=(0, 2, 3)) if bias.requires_grad else None
                res.append(db)
            return tuple(res)
        out.grad_fn = Node("Conv2dBackward", _bwd, inputs)
    return out


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    if stride is None:
        sh, sw = kh, kw
    else:
        sh, sw = (stride, stride) if isinstance(stride, int) else tuple(stride)
    ph, pw = (padding, padding) if isinstance(padding, int) else tuple(padding)
    out_nat, idx_nat = input._native.maxpool2d(kh, kw, sh, sw, ph, pw)
    req_grad = input.requires_grad and is_grad_enabled()
    out = Tensor._from_native(out_nat, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = input.device
        in_shape = list(input.shape)
        def _bwd(g: Tensor):
            dx = g.contiguous()._native.maxpool2d_backward(idx_nat, in_shape, kh, kw, sh, sw, ph, pw)
            return Tensor._from_native(dx, dev, False)
        out.grad_fn = Node("MaxPool2dBackward", _bwd, [input])
    return out


def avg_pool2d(input: Tensor, kernel_size, stride=None, padding=0, count_include_pad: bool = True) -> Tensor:
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    if stride is None:
        sh, sw = kh, kw
    else:
        sh, sw = (stride, stride) if isinstance(stride, int) else tuple(stride)
    ph, pw = (padding, padding) if isinstance(padding, int) else tuple(padding)
    out_nat = input._native.avgpool2d(kh, kw, sh, sw, ph, pw, count_include_pad, False, 1, 1)
    req_grad = input.requires_grad and is_grad_enabled()
    out = Tensor._from_native(out_nat, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = input.device
        in_shape = list(input.shape)
        def _bwd(g: Tensor):
            dx = g.contiguous()._native.avgpool2d_backward(in_shape, kh, kw, sh, sw, ph, pw, count_include_pad, False)
            return Tensor._from_native(dx, dev, False)
        out.grad_fn = Node("AvgPool2dBackward", _bwd, [input])
    return out


def layer_norm(
    input: Tensor,
    normalized_shape,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    norm_dims = len(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else 1
    out_nat = input._native.layernorm(
        weight._native if weight is not None else None,
        bias._native if bias is not None else None,
        float(eps), norm_dims,
    )
    req_grad = (
        input.requires_grad
        or (weight is not None and weight.requires_grad)
        or (bias is not None and bias.requires_grad)
    ) and is_grad_enabled()
    out = Tensor._from_native(out_nat, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = input.device
        inputs = [input] + ([weight] if weight is not None else []) + ([bias] if bias is not None else [])
        def _bwd(g: Tensor):
            dx, dw, db = input._native.layernorm_backward(
                g.contiguous()._native,
                weight._native if weight is not None else None,
                float(eps), norm_dims,
                input.requires_grad,
                weight is not None and weight.requires_grad,
                bias is not None and bias.requires_grad,
            )
            res = [Tensor._from_native(dx, dev, False) if input.requires_grad else None]
            if weight is not None:
                res.append(Tensor._from_native(dw, dev, False) if weight.requires_grad else None)
            if bias is not None:
                res.append(Tensor._from_native(db, dev, False) if bias.requires_grad else None)
            return tuple(res)
        out.grad_fn = Node("LayerNormBackward", _bwd, inputs)
    return out


def rms_norm(
    input: Tensor,
    normalized_shape,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    norm_dims = len(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else 1
    out_nat = input._native.rmsnorm(
        weight._native if weight is not None else None, float(eps), norm_dims
    )
    req_grad = (input.requires_grad or (weight is not None and weight.requires_grad)) and is_grad_enabled()
    out = Tensor._from_native(out_nat, input.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = input.device
        inputs = [input] + ([weight] if weight is not None else [])
        def _bwd(g: Tensor):
            dx, dw = input._native.rmsnorm_backward(
                g.contiguous()._native,
                weight._native if weight is not None else None,
                float(eps), norm_dims,
                input.requires_grad,
                weight is not None and weight.requires_grad,
            )
            res = [Tensor._from_native(dx, dev, False) if input.requires_grad else None]
            if weight is not None:
                res.append(Tensor._from_native(dw, dev, False) if weight.requires_grad else None)
            return tuple(res)
        out.grad_fn = Node("RMSNormBackward", _bwd, inputs)
    return out


def embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None) -> Tensor:
    pad = -1 if padding_idx is None else int(padding_idx)
    out_nat = input._native.embedding(weight._native, pad)
    req_grad = weight.requires_grad and is_grad_enabled()
    out = Tensor._from_native(out_nat, weight.device, req_grad)
    if req_grad:
        out.is_leaf = False
        dev = weight.device
        num_emb = weight.shape[0]
        idx_ref = input.detach()
        def _bwd(g: Tensor):
            dw = g.contiguous()._native.embedding_backward(idx_ref._native, num_emb, pad)
            return Tensor._from_native(dw, dev, False)
        out.grad_fn = Node("EmbeddingBackward", _bwd, [weight])
    return out


def interpolate(
    input: Tensor,
    size=None,
    scale_factor=None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    H, W = input.shape[-2], input.shape[-1]
    if size is not None:
        oh, ow = (size, size) if isinstance(size, int) else tuple(size)
    elif scale_factor is not None:
        sh, sw = (scale_factor, scale_factor) if isinstance(scale_factor, (int, float)) else tuple(scale_factor)
        oh, ow = int(H * sh), int(W * sw)
    else:
        raise ValueError("interpolate requires size or scale_factor")

    dev = input.device
    if mode == "nearest":
        if oh % H == 0 and ow % W == 0:
            return input.repeat_interleave(oh // H, dim=-2).repeat_interleave(ow // W, dim=-1)
        iy = vultorch.linspace(0, H - 1, oh, device=dev).round().clamp(0, H - 1).cast(vultorch.int32)
        ix = vultorch.linspace(0, W - 1, ow, device=dev).round().clamp(0, W - 1).cast(vultorch.int32)
        return input.index_select(-2, iy).index_select(-1, ix)

    # Bilinear interpolation using GPU index_select + lerp (full autograd)
    ac = bool(align_corners) if align_corners is not None else False
    if ac and oh > 1:
        ys = vultorch.linspace(0.0, float(H - 1), oh, device=dev)
    else:
        ys = ((vultorch.arange(oh, device=dev) + 0.5) * (float(H) / float(oh)) - 0.5).clamp(0.0, float(H - 1))
    if ac and ow > 1:
        xs = vultorch.linspace(0.0, float(W - 1), ow, device=dev)
    else:
        xs = ((vultorch.arange(ow, device=dev) + 0.5) * (float(W) / float(ow)) - 0.5).clamp(0.0, float(W - 1))

    y0 = ys.floor().clamp(0, H - 1).cast(vultorch.int32)
    y1 = (ys.floor() + 1.0).clamp(0, H - 1).cast(vultorch.int32)
    wy = (ys - ys.floor()).reshape((1, 1, oh, 1))

    x0 = xs.floor().clamp(0, W - 1).cast(vultorch.int32)
    x1 = (xs.floor() + 1.0).clamp(0, W - 1).cast(vultorch.int32)
    wx = (xs - xs.floor()).reshape((1, 1, 1, ow))

    top = input.index_select(-2, y0)
    bot = input.index_select(-2, y1)
    v_interp = top.lerp(bot, wy)
    left = v_interp.index_select(-1, x0)
    right = v_interp.index_select(-1, x1)
    return left.lerp(right, wx)
