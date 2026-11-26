"""
Linear Algebra Operations (vultorch.linalg)
"""

from typing import Optional, Union, Tuple
import vultorch
from .tensor import Tensor

def norm(A: Tensor, ord: Optional[Union[int, float, str]] = None, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Tensor:
    if dim is None:
        flat = A.reshape((-1,))
        if ord is None or ord == 2 or ord == 2.0:
            return (flat * flat).sum().sqrt()
        elif ord == 1 or ord == 1.0:
            return flat.abs().sum()
        elif ord == float("inf"):
            return flat.abs().max()
        elif ord == -float("inf"):
            return flat.abs().min()
        else:
            p = float(ord)
            return ((flat.abs() ** p).sum()) ** (1.0 / p)
    else:
        if ord is None or ord == 2 or ord == 2.0:
            return (A * A).sum(dim=dim, keepdim=keepdim).sqrt()
        elif ord == 1 or ord == 1.0:
            return A.abs().sum(dim=dim, keepdim=keepdim)
        elif ord == float("inf"):
            return A.abs().max(dim=dim, keepdim=keepdim)
        elif ord == -float("inf"):
            return A.abs().min(dim=dim, keepdim=keepdim)
        else:
            p = float(ord)
            return ((A.abs() ** p).sum(dim=dim, keepdim=keepdim)) ** (1.0 / p)

def vector_norm(x: Tensor, ord: Union[int, float, str] = 2, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Tensor:
    return norm(x, ord=ord, dim=dim, keepdim=keepdim)

def matrix_norm(A: Tensor, ord: Union[int, float, str] = "fro", dim: Tuple[int, int] = (-2, -1), keepdim: bool = False) -> Tensor:
    if ord == "fro":
        return (A * A).sum(dim=dim, keepdim=keepdim).sqrt()
    return norm(A, ord=ord, dim=dim, keepdim=keepdim)

def inv(A: Tensor) -> Tensor:
    """Matrix inverse using CPU/GPU numerical fallback for square matrices."""
    import numpy as np
    A_np = A.cpu().numpy()
    inv_np = np.linalg.inv(A_np)
    return vultorch.tensor(inv_np, device=A.device, dtype=A.dtype)

def solve(A: Tensor, B: Tensor) -> Tensor:
    """Solves A X = B."""
    import numpy as np
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()
    X_np = np.linalg.solve(A_np, B_np)
    return vultorch.tensor(X_np, device=A.device, dtype=A.dtype)

def qr(A: Tensor) -> Tuple[Tensor, Tensor]:
    """QR decomposition."""
    import numpy as np
    q_np, r_np = np.linalg.qr(A.cpu().numpy())
    return (
        vultorch.tensor(q_np, device=A.device, dtype=A.dtype),
        vultorch.tensor(r_np, device=A.device, dtype=A.dtype),
    )

def svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Singular value decomposition."""
    import numpy as np
    u, s, vh = np.linalg.svd(A.cpu().numpy())
    return (
        vultorch.tensor(u, device=A.device, dtype=A.dtype),
        vultorch.tensor(s, device=A.device, dtype=A.dtype),
        vultorch.tensor(vh, device=A.device, dtype=A.dtype),
    )
