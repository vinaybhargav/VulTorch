"""
Fast Fourier Transforms (vultorch.fft)
"""

from typing import Optional, Union, Tuple
import vultorch
from .tensor import Tensor

def fft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    import numpy as np
    arr = input.cpu().numpy()
    res = np.fft.fft(arr, n=n, axis=dim, norm=norm)
    # Pack complex as real/imag along trailing dim or float array
    return vultorch.tensor(res.real, device=input.device, dtype="float32")

def ifft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    import numpy as np
    arr = input.cpu().numpy()
    res = np.fft.ifft(arr, n=n, axis=dim, norm=norm)
    return vultorch.tensor(res.real, device=input.device, dtype="float32")

def rfft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    import numpy as np
    arr = input.cpu().numpy()
    res = np.fft.rfft(arr, n=n, axis=dim, norm=norm)
    return vultorch.tensor(res.real, device=input.device, dtype="float32")

def irfft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    import numpy as np
    arr = input.cpu().numpy()
    res = np.fft.irfft(arr, n=n, axis=dim, norm=norm)
    return vultorch.tensor(res.real, device=input.device, dtype="float32")
