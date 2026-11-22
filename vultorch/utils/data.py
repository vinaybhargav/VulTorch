"""
VulTorch Data Loading Utilities (vultorch.utils.data)
"""

import numpy as np
from typing import Iterator, Tuple, Optional, Union
from ..tensor import Tensor, Device


class Dataset:
    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping one or more Tensors or NumPy arrays along dimension 0."""

    def __init__(self, *tensors: Union[Tensor, np.ndarray], device: Optional[Union[str, Device]] = None):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor or array")
        converted = []
        for t in tensors:
            if isinstance(t, Tensor):
                converted.append(t.to(device) if device is not None else t)
            else:
                converted.append(Tensor(np.asarray(t), device=device))
        first_len = converted[0].shape[0]
        if not all(t.shape[0] == first_len for t in converted):
            raise ValueError("Size mismatch between tensors along dimension 0")
        self.tensors = tuple(converted)

    def __getitem__(self, index: int):
        if len(self.tensors) == 1:
            return (self.tensors[0][index],)
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]


class DataLoader:
    """
    Beginner-friendly, fast mini-batch iterator supporting TensorDataset fast-path slicing,
    automatic shuffling, `drop_last`, and optional on-the-fly device placement (`device=...`).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        device: Optional[Union[str, Device]] = None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.device = Device(device) if device is not None else None

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _move(self, batch):
        if self.device is None:
            return batch
        if isinstance(batch, Tensor):
            return batch.to(self.device)
        if isinstance(batch, tuple):
            return tuple(self._move(x) for x in batch)
        if isinstance(batch, list):
            return [self._move(x) for x in batch]
        return batch

    def __iter__(self) -> Iterator[Union[Tensor, Tuple[Tensor, ...]]]:
        import vultorch

        n = len(self.dataset)
        # Fast path for unshuffled TensorDataset: contiguous slice along dim 0 without per-item indexing
        if isinstance(self.dataset, TensorDataset) and not self.shuffle:
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                if self.drop_last and (end - start) < self.batch_size:
                    break
                batch = tuple(t[start:end] for t in self.dataset.tensors)
                yield self._move(batch)
            return

        indices = np.arange(n, dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(indices)

        if isinstance(self.dataset, TensorDataset):
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                if self.drop_last and (end - start) < self.batch_size:
                    break
                idx_chunk = indices[start:end]
                batched = []
                for t in self.dataset.tensors:
                    idx_t = vultorch.tensor(idx_chunk, device=t.device, dtype=vultorch.int32)
                    batched.append(t.index_select(0, idx_t))
                yield self._move(tuple(batched))
            return

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            if self.drop_last and (end - start) < self.batch_size:
                break
            batch_indices = indices[start:end]
            items = [self.dataset[int(idx)] for idx in batch_indices]

            if isinstance(items[0], tuple):
                num_fields = len(items[0])
                batched = []
                for field_idx in range(num_fields):
                    field_items = [item[field_idx] for item in items]
                    batched.append(vultorch.stack(field_items, dim=0))
                yield self._move(tuple(batched))
            else:
                yield self._move(vultorch.stack(items, dim=0))
