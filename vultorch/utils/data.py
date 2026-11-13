"""
VulTorch Data Loading Utilities (vultorch.utils.data)
"""

import numpy as np
from typing import Sequence, Iterator, Tuple, List
from ..tensor import Tensor

class Dataset:
    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index: int):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[Tensor, ...]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        import vultorch

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            items = [self.dataset[idx] for idx in batch_indices]

            # If items are tuples of tensors
            if isinstance(items[0], tuple):
                num_fields = len(items[0])
                batched = []
                for field_idx in range(num_fields):
                    field_items = [item[field_idx] for item in items]
                    batched.append(vultorch.stack(field_items, dim=0))
                yield tuple(batched)
            else:
                yield vultorch.stack(items, dim=0)
