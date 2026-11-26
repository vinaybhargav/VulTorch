"""
Safetensors and Checkpoint Serialization (vultorch.serialization)
"""

import json
import struct
import os
from typing import Dict, Any, Union, Optional
import numpy as np

import vultorch
from .tensor import Tensor, resolve_device

# Supported dtypes in safetensors
_DTYPE_TO_SAFETENSORS = {
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
}

_SAFETENSORS_TO_DTYPE = {v: k for k, v in _DTYPE_TO_SAFETENSORS.items()}


def save_safetensors(tensors: Dict[str, Tensor], filepath: Union[str, os.PathLike], metadata: Optional[Dict[str, str]] = None):
    """
    Saves a dict of tensors to safetensors binary format.
    Zero external dependencies.
    """
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    total_bytes = 0
    tensor_bytes_list = []

    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Expected Tensor, got {type(tensor)}")
        t_cpu = tensor.cpu()
        np_arr = np.ascontiguousarray(t_cpu.numpy())
        raw_data = np_arr.tobytes()
        t_dtype = tensor.dtype
        st_dtype = _DTYPE_TO_SAFETENSORS.get(t_dtype, "F32")
        
        start_offset = total_bytes
        end_offset = total_bytes + len(raw_data)
        total_bytes = end_offset

        header[name] = {
            "dtype": st_dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start_offset, end_offset],
        }
        tensor_bytes_list.append(raw_data)

    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    with open(filepath, "wb") as f:
        # Safetensors format: 8-byte uint64 little-endian header length, followed by header JSON, followed by raw buffer data
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for raw_b in tensor_bytes_list:
            f.write(raw_b)


def load_safetensors(filepath: Union[str, os.PathLike], device: Optional[Union[str, vultorch.device]] = None) -> Dict[str, Tensor]:
    """
    Loads tensors from a safetensors binary file.
    """
    target_device = resolve_device(device)
    with open(filepath, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"Corrupt safetensors file: {filepath}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)

        # Base offset where raw data starts
        base_offset = 8 + header_len
        result: Dict[str, Tensor] = {}

        for k, info in header.items():
            if k == "__metadata__":
                continue
            st_dtype = info["dtype"]
            shape = info["shape"]
            start_off, end_off = info["data_offsets"]
            byte_length = end_off - start_off

            f.seek(base_offset + start_off)
            raw = f.read(byte_length)

            np_dtype_map = {
                "F32": np.float32,
                "F16": np.float16,
                "I64": np.int64,
                "I32": np.int32,
                "I16": np.int16,
                "I8": np.int8,
                "U8": np.uint8,
                "BOOL": np.bool_,
            }
            np_dt = np_dtype_map.get(st_dtype, np.float32)
            np_arr = np.frombuffer(raw, dtype=np_dt).reshape(shape)
            
            vt_dtype = _SAFETENSORS_TO_DTYPE.get(st_dtype, "float32")
            t = vultorch.tensor(np_arr, dtype=vt_dtype, device=target_device)
            result[k] = t

        return result
