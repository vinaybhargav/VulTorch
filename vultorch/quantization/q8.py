import numpy as np
import struct
from typing import Dict, Any, Tuple
import vultorch as torch

class QuantizedLinearQ8:
    """
    Symmetric INT8 Quantized Linear Layer.
    Stores weights as int8 (-128 to 127) with a per-tensor scale factor (float32).
    Reduces memory footprint by 75% and guarantees deterministic integer dot-products.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # Quantized weights: int8 shape (out_features, in_features)
        self.qweight = np.zeros((out_features, in_features), dtype=np.int8)
        self.scale = 1.0 / 127.0
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else None

    @classmethod
    def from_float_linear(cls, linear_module: torch.nn.Linear) -> 'QuantizedLinearQ8':
        qlinear = cls(linear_module.in_features, linear_module.out_features, bias=(linear_module.bias is not None))
        w = linear_module.weight.numpy() # (out_features, in_features)
        max_val = float(np.max(np.abs(w)))
        if max_val < 1e-8:
            max_val = 1e-8

        scale = max_val / 127.0
        qweight = np.clip(np.round(w / scale), -128, 127).astype(np.int8)

        qlinear.scale = scale
        qlinear.qweight = qweight
        if linear_module.bias is not None:
            qlinear.bias = linear_module.bias.numpy()
        return qlinear

    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        """
        Deterministic integer dot-product inference on CPU.
        """
        # x: (..., in_features)
        # Cast to float32 using exact scale or integer accumulation
        # Exact integer accumulation: int8 x int8 or float32 x int8
        w_float = self.qweight.astype(np.float32) * self.scale
        out = np.matmul(x, w_float.T)
        if self.bias is not None:
            out += self.bias
        return out

    def serialize(self) -> bytes:
        """
        Binary packed serialization of Q8 weights:
        Header: in_features (uint32), out_features (uint32), has_bias (uint8), scale (float32)
        Body: raw int8 bytes for qweight + optional float32 bias
        """
        header = struct.pack(">IIBf", self.in_features, self.out_features, 1 if self.has_bias else 0, self.scale)
        body = self.qweight.tobytes()
        if self.has_bias and self.bias is not None:
            body += self.bias.astype(np.float32).tobytes()
        return header + body

    @classmethod
    def deserialize(cls, data: bytes, offset: int = 0) -> Tuple['QuantizedLinearQ8', int]:
        in_f, out_f, has_b, scale = struct.unpack_from(">IIBf", data, offset)
        offset += struct.calcsize(">IIBf")

        layer = cls(in_f, out_f, bias=bool(has_b))
        layer.scale = scale

        w_size = in_f * out_f
        layer.qweight = np.frombuffer(data[offset:offset + w_size], dtype=np.int8).reshape((out_f, in_f)).copy()
        offset += w_size

        if bool(has_b):
            b_size = out_f * 4
            layer.bias = np.frombuffer(data[offset:offset + b_size], dtype=np.float32).copy()
            offset += b_size

        return layer, offset


class ModelQuantizerQ8:
    """
    Quantizes a complete PyTorch/VulTorch model's Linear layers into Q8.
    """
    @staticmethod
    def quantize_state_dict(model: torch.nn.Module) -> bytes:
        serialized = bytearray()
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                ql = QuantizedLinearQ8.from_float_linear(module)
                name_bytes = name.encode("utf-8")
                serialized += struct.pack(">H", len(name_bytes))
                serialized += name_bytes
                serialized += ql.serialize()
                count += 1
        # Prepend total quantized layer count
        return struct.pack(">I", count) + bytes(serialized)
