"""
High-Precision 32-bit / 64-bit Range Coder for Neural Entropy Compression
"""

import numpy as np

class RangeCoder:
    TOP = 1 << 24
    BOTTOM = 1 << 16

    @classmethod
    def encode(cls, symbols, prob_distributions):
        """
        Encodes a sequence of byte symbols using their corresponding probability distributions.
        prob_distributions: list or 2D array of shape [N, 256] with float probabilities summing to 1.
        Returns: bytes
        """
        low = 0
        range_ = 0xFFFFFFFF
        output = bytearray()

        for sym, probs in zip(symbols, prob_distributions):
            # Scale float probabilities to cumulative integer frequencies summing to 16384 (1 << 14)
            # Ensure each symbol has at least count 1
            total_count = 1 << 14
            counts = np.maximum(1, (probs * (total_count - 256)).astype(np.int32))
            # Adjust total count
            current_sum = np.sum(counts)
            diff = total_count - current_sum
            counts[0] += diff

            cum = np.zeros(257, dtype=np.uint32)
            cum[1:] = np.cumsum(counts)

            sym_low = int(cum[sym])
            sym_high = int(cum[sym + 1])

            range_ //= total_count
            low += sym_low * range_
            range_ *= (sym_high - sym_low)

            while (low ^ (low + range_)) < cls.TOP or (range_ < cls.BOTTOM and ((range_ := -low & (cls.BOTTOM - 1)), True)):
                output.append((low >> 24) & 0xFF)
                low = (low << 8) & 0xFFFFFFFF
                range_ = (range_ << 8) & 0xFFFFFFFF

        # Flush remaining bits
        for _ in range(4):
            output.append((low >> 24) & 0xFF)
            low = (low << 8) & 0xFFFFFFFF

        return bytes(output)

    @classmethod
    def decode(cls, byte_stream, get_probs_fn, num_symbols):
        """
        Decodes num_symbols using an autoregressive probability function get_probs_fn(decoded_so_far).
        """
        stream = bytearray(byte_stream)
        low = 0
        range_ = 0xFFFFFFFF
        code = 0
        stream_idx = 0

        for _ in range(4):
            b = stream[stream_idx] if stream_idx < len(stream) else 0
            stream_idx += 1
            code = (code << 8) | b

        decoded = bytearray()

        for _ in range(num_symbols):
            probs = get_probs_fn(decoded)
            total_count = 1 << 14
            counts = np.maximum(1, (probs * (total_count - 256)).astype(np.int32))
            diff = total_count - np.sum(counts)
            counts[0] += diff

            cum = np.zeros(257, dtype=np.uint32)
            cum[1:] = np.cumsum(counts)

            range_ //= total_count
            count_val = (code - low) // range_

            # Binary search for symbol
            sym = int(np.searchsorted(cum, count_val, side='right') - 1)
            sym = max(0, min(255, sym))

            sym_low = int(cum[sym])
            sym_high = int(cum[sym + 1])

            low += sym_low * range_
            range_ *= (sym_high - sym_low)

            while (low ^ (low + range_)) < cls.TOP or (range_ < cls.BOTTOM and ((range_ := -low & (cls.BOTTOM - 1)), True)):
                b = stream[stream_idx] if stream_idx < len(stream) else 0
                stream_idx += 1
                code = ((code << 8) | b) & 0xFFFFFFFF
                low = (low << 8) & 0xFFFFFFFF
                range_ = (range_ << 8) & 0xFFFFFFFF

            decoded.append(sym)

        return bytes(decoded)
