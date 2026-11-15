// Universal strided dtype cast across all 7 VulTorch dtypes:
//   0: float32  (4B)
//   1: float16  (2B)
//   2: bfloat16 (2B)
//   3: int32    (4B)
//   4: int64    (8B)
//   5: bool     (1B)
//   6: uint8    (1B)
//
// Uses pure 32-bit word stores on dataOut so 1B, 2B, 4B, and 8B outputs
// are completely race-free and require no 8-bit/16-bit/int64 extensions.

struct CastParams {
    uint total_elements;
    uint src_dtype;
    uint dst_dtype;
    uint src_offset;
    uint ndim;
    uint shape[6];
    uint src_stride[6];
};

[[vk::push_constant]] CastParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dataOut;

uint compute_src_idx(uint dst_idx) {
    uint rem = dst_idx;
    uint src_idx = pc.src_offset;
    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        src_idx += coord * pc.src_stride[d];
    }
    return src_idx;
}

float fp16_to_fp32(uint h) {
    uint sign = (h & 0x8000u) << 16u;
    uint exp16 = (h >> 10u) & 0x1Fu;
    uint mant16 = h & 0x03FFu;
    uint out_bits = 0u;
    if (exp16 == 0u) {
        if (mant16 == 0u) {
            out_bits = sign;
        } else {
            uint exp32 = 127u - 14u;
            uint mant = mant16;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1u;
                exp32 -= 1u;
            }
            mant &= 0x03FFu;
            out_bits = sign | (exp32 << 23u) | (mant << 13u);
        }
    } else if (exp16 == 31u) {
        out_bits = sign | 0x7F800000u | (mant16 << 13u);
    } else {
        uint exp32 = exp16 + (127u - 15u);
        out_bits = sign | (exp32 << 23u) | (mant16 << 13u);
    }
    return asfloat(out_bits);
}

uint fp32_to_fp16(float value) {
    uint bits = asuint(value);
    uint sign = (bits >> 16u) & 0x8000u;
    uint abs_bits = bits & 0x7FFFFFFFu;

    if (abs_bits >= 0x7F800000u) {
        if (abs_bits > 0x7F800000u) {
            uint payload = (abs_bits >> 13u) & 0x03FFu;
            return sign | 0x7C00u | (payload != 0u ? payload : 0x0200u);
        }
        return sign | 0x7C00u;
    }
    if (abs_bits >= 0x47800000u) {
        return sign | 0x7C00u;
    }
    if (abs_bits < 0x38800000u) {
        if (abs_bits < 0x33000000u) {
            return sign;
        }
        uint exp_val = abs_bits >> 23u;
        uint mant = (abs_bits & 0x007FFFFFu) | 0x00800000u;
        uint shift = 113u - exp_val + 13u;
        if (shift >= 32u) return sign;
        uint half_mant = mant >> shift;
        uint rem_mask = (1u << shift) - 1u;
        uint rem = mant & rem_mask;
        uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_mant & 1u) != 0u)) {
            half_mant += 1u;
        }
        return sign | half_mant;
    }

    uint mant = abs_bits & 0x007FFFFFu;
    uint rounding_bias = 0x00000FFFu + ((mant >> 13u) & 1u);
    uint rounded = abs_bits + rounding_bias;
    uint exp16 = (rounded >> 23u) - 112u;
    if (exp16 >= 31u) {
        return sign | 0x7C00u;
    }
    return sign | (exp16 << 10u) | ((rounded >> 13u) & 0x03FFu);
}

float bf16_to_fp32(uint bf) {
    return asfloat(bf << 16u);
}

uint fp32_to_bf16(float value) {
    uint bits = asuint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (bits >> 16u) | 0x0040u;
    }
    uint rounding_bias = 0x00007FFFu + ((bits >> 16u) & 1u);
    bits += rounding_bias;
    return bits >> 16u;
}

void read_src_element(uint s_idx, out float out_f, out int out_i32, out uint out_i64_lo, out uint out_i64_hi, out uint out_nonzero) {
    out_f = 0.0f;
    out_i32 = 0;
    out_i64_lo = 0u;
    out_i64_hi = 0u;
    out_nonzero = 0u;

    if (pc.src_dtype == 0u) {
        float f = asfloat(dataIn[s_idx]);
        out_f = f;
        out_i32 = int(f);
        out_nonzero = (f == 0.0f) ? 0u : 1u;
        bool neg = (f < 0.0f);
        float af = abs(f);
        uint hi = uint(floor(af / 4294967296.0f));
        uint lo = uint(trunc(af - float(hi) * 4294967296.0f));
        if (neg) {
            lo = (~lo) + 1u;
            hi = (~hi) + ((lo == 0u) ? 1u : 0u);
        }
        out_i64_lo = lo;
        out_i64_hi = hi;
        return;
    }

    if (pc.src_dtype == 1u || pc.src_dtype == 2u) {
        uint h = (dataIn[s_idx >> 1u] >> ((s_idx & 1u) * 16u)) & 0xFFFFu;
        float f = (pc.src_dtype == 1u) ? fp16_to_fp32(h) : bf16_to_fp32(h);
        out_f = f;
        out_i32 = int(f);
        out_nonzero = (f == 0.0f) ? 0u : 1u;
        bool neg = (f < 0.0f);
        float af = abs(f);
        uint hi = uint(floor(af / 4294967296.0f));
        uint lo = uint(trunc(af - float(hi) * 4294967296.0f));
        if (neg) {
            lo = (~lo) + 1u;
            hi = (~hi) + ((lo == 0u) ? 1u : 0u);
        }
        out_i64_lo = lo;
        out_i64_hi = hi;
        return;
    }

    if (pc.src_dtype == 3u) {
        int v = int(dataIn[s_idx]);
        out_f = float(v);
        out_i32 = v;
        out_i64_lo = uint(v);
        out_i64_hi = (v < 0) ? 0xFFFFFFFFu : 0u;
        out_nonzero = (v != 0) ? 1u : 0u;
        return;
    }

    if (pc.src_dtype == 4u) {
        uint lo = dataIn[s_idx * 2u];
        uint hi = dataIn[s_idx * 2u + 1u];
        out_i64_lo = lo;
        out_i64_hi = hi;
        out_i32 = int(lo);
        out_nonzero = (lo != 0u || hi != 0u) ? 1u : 0u;
        bool neg = (hi & 0x80000000u) != 0u;
        uint m_lo = lo;
        uint m_hi = hi;
        if (neg) {
            m_lo = (~lo) + 1u;
            m_hi = (~hi) + ((m_lo == 0u) ? 1u : 0u);
        }
        float mag = float(m_hi) * 4294967296.0f + float(m_lo);
        out_f = neg ? -mag : mag;
        return;
    }

    // 5: Bool, 6: UInt8
    uint byte_val = (dataIn[s_idx >> 2u] >> ((s_idx & 3u) * 8u)) & 0xFFu;
    if (pc.src_dtype == 5u) {
        byte_val = (byte_val != 0u) ? 1u : 0u;
    }
    out_f = float(byte_val);
    out_i32 = int(byte_val);
    out_i64_lo = byte_val;
    out_i64_hi = 0u;
    out_nonzero = (byte_val != 0u) ? 1u : 0u;
}

uint convert_to_u32(float f, int i32, uint i64_lo, uint nonzero) {
    if (pc.dst_dtype == 0u) return asuint(f);
    if (pc.dst_dtype == 1u) return fp32_to_fp16(f);
    if (pc.dst_dtype == 2u) return fp32_to_bf16(f);
    if (pc.dst_dtype == 3u) return (pc.src_dtype == 4u) ? i64_lo : uint(i32);
    if (pc.dst_dtype == 5u) return nonzero;
    // UInt8
    return uint(i32) & 0xFFu;
}

[shader("compute")] [numthreads(256, 1, 1)]
void cast_dtype(uint3 tid : SV_DispatchThreadID) {
    uint word_idx = tid.x;

    // 8-byte Int64 output: 2 words per element
    if (pc.dst_dtype == 4u) {
        uint elem_idx = word_idx >> 1u;
        if (elem_idx >= pc.total_elements) return;
        uint sub = word_idx & 1u;
        uint s = compute_src_idx(elem_idx);
        float f; int i32; uint lo; uint hi; uint nz;
        read_src_element(s, f, i32, lo, hi, nz);
        dataOut[word_idx] = (sub == 0u) ? lo : hi;
        return;
    }

    // 4-byte Float32 / Int32 output
    if (pc.dst_dtype == 0u || pc.dst_dtype == 3u) {
        if (word_idx >= pc.total_elements) return;
        uint s = compute_src_idx(word_idx);
        float f; int i32; uint lo; uint hi; uint nz;
        read_src_element(s, f, i32, lo, hi, nz);
        dataOut[word_idx] = convert_to_u32(f, i32, lo, nz);
        return;
    }

    // 2-byte Float16 / BFloat16 output
    if (pc.dst_dtype == 1u || pc.dst_dtype == 2u) {
        uint idx0 = word_idx * 2u;
        if (idx0 >= pc.total_elements) return;

        float f0; int i0; uint lo0; uint hi0; uint nz0;
        read_src_element(compute_src_idx(idx0), f0, i0, lo0, hi0, nz0);
        uint low = convert_to_u32(f0, i0, lo0, nz0) & 0xFFFFu;

        uint high = 0u;
        uint idx1 = idx0 + 1u;
        if (idx1 < pc.total_elements) {
            float f1; int i1; uint lo1; uint hi1; uint nz1;
            read_src_element(compute_src_idx(idx1), f1, i1, lo1, hi1, nz1);
            high = convert_to_u32(f1, i1, lo1, nz1) & 0xFFFFu;
        }
        dataOut[word_idx] = low | (high << 16u);
        return;
    }

    // 1-byte Bool / UInt8 output
    uint base_idx = word_idx * 4u;
    if (base_idx >= pc.total_elements) return;
    uint packed = 0u;
    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
        uint elem = base_idx + k;
        if (elem < pc.total_elements) {
            float fk; int ik; uint lok; uint hik; uint nzk;
            read_src_element(compute_src_idx(elem), fk, ik, lok, hik, nzk);
            uint b = convert_to_u32(fk, ik, lok, nzk) & 0xFFu;
            packed |= (b << (k * 8u));
        }
    }
    dataOut[word_idx] = packed;
}
