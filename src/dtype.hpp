#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vultorch
{

enum class ScalarType : uint32_t
{
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int64 = 4,
    Bool = 5,
    UInt8 = 6,
};

inline size_t element_size(ScalarType dt)
{
    switch (dt)
    {
    case ScalarType::Float32:
        return 4;
    case ScalarType::Float16:
        return 2;
    case ScalarType::BFloat16:
        return 2;
    case ScalarType::Int32:
        return 4;
    case ScalarType::Int64:
        return 8;
    case ScalarType::Bool:
        return 1;
    case ScalarType::UInt8:
        return 1;
    }
    throw std::runtime_error("Unknown ScalarType");
}

inline const char *dtype_name(ScalarType dt)
{
    switch (dt)
    {
    case ScalarType::Float32:
        return "float32";
    case ScalarType::Float16:
        return "float16";
    case ScalarType::BFloat16:
        return "bfloat16";
    case ScalarType::Int32:
        return "int32";
    case ScalarType::Int64:
        return "int64";
    case ScalarType::Bool:
        return "bool";
    case ScalarType::UInt8:
        return "uint8";
    }
    throw std::runtime_error("Unknown ScalarType");
}

inline ScalarType parse_dtype(const std::string &name)
{
    if (name == "float32" || name == "float" || name == "f32" || name == "vultorch.float32")
    {
        return ScalarType::Float32;
    }
    if (name == "float16" || name == "half" || name == "f16" || name == "vultorch.float16")
    {
        return ScalarType::Float16;
    }
    if (name == "bfloat16" || name == "bf16" || name == "vultorch.bfloat16")
    {
        return ScalarType::BFloat16;
    }
    if (name == "int32" || name == "int" || name == "i32" || name == "vultorch.int32")
    {
        return ScalarType::Int32;
    }
    if (name == "int64" || name == "long" || name == "i64" || name == "vultorch.int64")
    {
        return ScalarType::Int64;
    }
    if (name == "bool" || name == "vultorch.bool")
    {
        return ScalarType::Bool;
    }
    if (name == "uint8" || name == "byte" || name == "u8" || name == "vultorch.uint8")
    {
        return ScalarType::UInt8;
    }
    throw std::runtime_error("Unsupported dtype string: " + name);
}

inline bool is_floating_point(ScalarType dt)
{
    return dt == ScalarType::Float32 || dt == ScalarType::Float16 || dt == ScalarType::BFloat16;
}

inline bool is_integer_or_bool(ScalarType dt)
{
    return !is_floating_point(dt);
}

inline ScalarType promote_dtype(ScalarType a, ScalarType b)
{
    if (a == b)
        return a;
    if (a == ScalarType::Float32 || b == ScalarType::Float32)
        return ScalarType::Float32;
    if ((a == ScalarType::Float16 && b == ScalarType::BFloat16) ||
        (a == ScalarType::BFloat16 && b == ScalarType::Float16))
    {
        return ScalarType::Float32;
    }
    if (a == ScalarType::Float16 || b == ScalarType::Float16)
        return ScalarType::Float16;
    if (a == ScalarType::BFloat16 || b == ScalarType::BFloat16)
        return ScalarType::BFloat16;
    if (a == ScalarType::Int64 || b == ScalarType::Int64)
        return ScalarType::Int64;
    if (a == ScalarType::Int32 || b == ScalarType::Int32)
        return ScalarType::Int32;
    if (a == ScalarType::UInt8 || b == ScalarType::UInt8)
        return ScalarType::UInt8;
    return ScalarType::Bool;
}

// IEEE 754 binary32 <-> binary16 conversion with round-to-nearest-even.
inline uint16_t fp32_to_fp16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t abs_bits = bits & 0x7FFFFFFFu;

    // NaN or Inf
    if (abs_bits >= 0x7F800000u)
    {
        if (abs_bits > 0x7F800000u)
        {
            uint16_t payload = static_cast<uint16_t>((abs_bits >> 13) & 0x03FFu);
            return static_cast<uint16_t>(sign | 0x7C00u | (payload ? payload : 0x0200u));
        }
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Overflow to Inf (half max finite is 65504 -> ~0x477FF000)
    if (abs_bits >= 0x47800000u)
    {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Subnormal or underflow
    if (abs_bits < 0x38800000u)
    {
        if (abs_bits < 0x33000000u)
        {
            return static_cast<uint16_t>(sign);
        }
        uint32_t exp_val = abs_bits >> 23;
        uint32_t mant = (abs_bits & 0x007FFFFFu) | 0x00800000u;
        uint32_t shift = 113u - exp_val + 13u; // shift in [14, 24]
        if (shift >= 32u)
        {
            return static_cast<uint16_t>(sign);
        }
        uint32_t half_mant = mant >> shift;
        uint32_t rem_mask = (1u << shift) - 1u;
        uint32_t rem = mant & rem_mask;
        uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_mant & 1u) != 0u))
        {
            half_mant += 1u;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    // Normalized number
    uint32_t mant = abs_bits & 0x007FFFFFu;
    uint32_t exp16 = (abs_bits >> 23) - 127u + 15u;
    uint32_t rounding_bias = 0x00000FFFu + ((mant >> 13) & 1u);
    uint32_t rounded = abs_bits + rounding_bias;
    // Check overflow past exponent 30
    exp16 = ((rounded >> 23) - 112u);
    if (exp16 >= 31u)
    {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (exp16 << 10) | ((rounded >> 13) & 0x03FFu));
}

inline float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp16 = (h >> 10) & 0x1Fu;
    uint32_t mant16 = h & 0x03FFu;
    uint32_t out_bits = 0;

    if (exp16 == 0u)
    {
        if (mant16 == 0u)
        {
            out_bits = sign;
        }
        else
        {
            // Normalize subnormal
            uint32_t exp32 = 127u - 14u;
            uint32_t mant = mant16;
            while ((mant & 0x0400u) == 0u)
            {
                mant <<= 1;
                --exp32;
            }
            mant &= 0x03FFu;
            out_bits = sign | (exp32 << 23) | (mant << 13);
        }
    }
    else if (exp16 == 31u)
    {
        out_bits = sign | 0x7F800000u | (mant16 << 13);
    }
    else
    {
        uint32_t exp32 = exp16 + (127u - 15u);
        out_bits = sign | (exp32 << 23) | (mant16 << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &out_bits, sizeof(out));
    return out;
}

// bfloat16 <-> fp32 with round-to-nearest-even
inline uint16_t fp32_to_bf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
    {
        // Preserve NaN
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    uint32_t rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_fp32(uint16_t bf)
{
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline double read_scalar_as_double(const void *base, size_t elem_idx, ScalarType dt)
{
    const auto *ptr = static_cast<const uint8_t *>(base);
    switch (dt)
    {
    case ScalarType::Float32:
    {
        float v = 0.0f;
        std::memcpy(&v, ptr + elem_idx * 4, 4);
        return static_cast<double>(v);
    }
    case ScalarType::Float16:
    {
        uint16_t h = 0;
        std::memcpy(&h, ptr + elem_idx * 2, 2);
        return static_cast<double>(fp16_to_fp32(h));
    }
    case ScalarType::BFloat16:
    {
        uint16_t bf = 0;
        std::memcpy(&bf, ptr + elem_idx * 2, 2);
        return static_cast<double>(bf16_to_fp32(bf));
    }
    case ScalarType::Int32:
    {
        int32_t v = 0;
        std::memcpy(&v, ptr + elem_idx * 4, 4);
        return static_cast<double>(v);
    }
    case ScalarType::Int64:
    {
        int64_t v = 0;
        std::memcpy(&v, ptr + elem_idx * 8, 8);
        return static_cast<double>(v);
    }
    case ScalarType::Bool:
    {
        return ptr[elem_idx] != 0 ? 1.0 : 0.0;
    }
    case ScalarType::UInt8:
    {
        return static_cast<double>(ptr[elem_idx]);
    }
    }
    throw std::runtime_error("Unsupported ScalarType in read_scalar_as_double");
}

inline float read_scalar_as_float(const void *base, size_t elem_idx, ScalarType dt)
{
    return static_cast<float>(read_scalar_as_double(base, elem_idx, dt));
}

inline void write_scalar_from_double(void *base, size_t elem_idx, ScalarType dt, double value)
{
    auto *ptr = static_cast<uint8_t *>(base);
    switch (dt)
    {
    case ScalarType::Float32:
    {
        float v = static_cast<float>(value);
        std::memcpy(ptr + elem_idx * 4, &v, 4);
        return;
    }
    case ScalarType::Float16:
    {
        uint16_t h = fp32_to_fp16(static_cast<float>(value));
        std::memcpy(ptr + elem_idx * 2, &h, 2);
        return;
    }
    case ScalarType::BFloat16:
    {
        uint16_t bf = fp32_to_bf16(static_cast<float>(value));
        std::memcpy(ptr + elem_idx * 2, &bf, 2);
        return;
    }
    case ScalarType::Int32:
    {
        int32_t v = static_cast<int32_t>(value);
        std::memcpy(ptr + elem_idx * 4, &v, 4);
        return;
    }
    case ScalarType::Int64:
    {
        int64_t v = static_cast<int64_t>(value);
        std::memcpy(ptr + elem_idx * 8, &v, 8);
        return;
    }
    case ScalarType::Bool:
    {
        uint8_t v = (value != 0.0) ? 1u : 0u;
        ptr[elem_idx] = v;
        return;
    }
    case ScalarType::UInt8:
    {
        uint8_t v = static_cast<uint8_t>(value);
        ptr[elem_idx] = v;
        return;
    }
    }
    throw std::runtime_error("Unsupported ScalarType in write_scalar_from_double");
}

inline void write_scalar_from_float(void *base, size_t elem_idx, ScalarType dt, float value)
{
    write_scalar_from_double(base, elem_idx, dt, static_cast<double>(value));
}

} // namespace vultorch
