#pragma once

#include "tensor.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vultorch
{

// Single source of truth for elementwise op semantics. Both the CPU fallback
// and the shader paths use the same op numbering.
inline float apply_binary_op(uint32_t op, float a, float b)
{
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kGeluCoeff = 0.044715f;
    switch (op)
    {
    case 0:
        return a + b;
    case 1:
        return a - b;
    case 2:
        return a * b;
    case 3:
        return a / b;
    case 4:
        return std::pow(a, b);
    case 5:
        return std::max(a, b);
    case 6:
        return std::min(a, b);
    case 7:
        return (a > b) ? 1.0f : 0.0f;
    case 8:
        return (a < b) ? 1.0f : 0.0f;
    case 9:
        return (a == b) ? 1.0f : 0.0f;
    case 10:
        return (b > 0.0f) ? a : 0.0f;
    case 11:
    {
        const float x2 = b * b;
        const float inner = kSqrt2OverPi * (b + kGeluCoeff * x2 * b);
        const float t = std::tanh(inner);
        const float dt = (1.0f - t * t) * kSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x2);
        return a * (0.5f * (1.0f + t) + 0.5f * b * dt);
    }
    case 12:
    {
        const float s = 1.0f / (1.0f + std::exp(-b));
        return a * s * (1.0f + b * (1.0f - s));
    }
    case 13:
        return (b > 0.0f) ? a : ((b < 0.0f) ? -a : 0.0f);
    case 14:
        return (a == b) ? 1.0f : 0.0f;
    case 15:
        return (a >= b) ? 1.0f : 0.0f;
    case 16:
        return (a <= b) ? 1.0f : 0.0f;
    default:
        throw std::runtime_error("Unsupported binary op_type: " + std::to_string(op));
    }
}

inline float apply_unary_op(uint32_t op, float x, float p1 = 0.0f, float p2 = 0.0f)
{
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kGeluCoeff = 0.044715f;

    switch (op)
    {
    case 0:
        return -x;
    case 1:
        return std::exp(x);
    case 2:
        return std::log(x);
    case 3:
        return std::sqrt(x);
    case 4:
        return 1.0f / std::sqrt(x);
    case 5:
        return std::abs(x);
    case 6:
        return std::sin(x);
    case 7:
        return std::cos(x);
    case 8:
        return std::tanh(x);
    case 9:
        return 1.0f / (1.0f + std::exp(-x));
    case 10:
        return std::max(x, 0.0f);
    case 11:
    {
        const float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
        return 0.5f * x * (1.0f + std::tanh(inner));
    }
    case 12:
        return x / (1.0f + std::exp(-x));
    case 13:
        return std::min(std::max(x, p1), p2);
    case 14:
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    case 15:
        return x * p1 + p2;
    case 16:
        return std::nearbyint(x);
    case 17:
        return std::ceil(x);
    case 18:
        return std::floor(x);
    case 19:
        return std::trunc(x);
    case 20:
        return x - std::trunc(x);
    case 21:
        return x * x;
    case 22:
        return 1.0f / x;
    case 23:
        return std::exp2(x);
    case 24:
        return std::expm1(x);
    case 25:
        return std::log2(x);
    case 26:
        return std::log10(x);
    case 27:
        return std::log1p(x);
    case 28:
        return std::tan(x);
    case 29:
        return std::asin(x);
    case 30:
        return std::acos(x);
    case 31:
        return std::atan(x);
    case 32:
        return std::sinh(x);
    case 33:
        return std::cosh(x);
    case 34:
        return std::asinh(x);
    case 35:
        return std::acosh(x);
    case 36:
        return std::atanh(x);
    case 37:
        return std::erf(x);
    case 38:
        return std::erfc(x);
    case 39:
        return std::isnan(x) ? 1.0f : 0.0f;
    case 40:
        return std::isinf(x) ? 1.0f : 0.0f;
    case 41:
        return std::isfinite(x) ? 1.0f : 0.0f;
    default:
        throw std::runtime_error("Unsupported unary op_type: " + std::to_string(op));
    }
}

// Broadcasting plan shared by the CPU fallback and the GPU broadcast kernel.
struct BroadcastPlan
{
    std::vector<int64_t> out_shape;
    std::vector<int64_t> strides_a;
    std::vector<int64_t> strides_b;
};

inline BroadcastPlan make_broadcast_plan(const std::vector<int64_t> &a_shape,
                                         const std::vector<int64_t> &a_strides,
                                         const std::vector<int64_t> &b_shape,
                                         const std::vector<int64_t> &b_strides)
{
    const size_t max_ndim = std::max(a_shape.size(), b_shape.size());
    if (max_ndim > 6)
    {
        throw std::runtime_error("Broadcasting supports up to 6 dimensions");
    }

    BroadcastPlan plan;
    plan.out_shape.assign(max_ndim, 1);
    plan.strides_a.assign(max_ndim, 0);
    plan.strides_b.assign(max_ndim, 0);

    for (size_t k = 0; k < max_ndim; ++k)
    {
        const int ai = static_cast<int>(a_shape.size()) - 1 - static_cast<int>(k);
        const int bi = static_cast<int>(b_shape.size()) - 1 - static_cast<int>(k);
        const size_t oi = max_ndim - 1 - k;

        const int64_t dim_a = (ai >= 0) ? a_shape[ai] : 1;
        const int64_t dim_b = (bi >= 0) ? b_shape[bi] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
        {
            throw std::runtime_error("Cannot broadcast shapes: incompatible dimension " +
                                     std::to_string(dim_a) + " vs " + std::to_string(dim_b));
        }

        plan.out_shape[oi] = std::max(dim_a, dim_b);
        if (ai >= 0 && dim_a > 1)
            plan.strides_a[oi] = a_strides[ai];
        if (bi >= 0 && dim_b > 1)
            plan.strides_b[oi] = b_strides[bi];
    }
    return plan;
}

inline bool compute_view_strides(const std::vector<int64_t> &old_shape,
                                 const std::vector<int64_t> &old_strides, size_t numel,
                                 const std::vector<int64_t> &new_shape,
                                 std::vector<int64_t> &out_strides)
{
    out_strides.assign(new_shape.size(), 0);
    if (numel == 0)
    {
        int64_t s = 1;
        for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i)
        {
            out_strides[static_cast<size_t>(i)] = s;
            s *= new_shape[static_cast<size_t>(i)];
        }
        return true;
    }
    if (new_shape.empty())
    {
        return numel == 1;
    }

    struct Chunk
    {
        int64_t size;
        int64_t stride;
    };
    std::vector<Chunk> chunks;
    for (int i = static_cast<int>(old_shape.size()) - 1; i >= 0; --i)
    {
        int64_t d = old_shape[static_cast<size_t>(i)];
        int64_t st = old_strides[static_cast<size_t>(i)];
        if (d == 1)
            continue;
        if (!chunks.empty() && st == chunks.back().size * chunks.back().stride)
        {
            chunks.back().size *= d;
        }
        else
        {
            chunks.push_back({d, st});
        }
    }
    if (chunks.empty())
    {
        chunks.push_back({1, 1});
    }

    size_t chunk_idx = 0;
    int64_t chunk_rem = chunks[0].size;
    int64_t cur_stride = chunks[0].stride;

    for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i)
    {
        int64_t d = new_shape[static_cast<size_t>(i)];
        if (d == 1)
        {
            if (i + 1 < static_cast<int>(new_shape.size()))
            {
                out_strides[static_cast<size_t>(i)] =
                    out_strides[static_cast<size_t>(i + 1)] * new_shape[static_cast<size_t>(i + 1)];
            }
            else
            {
                out_strides[static_cast<size_t>(i)] = 1;
            }
            continue;
        }
        while (chunk_rem == 1)
        {
            ++chunk_idx;
            if (chunk_idx >= chunks.size())
                return false;
            chunk_rem = chunks[chunk_idx].size;
            cur_stride = chunks[chunk_idx].stride;
        }
        if (chunk_rem % d != 0)
        {
            return false;
        }
        out_strides[static_cast<size_t>(i)] = cur_stride;
        cur_stride *= d;
        chunk_rem /= d;
    }
    return chunk_rem == 1 && (chunk_idx + 1 == chunks.size());
}

inline void advance_odometer(size_t ndim, const std::vector<int64_t> &shape,
                             const std::vector<int64_t> &strides, std::vector<int64_t> &coord,
                             int64_t &offset_accum)
{
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d)
    {
        size_t ud = static_cast<size_t>(d);
        offset_accum += strides[ud];
        if (++coord[ud] < shape[ud])
        {
            return;
        }
        offset_accum -= strides[ud] * shape[ud];
        coord[ud] = 0;
    }
}

inline uint32_t packed_word_count(size_t num_elems, size_t elem_bytes)
{
    if (elem_bytes == 4)
        return static_cast<uint32_t>(num_elems);
    if (elem_bytes == 8)
        return static_cast<uint32_t>(num_elems * 2);
    if (elem_bytes == 2)
        return static_cast<uint32_t>((num_elems + 1) / 2);
    return static_cast<uint32_t>((num_elems + 3) / 4);
}

inline bool all_strides_non_negative(const std::vector<int64_t> &st)
{
    for (int64_t v : st)
    {
        if (v < 0)
            return false;
    }
    return true;
}

inline int64_t normalize_dim(int64_t dim, int64_t ndim)
{
    int64_t d = (dim < 0) ? (dim + ndim) : dim;
    if (d < 0 || d >= ndim)
    {
        throw std::out_of_range("Dimension out of range: " + std::to_string(dim));
    }
    return d;
}

inline std::vector<int64_t> align_strides_for_broadcast(const std::vector<int64_t> &src_shape,
                                                        const std::vector<int64_t> &src_strides,
                                                        const std::vector<int64_t> &target_shape)
{
    const size_t ndim = target_shape.size();
    std::vector<int64_t> out(ndim, 0);
    for (size_t k = 0; k < src_shape.size(); ++k)
    {
        const size_t si = src_shape.size() - 1 - k;
        const size_t ti = ndim - 1 - k;
        if (src_shape[si] > 1)
        {
            out[ti] = src_strides[si];
        }
    }
    return out;
}

inline int reflect_index_cpu(int64_t c, int64_t len)
{
    if (len <= 1)
        return 0;
    while (c < 0 || c >= len)
    {
        if (c < 0)
            c = -c;
        if (c >= len)
            c = 2 * (len - 1) - c;
    }
    return static_cast<int>(c);
}

inline std::shared_ptr<NativeTensor>
ensure_int32_for_index(const std::shared_ptr<NativeTensor> &idx)
{
    auto cont = idx->is_contiguous() ? idx : idx->contiguous();
    if (cont->dtype == ScalarType::Int32)
        return cont;
    return cont->cast(ScalarType::Int32);
}

} // namespace vultorch
