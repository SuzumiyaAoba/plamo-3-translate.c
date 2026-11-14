#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#pragma clang fp contract(off)

struct Linear {
    uint rows, columns, tokens, input_stride, output_stride, aligned_weights, weight_offset;
};
inline float bf16(ushort x) {
    return as_type<float>(uint(x) << 16);
}

// A cache representation, not quantization: all sixteen original BF16 bits
// are recovered before the same FP32 arithmetic as the ordinary kernels.
#define PACK_BLOCK 512u
struct PackBlock {
    uint offset, exponent;
};
inline uint exponent_delta(device const uchar *data, uint i, uint bits) {
    if (!bits)
        return 0;
    uint bit = i * bits, shift = bit & 31;
    device const uint *words = (device const uint *)(data + PACK_BLOCK);
    uint value = words[bit >> 5] >> shift;
    if (shift + bits > 32)
        value |= words[(bit >> 5) + 1] << (32 - shift);
    return value & ((1u << bits) - 1);
}
template <bool Packed>
inline ushort weight_bits(device const ushort *w, ulong i, constant Linear &p) {
    if (!Packed)
        return w[i];
    i += p.weight_offset;
    PackBlock h = ((device const PackBlock *)w)[i / PACK_BLOCK];
    device const uchar *data = (device const uchar *)w + h.offset;
    uint j = uint(i) & (PACK_BLOCK - 1), low = data[j];
    uint exponent = (h.exponent & 255) + exponent_delta(data, j, h.exponent >> 8);
    return ushort((low & 127) | ((low & 128) << 8) | (exponent << 7));
}
// Approximate weights are confined to a separate proposal model. The 8B
// verifier continues to use original BF16 values and FP32 arithmetic.
template <uint Storage>
inline float weight_value(device const ushort *w, ulong i, constant Linear &p) {
    if (Storage == 2) {
        i += p.weight_offset;
        device const uchar *block = (device const uchar *)w + (i / 32) * 20;
        uint j = uint(i) & 31, value = block[4 + j / 2];
        value = (value >> ((j & 1) * 4)) & 15;
        return float(int(value) - 8) * *((device const float *)block);
    }
    return bf16(weight_bits<Storage == 1>(w, i, p));
}
template <uint Packed> inline float4 weight4(device const ushort *w, ulong i, constant Linear &p) {
    if (Packed == 2) {
        i += p.weight_offset;
        device const uchar *block = (device const uchar *)w + (i / 32) * 20;
        uint j = uint(i) & 31;
        uint code = *((device const ushort *)(block + 4 + j / 2));
        uint4 value = (uint4(code) >> uint4(0, 4, 8, 12)) & 15;
        return float4(int4(value) - 8) * *((device const float *)block);
    }
    if (!Packed)
        return as_type<float4>(uint4(*((device const ushort4 *)(w + i))) << 16);
    i += p.weight_offset;
    PackBlock h = ((device const PackBlock *)w)[i / PACK_BLOCK];
    device const uchar *data = (device const uchar *)w + h.offset;
    uint j = uint(i) & (PACK_BLOCK - 1), bits = h.exponent >> 8;
    uint4 low = uint4(*((device const uchar4 *)(data + j))), delta;
    if (!bits)
        delta = 0;
    else {
        // Four adjacent exponent deltas occupy at most 32 bits. Read their
        // complete bit field once, including a possible word boundary.
        uint bit = j * bits, shift = bit & 31;
        device const uint *words = (device const uint *)(data + PACK_BLOCK);
        uint v = words[bit >> 5] >> shift;
        if (shift + 4 * bits > 32)
            v |= words[(bit >> 5) + 1] << (32 - shift);
        delta = (uint4(v) >> uint4(0, bits, 2 * bits, 3 * bits)) & ((1u << bits) - 1);
    }
    uint4 exponent = delta + (h.exponent & 255);
    return as_type<float4>(((low & 127) | ((low & 128) << 8) | (exponent << 7)) << 16);
}

// Integer texture reads preserve all BF16 encodings, including NaN payloads.
// Each cache image has a power-of-two width; aliases remain linear word offsets.
inline ushort4 texture_words(texture2d<ushort, access::read> w, uint pixel) {
    uint width = w.get_width();
    return w.read(uint2(pixel & (width - 1), pixel >> (31 - clz(width))));
}
template <uint Storage>
inline float weight_value(texture2d<ushort, access::read> w, ulong i, constant Linear &p) {
    uint index = uint(i) + p.weight_offset, bits = texture_words(w, index >> 2)[index & 3];
    return bf16(ushort((bits >> 1) | ((bits & 1) << 15)));
}
template <uint Storage>
inline float4 weight4(texture2d<ushort, access::read> w, ulong i, constant Linear &p) {
    uint index = uint(i) + p.weight_offset;
    if (index & 3)
        return float4(weight_value<Storage>(w, i, p), weight_value<Storage>(w, i + 1, p),
                      weight_value<Storage>(w, i + 2, p), weight_value<Storage>(w, i + 3, p));
    uint4 bits = uint4(texture_words(w, index >> 2));
    return as_type<float4>(((bits >> 1) | ((bits & 1) << 15)) << 16);
}
template <uint Storage>
inline float4 weight_row4(device const ushort *w, uint row, uint col, constant Linear &p) {
    return weight4<Storage>(w, ulong(row) * p.columns + col, p);
}
template <uint Storage>
inline float4 weight_row4(texture2d<ushort, access::read> w, uint row, uint col,
                          constant Linear &p) {
    if (Storage != 4)
        return weight4<Storage>(w, ulong(row) * p.columns + col, p);
    // The encoder selects this form only for complete, unaliased image rows.
    uint4 bits = uint4(w.read(uint2(col >> 2, row)));
    return as_type<float4>(((bits >> 1) | ((bits & 1) << 15)) << 16);
}

template <uint Storage>
inline ushort4 weight_row_bits4(device const ushort *w, uint row, uint col, constant Linear &p) {
    return ushort4(as_type<uint4>(weight_row4<Storage>(w, row, col, p)) >> 16);
}
template <uint Storage>
inline ushort4 weight_row_bits4(texture2d<ushort, access::read> w, uint row, uint col,
                                constant Linear &p) {
    ushort4 bits;
    if (Storage == 4)
        bits = w.read(uint2(col >> 2, row));
    else
        bits = texture_words(w, (row * p.columns + col + p.weight_offset) >> 2);
    // Keep the read and inverse rotation in sixteen-bit registers until the
    // matrix fragment is expanded for FP32 arithmetic.
    return (bits >> 1) | (bits << 15);
}

kernel void plamo_pack_analyze(device const ushort *w [[buffer(0)]],
                               device PackBlock *headers [[buffer(1)]],
                               constant uint &count [[buffer(2)]],
                               uint block [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    uint minimum = 255, maximum = 0;
    for (uint j = tid; j < PACK_BLOCK; j += 128) {
        ulong i = ulong(block) * PACK_BLOCK + j;
        if (i < count) {
            uint e = (w[i] >> 7) & 255;
            minimum = min(minimum, e);
            maximum = max(maximum, e);
        }
    }
    threadgroup uint minima[4], maxima[4];
    minimum = simd_min(minimum);
    maximum = simd_max(maximum);
    if (!lane) {
        minima[simd] = minimum;
        maxima[simd] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) {
        minimum = min(min(minima[0], minima[1]), min(minima[2], minima[3]));
        maximum = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
        uint span = maximum - minimum, bits = span ? 32 - clz(span) : 0;
        headers[block] = {0, minimum | (bits << 8)};
    }
}

kernel void plamo_pack_encode(device const ushort *w [[buffer(0)]],
                              device const PackBlock *headers [[buffer(1)]],
                              device uchar *output [[buffer(2)]],
                              constant uint &count [[buffer(3)]],
                              uint block [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    PackBlock h = headers[block];
    if (!tid)
        ((device PackBlock *)output)[block] = h;
    device uchar *data = output + h.offset;
    ulong first = ulong(block) * PACK_BLOCK;
    for (uint j = tid; j < PACK_BLOCK; j += 128) {
        ushort v = first + j < count ? w[first + j] : 0;
        data[j] = uchar((v & 127) | ((v >> 8) & 128));
    }
    uint bits = h.exponent >> 8, base = h.exponent & 255;
    for (uint word = tid; word < bits * (PACK_BLOCK / 32); word += 128) {
        uint begin = word * 32, end = begin + 32, value = 0;
        for (uint j = begin / bits; j < min(PACK_BLOCK, (end + bits - 1) / bits); ++j) {
            uint e = first + j < count ? ((w[first + j] >> 7) & 255) - base : 0;
            int shift = int(j * bits) - int(begin);
            value |= shift < 0 ? e >> uint(-shift) : e << uint(shift);
        }
        ((device uint *)(data + PACK_BLOCK))[word] = value;
    }
}

// Make each retained page accessible to the GPU before API readiness. Reading
// one byte per OS page avoids a dummy model inference or a full tensor scan.
kernel void plamo_cache_touch(device const uchar *weights [[buffer(0)]],
                              device uint *sink [[buffer(1)]], constant uint2 &p [[buffer(2)]],
                              uint page [[thread_position_in_grid]]) {
    ulong offset = ulong(page) * p.y;
    if (offset < p.x)
        sink[page] = weights[offset];
}

// One SIMD group reduces a row. Adjacent lanes read adjacent BF16 values.
template <uint Packed>
inline void matvec(device const ushort *w, device const float *x, device float *y,
                   constant Linear &p, uint3 group, uint simd, uint lane) {
    uint row = group.x * 4 + simd;
    if (row >= p.rows)
        return;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    if (Packed == 2 && !(p.columns % 32) && !(p.weight_offset % 32)) {
        for (uint k = lane * 8; k < p.columns; k += 256) {
            uint i = row * p.columns + k + p.weight_offset, j = i & 31;
            device const uchar *block = (device const uchar *)w + ulong(i >> 5) * 20;
            float scale = *((device const float *)block);
            uint code = *((device const uint *)(block + 4 + j / 2));
            uint4 low = (uint4(code) >> uint4(0, 4, 8, 12)) & 15;
            uint4 high = (uint4(code) >> uint4(16, 20, 24, 28)) & 15;
            float4 input0 = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k));
            float4 input1 = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k + 4));
            sum0 = fma(float4(int4(low) - 8) * scale, input0, sum0);
            sum1 = fma(float4(int4(high) - 8) * scale, input1, sum1);
        }
    } else {
        for (uint k = lane * 8; k < p.columns; k += 256) {
            if (!(p.columns % 8) && p.aligned_weights) {
                float4 weights = weight4<Packed>(w, ulong(row) * p.columns + k, p);
                float4 input = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k));
                sum0 = fma(weights, input, sum0);
                weights = weight4<Packed>(w, ulong(row) * p.columns + k + 4, p);
                input = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k + 4));
                sum1 = fma(weights, input, sum1);
            } else {
                for (uint j = 0; j < 4; ++j) {
                    if (k + j < p.columns)
                        sum0[j] = fma(weight_value<Packed>(w, ulong(row) * p.columns + k + j, p),
                                      x[ulong(group.y) * p.input_stride + k + j], sum0[j]);
                    if (k + j + 4 < p.columns)
                        sum1[j] =
                            fma(weight_value<Packed>(w, ulong(row) * p.columns + k + j + 4, p),
                                x[ulong(group.y) * p.input_stride + k + j + 4], sum1[j]);
                }
            }
        }
    }
    float4 sum = sum0 + sum1;
    float value = simd_sum((sum.x + sum.y) + (sum.z + sum.w));
    if (!lane)
        y[ulong(group.y) * p.output_stride + row] = value;
}

// Narrow prompts benefit from independent SIMD reductions without matrix
// tile barriers. Eight tokens reuse each BF16 load; operands stay FP32.
template <uint Packed, typename Weights>
inline void matmul_vector(Weights w, device const float *x, device float *y, constant Linear &p,
                          uint3 group, uint simd, uint lane) {
    uint row = group.x * 4 + simd, first = group.y * 8;
    if (row >= p.rows)
        return;
    float4 sums[8];
#pragma unroll
    for (uint t = 0; t < 8; ++t)
        sums[t] = 0.0f;
    for (uint k = lane * 4; k < p.columns; k += 128) {
        float4 weights = 0.0f;
        if (!(p.columns % 4) && p.aligned_weights) {
            weights = weight4<Packed>(w, ulong(row) * p.columns + k, p);
        } else {
            for (uint j = 0; j < 4; ++j)
                if (k + j < p.columns)
                    weights[j] = weight_value<Packed>(w, ulong(row) * p.columns + k + j, p);
        }
#pragma unroll
        for (uint t = 0; t < 8; ++t)
            if (first + t < p.tokens) {
                float4 input = 0.0f;
                if (!(p.columns % 4))
                    input = *((device const float4 *)(x + ulong(first + t) * p.input_stride + k));
                else
                    for (uint j = 0; j < 4; ++j)
                        if (k + j < p.columns)
                            input[j] = x[ulong(first + t) * p.input_stride + k + j];
                sums[t] = fma(weights, input, sums[t]);
            }
    }
#pragma unroll
    for (uint t = 0; t < 8; ++t)
        if (first + t < p.tokens) {
            float value = simd_sum((sums[t].x + sums[t].y) + (sums[t].z + sums[t].w));
            if (!lane)
                y[ulong(first + t) * p.output_stride + row] = value;
        }
}

// 32 output rows x 8/16/24/32 prompt tokens. BF16 expands exactly to FP32; both
// operands and accumulation remain FP32 in every temporary tile.
template <uint Packed, uint N, uint Parts = 1, typename Weights>
inline void matmul_tile(Weights w, device const float *x, device float *y, constant Linear &p,
                        uint3 group, uint tid, uint simd, threadgroup float *weights,
                        threadgroup float *inputs, threadgroup float *outputs) {
    simdgroup_float8x8 total0(0.0f), total1(0.0f), total2(0.0f), total3(0.0f);
    y += ulong(group.z) * p.rows * p.tokens;
    for (uint base = group.z * (p.columns / Parts); base < (group.z + 1) * (p.columns / Parts);
         base += 32) {
        for (uint i = tid; i < 32 * 8; i += 128) {
            uint row = group.x * 32 + i / 8, col = base + (i % 8) * 4;
            float4 value = 0.0f;
            if (row < p.rows && col + 3 < p.columns && !(p.columns % 4) && p.aligned_weights)
                value = weight_row4<Packed>(w, row, col, p);
            else
                for (uint j = 0; j < 4; ++j)
                    if (row < p.rows && col + j < p.columns)
                        value[j] = weight_value<Packed>(w, ulong(row) * p.columns + col + j, p);
            for (uint j = 0; j < 4; ++j)
                weights[i * 4 + j] = value[j];
        }
        for (uint i = tid; i < N * 8; i += 128) {
            uint token = group.y * N + i / 8, col = base + (i % 8) * 4;
            float4 value = 0.0f;
            if (token < p.tokens && col + 3 < p.columns && !(p.input_stride % 4))
                value = *((device const float4 *)(x + ulong(token) * p.input_stride + col));
            else
                for (uint j = 0; j < 4; ++j)
                    if (token < p.tokens && col + j < p.columns)
                        value[j] = x[ulong(token) * p.input_stride + col + j];
            for (uint j = 0; j < 4; ++j)
                inputs[i * 4 + j] = value[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 a, b0, b1, b2, b3;
            simdgroup_load(a, weights + simd * 8 * 32 + k, 32);
            simdgroup_load(b0, inputs + k, 32, ulong2(0), true);
            if (N > 8)
                simdgroup_load(b1, inputs + 8 * 32 + k, 32, ulong2(0), true);
            if (N > 16)
                simdgroup_load(b2, inputs + 16 * 32 + k, 32, ulong2(0), true);
            if (N > 24)
                simdgroup_load(b3, inputs + 24 * 32 + k, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(total0, a, b0, total0);
            if (N > 8)
                simdgroup_multiply_accumulate(total1, a, b1, total1);
            if (N > 16)
                simdgroup_multiply_accumulate(total2, a, b2, total2);
            if (N > 24)
                simdgroup_multiply_accumulate(total3, a, b3, total3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(total0, outputs + simd * 8 * N, N);
    if (N > 8)
        simdgroup_store(total1, outputs + simd * 8 * N + 8, N);
    if (N > 16)
        simdgroup_store(total2, outputs + simd * 8 * N + 16, N);
    if (N > 24)
        simdgroup_store(total3, outputs + simd * 8 * N + 24, N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * N; i += 128) {
        uint row = group.x * 32 + i / N, token = group.y * N + i % N;
        if (row < p.rows && token < p.tokens)
            y[ulong(token) * p.output_stride + row] = outputs[i];
    }
}

// Keep all original BF16 bits in threadgroup memory. A half matrix is only
// a 16-bit carrier: expand its raw bits into float matrix fragments before
// the same FP32 MMA sequence. No half arithmetic or weight rounding occurs.
template <uint Packed, uint N, uint K, uint Parts, uint R, bool Complete = false, typename Weights>
inline void matmul_compact(Weights w, device const float *x, device float *y, constant Linear &p,
                           uint3 group, uint tid, uint simd, threadgroup ushort *weights,
                           threadgroup float *inputs, threadgroup float *outputs) {
    simdgroup_float8x8 total0(0.0f), total1(0.0f), total2(0.0f), total3(0.0f);
    y += ulong(group.z) * p.rows * p.tokens;
    for (uint base = group.z * (p.columns / Parts); base < (group.z + 1) * (p.columns / Parts);
         base += K) {
        for (uint i = tid; i < R * (K / 4); i += R * 4) {
            uint row = group.x * R + i / (K / 4), col = base + (i % (K / 4)) * 4;
            ushort4 value = 0;
            if (Complete ||
                (row < p.rows && col + 3 < p.columns && !(p.columns % 4) && p.aligned_weights))
                value = weight_row_bits4<Packed>(w, row, col, p);
            else
                for (uint j = 0; j < 4; ++j)
                    if (row < p.rows && col + j < p.columns)
                        value[j] = ushort(as_type<uint>(weight_value<Packed>(
                                              w, ulong(row) * p.columns + col + j, p)) >>
                                          16);
            *((threadgroup ushort4 *)(weights + i * 4)) = value;
        }
        for (uint i = tid; i < N * (K / 4); i += R * 4) {
            uint token = group.y * N + i / (K / 4), col = base + (i % (K / 4)) * 4;
            float4 value = 0.0f;
            if (Complete || (token < p.tokens && col + 3 < p.columns && !(p.input_stride % 4)))
                value = *((device const float4 *)(x + ulong(token) * p.input_stride + col));
            else
                for (uint j = 0; j < 4; ++j)
                    if (token < p.tokens && col + j < p.columns)
                        value[j] = x[ulong(token) * p.input_stride + col + j];
            for (uint j = 0; j < 4; ++j)
                inputs[i * 4 + j] = value[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < K; k += 8) {
            simdgroup_float8x8 a, b0, b1, b2, b3;
            simdgroup_half8x8 bits;
            simdgroup_load(bits, (threadgroup half *)weights + simd * 8 * K + k, K);
            for (uint i = 0; i < 2; ++i)
                a.thread_elements()[i] = bf16(as_type<ushort>(bits.thread_elements()[i]));
            simdgroup_load(b0, inputs + k, K, ulong2(0), true);
            if (N > 8)
                simdgroup_load(b1, inputs + 8 * K + k, K, ulong2(0), true);
            if (N > 16)
                simdgroup_load(b2, inputs + 16 * K + k, K, ulong2(0), true);
            if (N > 24)
                simdgroup_load(b3, inputs + 24 * K + k, K, ulong2(0), true);
            simdgroup_multiply_accumulate(total0, a, b0, total0);
            if (N > 8)
                simdgroup_multiply_accumulate(total1, a, b1, total1);
            if (N > 16)
                simdgroup_multiply_accumulate(total2, a, b2, total2);
            if (N > 24)
                simdgroup_multiply_accumulate(total3, a, b3, total3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(total0, outputs + simd * 8 * N, N);
    if (N > 8)
        simdgroup_store(total1, outputs + simd * 8 * N + 8, N);
    if (N > 16)
        simdgroup_store(total2, outputs + simd * 8 * N + 16, N);
    if (N > 24)
        simdgroup_store(total3, outputs + simd * 8 * N + 24, N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < R * N; i += R * 4) {
        uint row = group.x * R + i / N, token = group.y * N + i % N;
        if (Complete || (row < p.rows && token < p.tokens))
            y[ulong(token) * p.output_stride + row] = outputs[i];
    }
}
kernel void plamo_matmul8_compact(device const ushort *w [[buffer(0)]],
                                  device const float *x [[buffer(1)]],
                                  device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup ushort4 weights[512];
    threadgroup float inputs[8 * 64];
    matmul_compact<0, 8, 64, 1, 32>(w, x, y, p, group, tid, simd, (threadgroup ushort *)weights,
                                    inputs, (threadgroup float *)weights);
}
kernel void plamo_matmul16_compact(device const ushort *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup ushort4 weights[512];
    threadgroup float inputs[16 * 32];
    matmul_compact<0, 16, 32, 1, 64>(w, x, y, p, group, tid, simd, (threadgroup ushort *)weights,
                                     inputs, (threadgroup float *)weights);
}
kernel void plamo_matmul16_split_compact(device const ushort *w [[buffer(0)]],
                                         device const float *x [[buffer(1)]],
                                         device float *y [[buffer(2)]],
                                         constant Linear &p [[buffer(3)]],
                                         uint3 group [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup ushort4 weights[256];
    threadgroup float inputs[16 * 32];
    matmul_compact<0, 16, 32, 4, 32>(w, x, y, p, group, tid, simd, (threadgroup ushort *)weights,
                                     inputs, (threadgroup float *)weights);
}

#define VECTOR_KERNEL(NAME, BODY, PACKED)                                                          \
    kernel void NAME(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]],    \
                     device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],              \
                     uint3 group [[threadgroup_position_in_grid]],                                 \
                     uint simd [[simdgroup_index_in_threadgroup]],                                 \
                     uint lane [[thread_index_in_simdgroup]]) {                                    \
        BODY<PACKED>(w, x, y, p, group, simd, lane);                                               \
    }
VECTOR_KERNEL(plamo_matvec, matvec, false)
VECTOR_KERNEL(plamo_matvec_packed, matvec, true)
VECTOR_KERNEL(plamo_matvec_q4, matvec, 2)
VECTOR_KERNEL(plamo_matmul_vector, matmul_vector, false)
VECTOR_KERNEL(plamo_matmul_vector_packed, matmul_vector, true)
VECTOR_KERNEL(plamo_matmul_vector_q4, matmul_vector, 2)

// The proposal model's wide output matrices benefit from four independent
// eight-lane rows per SIMD group. Narrow/wide-input matrices keep matvec.
kernel void plamo_matvec_q4_rows(device const ushort *w [[buffer(0)]],
                                 device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                                 constant Linear &p [[buffer(3)]],
                                 uint3 group [[threadgroup_position_in_grid]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    uint inner = lane & 7, row = group.x * 16 + simd * 4 + lane / 8;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    for (uint k = inner * 8; k < p.columns; k += 64) {
        if (row >= p.rows)
            continue;
        uint i = row * p.columns + k + p.weight_offset, j = i & 31;
        device const uchar *block = (device const uchar *)w + ulong(i >> 5) * 20;
        float scale = *((device const float *)block);
        uint code = *((device const uint *)(block + 4 + j / 2));
        uint4 low = (uint4(code) >> uint4(0, 4, 8, 12)) & 15;
        uint4 high = (uint4(code) >> uint4(16, 20, 24, 28)) & 15;
        float4 a = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k));
        float4 b = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k + 4));
        sum0 = fma(float4(int4(low) - 8) * scale, a, sum0);
        sum1 = fma(float4(int4(high) - 8) * scale, b, sum1);
    }
    float4 sum = sum0 + sum1;
    float value = (sum.x + sum.y) + (sum.z + sum.w);
#pragma unroll
    for (uint delta = 4; delta; delta /= 2)
        value += simd_shuffle_xor(value, delta);
    if (!inner && row < p.rows)
        y[ulong(group.y) * p.output_stride + row] = value;
}

#define MATRIX_KERNEL(NAME, PACKED, N)                                                             \
    kernel void NAME(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]],    \
                     device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],              \
                     uint3 group [[threadgroup_position_in_grid]],                                 \
                     uint tid [[thread_index_in_threadgroup]],                                     \
                     uint simd [[simdgroup_index_in_threadgroup]]) {                               \
        threadgroup float weights[32 * 32], inputs[N * 32];                                        \
        matmul_tile<PACKED, N>(w, x, y, p, group, tid, simd, weights, inputs, weights);            \
    }
MATRIX_KERNEL(plamo_matmul, false, 32)
MATRIX_KERNEL(plamo_matmul8, false, 8)
MATRIX_KERNEL(plamo_matmul16, false, 16)
MATRIX_KERNEL(plamo_matmul_short, false, 24)
MATRIX_KERNEL(plamo_matmul_packed, true, 32)
MATRIX_KERNEL(plamo_matmul8_packed, true, 8)
MATRIX_KERNEL(plamo_matmul16_packed, true, 16)
MATRIX_KERNEL(plamo_matmul_short_packed, true, 24)
MATRIX_KERNEL(plamo_matmul_q4, 2, 32)
MATRIX_KERNEL(plamo_matmul8_q4, 2, 8)
MATRIX_KERNEL(plamo_matmul16_q4, 2, 16)
MATRIX_KERNEL(plamo_matmul_short_q4, 2, 24)

#define SPLIT_MATRIX_KERNEL(NAME, PACKED, N)                                                       \
    kernel void NAME(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]],    \
                     device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],              \
                     uint3 group [[threadgroup_position_in_grid]],                                 \
                     uint tid [[thread_index_in_threadgroup]],                                     \
                     uint simd [[simdgroup_index_in_threadgroup]]) {                               \
        threadgroup float weights[32 * 32], inputs[N * 32];                                        \
        matmul_tile<PACKED, N, 4>(w, x, y, p, group, tid, simd, weights, inputs, weights);         \
    }
SPLIT_MATRIX_KERNEL(plamo_matmul8_split, 0, 8)
SPLIT_MATRIX_KERNEL(plamo_matmul16_split, 0, 16)
SPLIT_MATRIX_KERNEL(plamo_matmul24_split, 0, 24)
SPLIT_MATRIX_KERNEL(plamo_matmul32_split, 0, 32)
SPLIT_MATRIX_KERNEL(plamo_matmul8_split_packed, 1, 8)
SPLIT_MATRIX_KERNEL(plamo_matmul16_split_packed, 1, 16)
SPLIT_MATRIX_KERNEL(plamo_matmul24_split_packed, 1, 24)
SPLIT_MATRIX_KERNEL(plamo_matmul32_split_packed, 1, 32)
SPLIT_MATRIX_KERNEL(plamo_matmul8_split_q4, 2, 8)
SPLIT_MATRIX_KERNEL(plamo_matmul16_split_q4, 2, 16)
SPLIT_MATRIX_KERNEL(plamo_matmul24_split_q4, 2, 24)
SPLIT_MATRIX_KERNEL(plamo_matmul32_split_q4, 2, 32)

template <uint Storage>
inline void texture_matvec(texture2d<ushort, access::read> w, device const float *x,
                           device float *y, constant Linear &p, uint3 group, uint simd, uint lane) {
    uint row = group.x * 4 + simd;
    if (row >= p.rows)
        return;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    for (uint k = lane * 8; k < p.columns; k += 256) {
        float4 a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
        if (!(p.columns % 8) && p.aligned_weights) {
            a = weight4<Storage>(w, ulong(row) * p.columns + k, p);
            b = weight4<Storage>(w, ulong(row) * p.columns + k + 4, p);
            c = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k));
            d = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k + 4));
        } else
            for (uint j = 0; j < 4; ++j) {
                if (k + j < p.columns) {
                    a[j] = weight_value<Storage>(w, ulong(row) * p.columns + k + j, p);
                    c[j] = x[ulong(group.y) * p.input_stride + k + j];
                }
                if (k + j + 4 < p.columns) {
                    b[j] = weight_value<Storage>(w, ulong(row) * p.columns + k + j + 4, p);
                    d[j] = x[ulong(group.y) * p.input_stride + k + j + 4];
                }
            }
        sum0 = fma(a, c, sum0);
        sum1 = fma(b, d, sum1);
    }
    float4 sum = sum0 + sum1;
    float value = simd_sum((sum.x + sum.y) + (sum.z + sum.w));
    if (!lane)
        y[ulong(group.y) * p.output_stride + row] = value;
}
#define TEXTURE_VECTOR_KERNEL(NAME, BODY)                                                          \
    kernel void NAME(                                                                              \
        texture2d<ushort, access::read> w [[texture(0)]], device const float *x [[buffer(1)]],     \
        device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],                           \
        uint3 group [[threadgroup_position_in_grid]],                                              \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {   \
        BODY<3>(w, x, y, p, group, simd, lane);                                                    \
    }
TEXTURE_VECTOR_KERNEL(plamo_matvec_texture, texture_matvec)
TEXTURE_VECTOR_KERNEL(plamo_matmul_vector_texture, matmul_vector)
#define TEXTURE_MATRIX_KERNEL(NAME, N, PARTS)                                                      \
    kernel void NAME(                                                                              \
        texture2d<ushort, access::read> w [[texture(0)]], device const float *x [[buffer(1)]],     \
        device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],                           \
        uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],    \
        uint simd [[simdgroup_index_in_threadgroup]]) {                                            \
        threadgroup float weights[32 * 32], inputs[N * 32];                                        \
        matmul_tile<3, N, PARTS>(w, x, y, p, group, tid, simd, weights, inputs, weights);          \
    }
TEXTURE_MATRIX_KERNEL(plamo_matmul8_texture, 8, 1)
TEXTURE_MATRIX_KERNEL(plamo_matmul16_texture, 16, 1)
TEXTURE_MATRIX_KERNEL(plamo_matmul_short_texture, 24, 1)
TEXTURE_MATRIX_KERNEL(plamo_matmul_texture, 32, 1)
TEXTURE_MATRIX_KERNEL(plamo_matmul8_split_texture, 8, 4)
TEXTURE_MATRIX_KERNEL(plamo_matmul16_split_texture, 16, 4)
TEXTURE_MATRIX_KERNEL(plamo_matmul24_split_texture, 24, 4)
TEXTURE_MATRIX_KERNEL(plamo_matmul32_split_texture, 32, 4)
#define TEXTURE_COMPACT_KERNEL(NAME, N, K, PARTS, R, STORAGE, COMPLETE)                            \
    kernel void NAME(                                                                              \
        texture2d<ushort, access::read> w [[texture(0)]], device const float *x [[buffer(1)]],     \
        device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],                           \
        uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],    \
        uint simd [[simdgroup_index_in_threadgroup]]) {                                            \
        threadgroup ushort4 weights[(R * K > R * N * 2 ? R * K : R * N * 2) / 4];                  \
        threadgroup float inputs[N * K];                                                           \
        matmul_compact<STORAGE, N, K, PARTS, R, COMPLETE>(w, x, y, p, group, tid, simd,            \
                                                          (threadgroup ushort *)weights, inputs,   \
                                                          (threadgroup float *)weights);           \
    }
TEXTURE_COMPACT_KERNEL(plamo_matmul8_compact_texture, 8, 64, 1, 32, 3, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul16_compact_texture, 16, 32, 1, 64, 3, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul24_compact_texture, 24, 32, 1, 64, 3, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul32_compact_texture, 32, 32, 1, 64, 3, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul8_split_compact_texture, 8, 32, 4, 32, 3, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul16_split_compact_texture, 16, 32, 4, 32, 3, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul8_compact_texture_direct, 8, 64, 1, 32, 4, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul16_compact_texture_direct, 16, 32, 1, 64, 4, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul24_compact_texture_direct, 24, 32, 1, 64, 4, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul32_compact_texture_direct, 32, 32, 1, 64, 4, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul8_split_compact_texture_direct, 8, 32, 4, 32, 4, false)
TEXTURE_COMPACT_KERNEL(plamo_matmul16_split_compact_texture_direct, 16, 32, 4, 32, 4, false)
// The host proves complete N8/R32/K32 tiles before selecting these kernels.
TEXTURE_COMPACT_KERNEL(plamo_matmul8_full_texture, 8, 32, 4, 32, 3, true)
TEXTURE_COMPACT_KERNEL(plamo_matmul8_full_texture_direct, 8, 32, 4, 32, 4, true)

kernel void plamo_split_reduce(device const float *partial [[buffer(0)]],
                               device float *out [[buffer(1)]], constant Linear &p [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    uint count = p.rows * p.tokens;
    if (i >= count)
        return;
    float value = partial[i];
    for (uint part = 1; part < 4; ++part)
        value += partial[ulong(part) * count + i];
    out[ulong(i / p.rows) * p.output_stride + i % p.rows] = value;
}

// RMSNorm stays entirely FP32; exponentials use the CPU polynomial.
// Pairwise FP32 reduction of nonnegative squares. Each lane's partial and the
// four SIMD-group sums stay FP32; the shallower sum avoids the long serial
// rounding chain as well as thousands of idle threads across decode layers.
inline void rms_inverse(device const float *x, uint hidden, uint tid, threadgroup float *partials,
                        threadgroup float &inverse) {
    float sum = 0.0f;
    for (uint i = tid; i < hidden; i += 128)
        sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (!(tid & 31))
        partials[tid >> 5] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid)
        inverse = 1.0f /
                  sqrt(((partials[0] + partials[1]) + (partials[2] + partials[3])) / float(hidden) +
                       0.000001f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct MLP {
    uint hidden, intermediate, tokens;
};
kernel void plamo_mlp_norm(device const float *x [[buffer(0)]],
                           device const ushort *w [[buffer(1)]], device float *y [[buffer(2)]],
                           constant MLP &p [[buffer(3)]],
                           uint token [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partials[4], inverse;
    ulong base = ulong(token) * p.hidden;
    rms_inverse(x + base, p.hidden, tid, partials, inverse);
    for (uint i = tid; i < p.hidden; i += 128)
        y[base + i] = (x[base + i] * inverse) * (bf16(w[i]) + 1.0f);
}

inline float plamo_exp(float x) {
    if (x < -87.0f)
        return 0.0f;
    if (x > 88.0f)
        return INFINITY;
    float n = rint(x * 1.4426950408889634f);
    float r = fma(-n, 0.6931471805599453f, x);
    float y = fma(0.0001984126984126984f, r, 0.001388888888888889f);
    y = fma(y, r, 0.008333333333333333f);
    y = fma(y, r, 0.041666666666666664f);
    y = fma(y, r, 0.16666666666666666f);
    y = fma(y, r, 0.5f);
    y = fma(y, r, 1.0f);
    y = fma(y, r, 1.0f);
    return y * as_type<float>(uint(int(n) + 127) << 23);
}

kernel void plamo_mlp_swiglu(device float *x [[buffer(0)]], constant MLP &p [[buffer(1)]],
                             uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.intermediate || pos.y >= p.tokens)
        return;
    ulong i = ulong(pos.y) * 2 * p.intermediate + pos.x;
    x[i] = (x[i] / (1.0f + plamo_exp(-x[i]))) * x[i + p.intermediate];
}

// Reuse the consumed normalization buffer for contiguous MLP inputs. Wide
// prompts otherwise retain an unused up-projection row between useful rows.
kernel void plamo_mlp_swiglu_compact(device const float *x [[buffer(0)]],
                                     constant MLP &p [[buffer(1)]], device float *out [[buffer(2)]],
                                     uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.intermediate || pos.y >= p.tokens)
        return;
    ulong i = ulong(pos.y) * 2 * p.intermediate + pos.x;
    out[ulong(pos.y) * p.intermediate + pos.x] =
        (x[i] / (1.0f + plamo_exp(-x[i]))) * x[i + p.intermediate];
}

kernel void plamo_mlp_residual(device float *state [[buffer(0)]],
                               device const ushort *w [[buffer(1)]],
                               device const float *branch [[buffer(2)]],
                               constant MLP &p [[buffer(3)]],
                               uint token [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partials[4], inverse;
    ulong base = ulong(token) * p.hidden;
    rms_inverse(branch + base, p.hidden, tid, partials, inverse);
    for (uint i = tid; i < p.hidden; i += 128)
        state[base + i] += (branch[base + i] * inverse) * (bf16(w[i]) + 0.08944271909999159f);
}

struct Decode {
    uint qh, kh, dim, count;
};
inline float2 rotate_pair(float2 x, float2 cs) {
    return float2(x.x * cs.x - x.y * cs.y, x.y * cs.x + x.x * cs.y);
}

// Shift a full sliding cache, or rebuild a stale rotated cache. Each thread
// owns one coordinate pair, so the overlapping forward copy has no races.
kernel void plamo_decode_keys(device float *k [[buffer(0)]], device float *v [[buffer(1)]],
                              device float *rotated [[buffer(2)]],
                              device const float *table [[buffer(3)]],
                              constant Decode &p [[buffer(4)]], constant uint &shift [[buffer(5)]],
                              uint pair [[thread_position_in_grid]]) {
    uint half_dim = p.dim / 2, head = pair / half_dim, col = pair % half_dim;
    if (head >= p.kh)
        return;
    uint width = p.kh * p.dim, component = head * p.dim + col;
    for (uint row = 0; row + 1 < p.count; ++row) {
        ulong source = ulong(row + shift) * width + component,
              dest = ulong(row) * width + component;
        float2 key = float2(k[source], k[source + half_dim]);
        if (shift) {
            k[dest] = key.x;
            k[dest + half_dim] = key.y;
            v[dest] = v[source];
            v[dest + half_dim] = v[source + half_dim];
        }
        ulong t = ulong(row) * p.dim + col;
        float2 r = rotate_pair(key, float2(table[t], table[t + half_dim]));
        rotated[dest] = r.x;
        rotated[dest + half_dim] = r.y;
    }
}

// Normalize Q/K with the CPU's sequential FP32 sum, rotate Q and the appended
// K, and append V. CPU and GPU refer to the same original KV allocations.
kernel void plamo_decode_qkv(device float *qkv [[buffer(0)]], device const ushort *qn [[buffer(1)]],
                             device const ushort *kn [[buffer(2)]], device float *k [[buffer(3)]],
                             device float *v [[buffer(4)]], device float *rotated [[buffer(5)]],
                             device const float *table [[buffer(6)]],
                             constant Decode &p [[buffer(7)]],
                             uint head [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float inverse;
    ulong base = ulong(head) * p.dim;
    if (!tid) {
        float sum = 0.0f;
        for (uint i = 0; i < p.dim; ++i)
            sum += qkv[base + i] * qkv[base + i];
        inverse = 1.0f / sqrt(sum / float(p.dim) + 0.000001f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint half_dim = p.dim / 2;
    device const ushort *w = head < p.qh ? qn : kn;
    for (uint col = tid; col < half_dim; col += 128) {
        float2 raw =
            float2((qkv[base + col] * inverse) * (bf16(w[col]) + 1.0f),
                   (qkv[base + col + half_dim] * inverse) * (bf16(w[col + half_dim]) + 1.0f));
        ulong t = ulong(p.count - 1) * p.dim + col;
        float2 r = rotate_pair(raw, float2(table[t], table[t + half_dim]));
        if (head < p.qh) {
            qkv[base + col] = r.x;
            qkv[base + col + half_dim] = r.y;
        } else {
            ulong dest = (ulong(p.count - 1) * p.kh + head - p.qh) * p.dim + col;
            k[dest] = raw.x;
            k[dest + half_dim] = raw.y;
            rotated[dest] = r.x;
            rotated[dest + half_dim] = r.y;
            ulong vi = ulong(p.qh + p.kh + head - p.qh) * p.dim + col;
            v[dest] = qkv[vi];
            v[dest + half_dim] = qkv[vi + half_dim];
        }
    }
}

// Parallel heads and score rows, with the original per-dot FMA order,
// sequential softmax denominator and weighted-value accumulation preserved.
kernel void
plamo_decode_attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                       device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
                       device float *scores [[buffer(4)]], constant Decode &p [[buffer(5)]],
                       uint head [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float maximum, denominator;
    uint kh = head / (p.qh / p.kh), width = p.kh * p.dim;
    device float *s = scores + ulong(head) * p.count;
    float scale = 1.0f / sqrt(float(p.dim));
    for (uint row = tid; row < p.count; row += 128) {
        float sum = 0.0f;
        for (uint col = 0; col < p.dim; ++col)
            sum = fma(q[ulong(head) * p.dim + col], k[(ulong(row) * p.kh + kh) * p.dim + col], sum);
        s[row] = sum * scale;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float m = s[0];
        for (uint row = 1; row < p.count; ++row)
            m = max(m, s[row]);
        maximum = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < p.count; row += 128)
        s[row] = plamo_exp(s[row] - maximum);
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float sum = 0.0f;
        for (uint row = 0; row < p.count; ++row)
            sum += s[row];
        denominator = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < p.count; row += 128)
        s[row] /= denominator;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint col = tid; col < p.dim; col += 128) {
        float sum = 0.0f;
        for (uint row = 0; row < p.count; ++row)
            sum = fma(s[row], v[ulong(row) * width + kh * p.dim + col], sum);
        out[ulong(head) * p.dim + col] = sum;
    }
}

#define DECODE_RESIDUAL(NAME, OFFSET)                                                              \
    kernel void NAME(device float *state [[buffer(0)]], device const ushort *w [[buffer(1)]],      \
                     device const float *branch [[buffer(2)]], constant MLP &p [[buffer(3)]],      \
                     uint token [[threadgroup_position_in_grid]],                                  \
                     uint tid [[thread_index_in_threadgroup]]) {                                   \
        threadgroup float partials[4], inverse;                                                    \
        ulong base = ulong(token) * p.hidden;                                                      \
        rms_inverse(branch + base, p.hidden, tid, partials, inverse);                              \
        for (uint i = tid; i < p.hidden; i += 128)                                                 \
            state[base + i] += (branch[base + i] * inverse) * (bf16(w[i]) + OFFSET);               \
    }
DECODE_RESIDUAL(plamo_decode_residual, 0.2f)

struct Prefill {
    uint qh, kh, dim, prefix, tokens, capacity, query_offset, speculative;
};

// Normalize all prompt Q/K heads in parallel. For a sliding batch that would
// overwrite a prefix still needed by earlier queries, retain normalized K/V
// in QKV until attention has finished. Otherwise use canonical KV directly.
kernel void plamo_prefill_qkv(device float *qkv [[buffer(0)]],
                              device const ushort *qn [[buffer(1)]],
                              device const ushort *kn [[buffer(2)]], device float *k [[buffer(3)]],
                              device float *v [[buffer(4)]], device float *rotated [[buffer(5)]],
                              device const float *table [[buffer(6)]],
                              constant Prefill &p [[buffer(7)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    uint head = group.x, token = group.y, half_dim = p.dim / 2;
    ulong width = ulong(p.qh + 2 * p.kh) * p.dim, base = ulong(token) * width + ulong(head) * p.dim;
    threadgroup float inverse;
    if (!tid) {
        float sum = 0.0f;
        for (uint col = 0; col < p.dim; ++col)
            sum += qkv[base + col] * qkv[base + col];
        inverse = 1.0f / sqrt(sum / float(p.dim) + 0.000001f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ushort *w = head < p.qh ? qn : kn;
    for (uint col = tid; col < half_dim; col += 128) {
        float2 raw =
            float2((qkv[base + col] * inverse) * (bf16(w[col]) + 1.0f),
                   (qkv[base + col + half_dim] * inverse) * (bf16(w[col + half_dim]) + 1.0f));
        uint position = min(p.prefix + token, p.capacity - 1);
        ulong t = ulong(position) * p.dim + col;
        float2 r = rotate_pair(raw, float2(table[t], table[t + half_dim]));
        if (head < p.qh) {
            qkv[base + col] = r.x;
            qkv[base + col + half_dim] = r.y;
        } else if (p.speculative) {
            qkv[base + col] = raw.x;
            qkv[base + col + half_dim] = raw.y;
            ulong dest = (ulong(token) * p.kh + head - p.qh) * p.dim + col;
            rotated[dest] = r.x;
            rotated[dest + half_dim] = r.y;
        } else if (p.prefix + p.tokens <= p.capacity) {
            ulong dest = (ulong(p.prefix + token) * p.kh + head - p.qh) * p.dim + col;
            k[dest] = raw.x;
            k[dest + half_dim] = raw.y;
            rotated[dest] = r.x;
            rotated[dest + half_dim] = r.y;
            v[dest] = qkv[base + ulong(p.kh) * p.dim + col];
            v[dest + half_dim] = qkv[base + ulong(p.kh) * p.dim + col + half_dim];
        } else {
            qkv[base + col] = raw.x;
            qkv[base + col + half_dim] = raw.y;
        }
    }
}

// A group normalizes eight heads, preserving each head's sequential FP32 sum.
// The transposed tile keeps the other lanes useful during that serial reduction.
kernel void plamo_prefill_qkv8(device float *qkv [[buffer(0)]],
                               device const ushort *qn [[buffer(1)]],
                               device const ushort *kn [[buffer(2)]], device float *k [[buffer(3)]],
                               device float *v [[buffer(4)]], device float *rotated [[buffer(5)]],
                               device const float *table [[buffer(6)]],
                               constant Prefill &p [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    uint token = group.y, first_head = group.x * 8;
    ulong width = ulong(p.qh + 2 * p.kh) * 128;
    threadgroup float values[128 * 9], inverses[8];
    for (uint i = tid; i < 8 * 128; i += 128) {
        uint head = first_head + i / 128, col = i % 128;
        values[col * 9 + i / 128] =
            head < p.qh + p.kh ? qkv[ulong(token) * width + head * 128 + col] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float sum = 0.0f;
        for (uint col = 0; col < 128; ++col) {
            float value = values[col * 9 + tid];
            sum += value * value;
        }
        inverses[tid] = 1.0f / sqrt(sum / 128.0f + 0.000001f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 8 * 64; i += 128) {
        uint head = first_head + i / 64, col = i % 64, half_dim = 64;
        if (head >= p.qh + p.kh)
            continue;
        ulong base = ulong(token) * width + head * 128;
        float inverse = inverses[i / 64];
        device const ushort *w = head < p.qh ? qn : kn;
        float2 raw =
            float2((values[col * 9 + i / 64] * inverse) * (bf16(w[col]) + 1.0f),
                   (values[(col + 64) * 9 + i / 64] * inverse) * (bf16(w[col + 64]) + 1.0f));
        uint position = min(p.prefix + token, p.capacity - 1);
        ulong t = ulong(position) * p.dim + col;
        float2 r = rotate_pair(raw, float2(table[t], table[t + half_dim]));
        if (head < p.qh) {
            qkv[base + col] = r.x;
            qkv[base + col + half_dim] = r.y;
        } else if (p.speculative) {
            qkv[base + col] = raw.x;
            qkv[base + col + half_dim] = raw.y;
            ulong dest = (ulong(token) * p.kh + head - p.qh) * p.dim + col;
            rotated[dest] = r.x;
            rotated[dest + half_dim] = r.y;
        } else if (p.prefix + p.tokens <= p.capacity) {
            ulong dest = (ulong(p.prefix + token) * p.kh + head - p.qh) * p.dim + col;
            k[dest] = raw.x;
            k[dest + half_dim] = raw.y;
            rotated[dest] = r.x;
            rotated[dest + half_dim] = r.y;
            v[dest] = qkv[base + ulong(p.kh) * p.dim + col];
            v[dest + half_dim] = qkv[base + ulong(p.kh) * p.dim + col + half_dim];
        } else {
            qkv[base + col] = raw.x;
            qkv[base + col + half_dim] = raw.y;
        }
    }
}

// Each head/query owns one score row. Query tiles reuse bounded score scratch
// without storing a tokens x tokens attention matrix for the whole prompt.
// Keep sequential FP32 dot products, softmax sum and value accumulation.
kernel void plamo_prefill_attention(
    device const float *qkv [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device const float *rotated [[buffer(3)]],
    device const float *table [[buffer(4)]], device float *out [[buffer(5)]],
    device float *scores [[buffer(6)]], constant Prefill &p [[buffer(7)]],
    device const float *proposed_rotated [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint head = group.x, token = p.query_offset + group.y, kh = head / (p.qh / p.kh);
    uint count = min(p.prefix + token + 1, p.capacity), first = p.prefix + token + 1 - count;
    uint half_dim = p.dim / 2, kv_width = p.kh * p.dim, width = (p.qh + 2 * p.kh) * p.dim;
    bool sliding = p.speculative || p.prefix + p.tokens > p.capacity;
    device const float *q = qkv + ulong(token) * width + ulong(head) * p.dim;
    device float *s =
        scores + (ulong(group.y) * p.qh + head) * min(p.prefix + p.tokens, p.capacity);
    threadgroup float maximum, denominator;
    float scale = 1.0f / sqrt(float(p.dim));
    for (uint row = tid; row < count; row += 128) {
        float sum = 0.0f;
        if (p.prefix + p.tokens <= p.capacity) {
            device const float *key =
                p.speculative && row >= p.prefix
                    ? proposed_rotated + (ulong(row - p.prefix) * p.kh + kh) * p.dim
                    : rotated + (ulong(row) * p.kh + kh) * p.dim;
            for (uint col = 0; col < p.dim; ++col)
                sum = fma(q[col], key[col], sum);
        } else {
            uint source = first + row;
            device const float *key =
                source < p.prefix ? k + ulong(source) * kv_width + kh * p.dim
                                  : qkv + ulong(source - p.prefix) * width + (p.qh + kh) * p.dim;
            // Sliding RoPE uses local positions after each eviction, matching
            // CPU rebasing even across several window lengths in one batch.
            for (uint col = 0; col < p.dim; ++col) {
                uint pair = col % half_dim;
                ulong t = ulong(row) * p.dim + pair;
                float2 r = rotate_pair(float2(key[pair], key[pair + half_dim]),
                                       float2(table[t], table[t + half_dim]));
                sum = fma(q[col], col < half_dim ? r.x : r.y, sum);
            }
        }
        s[row] = sum * scale;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float value = s[0];
        for (uint row = 1; row < count; ++row)
            value = max(value, s[row]);
        maximum = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < count; row += 128)
        s[row] = plamo_exp(s[row] - maximum);
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float sum = 0.0f;
        for (uint row = 0; row < count; ++row)
            sum += s[row];
        denominator = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < count; row += 128)
        s[row] /= denominator;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint col = tid; col < p.dim; col += 128) {
        float sum = 0.0f;
        for (uint row = 0; row < count; ++row) {
            uint source = first + row;
            float value =
                !sliding || source < p.prefix
                    ? v[ulong(sliding ? source : row) * kv_width + kh * p.dim + col]
                    : qkv[ulong(source - p.prefix) * width + (p.qh + p.kh + kh) * p.dim + col];
            sum = fma(s[row], value, sum);
        }
        out[(ulong(token) * p.qh + head) * p.dim + col] = sum;
    }
}

// A causal score tile shares sixteen keys across four queries. Each score
// retains the original column-by-column FP32 FMA order. Padding avoids bank
// conflicts while loading row-major keys into the transposed shared tile.
kernel void plamo_verify_scores(device const float *qkv [[buffer(0)]],
                                device const float *rotated [[buffer(1)]],
                                device const float *proposed [[buffer(2)]],
                                device float *scores [[buffer(3)]],
                                constant Prefill &p [[buffer(4)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float keys[128 * 17], queries[4 * 129];
    uint head = group.x, kh = head / (p.qh / p.kh), width = (p.qh + 2 * p.kh) * 128;
    uint first_query = group.y * 4, first_key = group.z * 16, count = p.prefix + p.tokens;
    for (uint i = tid; i < 4 * 128; i += 64) {
        uint token = first_query + i / 128, col = i % 128;
        queries[(i / 128) * 129 + col] =
            token < p.tokens ? qkv[ulong(token) * width + head * 128 + col] : 0.0f;
    }
    for (uint i = tid; i < 16 * 128; i += 64) {
        uint row = first_key + i / 128, col = i % 128;
        float value = 0.0f;
        if (row < count) {
            device const float *key = row < p.prefix
                                          ? rotated + (ulong(row) * p.kh + kh) * 128
                                          : proposed + (ulong(row - p.prefix) * p.kh + kh) * 128;
            value = key[col];
        }
        keys[col * 17 + i / 128] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint query = tid / 16, key = tid % 16, token = first_query + query, row = first_key + key;
    if (token < p.tokens && row < p.prefix + token + 1) {
        float sum = 0.0f;
        for (uint col = 0; col < 128; ++col)
            sum = fma(queries[query * 129 + col], keys[col * 17 + key], sum);
        scores[(ulong(token) * p.qh + head) * count + row] = sum * (1.0f / sqrt(128.0f));
    }
}

kernel void plamo_verify_softmax(device float *scores [[buffer(0)]],
                                 constant Prefill &p [[buffer(1)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    uint count = p.prefix + group.y + 1;
    device float *s = scores + (ulong(group.y) * p.qh + group.x) * (p.prefix + p.tokens);
    threadgroup float maximum, denominator;
    if (!tid) {
        float value = s[0];
        for (uint row = 1; row < count; ++row)
            value = max(value, s[row]);
        maximum = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < count; row += 128)
        s[row] = plamo_exp(s[row] - maximum);
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float sum = 0.0f;
        for (uint row = 0; row < count; ++row)
            sum += s[row];
        denominator = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < count; row += 128)
        s[row] /= denominator;
}

// Two queries reuse each 32-key value tile. The row loop is still ascending
// and sequential for each output component; this is a memory transformation.
kernel void plamo_verify_values(device const float *qkv [[buffer(0)]],
                                device const float *v [[buffer(1)]],
                                device const float *scores [[buffer(2)]],
                                device float *out [[buffer(3)]], constant Prefill &p [[buffer(4)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[32 * 65], weights[2 * 32];
    uint head = group.x, kh = head / (p.qh / p.kh), first_query = group.y * 2,
         first_col = group.z * 64;
    uint width = (p.qh + 2 * p.kh) * 128, query = tid / 64, col = tid % 64,
         token = first_query + query;
    uint maximum = p.prefix + p.tokens, count = p.prefix + token + 1;
    float sum = 0.0f;
    for (uint first = 0; first < maximum; first += 32) {
        for (uint i = tid; i < 32 * 64; i += 128) {
            uint row = first + i / 64, component = first_col + i % 64;
            float value = 0.0f;
            if (row < maximum)
                value =
                    row < p.prefix
                        ? v[(ulong(row) * p.kh + kh) * 128 + component]
                        : qkv[ulong(row - p.prefix) * width + (p.qh + p.kh + kh) * 128 + component];
            values[(i / 64) * 65 + i % 64] = value;
        }
        if (tid < 64) {
            uint q = tid / 32, row = first + tid % 32, qt = first_query + q;
            weights[tid] = qt < p.tokens && row < p.prefix + qt + 1
                               ? scores[(ulong(qt) * p.qh + head) * maximum + row]
                               : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (token < p.tokens)
            for (uint row = 0; row < 32 && first + row < count; ++row)
                sum = fma(weights[query * 32 + row], values[row * 65 + col], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token < p.tokens)
        out[(ulong(token) * p.qh + head) * 128 + first_col + col] = sum;
}

// Commit only the final sliding window after every query has consumed the old
// prefix. A thread owns a coordinate pair and copies rows forward in order.
kernel void plamo_prefill_store(device const float *qkv [[buffer(0)]],
                                device float *k [[buffer(1)]], device float *v [[buffer(2)]],
                                device float *rotated [[buffer(3)]],
                                device const float *table [[buffer(4)]],
                                constant Prefill &p [[buffer(5)]],
                                uint pair [[thread_position_in_grid]]) {
    uint half_dim = p.dim / 2, head = pair / half_dim, col = pair % half_dim;
    if (head >= p.kh)
        return;
    uint count = min(p.prefix + p.tokens, p.capacity), first = p.prefix + p.tokens - count;
    uint width = (p.qh + 2 * p.kh) * p.dim;
    for (uint row = 0; row < count; ++row) {
        uint source = first + row;
        ulong dest = (ulong(row) * p.kh + head) * p.dim + col;
        device const float *key =
            source < p.prefix ? k + (ulong(source) * p.kh + head) * p.dim
                              : qkv + ulong(source - p.prefix) * width + (p.qh + head) * p.dim;
        device const float *value = source < p.prefix ? v + (ulong(source) * p.kh + head) * p.dim
                                                      : qkv + ulong(source - p.prefix) * width +
                                                            (p.qh + p.kh + head) * p.dim;
        float2 raw = float2(key[col], key[col + half_dim]),
               val = float2(value[col], value[col + half_dim]);
        ulong t = ulong(row) * p.dim + col;
        float2 r = rotate_pair(raw, float2(table[t], table[t + half_dim]));
        k[dest] = raw.x;
        k[dest + half_dim] = raw.y;
        v[dest] = val.x;
        v[dest + half_dim] = val.y;
        rotated[dest] = r.x;
        rotated[dest + half_dim] = r.y;
    }
}

// Q4 proposal-only cache: FP32 scale + 16 bytes of adjacent code pairs.
kernel void plamo_draft_quantize4(device const ushort *w [[buffer(0)]],
                                  device uchar *out [[buffer(1)]],
                                  constant uint &count [[buffer(2)]],
                                  uint block [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    uint i = block * 32 + lane;
    float value = i < count ? bf16(w[i]) : 0.0f;
    float maximum = simd_max(abs(value)), scale = maximum / 7.0f;
    device uchar *data = out + ulong(block) * 20;
    if (!lane)
        *((device float *)data) = scale;
    if (lane < 16) {
        uint first = block * 32 + lane * 2, second = first + 1;
        float a = first < count ? bf16(w[first]) : 0.0f;
        float b = second < count ? bf16(w[second]) : 0.0f;
        uint qa = scale ? uint(clamp(int(rint(a / scale)), -8, 7) + 8) : 8;
        uint qb = scale ? uint(clamp(int(rint(b / scale)), -8, 7) + 8) : 8;
        data[4 + lane] = uchar(qa | qb << 4);
    }
}

// Adjacent Q4 codes become exact half integers; FP32 block scales and sums
// avoid rounding small weight scales. Only auxiliary activations use half.
template <uint Width>
inline void draft_dot(device const uchar *w, device const half *x, device float *y,
                      constant Linear &p, uint3 group, uint simd, uint lane) {
    uint inner = lane % Width, row = group.x * (128 / Width) + simd * (32 / Width) + lane / Width;
    float sum = 0.0f;
    for (uint k = inner * 8; k < p.columns; k += Width * 8) {
        if (row >= p.rows)
            continue;
        uint i = row * p.columns + k + p.weight_offset, j = i & 31;
        device const uchar *block = w + ulong(i >> 5) * 20;
        float scale = *((device const float *)block);
        uint code = as_type<uint>(*((device const packed_uchar4 *)(block + 4 + j / 2)));
        uint4 bytes = (uint4(code) >> uint4(0, 8, 16, 24)) & 255;
        uint4 bits = (bytes & 15) | ((bytes & 240) << 12) | 0x64006400;
        half4 c = as_type<half4>(bits.xy) - half4(1032), d = as_type<half4>(bits.zw) - half4(1032);
        ulong index = ulong(group.y) * p.input_stride + k;
        half4 a = *((device const half4 *)(x + index)),
              b = *((device const half4 *)(x + index + 4));
        half4 products = fma(c, a, d * b);
        half dot = (products.x + products.y) + (products.z + products.w);
        sum = fma(float(dot), scale, sum);
    }
    if (Width == 32)
        sum = simd_sum(sum);
    else {
#pragma unroll
        for (uint delta = Width / 2; delta; delta /= 2)
            sum += simd_shuffle_xor(sum, delta);
    }
    if (!inner && row < p.rows)
        y[ulong(group.y) * p.output_stride + row] = sum;
}
#define DRAFT_DOT_KERNEL(NAME, WIDTH)                                                              \
    kernel void NAME(device const uchar *w [[buffer(0)]], device const half *x [[buffer(1)]],      \
                     device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],              \
                     uint3 group [[threadgroup_position_in_grid]],                                 \
                     uint simd [[simdgroup_index_in_threadgroup]],                                 \
                     uint lane [[thread_index_in_simdgroup]]) {                                    \
        draft_dot<WIDTH>(w, x, y, p, group, simd, lane);                                           \
    }
DRAFT_DOT_KERNEL(plamo_draft_dot32, 32)
DRAFT_DOT_KERNEL(plamo_draft_dot8, 8)

kernel void plamo_draft_norm(device const float *x [[buffer(0)]],
                             device const ushort *w [[buffer(1)]], device float *y [[buffer(2)]],
                             constant MLP &p [[buffer(3)]], device half *copy [[buffer(4)]],
                             uint token [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partials[4], inverse;
    ulong base = ulong(token) * p.hidden;
    rms_inverse(x + base, p.hidden, tid, partials, inverse);
    for (uint i = tid; i < p.hidden; i += 128) {
        float value = (x[base + i] * inverse) * (bf16(w[i]) + 1.0f);
        y[base + i] = value;
        copy[base + i] = half(value);
    }
}
kernel void plamo_draft_swiglu(device float *x [[buffer(0)]], constant MLP &p [[buffer(1)]],
                               device half *copy [[buffer(2)]],
                               uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.intermediate || pos.y >= p.tokens)
        return;
    ulong i = ulong(pos.y) * 2 * p.intermediate + pos.x;
    float value = (x[i] / (1.0f + plamo_exp(-x[i]))) * x[i + p.intermediate];
    x[i] = value;
    copy[ulong(pos.y) * p.intermediate + pos.x] = half(value);
}

// Probe the compiler's opaque 8x8 matrix fragment layout before using the
// direct element path. A different layout keeps the matrix-load fallback.
kernel void plamo_matrix_layout(device float *out [[buffer(0)]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tile[64];
    tile[lane * 2] = float(lane * 2);
    tile[lane * 2 + 1] = float(lane * 2 + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 matrix;
    simdgroup_load(matrix, tile, 8);
    out[lane * 2] = matrix.thread_elements()[0];
    out[lane * 2 + 1] = matrix.thread_elements()[1];
}

// Prompt projections share a 64-row BF16 tile across 24, 32, 40 or 64 tokens. Only
// the small input tile is FP32; every product and accumulator remains FP32.
// Direct fragments avoid matrix-load transposes and shared output staging.
// Narrow panels reduce padding without changing the accumulation order.
template <uint Storage, uint N, bool TwoRows = false>
inline void prompt_matrix(texture2d<ushort, access::read> w, device const float *x, device float *y,
                          constant Linear &p, uint3 group, uint tid, uint simd, uint lane,
                          threadgroup ushort *weights, threadgroup float *inputs) {
    constexpr uint R = 64, K = TwoRows ? 8 : 16, SR = TwoRows ? 4 : 8, MR = 8 / SR,
                   Threads = SR * 32;
    uint lr = ((lane & 6) >> 1) | ((lane & 16) >> 2);
    uint lc = ((lane & 1) << 1) | ((lane & 8) >> 1);
    uint br = group.x * R + (simd % SR) * MR * 8;
    uint bt = group.y * N + (simd / SR) * N;
    simdgroup_float8x8 c0(0.0f), c1(0.0f), c2(0.0f), c3(0.0f);
    simdgroup_float8x8 d0(0.0f), d1(0.0f), d2(0.0f), d3(0.0f);
    simdgroup_float8x8 c4(0.0f), c5(0.0f), c6(0.0f), c7(0.0f);
    simdgroup_float8x8 d4(0.0f), d5(0.0f), d6(0.0f), d7(0.0f);
    for (uint base = 0; base < p.columns; base += K) {
        for (uint i = tid; i < R * (K / 4); i += Threads) {
            uint row = group.x * R + i / (K / 4), col = base + (i % (K / 4)) * 4;
            ushort4 bits = row < p.rows ? weight_row_bits4<Storage>(w, row, col, p) : ushort4(0);
            *((threadgroup ushort4 *)(weights + i * 4)) = bits;
        }
        for (uint i = tid; i < N * (K / 4); i += Threads) {
            uint token = group.y * N + i / (K / 4), col = base + (i % (K / 4)) * 4;
            float4 values =
                token < p.tokens
                    ? *((device const float4 *)(x + ulong(token) * p.input_stride + col))
                    : float4(0);
            *((threadgroup float4 *)(inputs + i * 4)) = values;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < K; k += 8) {
            simdgroup_float8x8 a, second;
            uint first = ((simd % SR) * MR * 8 + lr) * K + k + lc;
            // One aligned load recovers both original BF16 encodings.
            uint bits = *((threadgroup uint *)(weights + first));
            a.thread_elements()[0] = as_type<float>(bits << 16);
            a.thread_elements()[1] = as_type<float>(bits & 0xffff0000u);
            if (TwoRows) {
                uint bits = *((threadgroup uint *)(weights + first + 8 * K));
                second.thread_elements()[0] = as_type<float>(bits << 16);
                second.thread_elements()[1] = as_type<float>(bits & 0xffff0000u);
            }
#define PROMPT_ACCUMULATE(I, C, D)                                                                 \
    {                                                                                              \
        simdgroup_float8x8 b;                                                                      \
        uint pos = ((simd / SR) * N + I * 8 + lc) * K + k + lr;                                    \
        b.thread_elements()[0] = inputs[pos];                                                      \
        b.thread_elements()[1] = inputs[pos + K];                                                  \
        simdgroup_multiply_accumulate(C, a, b, C);                                                 \
        if (TwoRows)                                                                               \
            simdgroup_multiply_accumulate(D, second, b, D);                                        \
    }
            PROMPT_ACCUMULATE(0, c0, d0)
            PROMPT_ACCUMULATE(1, c1, d1)
            PROMPT_ACCUMULATE(2, c2, d2)
            if (N >= 32) {
                PROMPT_ACCUMULATE(3, c3, d3)
            }
            if (N >= 40) {
                PROMPT_ACCUMULATE(4, c4, d4)
            }
            if (N == 64) {
                PROMPT_ACCUMULATE(5, c5, d5)
                PROMPT_ACCUMULATE(6, c6, d6)
                PROMPT_ACCUMULATE(7, c7, d7)
            }
#undef PROMPT_ACCUMULATE
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#define PROMPT_STORE(I, C, D)                                                                      \
    {                                                                                              \
        uint row = br + lr, token = bt + I * 8 + lc;                                               \
        if (row < p.rows && token < p.tokens)                                                      \
            y[ulong(token) * p.output_stride + row] = C.thread_elements()[0];                      \
        if (row < p.rows && token + 1 < p.tokens)                                                  \
            y[ulong(token + 1) * p.output_stride + row] = C.thread_elements()[1];                  \
        if (TwoRows) {                                                                             \
            if (row + 8 < p.rows && token < p.tokens)                                              \
                y[ulong(token) * p.output_stride + row + 8] = D.thread_elements()[0];              \
            if (row + 8 < p.rows && token + 1 < p.tokens)                                          \
                y[ulong(token + 1) * p.output_stride + row + 8] = D.thread_elements()[1];          \
        }                                                                                          \
    }
    PROMPT_STORE(0, c0, d0)
    PROMPT_STORE(1, c1, d1)
    PROMPT_STORE(2, c2, d2)
    if (N >= 32) {
        PROMPT_STORE(3, c3, d3)
    }
    if (N >= 40) {
        PROMPT_STORE(4, c4, d4)
    }
    if (N == 64) {
        PROMPT_STORE(5, c5, d5)
        PROMPT_STORE(6, c6, d6)
        PROMPT_STORE(7, c7, d7)
    }
#undef PROMPT_STORE
}
#define PROMPT_MATRIX(NAME, STORAGE, N, TWO_ROWS)                                                  \
    kernel void NAME(                                                                              \
        texture2d<ushort, access::read> w [[texture(0)]], device const float *x [[buffer(1)]],     \
        device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]],                           \
        uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],    \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {   \
        threadgroup ushort weights[64 * (TWO_ROWS ? 8 : 16)];                                      \
        threadgroup float inputs[N * (TWO_ROWS ? 8 : 16)];                                         \
        prompt_matrix<STORAGE, N, TWO_ROWS>(w, x, y, p, group, tid, simd, lane, weights, inputs);  \
    }
PROMPT_MATRIX(plamo_prompt32_texture, 3, 32, false)
PROMPT_MATRIX(plamo_prompt64_texture, 3, 64, true)
PROMPT_MATRIX(plamo_prompt24_texture, 3, 24, true)
PROMPT_MATRIX(plamo_prompt40_texture, 3, 40, true)
PROMPT_MATRIX(plamo_prompt32_texture_direct, 4, 32, false)
PROMPT_MATRIX(plamo_prompt64_texture_direct, 4, 64, true)
PROMPT_MATRIX(plamo_prompt24_texture_direct, 4, 24, true)
PROMPT_MATRIX(plamo_prompt40_texture_direct, 4, 40, true)

template <uint Capacity>
inline void prompt_scores_matrix(device const float *rotated, threadgroup const float *queries,
                                 threadgroup float *scores, constant Prefill &p, uint kh,
                                 uint first_query, uint last_key, uint tid) {
    uint lr = ((tid & 6) >> 1) | ((tid & 16) >> 2);
    uint lc = ((tid & 1) << 1) | ((tid & 8) >> 1), simd = tid / 32;
    for (uint first = 0; first < last_key; first += 64) {
        simdgroup_float8x8 accum(0.0f);
        for (uint col = 0; col < 128; col += 8) {
            simdgroup_float8x8 q, key;
            q.thread_elements()[0] = queries[lr * 129 + col + lc];
            q.thread_elements()[1] = queries[lr * 129 + col + lc + 1];
            uint row = first + simd * 8 + lc;
            key.thread_elements()[0] =
                row < last_key ? rotated[(ulong(row) * p.kh + kh) * 128 + col + lr] : 0.0f;
            key.thread_elements()[1] =
                row + 1 < last_key ? rotated[(ulong(row + 1) * p.kh + kh) * 128 + col + lr] : 0.0f;
            simdgroup_multiply_accumulate(accum, q, key, accum);
        }
        uint row = first + simd * 8 + lc, count = p.prefix + first_query + lr + 1;
        if (first_query + lr < p.tokens && row < count)
            scores[lr * Capacity + row] = accum.thread_elements()[0] * (1.0f / sqrt(128.0f));
        if (first_query + lr < p.tokens && row + 1 < count)
            scores[lr * Capacity + row + 1] = accum.thread_elements()[1] * (1.0f / sqrt(128.0f));
    }
}

template <uint Capacity>
inline void prompt_values_matrix(device const float *v, device float *out,
                                 threadgroup const float *scores, constant Prefill &p, uint head,
                                 uint kh, uint first_query, uint tid) {
    uint lr = ((tid & 6) >> 1) | ((tid & 16) >> 2);
    uint lc = ((tid & 1) << 1) | ((tid & 8) >> 1), column = (tid / 32) * 16 + lc;
    uint token = first_query + lr, common = (p.prefix + first_query) / 8 * 8;
    simdgroup_float8x8 c0(0.0f), c1(0.0f);
    for (uint first = 0; first < common; first += 8) {
        simdgroup_float8x8 a, b;
        a.thread_elements()[0] = token < p.tokens ? scores[lr * Capacity + first + lc] : 0.0f;
        a.thread_elements()[1] = token < p.tokens ? scores[lr * Capacity + first + lc + 1] : 0.0f;
        b.thread_elements()[0] = v[(ulong(first + lr) * p.kh + kh) * 128 + column];
        b.thread_elements()[1] = v[(ulong(first + lr) * p.kh + kh) * 128 + column + 1];
        simdgroup_multiply_accumulate(c0, a, b, c0);
        b.thread_elements()[0] = v[(ulong(first + lr) * p.kh + kh) * 128 + column + 8];
        b.thread_elements()[1] = v[(ulong(first + lr) * p.kh + kh) * 128 + column + 9];
        simdgroup_multiply_accumulate(c1, a, b, c1);
    }
    if (token >= p.tokens)
        return;
    float v0 = c0.thread_elements()[0], v1 = c0.thread_elements()[1];
    float v2 = c1.thread_elements()[0], v3 = c1.thread_elements()[1];
    // Finish the causal tail without padding the accumulation with future keys.
    for (uint row = common; row < p.prefix + token + 1; ++row) {
        float weight = scores[lr * Capacity + row];
        v0 = fma(weight, v[(ulong(row) * p.kh + kh) * 128 + column], v0);
        v1 = fma(weight, v[(ulong(row) * p.kh + kh) * 128 + column + 1], v1);
        v2 = fma(weight, v[(ulong(row) * p.kh + kh) * 128 + column + 8], v2);
        v3 = fma(weight, v[(ulong(row) * p.kh + kh) * 128 + column + 9], v3);
    }
    out[(ulong(token) * p.qh + head) * 128 + column] = v0;
    out[(ulong(token) * p.qh + head) * 128 + column + 1] = v1;
    out[(ulong(token) * p.qh + head) * 128 + column + 8] = v2;
    out[(ulong(token) * p.qh + head) * 128 + column + 9] = v3;
}

// Eight causal queries share keys and values; softmax stays in threadgroup
// memory. Dot products, softmax sums and value FMAs retain their original order.
// Matrix fragments use the same lane-layout probe as the dense projections.
template <uint Capacity, bool Matrix>
inline void prompt_attention(device const float *qkv, device const float *rotated,
                             device const float *v, device float *out, constant Prefill &p,
                             uint2 group, uint tid, threadgroup float *keys,
                             threadgroup float *queries, threadgroup float *scores,
                             threadgroup float *maximum, threadgroup float *denominator) {
    constexpr uint Threads = Matrix ? 256 : 128;
    uint head = group.x, kh = head / (p.qh / p.kh), width = (p.qh + 2 * p.kh) * 128,
         first_query = group.y * 8, last = min(p.tokens, first_query + 8),
         last_key = p.prefix + last;
    for (uint i = tid; i < 8 * 128; i += Threads) {
        uint token = first_query + i / 128, col = i % 128;
        queries[(i / 128) * 129 + col] =
            token < p.tokens ? qkv[ulong(token) * width + head * 128 + col] : 0.0f;
    }
    if (Matrix) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prompt_scores_matrix<Capacity>(rotated, queries, scores, p, kh, first_query, last_key, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        uint query = tid / 16, key = tid % 16, token = first_query + query,
             count = p.prefix + token + 1;
        for (uint first = 0; first < last_key; first += 16) {
            for (uint i = tid; i < 16 * 128; i += 128) {
                uint row = first + i / 128, col = i % 128;
                keys[col * 17 + i / 128] =
                    row < last_key ? rotated[(ulong(row) * p.kh + kh) * 128 + col] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint row = first + key;
            if (token < p.tokens && row < count) {
                float sum = 0.0f;
                for (uint col = 0; col < 128; ++col)
                    sum = fma(queries[query * 129 + col], keys[col * 17 + key], sum);
                scores[query * Capacity + row] = sum * (1.0f / sqrt(128.0f));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < 8 && first_query + tid < p.tokens) {
        uint count = p.prefix + first_query + tid + 1;
        float value = scores[tid * Capacity];
        for (uint row = 1; row < count; ++row)
            value = max(value, scores[tid * Capacity + row]);
        maximum[tid] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 8 * Capacity; i += Threads) {
        uint q = i / Capacity, row = i % Capacity;
        if (first_query + q < p.tokens && row < p.prefix + first_query + q + 1)
            scores[i] = plamo_exp(scores[i] - maximum[q]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8 && first_query + tid < p.tokens) {
        uint count = p.prefix + first_query + tid + 1;
        float sum = 0.0f;
        for (uint row = 0; row < count; ++row)
            sum += scores[tid * Capacity + row];
        denominator[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 8 * Capacity; i += Threads) {
        uint q = i / Capacity, row = i % Capacity;
        if (first_query + q < p.tokens && row < p.prefix + first_query + q + 1)
            scores[i] /= denominator[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (Matrix) {
        prompt_values_matrix<Capacity>(v, out, scores, p, head, kh, first_query, tid);
        return;
    }
    for (uint col = tid; col < 128; col += Threads) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sum4 = 0.0f;
        float sum5 = 0.0f;
        float sum6 = 0.0f;
        float sum7 = 0.0f;
        // Complete query tiles share this causal prefix. Only the final eight
        // rows need masks; keep each accumulator's ascending FMA sequence.
        uint common = last - first_query == 8 ? p.prefix + first_query : 0;
        for (uint row = 0; row < common; ++row) {
            float value = v[(ulong(row) * p.kh + kh) * 128 + col];
            sum0 = fma(scores[0 * Capacity + row], value, sum0);
            sum1 = fma(scores[1 * Capacity + row], value, sum1);
            sum2 = fma(scores[2 * Capacity + row], value, sum2);
            sum3 = fma(scores[3 * Capacity + row], value, sum3);
            sum4 = fma(scores[4 * Capacity + row], value, sum4);
            sum5 = fma(scores[5 * Capacity + row], value, sum5);
            sum6 = fma(scores[6 * Capacity + row], value, sum6);
            sum7 = fma(scores[7 * Capacity + row], value, sum7);
        }
        for (uint row = common; row < last_key; ++row) {
            float value = v[(ulong(row) * p.kh + kh) * 128 + col];
            if (first_query + 0 < p.tokens && row < p.prefix + first_query + 0 + 1)
                sum0 = fma(scores[0 * Capacity + row], value, sum0);
            if (first_query + 1 < p.tokens && row < p.prefix + first_query + 1 + 1)
                sum1 = fma(scores[1 * Capacity + row], value, sum1);
            if (first_query + 2 < p.tokens && row < p.prefix + first_query + 2 + 1)
                sum2 = fma(scores[2 * Capacity + row], value, sum2);
            if (first_query + 3 < p.tokens && row < p.prefix + first_query + 3 + 1)
                sum3 = fma(scores[3 * Capacity + row], value, sum3);
            if (first_query + 4 < p.tokens && row < p.prefix + first_query + 4 + 1)
                sum4 = fma(scores[4 * Capacity + row], value, sum4);
            if (first_query + 5 < p.tokens && row < p.prefix + first_query + 5 + 1)
                sum5 = fma(scores[5 * Capacity + row], value, sum5);
            if (first_query + 6 < p.tokens && row < p.prefix + first_query + 6 + 1)
                sum6 = fma(scores[6 * Capacity + row], value, sum6);
            if (first_query + 7 < p.tokens && row < p.prefix + first_query + 7 + 1)
                sum7 = fma(scores[7 * Capacity + row], value, sum7);
        }
        if (first_query + 0 < p.tokens)
            out[((ulong(first_query) + 0) * p.qh + head) * 128 + col] = sum0;
        if (first_query + 1 < p.tokens)
            out[((ulong(first_query) + 1) * p.qh + head) * 128 + col] = sum1;
        if (first_query + 2 < p.tokens)
            out[((ulong(first_query) + 2) * p.qh + head) * 128 + col] = sum2;
        if (first_query + 3 < p.tokens)
            out[((ulong(first_query) + 3) * p.qh + head) * 128 + col] = sum3;
        if (first_query + 4 < p.tokens)
            out[((ulong(first_query) + 4) * p.qh + head) * 128 + col] = sum4;
        if (first_query + 5 < p.tokens)
            out[((ulong(first_query) + 5) * p.qh + head) * 128 + col] = sum5;
        if (first_query + 6 < p.tokens)
            out[((ulong(first_query) + 6) * p.qh + head) * 128 + col] = sum6;
        if (first_query + 7 < p.tokens)
            out[((ulong(first_query) + 7) * p.qh + head) * 128 + col] = sum7;
    }
}

#define PROMPT_ATTENTION(NAME, CAPACITY, MATRIX)                                                   \
    kernel void NAME(                                                                              \
        device const float *qkv [[buffer(0)]], device const float *rotated [[buffer(1)]],          \
        device const float *v [[buffer(2)]], device float *out [[buffer(3)]],                      \
        constant Prefill &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],           \
        uint tid [[thread_index_in_threadgroup]]) {                                                \
        threadgroup float keys[MATRIX ? 1 : 128 * 17], queries[8 * 129], scores[8 * CAPACITY];     \
        threadgroup float maximum[8], denominator[8];                                              \
        prompt_attention<CAPACITY, MATRIX>(qkv, rotated, v, out, p, group, tid, keys, queries,     \
                                           scores, maximum, denominator);                          \
    }
PROMPT_ATTENTION(plamo_prompt_attention128, 128, false)
PROMPT_ATTENTION(plamo_prompt_attention512, 512, false)
PROMPT_ATTENTION(plamo_prompt_attention_matrix256, 256, true)
PROMPT_ATTENTION(plamo_prompt_attention_matrix512, 512, true)
