#pragma once

#include "rocmfp4_hip_scale.cuh"

#include <cstdint>
#include <cstring>

#ifndef GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD
#define GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD 1
#endif

static __device__ __forceinline__ int rocmfp4_get_qs_i32(const void * x, const int & i32) {
#if defined(GGML_USE_HIP) && GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD
    return *((const int *) ((const uint8_t *) x + 4*i32));
#else
    const uint8_t * x8 = (const uint8_t *) x;

    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    return x32;
#endif
}

// AMD-specific fast path for expanding eight packed ROCmFP4 nibbles into two
// int32 DP4A operands. This encodes the Codebook10 table directly as four
// 32-bit constants:
//   [0, 1, 2, 3], [4, 6, 8, 10], [0, -1, -2, -3], [-4, -6, -8, -10]
// Avoiding the table pointer keeps the ROCm/HIP MMVQ/MMQ hot path fully local
// to this format.
static __device__ __forceinline__ int2 rocmfp4_get_int_from_codebook_16(const int & q4, const int8_t * fallback_table) {
#if defined(__AMDGCN__) || defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    const uint32_t v_even_low  = __builtin_amdgcn_perm(values1, values0, q_even & 0x07070707u);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values1, values0, q_odd  & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values3, values2, q_even & 0x07070707u);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values3, values2, q_odd  & 0x07070707u);

    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    const uint32_t mask_odd  = 0x03020100u | ((q_odd  & 0x08080808u) >> 1);

    return make_int2(
        __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
        __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
#else
    int2 res;
    if (fallback_table) {
        uint8_t b0 = static_cast<uint8_t>(fallback_table[(q4 >> 0) & 0xf]);
        uint8_t b1 = static_cast<uint8_t>(fallback_table[(q4 >> 8) & 0xf]);
        uint8_t b2 = static_cast<uint8_t>(fallback_table[(q4 >> 16) & 0xf]);
        uint8_t b3 = static_cast<uint8_t>(fallback_table[(q4 >> 24) & 0xf]);
        res.x = static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        uint8_t b4 = static_cast<uint8_t>(fallback_table[(q4 >> 4) & 0xf]);
        uint8_t b5 = static_cast<uint8_t>(fallback_table[(q4 >> 12) & 0xf]);
        uint8_t b6 = static_cast<uint8_t>(fallback_table[(q4 >> 20) & 0xf]);
        uint8_t b7 = static_cast<uint8_t>(fallback_table[(q4 >> 28) & 0xf]);
        res.y = static_cast<int>(b4 | (b5 << 8) | (b6 << 16) | (b7 << 24));
    } else {
        res.x = 0;
        res.y = 0;
    }
    return res;
#endif
}

static __device__ __forceinline__ int rocmfp4_get_low_int_from_codebook_16(const int & q4, const int8_t * fallback_table) {
#if defined(__AMDGCN__) || defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q = q4;

    const uint32_t v_low  = __builtin_amdgcn_perm(values1, values0, q & 0x07070707u);
    const uint32_t v_high = __builtin_amdgcn_perm(values3, values2, q & 0x07070707u);
    const uint32_t mask   = 0x03020100u | ((q & 0x08080808u) >> 1);

    return __builtin_amdgcn_perm(v_high, v_low, mask);
#else
    if (fallback_table) {
        uint8_t b0 = static_cast<uint8_t>(fallback_table[(q4 >> 0) & 0xf]);
        uint8_t b1 = static_cast<uint8_t>(fallback_table[(q4 >> 8) & 0xf]);
        uint8_t b2 = static_cast<uint8_t>(fallback_table[(q4 >> 16) & 0xf]);
        uint8_t b3 = static_cast<uint8_t>(fallback_table[(q4 >> 24) & 0xf]);
        return static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
    return 0;
#endif
}
