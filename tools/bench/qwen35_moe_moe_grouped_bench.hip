// Correctness + perf bench for the WMMA grouped MoE FFN kernels against the
// production scalar grouped kernels (gathered activations through
// sorted_slots, swiglu store / weighted atomic accumulate).
//
// Build (Windows TheRock toolchain):
//   clang-cl.exe -x hip qwen35_moe_moe_grouped_bench.hip --offload-arch=gfx1151 \
//     -O3 /fp:fast /clang:-ffast-math -link <therock>/lib/amdhip64.lib
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    hipError_t err_ = (expr);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

namespace quant {
struct block_rocmfp4_fast {
  uint8_t qs[16];
  uint8_t e;
};
static_assert(sizeof(block_rocmfp4_fast) == 17, "17-byte block");
}  // namespace quant

__device__ __forceinline__ int Rocmfp4DecodeNibble(uint8_t q) {
  const int mag3 = q & 7;
  const int mag = mag3 <= 4 ? mag3 : (2 * mag3 - 4);
  return (q & 8) ? -mag : mag;
}
__device__ __forceinline__ float rocmfp4_ue4m3_to_fp32_half_finite(uint8_t x) {
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  if (exp == 0) return (float)man * (1.0f / 1024.0f);
  const uint32_t bits = ((uint32_t)exp + 119u) << 23 | ((uint32_t)man << 20);
  return __uint_as_float(bits);
}

// ---------------------------------------------------------------------------
// Shared WMMA helpers (copied from src/models/qwen35_moe/hip/kernels.hip)
// ---------------------------------------------------------------------------
__device__ __forceinline__ v16h WmmaHalf(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v16h LoadSwizzled(const short* p) {
  union { v16h f; uint4 q[2]; } bits;
  const unsigned row = threadIdx.x & 15, shift = (row ^ (row >> 2)) & 1;
  bits.q[0] = reinterpret_cast<const uint4*>(p)[shift];
  bits.q[1] = reinterpret_cast<const uint4*>(p)[shift ^ 1];
  return bits.f;
}
__device__ __forceinline__ void StoreSwizzled(short* p, uint4 lo, uint4 hi,
                                              unsigned row) {
  const unsigned shift = (row ^ (row >> 2)) & 1;
  reinterpret_cast<uint4*>(p)[shift] = lo;
  reinterpret_cast<uint4*>(p)[shift ^ 1] = hi;
}
struct HalfWeights {
  uint4 low, high;
};
__device__ __forceinline__ HalfWeights DecodeRocmFp4Sub(
    const uint8_t* blk, unsigned ks) {
  const unsigned* wp = reinterpret_cast<const unsigned*>(blk);
  const unsigned words[4] = {__ldg(wp), __ldg(wp + 1), __ldg(wp + 2), __ldg(wp + 3)};
  const unsigned shift = (ks & 1) * 4;
  const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk[16]);
  union { v16h f; uint4 u[2]; } result;
#pragma unroll
  for (unsigned wi = 0; wi < 4; ++wi) {
    const unsigned n = (words[wi] >> shift) & 0x0F0F0F0FU;
    const unsigned index = n & 0x07070707U;
    const unsigned sign_bits = (n & 0x08080808U) << 4;
    const unsigned hi =
        __builtin_amdgcn_perm(0x49484644U, 0x42403C00U, index) | sign_bits;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const _Float16 mag = __ushort_as_half(((hi >> (8 * j)) & 0xFFU) << 8);
      result.f[4 * wi + j] =
          static_cast<_Float16>(__fmul_rn(scale, __half2float(mag)));
    }
  }
  return {result.u[0], result.u[1]};
}

__global__ void MoeBuildTilesWmmaKernel(
    const uint32_t* __restrict__ counts, uint32_t* __restrict__ tile_expert,
    uint32_t* __restrict__ tile_tok_offset, uint32_t* __restrict__ num_tiles_out,
    uint32_t num_experts) {
  if (threadIdx.x == 0) {
    uint32_t tile_idx = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
      const uint32_t c = counts[e];
      for (uint32_t tok = 0; tok < c && tile_idx < 8192u; tok += 64) {
        tile_expert[tile_idx] = e;
        tile_tok_offset[tile_idx] = tok;
        tile_idx++;
      }
    }
    *num_tiles_out = tile_idx;
  }
}

enum class MoeWmmaMode { kGateUp, kDown };

template <int BM, int BN, int WM, int WN, MoeWmmaMode Mode>
__launch_bounds__(WM *WN * 32) __global__ void MoeGemmWmmaKernel(
    const void* __restrict__ gate_data, const void* __restrict__ up_data,
    uint64_t expert_stride_gate, uint64_t expert_stride_up,
    const float* __restrict__ x, const float* __restrict__ expert_weights,
    const uint32_t* __restrict__ counts, const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ sorted_slots,
    const uint32_t* __restrict__ tile_expert,
    const uint32_t* __restrict__ tile_tok_offset,
    const uint32_t* __restrict__ num_tiles_out, uint32_t num_experts_used,
    uint32_t ff, uint32_t H, float* __restrict__ out) {
  constexpr int BK = 4;
  constexpr int WaveSize = 32;
  constexpr int kThreads = WM * WN * WaveSize;
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int KS = 16;
  constexpr int PFA = (BM * BK + kThreads - 1) / kThreads;
  constexpr int PFB = (BN * BK + kThreads - 1) / kThreads;
  constexpr bool kGateUp = Mode == MoeWmmaMode::kGateUp;

  const int k_len = static_cast<int>(kGateUp ? H : ff);
  const int m_len = static_cast<int>(kGateUp ? ff : H);

  __shared__ short s_a[BK][RS][16][KS];
  __shared__ short s_a2[kGateUp ? BK : 1][RS][16][KS];
  __shared__ short s_b[2 * BK][TS][16][KS];

  const uint32_t tile_id = blockIdx.y;
  if (tile_id >= *num_tiles_out) return;
  const uint32_t expert_id = tile_expert[tile_id];
  const uint32_t tok_offset = tile_tok_offset[tile_id];
  const uint32_t count = counts[expert_id];
  if (tok_offset >= count) return;
  const uint32_t base_offset = offsets[expert_id];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / WaveSize;
  const int lane = tid % WaveSize;
  const int sl = lane & 15;
  const int wr = wave / WN;
  const int wt = wave % WN;

  const int r_block = static_cast<int>(blockIdx.x) * BM;
  const int ksteps = k_len / 16;

  v8f acc[WRS][WTS] = {};
  v8f acc2[WRS][WTS] = {};

  const std::size_t row_bytes =
      (static_cast<std::size_t>(k_len) / 32) * sizeof(quant::block_rocmfp4_fast);
  const std::uint8_t* w_row[PFA];
  const std::uint8_t* w_row2[PFA];
  const std::uint8_t* expert_base = static_cast<const std::uint8_t*>(gate_data) +
                                    static_cast<std::size_t>(expert_id) * expert_stride_gate;
  const std::uint8_t* expert_base2 = static_cast<const std::uint8_t*>(up_data) +
                                     static_cast<std::size_t>(expert_id) * expert_stride_up;
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int r = r_block + (((p * kThreads) + tid) / BK);
    const std::size_t off =
        static_cast<std::size_t>((r < m_len ? r : (m_len - 1))) * row_bytes;
    w_row[p] = expert_base + off;
    w_row2[p] = expert_base2 + off;
  }

  uint4 ra[PFA][2];
  uint4 ra2[PFA][2];
  float4 rb[PFB][4];

  const auto fetch = [&](int ks0) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BM * BK && kk < ksteps) {
        const auto decoded =
            DecodeRocmFp4Sub(w_row[p] + (kk / 2) * 17, static_cast<unsigned>(kk));
        ra[p][0] = decoded.low;
        ra[p][1] = decoded.high;
        if constexpr (kGateUp) {
          const auto decoded2 = DecodeRocmFp4Sub(w_row2[p] + (kk / 2) * 17,
                                                 static_cast<unsigned>(kk));
          ra2[p][0] = decoded2.low;
          ra2[p][1] = decoded2.high;
        }
      } else {
        ra[p][0] = make_uint4(0, 0, 0, 0);
        ra[p][1] = make_uint4(0, 0, 0, 0);
        if constexpr (kGateUp) {
          ra2[p][0] = make_uint4(0, 0, 0, 0);
          ra2[p][1] = make_uint4(0, 0, 0, 0);
        }
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BN * BK && kk < ksteps) {
        const int local = idx / BK;
        const uint32_t pos = tok_offset + static_cast<uint32_t>(local);
        const uint32_t slot_pos = pos < count ? pos : (count > 0 ? count - 1u : 0u);
        const uint32_t slot = __ldg(&sorted_slots[base_offset + slot_pos]);
        const std::size_t a_off =
            kGateUp ? static_cast<std::size_t>(slot / num_experts_used) * k_len
                    : static_cast<std::size_t>(slot) * k_len;
        const float* src = x + a_off + static_cast<std::size_t>(kk) * 16;
        rb[p][0] = reinterpret_cast<const float4*>(src)[0];
        rb[p][1] = reinterpret_cast<const float4*>(src)[1];
        rb[p][2] = reinterpret_cast<const float4*>(src)[2];
        rb[p][3] = reinterpret_cast<const float4*>(src)[3];
      } else {
        rb[p][0] = make_float4(0, 0, 0, 0);
        rb[p][1] = make_float4(0, 0, 0, 0);
        rb[p][2] = make_float4(0, 0, 0, 0);
        rb[p][3] = make_float4(0, 0, 0, 0);
      }
    }
  };
  const auto commit = [&]() {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BM * BK) {
        const int placed = (row ^ (ks & 3));
        StoreSwizzled(&s_a[ks][placed / 16][placed % 16][0], ra[p][0],
                      ra[p][1], row);
        if constexpr (kGateUp)
          StoreSwizzled(&s_a2[ks][placed / 16][placed % 16][0], ra2[p][0],
                        ra2[p][1], row);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BN * BK) {
        const int placed = (row ^ (ks & 3));
        union { v16h f; uint4 u[2]; } hiP, loP;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const float4 v = rb[p][q4];
          const float vs[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const _Float16 h = __float2half_rn(vs[e]);
            hiP.f[q4 * 4 + e] = h;
            loP.f[q4 * 4 + e] =
                __float2half_rn(__fadd_rn(vs[e], -__half2float(h)));
          }
        }
        StoreSwizzled(&s_b[ks][placed / 16][placed % 16][0], hiP.u[0],
                      hiP.u[1], row);
        StoreSwizzled(&s_b[BK + ks][placed / 16][placed % 16][0], loP.u[0],
                      loP.u[1], row);
      }
    }
  };
  fetch(0);
  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
    commit();
    __syncthreads();
    if (ks0 + BK < ksteps) fetch(ks0 + BK);
#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      v16h a[WRS], a2[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        a[i] = LoadSwizzled(&s_a[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
        if constexpr (kGateUp)
          a2[i] = LoadSwizzled(&s_a2[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const v16h b =
            LoadSwizzled(&s_b[ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
        const v16h bl =
            LoadSwizzled(&s_b[BK + ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
          acc[i][j] = WmmaHalf(bl, a[i], acc[i][j]);
          if constexpr (kGateUp) {
            acc2[i][j] = WmmaHalf(b, a2[i], acc2[i][j]);
            acc2[i][j] = WmmaHalf(bl, a2[i], acc2[i][j]);
          }
        }
      }
    }
    __syncthreads();
  }

  const int hi_lane = lane / 16;
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
      const int row = r_block + ((wr * WRS + i) * 16) + sl;
      const int t0 = ((wt * WTS + j) * 16);
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const int local = t0 + 2 * l + hi_lane;
        if (row < m_len && tok_offset + static_cast<uint32_t>(local) < count) {
          const uint32_t slot =
              __ldg(&sorted_slots[base_offset + tok_offset + local]);
          if constexpr (kGateUp) {
            const float sg = acc[i][j][l], su = acc2[i][j][l];
            const float silu_gate = sg / (1.0f + __expf(-sg));
            out[static_cast<std::size_t>(slot) * ff + row] = silu_gate * su;
          } else {
            const float weight = expert_weights[slot];
            if (weight != 0.0f) {
              const uint32_t t = slot / num_experts_used;
              atomicAdd(&out[static_cast<std::size_t>(t) * H + row],
                        acc[i][j][l] * weight);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Production scalar grouped kernels (verbatim), T_BATCH=16 stride-16 tiles.
// ---------------------------------------------------------------------------
template <uint32_t WavesPerBlock = 8, uint32_t T_BATCH = 16>
__launch_bounds__(WavesPerBlock * 32)
__global__ void MoeGatedSwigluFastGroupedKernel(
    const void* __restrict__ gate_data, const void* __restrict__ up_data,
    uint64_t expert_stride_gate, uint64_t expert_stride_up,
    const float* __restrict__ x, const uint32_t* __restrict__ counts,
    const uint32_t* __restrict__ offsets, const uint32_t* __restrict__ sorted_slots,
    const uint32_t* __restrict__ tile_expert,
    const uint32_t* __restrict__ tile_tok_offset,
    const uint32_t* __restrict__ num_tiles_out, uint32_t B,
    uint32_t num_experts, uint32_t num_experts_used, uint32_t expert_ff,
    uint32_t H, float* __restrict__ act_out) {
  const uint32_t tile_id = blockIdx.y;
  if (tile_id >= *num_tiles_out) return;
  const uint32_t expert_id = tile_expert[tile_id];
  const uint32_t tok_offset = tile_tok_offset[tile_id];
  const uint32_t count = counts[expert_id];
  if (tok_offset >= count) return;
  const uint32_t cur_batch = min(T_BATCH, count - tok_offset);
  const uint32_t base_offset = offsets[expert_id];

  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;

  const uint32_t num_blocks = H / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* expert_gate_base = static_cast<const uint8_t*>(gate_data) +
                                    (static_cast<size_t>(expert_id) * expert_stride_gate);
  const uint8_t* expert_up_base = static_cast<const uint8_t*>(up_data) +
                                  (static_cast<size_t>(expert_id) * expert_stride_up);
  const uint8_t* gate_row = (m < expert_ff) ? (expert_gate_base + (static_cast<size_t>(m) * row_bytes)) : nullptr;
  const uint8_t* up_row = (m < expert_ff) ? (expert_up_base + (static_cast<size_t>(m) * row_bytes)) : nullptr;
  const auto* gate_blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(gate_row);
  const auto* up_blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(up_row);

  __shared__ float s_x[T_BATCH][64];
  float sum_gate[T_BATCH] = {0.0f};
  float sum_up[T_BATCH] = {0.0f};
  const uint32_t half = lane_id >= 16 ? 1 : 0;
  const uint32_t j = lane_id & 15u;

  for (uint32_t b2 = 0; b2 < num_blocks; b2 += 2) {
    constexpr uint32_t total_floats = T_BATCH * 64;
    for (uint32_t idx = tid; idx < total_floats; idx += blockDim.x) {
      const uint32_t tb = idx >> 6u;
      const uint32_t k_idx = idx & 63u;
      if (tb < cur_batch && ((b2 * 32) + k_idx) < H) {
        const uint32_t slot = sorted_slots[base_offset + tok_offset + tb];
        const uint32_t t = slot / num_experts_used;
        s_x[tb][k_idx] = x[(static_cast<size_t>(t) * H) + (b2 * 32) + k_idx];
      } else {
        s_x[tb][k_idx] = 0.0f;
      }
    }
    __syncthreads();
    if (m < expert_ff) {
      const uint32_t b = b2 + half;
      if (b < num_blocks) {
        const auto& g_blk = gate_blocks[b];
        const float g_scale = rocmfp4_ue4m3_to_fp32_half_finite(g_blk.e);
        const uint8_t g_byte = g_blk.qs[j];
        const float gw0 = static_cast<float>(Rocmfp4DecodeNibble(g_byte & 0x0Fu)) * g_scale;
        const float gw1 = static_cast<float>(Rocmfp4DecodeNibble(g_byte >> 4u)) * g_scale;
        const auto& u_blk = up_blocks[b];
        const float u_scale = rocmfp4_ue4m3_to_fp32_half_finite(u_blk.e);
        const uint8_t u_byte = u_blk.qs[j];
        const float uw0 = static_cast<float>(Rocmfp4DecodeNibble(u_byte & 0x0Fu)) * u_scale;
        const float uw1 = static_cast<float>(Rocmfp4DecodeNibble(u_byte >> 4u)) * u_scale;
        const uint32_t offset_k = half * 32;
#pragma unroll
        for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
          const float x0 = s_x[tb][offset_k + j];
          const float x1 = s_x[tb][offset_k + j + 16];
          sum_gate[tb] += (gw0 * x0) + (gw1 * x1);
          sum_up[tb] += (uw0 * x0) + (uw1 * x1);
        }
      }
    }
    __syncthreads();
  }
  if (m < expert_ff) {
#pragma unroll
    for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
      if (tb < cur_batch) {
        float sg = sum_gate[tb], su = sum_up[tb];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          sg += __shfl_xor(sg, offset);
          su += __shfl_xor(su, offset);
        }
        if (lane_id == 0) {
          const uint32_t slot = sorted_slots[base_offset + tok_offset + tb];
          const float silu_gate = sg / (1.0f + __expf(-sg));
          act_out[(static_cast<size_t>(slot) * expert_ff) + m] = silu_gate * su;
        }
      }
    }
  }
}

template <uint32_t WavesPerBlock = 8, uint32_t T_BATCH = 16>
__launch_bounds__(WavesPerBlock * 32)
__global__ void MoeDownAccumFastGroupedKernel(
    const void* __restrict__ down_data, uint64_t expert_stride_down,
    const float* __restrict__ act_in, const float* __restrict__ expert_weights,
    const uint32_t* __restrict__ counts, const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ sorted_slots,
    const uint32_t* __restrict__ tile_expert,
    const uint32_t* __restrict__ tile_tok_offset,
    const uint32_t* __restrict__ num_tiles_out, uint32_t B,
    uint32_t num_experts, uint32_t num_experts_used, uint32_t expert_ff,
    uint32_t H, float* __restrict__ routed_accum) {
  const uint32_t tile_id = blockIdx.y;
  if (tile_id >= *num_tiles_out) return;
  const uint32_t expert_id = tile_expert[tile_id];
  const uint32_t tok_offset = tile_tok_offset[tile_id];
  const uint32_t count = counts[expert_id];
  if (tok_offset >= count) return;
  const uint32_t cur_batch = min(T_BATCH, count - tok_offset);
  const uint32_t base_offset = offsets[expert_id];

  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;

  const uint32_t num_blocks = expert_ff / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* expert_down_base = static_cast<const uint8_t*>(down_data) +
                                    (static_cast<size_t>(expert_id) * expert_stride_down);
  const uint8_t* row_base = (m < H) ? (expert_down_base + (static_cast<size_t>(m) * row_bytes)) : nullptr;
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(row_base);

  __shared__ float s_act[T_BATCH][64];
  float sum[T_BATCH] = {0.0f};
  const uint32_t half = lane_id >= 16 ? 1 : 0;
  const uint32_t j = lane_id & 15u;

  for (uint32_t b2 = 0; b2 < num_blocks; b2 += 2) {
    constexpr uint32_t total_floats = T_BATCH * 64;
    for (uint32_t idx = tid; idx < total_floats; idx += blockDim.x) {
      const uint32_t tb = idx >> 6u;
      const uint32_t k_idx = idx & 63u;
      if (tb < cur_batch && ((b2 * 32) + k_idx) < expert_ff) {
        const uint32_t slot = sorted_slots[base_offset + tok_offset + tb];
        s_act[tb][k_idx] = act_in[(static_cast<size_t>(slot) * expert_ff) + (b2 * 32) + k_idx];
      } else {
        s_act[tb][k_idx] = 0.0f;
      }
    }
    __syncthreads();
    if (m < H) {
      const uint32_t b = b2 + half;
      if (b < num_blocks) {
        const auto& blk = blocks[b];
        const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
        const uint8_t byte_val = blk.qs[j];
        const float w0 = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * scale;
        const float w1 = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * scale;
        const uint32_t offset_k = half * 32;
#pragma unroll
        for (uint32_t tb = 0; tb < T_BATCH; ++tb)
          sum[tb] += (w0 * s_act[tb][offset_k + j]) + (w1 * s_act[tb][offset_k + j + 16]);
      }
    }
    __syncthreads();
  }
  if (m < H) {
#pragma unroll
    for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
      if (tb < cur_batch) {
        float s = sum[tb];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_xor(s, offset);
        if (lane_id == 0) {
          const uint32_t slot = sorted_slots[base_offset + tok_offset + tb];
          const float weight = expert_weights[slot];
          if (weight != 0.0f) {
            const uint32_t t = slot / num_experts_used;
            atomicAdd(&routed_accum[(static_cast<size_t>(t) * H) + m], s * weight);
          }
        }
      }
    }
  }
}

namespace {

float frand(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return (static_cast<float>(s >> 8) / 8388608.0f) - 1.0f;
}

struct Setup {
  uint32_t experts, n_eu, B, ff, H;
  std::vector<uint8_t> gate, up, down;
  std::vector<float> x, weights;
  std::vector<uint32_t> counts, offsets, sorted;
  std::vector<uint32_t> tiles16_e, tiles16_o, ntiles16;
  std::vector<uint32_t> tiles64_e, tiles64_o, ntiles64;
  size_t stride_w;  // bytes per expert matrix
};

Setup make_setup(uint32_t experts, uint32_t n_eu, uint32_t B, uint32_t ff,
                 uint32_t H) {
  Setup s{experts, n_eu, B, ff, H, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
  uint32_t seed = 1234u ^ (B * 2654435761u);
  s.stride_w = (size_t)(H / 32) * 17 * ff;  // gate/up rows K=H
  s.gate.resize((size_t)experts * s.stride_w);
  s.up.resize(s.gate.size());
  s.down.resize((size_t)experts * (ff / 32) * 17 * H);
  for (auto* v : {&s.gate, &s.up})
    for (size_t i = 0; i < v->size(); ++i) {
      seed = seed * 1664525u + 1013904223u;
      (*v)[i] = (uint8_t)(seed >> 16);
      if (i % 17 == 16) (*v)[i] &= 0x3C;  // moderate ue4m3 scales
    }
  for (size_t i = 0; i < s.down.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    s.down[i] = (uint8_t)(seed >> 16);
    if (i % 17 == 16) s.down[i] &= 0x3C;
  }
  s.x.resize((size_t)B * H);
  for (auto& v : s.x) v = frand(seed) * 0.5f;
  const uint32_t slots = B * n_eu;
  s.weights.resize(slots);
  for (auto& v : s.weights) v = (frand(seed) * 0.5f + 0.5f) / n_eu;
  s.counts.assign(experts, 0);
  std::vector<uint32_t> eid(slots);
  for (uint32_t sl = 0; sl < slots; ++sl) {
    seed = seed * 1664525u + 1013904223u;
    const uint32_t e = (seed >> 8) % experts;
    eid[sl] = e;
    s.counts[e]++;
  }
  s.offsets.resize(experts);
  uint32_t acc = 0;
  for (uint32_t e = 0; e < experts; ++e) { s.offsets[e] = acc; acc += s.counts[e]; }
  s.sorted.resize(slots);
  std::vector<uint32_t> cursor(experts, 0);
  for (uint32_t sl = 0; sl < slots; ++sl) {
    s.sorted[s.offsets[eid[sl]] + cursor[eid[sl]]++] = sl;
  }
  s.ntiles16 = {0}; s.ntiles64 = {0};
  for (uint32_t e = 0; e < experts; ++e) {
    for (uint32_t t = 0; t < s.counts[e]; t += 16) { s.tiles16_e.push_back(e); s.tiles16_o.push_back(t); s.ntiles16[0]++; }
    for (uint32_t t = 0; t < s.counts[e]; t += 64) { s.tiles64_e.push_back(e); s.tiles64_o.push_back(t); s.ntiles64[0]++; }
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t B = 2658, experts = 256, n_eu = 8, ff = 1024, H = 3072;
  int iters = 5;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-b") && i + 1 < argc) B = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-e") && i + 1 < argc) experts = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
  }
  const Setup s = make_setup(experts, n_eu, B, ff, H);
  const uint32_t slots = B * n_eu;
  printf("experts=%u n_eu=%u B=%u ff=%u H=%u slots=%u tiles16=%u tiles64=%u\n",
         experts, n_eu, B, ff, H, slots, s.ntiles16[0], s.ntiles64[0]);

  void *d_gate, *d_up, *d_down;
  float *d_x, *d_w;
  uint32_t *d_counts, *d_offsets, *d_sorted, *d_t16e, *d_t16o, *d_nt16,
      *d_t64e, *d_t64o, *d_nt64;
  float *d_act_ref, *d_act_new, *d_acc_ref, *d_acc_new;
  HIP_OK(hipMalloc(&d_gate, s.gate.size()));
  HIP_OK(hipMalloc(&d_up, s.up.size()));
  HIP_OK(hipMalloc(&d_down, s.down.size()));
  HIP_OK(hipMalloc(&d_x, s.x.size() * 4));
  HIP_OK(hipMalloc(&d_w, s.weights.size() * 4));
  HIP_OK(hipMalloc(&d_counts, experts * 4));
  HIP_OK(hipMalloc(&d_offsets, experts * 4));
  HIP_OK(hipMalloc(&d_sorted, slots * 4));
  HIP_OK(hipMalloc(&d_t16e, 8192 * 4));
  HIP_OK(hipMalloc(&d_t16o, 8192 * 4));
  HIP_OK(hipMalloc(&d_nt16, 4));
  HIP_OK(hipMalloc(&d_t64e, 8192 * 4));
  HIP_OK(hipMalloc(&d_t64o, 8192 * 4));
  HIP_OK(hipMalloc(&d_nt64, 4));
  HIP_OK(hipMalloc(&d_act_ref, (size_t)slots * ff * 4));
  HIP_OK(hipMalloc(&d_act_new, (size_t)slots * ff * 4));
  HIP_OK(hipMalloc(&d_acc_ref, (size_t)B * H * 4));
  HIP_OK(hipMalloc(&d_acc_new, (size_t)B * H * 4));
  HIP_OK(hipMemcpy(d_gate, s.gate.data(), s.gate.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_up, s.up.data(), s.up.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_down, s.down.data(), s.down.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_x, s.x.data(), s.x.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_w, s.weights.data(), s.weights.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_counts, s.counts.data(), experts * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_offsets, s.offsets.data(), experts * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_sorted, s.sorted.data(), slots * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_t16e, s.tiles16_e.data(), s.tiles16_e.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_t16o, s.tiles16_o.data(), s.tiles16_o.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_nt16, s.ntiles16.data(), 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_t64e, s.tiles64_e.data(), s.tiles64_e.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_t64o, s.tiles64_o.data(), s.tiles64_o.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_nt64, s.ntiles64.data(), 4, hipMemcpyHostToDevice));

  auto timeit = [&](const char* name, auto launch) -> double {
    launch();
    if (hipGetLastError() != hipSuccess) { printf("%s LAUNCH FAIL\n", name); return -1; }
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t a, b;
    HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
    launch(); HIP_OK(hipDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
      HIP_OK(hipEventRecord(a));
      launch();
      HIP_OK(hipEventRecord(b));
      HIP_OK(hipEventSynchronize(b));
      float ms; HIP_OK(hipEventElapsedTime(&ms, a, b));
      best = std::min(best, ms);
    }
    HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
    return best;
  };
  auto compare = [&](const char* name, float* ref, float* nw, size_t n) {
    std::vector<float> hr(n), hn(n);
    HIP_OK(hipMemcpy(hr.data(), ref, n * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(hn.data(), nw, n * 4, hipMemcpyDeviceToHost));
    double worst = 0.0, mag = 0.0, rel = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double d = std::fabs((double)hr[i] - hn[i]);
      if (d > worst) worst = d;
      const double rr = d / std::max(1.0, std::fabs((double)hr[i]));
      if (rr > rel) rel = rr;
      mag = std::max(mag, std::fabs((double)hr[i]));
    }
    printf("  %-22s max|diff| %.3e (ref mag %.3g) max rel %.3e\n", name, worst, mag, rel);
  };

  const size_t stride_gate = s.stride_w;
  const size_t stride_down = (size_t)(ff / 32) * 17 * H;

  // ---- gate/up -------------------------------------------------------------
  const uint32_t mt16 = std::min(8192u, (slots + 15u) / 16u + experts);
  const uint32_t mt64 = std::min(8192u, (slots + 63u) / 64u + experts);
  HIP_OK(hipMemset(d_act_ref, 0, (size_t)slots * ff * 4));
  const double t_scalar_gu = timeit("scalar gu", [&]() {
    MoeGatedSwigluFastGroupedKernel<8, 16><<<dim3((ff + 7) / 8, mt16), 256>>>(
        d_gate, d_up, stride_gate, stride_gate, d_x, d_counts, d_offsets,
        d_sorted, d_t16e, d_t16o, d_nt16, B, experts, n_eu, ff, H, d_act_ref);
  });
  HIP_OK(hipMemset(d_act_new, 0, (size_t)slots * ff * 4));
  const double t_wmma_gu = timeit("wmma gu", [&]() {
    MoeBuildTilesWmmaKernel<<<1, 32>>>(d_counts, d_t64e, d_t64o, d_nt64, experts);
    MoeGemmWmmaKernel<64, 64, 2, 2, MoeWmmaMode::kGateUp>
        <<<dim3((ff + 63) / 64, mt64), 128>>>(d_gate, d_up, stride_gate,
                                              stride_gate, d_x, nullptr,
                                              d_counts, d_offsets, d_sorted,
                                              d_t64e, d_t64o, d_nt64, n_eu, ff,
                                              H, d_act_new);
  });
  printf("gate/up: scalar %.3f ms | wmma %.3f ms (%.2fx)\n", t_scalar_gu,
         t_wmma_gu, t_scalar_gu / t_wmma_gu);
  compare("act_out", d_act_ref, d_act_new, (size_t)slots * ff);

  // Verify device-built tile table against host expectation.
  {
    std::vector<uint32_t> he(s.tiles64_e.size()), ho(s.tiles64_o.size());
    uint32_t hn = 0;
    HIP_OK(hipMemcpy(&hn, d_nt64, 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(he.data(), d_t64e, he.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(ho.data(), d_t64o, ho.size() * 4, hipMemcpyDeviceToHost));
    bool ok = hn == s.ntiles64[0];
    for (uint32_t i = 0; ok && i < hn; ++i)
      ok = he[i] == s.tiles64_e[i] && ho[i] == s.tiles64_o[i];
    printf("tile64 table: %s (%u tiles)\n", ok ? "OK" : "MISMATCH", hn);
  }

  // ---- down ----------------------------------------------------------------
  HIP_OK(hipMemset(d_acc_ref, 0, (size_t)B * H * 4));
  const double t_scalar_dn = timeit("scalar dn", [&]() {
    MoeDownAccumFastGroupedKernel<8, 16><<<dim3((H + 7) / 8, mt16), 256>>>(
        d_down, stride_down, d_act_ref, d_w, d_counts, d_offsets, d_sorted,
        d_t16e, d_t16o, d_nt16, B, experts, n_eu, ff, H, d_acc_ref);
  });
  HIP_OK(hipMemset(d_acc_new, 0, (size_t)B * H * 4));
  const double t_wmma_dn = timeit("wmma dn", [&]() {
    MoeGemmWmmaKernel<128, 64, 2, 2, MoeWmmaMode::kDown>
        <<<dim3((H + 127) / 128, mt64), 128>>>(d_down, d_down, stride_down, 0,
                                               d_act_new, d_w, d_counts,
                                               d_offsets, d_sorted, d_t64e,
                                               d_t64o, d_nt64, n_eu, ff, H,
                                               d_acc_new);
  });
  printf("down:     scalar %.3f ms | wmma %.3f ms (%.2fx)\n", t_scalar_dn,
         t_wmma_dn, t_scalar_dn / t_wmma_dn);
  compare("routed_accum", d_acc_ref, d_acc_new, (size_t)B * H);
  return 0;
}
