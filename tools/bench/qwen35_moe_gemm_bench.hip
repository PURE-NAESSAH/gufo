// Standalone microbenchmark + numerics harness for the Ornith-1.5-122B
// (qwen35moe) dense Q4_0_ROCMFP4_FAST projection GEMM on gfx1151.
//
// Self-contained; compile with the TheRock clang-cl HIP toolchain:
//
//   clang-cl.exe -x hip qwen35_moe_gemm_bench.hip --offload-arch=gfx1151 \
//     -O3 /fp:fast /clang:-ffast-math -link <therock>/lib/amdhip64.lib
//
//   ./qwen35_moe_gemm_bench [-b 2658] [-i 5]
//
// Production kernel: one warp per output row m of A[M,K], a token batch of
// T_BATCH staged in LDS, scalar nibble decode + float FMA accumulation.
// This harness sweeps T_BATCH to expose the weight-reread vs register
// pressure trade at prefill batch sizes.
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

namespace quant {
struct block_rocmfp4_fast {
  uint8_t qs[16];
  uint8_t e;
};
static_assert(sizeof(block_rocmfp4_fast) == 17, "17-byte block");
}  // namespace quant

__device__ __forceinline__ int8_t Rocmfp4DecodeNibble(uint8_t q) {
  const int mag3 = q & 0x07u;
  const int mag = (mag3 <= 4) ? mag3 : (2 * mag3 - 4);
  return (q & 0x08u) ? -mag : mag;
}

__device__ __forceinline__ float rocmfp4_ue4m3_to_fp32_half_finite(uint8_t x) {
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  if (exp == 0) return (float)man * (1.0f / 1024.0f);
  const uint32_t bits = ((uint32_t)exp + 119u) << 23 | ((uint32_t)man << 20);
  return __uint_as_float(bits);
}

// ---------------------------------------------------------------------------
// Production kernel, copied from src/models/qwen35_moe/hip/kernels.hip.
// ---------------------------------------------------------------------------
template <uint32_t WavesPerBlock, uint32_t T_BATCH = 32>
__launch_bounds__(WavesPerBlock * 32)
__global__ void GemmLdsRocmFp4FastKernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t_base = blockIdx.y * T_BATCH;
  if (t_base >= B) return;
  const uint32_t cur_batch = min(T_BATCH, B - t_base);

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* base = m < M ? (static_cast<const uint8_t*>(A_data) +
                                (static_cast<size_t>(m) * row_bytes)) : nullptr;
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(base);

  __shared__ float s_x[T_BATCH][64];

  float sum[T_BATCH] = {0.0f};

  const uint32_t half = lane_id >= 16 ? 1 : 0;
  const uint32_t j = lane_id & 15u;

  for (uint32_t b2 = 0; b2 < num_blocks; b2 += 2) {
    constexpr uint32_t total_floats = T_BATCH * 64;
    for (uint32_t idx = tid; idx < total_floats; idx += blockDim.x) {
      const uint32_t tb = idx >> 6u;
      const uint32_t k_idx = idx & 63u;
      if (tb < cur_batch && ((b2 * 32) + k_idx) < K) {
        s_x[tb][k_idx] = x[(static_cast<size_t>(t_base + tb) * K) + (b2 * 32) + k_idx];
      } else {
        s_x[tb][k_idx] = 0.0f;
      }
    }
    __syncthreads();

    if (m < M) {
      const uint32_t b = b2 + half;
      if (b < num_blocks) {
        const auto& blk = blocks[b];
        const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
        const uint8_t byte_val = blk.qs[j];
        const float w0 = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * scale;
        const float w1 = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * scale;

        const uint32_t offset_k = half * 32;
#pragma unroll
        for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
          sum[tb] += (w0 * s_x[tb][offset_k + j]) + (w1 * s_x[tb][offset_k + j + 16]);
        }
      }
    }
    __syncthreads();
  }

  if (m < M) {
#pragma unroll
    for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
      if (tb < cur_batch) {
        float s = sum[tb];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          s += __shfl_xor(s, offset);
        }
        if (lane_id == 0) {
          y[(static_cast<size_t>(t_base + tb) * M) + m] = s;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Variant S: identical to production but the launch grid swaps roles so
// blockIdx.x selects the token batch and blockIdx.y the output rows. Block
// scheduling then keeps batch slices hot instead of row groups.
// ---------------------------------------------------------------------------
template <uint32_t WavesPerBlock, uint32_t T_BATCH = 32>
__launch_bounds__(WavesPerBlock * 32)
__global__ void GemmLdsRocmFp4SwapKernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.y * WavesPerBlock) + warp_id;
  const uint32_t t_base = blockIdx.x * T_BATCH;
  if (t_base >= B) return;
  const uint32_t cur_batch = min(T_BATCH, B - t_base);

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* base = m < M ? (static_cast<const uint8_t*>(A_data) +
                                (static_cast<size_t>(m) * row_bytes)) : nullptr;
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(base);

  __shared__ float s_x[T_BATCH][64];

  float sum[T_BATCH] = {0.0f};

  const uint32_t half = lane_id >= 16 ? 1 : 0;
  const uint32_t j = lane_id & 15u;

  for (uint32_t b2 = 0; b2 < num_blocks; b2 += 2) {
    constexpr uint32_t total_floats = T_BATCH * 64;
    for (uint32_t idx = tid; idx < total_floats; idx += blockDim.x) {
      const uint32_t tb = idx >> 6u;
      const uint32_t k_idx = idx & 63u;
      if (tb < cur_batch && ((b2 * 32) + k_idx) < K) {
        s_x[tb][k_idx] = x[(static_cast<size_t>(t_base + tb) * K) + (b2 * 32) + k_idx];
      } else {
        s_x[tb][k_idx] = 0.0f;
      }
    }
    __syncthreads();

    if (m < M) {
      const uint32_t b = b2 + half;
      if (b < num_blocks) {
        const auto& blk = blocks[b];
        const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
        const uint8_t byte_val = blk.qs[j];
        const float w0 = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * scale;
        const float w1 = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * scale;

        const uint32_t offset_k = half * 32;
#pragma unroll
        for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
          sum[tb] += (w0 * s_x[tb][offset_k + j]) + (w1 * s_x[tb][offset_k + j + 16]);
        }
      }
    }
    __syncthreads();
  }

  if (m < M) {
#pragma unroll
    for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
      if (tb < cur_batch) {
        float s = sum[tb];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          s += __shfl_xor(s, offset);
        }
        if (lane_id == 0) {
          y[(static_cast<size_t>(t_base + tb) * M) + m] = s;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Variant D (dual-row): each warp owns two consecutive output rows so every
// staged activation value loaded from LDS feeds four FMAs instead of two,
// halving LDS traffic per FLOP and interleaving two weight streams.
// ---------------------------------------------------------------------------
template <uint32_t WavesPerBlock, uint32_t T_BATCH = 32, bool SwapGrid = true>
__launch_bounds__(WavesPerBlock * 32)
__global__ void GemmLdsRocmFp4DualRowKernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m0 = (blockIdx.y * WavesPerBlock + warp_id) * 2u;
  const uint32_t m1 = m0 + 1u;
  const uint32_t t_base = SwapGrid ? blockIdx.x * T_BATCH : blockIdx.y * T_BATCH;
  if (t_base >= B) return;
  const uint32_t cur_batch = min(T_BATCH, B - t_base);

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* base_a = m0 < M ? (static_cast<const uint8_t*>(A_data) +
                                    (static_cast<size_t>(m0) * row_bytes)) : nullptr;
  const uint8_t* base_b = m1 < M ? (static_cast<const uint8_t*>(A_data) +
                                    (static_cast<size_t>(m1) * row_bytes)) : nullptr;
  const auto* blocks_a = reinterpret_cast<const quant::block_rocmfp4_fast*>(base_a);
  const auto* blocks_b = reinterpret_cast<const quant::block_rocmfp4_fast*>(base_b);

  __shared__ float s_x[T_BATCH][64];

  float sum_a[T_BATCH] = {0.0f};
  float sum_b[T_BATCH] = {0.0f};

  const uint32_t half = lane_id >= 16 ? 1 : 0;
  const uint32_t j = lane_id & 15u;

  for (uint32_t b2 = 0; b2 < num_blocks; b2 += 2) {
    constexpr uint32_t total_floats = T_BATCH * 64;
    for (uint32_t idx = tid; idx < total_floats; idx += blockDim.x) {
      const uint32_t tb = idx >> 6u;
      const uint32_t k_idx = idx & 63u;
      if (tb < cur_batch && ((b2 * 32) + k_idx) < K) {
        s_x[tb][k_idx] = x[(static_cast<size_t>(t_base + tb) * K) + (b2 * 32) + k_idx];
      } else {
        s_x[tb][k_idx] = 0.0f;
      }
    }
    __syncthreads();

    const uint32_t b = b2 + half;
    if (b < num_blocks) {
      const auto& blk_a = blocks_a[b];
      const float scale_a = rocmfp4_ue4m3_to_fp32_half_finite(blk_a.e);
      const uint8_t byte_a = blk_a.qs[j];
      const float w0a = static_cast<float>(Rocmfp4DecodeNibble(byte_a & 0x0Fu)) * scale_a;
      const float w1a = static_cast<float>(Rocmfp4DecodeNibble(byte_a >> 4u)) * scale_a;

      float w0b = 0.0f, w1b = 0.0f;
      if (m1 < M) {
        const auto& blk_b = blocks_b[b];
        const float scale_b = rocmfp4_ue4m3_to_fp32_half_finite(blk_b.e);
        const uint8_t byte_b = blk_b.qs[j];
        w0b = static_cast<float>(Rocmfp4DecodeNibble(byte_b & 0x0Fu)) * scale_b;
        w1b = static_cast<float>(Rocmfp4DecodeNibble(byte_b >> 4u)) * scale_b;
      }

      const uint32_t offset_k = half * 32;
#pragma unroll
      for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
        const float x0 = s_x[tb][offset_k + j];
        const float x1 = s_x[tb][offset_k + j + 16];
        sum_a[tb] += (w0a * x0) + (w1a * x1);
        sum_b[tb] += (w0b * x0) + (w1b * x1);
      }
    }
    __syncthreads();
  }

  if (m0 < M) {
#pragma unroll
    for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
      if (tb < cur_batch) {
        float s = sum_a[tb];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_xor(s, offset);
        if (lane_id == 0) y[(static_cast<size_t>(t_base + tb) * M) + m0] = s;
      }
    }
  }
  if (m1 < M) {
#pragma unroll
    for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
      if (tb < cur_batch) {
        float s = sum_b[tb];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_xor(s, offset);
        if (lane_id == 0) y[(static_cast<size_t>(t_base + tb) * M) + m1] = s;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Variant P (pipelined): production dataflow with double-buffered LDS x
// stages. The next batch slice is loaded into registers before the barrier
// so its global latency overlaps the current slice's dot-product work.
// ---------------------------------------------------------------------------
template <uint32_t WavesPerBlock, uint32_t T_BATCH = 32>
__launch_bounds__(WavesPerBlock * 32)
__global__ void GemmLdsRocmFp4PipeKernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t_base = blockIdx.y * T_BATCH;
  if (t_base >= B) return;
  const uint32_t cur_batch = min(T_BATCH, B - t_base);

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* base = m < M ? (static_cast<const uint8_t*>(A_data) +
                                (static_cast<size_t>(m) * row_bytes)) : nullptr;
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(base);

  __shared__ float s_x[2][T_BATCH][64];

  float sum[T_BATCH] = {0.0f};

  const uint32_t half = lane_id >= 16 ? 1 : 0;
  const uint32_t j = lane_id & 15u;

  constexpr uint32_t total_floats = T_BATCH * 64;
  static_assert(total_floats % (WavesPerBlock * 32u) == 0, "even staging");
  constexpr uint32_t kPerThread = total_floats / (WavesPerBlock * 32u);
  // Thread-local staging indices: idx = tid + p*blockDim.
  uint32_t st_idx[kPerThread];
#pragma unroll
  for (uint32_t p = 0; p < kPerThread; ++p) st_idx[p] = tid + p * WavesPerBlock * 32u;

  // Prologue: stage slice 0.
  {
    float v[kPerThread];
#pragma unroll
    for (uint32_t p = 0; p < kPerThread; ++p) {
      const uint32_t tb = st_idx[p] >> 6u;
      const uint32_t k_idx = st_idx[p] & 63u;
      v[p] = (tb < cur_batch && k_idx < K)
                 ? x[(static_cast<size_t>(t_base + tb) * K) + k_idx]
                 : 0.0f;
    }
#pragma unroll
    for (uint32_t p = 0; p < kPerThread; ++p)
      s_x[0][st_idx[p] >> 6u][st_idx[p] & 63u] = v[p];
  }

  uint32_t buf = 0;
  for (uint32_t b2 = 0, kk = 0; kk < num_blocks; b2 += 2, kk += 1, buf ^= 1u) {
    // Prefetch the next slice into registers.
    float nxt[kPerThread];
    const uint32_t nk = kk + 1;
    if (nk < num_blocks) {
#pragma unroll
      for (uint32_t p = 0; p < kPerThread; ++p) {
        const uint32_t tb = st_idx[p] >> 6u;
        const uint32_t k_idx = st_idx[p] & 63u;
        nxt[p] = (tb < cur_batch && ((nk * 32) + k_idx) < K)
                     ? x[(static_cast<size_t>(t_base + tb) * K) + (nk * 32) + k_idx]
                     : 0.0f;
      }
    }

    __syncthreads();  // previous stage visible; free other buffer
    if (nk < num_blocks) {
#pragma unroll
      for (uint32_t p = 0; p < kPerThread; ++p)
        s_x[buf ^ 1u][st_idx[p] >> 6u][st_idx[p] & 63u] = nxt[p];
    }
    __syncthreads();  // this slice ready

    if (m < M) {
      const uint32_t b = b2 + half;
      if (b < num_blocks) {
        const auto& blk = blocks[b];
        const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
        const uint8_t byte_val = blk.qs[j];
        const float w0 = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * scale;
        const float w1 = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * scale;

        const uint32_t offset_k = half * 32;
#pragma unroll
        for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
          sum[tb] += (w0 * s_x[buf][tb][offset_k + j]) + (w1 * s_x[buf][tb][offset_k + j + 16]);
        }
      }
    }
  }

  if (m < M) {
#pragma unroll
    for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
      if (tb < cur_batch) {
        float s = sum[tb];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          s += __shfl_xor(s, offset);
        }
        if (lane_id == 0) {
          y[(static_cast<size_t>(t_base + tb) * M) + m] = s;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Variant R (row-reuse token-lane): each warp owns R_ROWS output rows and a
// T_BATCH-token tile; a lane owns one token. Activations are staged once in
// LDS (row-major, +4-float padding for conflict-free b128 row reads) and
// every staged value feeds R_ROWS FMAs, cutting LDS bytes per FLOP by R_ROWS.
// Weights are dequantized per warp into an LDS slab read as uniform
// broadcasts, so the whole dataflow stays pure fp32 FFMA (bit-compatible
// accumulation order per (row, token)).
// ---------------------------------------------------------------------------
template <uint32_t WavesPerBlock, uint32_t T_BATCH = 32, uint32_t R_ROWS = 8>
__launch_bounds__(WavesPerBlock * 32)
__global__ void GemmRocmFp4RowReuseKernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B) {
  constexpr uint32_t XP = 64 + 4;  // padded row stride, keeps 16B alignment
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t row0 = (blockIdx.x * WavesPerBlock + warp_id) * R_ROWS;
  const uint32_t t_base = blockIdx.y * T_BATCH;
  if (t_base >= B) return;
  const uint32_t cur_batch = min(T_BATCH, B - t_base);

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const bool warp_live = row0 < M;

  __shared__ float s_xt[T_BATCH][XP];                      // activations [token][k64]
  __shared__ float s_w[WavesPerBlock][R_ROWS][64];         // per-warp weight slab

  float acc[R_ROWS] = {0.0f};

  const uint32_t half = lane_id >= 16 ? 1 : 0;
  const uint32_t j = lane_id & 15u;

  for (uint32_t q = 0; q < num_blocks; q += 2) {
    // Stage 64 k-values for all T_BATCH tokens (coalesced along k).
    constexpr uint32_t total_floats = T_BATCH * 64;
    for (uint32_t idx = tid; idx < total_floats; idx += WavesPerBlock * 32u) {
      const uint32_t tb = idx >> 6u;
      const uint32_t kk = idx & 63u;
      s_xt[tb][kk] = (tb < cur_batch && ((q * 32) + kk) < K)
                         ? x[(static_cast<size_t>(t_base + tb) * K) + (q * 32) + kk]
                         : 0.0f;
    }
    __syncthreads();

    if (warp_live) {
      // Dequantize this warp's R_ROWS weight rows for the 64-k window into
      // LDS. Row r covers quant blocks b=q and b=q+1; lanes 0-15 decode
      // block q's nibbles (k=j*2,j*2+1 pattern below), lanes 16-31 block q+1.
      const uint32_t r = half;  // interleave rows across halves
#pragma unroll
      for (uint32_t rr = 0; rr < R_ROWS; rr += 2) {
        const uint32_t m = row0 + rr + r;
        const uint32_t blk_idx = q + r;
        if (m < M && blk_idx < num_blocks) {
          const auto* row = reinterpret_cast<const quant::block_rocmfp4_fast*>(
              static_cast<const uint8_t*>(A_data) +
              (static_cast<size_t>(m) * row_bytes));
          const auto& blk = row[blk_idx];
          const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
          const uint8_t byte_val = blk.qs[j];
          float2 vals;
          vals.x = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * scale;
          vals.y = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * scale;
          reinterpret_cast<float2*>(&s_w[warp_id][rr + r][(r * 32) + j * 2])[0] = vals;
        } else {
          reinterpret_cast<float2*>(&s_w[warp_id][rr + r][(r * 32) + j * 2])[0] =
              make_float2(0.0f, 0.0f);
        }
      }
    }
    __syncthreads();

    if (lane_id < cur_batch) {
      // Four FMAs per b128 activation quad; weights arrive as broadcasts.
#pragma unroll
      for (uint32_t kk4 = 0; kk4 < 16; ++kk4) {
        const float4 xv = reinterpret_cast<const float4*>(&s_xt[lane_id][kk4 * 4])[0];
        const float4* wsrc = reinterpret_cast<const float4*>(&s_w[warp_id][0][kk4 * 4]);
#pragma unroll
        for (uint32_t rr = 0; rr < R_ROWS; ++rr) {
          const float4 wv = wsrc[rr * 16];  // row stride 64 floats = 16 quads
          acc[rr] += wv.x * xv.x;
          acc[rr] += wv.y * xv.y;
          acc[rr] += wv.z * xv.z;
          acc[rr] += wv.w * xv.w;
        }
      }
    }
    __syncthreads();
  }

  if (warp_live && lane_id < cur_batch) {
#pragma unroll
    for (uint32_t rr = 0; rr < R_ROWS; ++rr) {
      const uint32_t m = row0 + rr;
      if (m < M) y[(static_cast<size_t>(t_base + lane_id) * M) + m] = acc[rr];
    }
  }
}

namespace {

float frand(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return (static_cast<float>(s >> 8) / 8388608.0f) - 1.0f;
}

void dequant_row(const uint8_t* row, uint32_t K, std::vector<double>& out) {
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(row);
  for (uint32_t b = 0; b < K / 32; ++b) {
    double d;
    {
      const uint8_t x = blocks[b].e;
      const int exp = (x >> 3) & 0xF;
      const int man = x & 0x7;
      d = exp == 0 ? man / 1024.0 : std::ldexp((1.0 + man / 8.0), exp - 8);
    }
    for (uint32_t i = 0; i < 16; ++i) {
      const uint8_t q = blocks[b].qs[i];
      auto dec = [](uint8_t c) -> double {
        const int mag3 = c & 7;
        const int mag = mag3 <= 4 ? mag3 : (2 * mag3 - 4);
        return (c & 8) ? -mag : mag;
      };
      out[b * 32 + i * 2 + 0] = d * dec(q & 0xF);
      out[b * 32 + i * 2 + 1] = d * dec(q >> 4);
    }
  }
}

template <typename LaunchFn>
void run_launch(const char* name, LaunchFn launch, uint32_t M, uint32_t K, uint32_t B,
                int iters, float* d_y, const std::vector<float>* ref, double& max_abs) {
  HIP_OK(hipMemsetAsync(d_y, 0, static_cast<size_t>(M) * B * sizeof(float)));
  launch();
  if (hipGetLastError() != hipSuccess) {
    std::printf("  %-12s LAUNCH FAILED\n", name);
    max_abs = 1e30;
    return;
  }
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> h(static_cast<size_t>(M) * B);
  HIP_OK(hipMemcpy(h.data(), d_y, h.size() * sizeof(float), hipMemcpyDeviceToHost));
  max_abs = 0.0;
  if (ref) {
    for (size_t i = 0; i < h.size(); ++i)
      max_abs = std::max(max_abs, std::fabs(static_cast<double>(h[i]) - static_cast<double>((*ref)[i])));
  }

  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  launch();
  HIP_OK(hipDeviceSynchronize());
  float best = 1e30f;
  for (int i = 0; i < iters; ++i) {
    HIP_OK(hipEventRecord(e0));
    launch();
    HIP_OK(hipEventRecord(e1));
    HIP_OK(hipEventSynchronize(e1));
    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
    best = std::min(best, ms);
  }
  const double gflop = 2.0 * M * K * B / 1e9;
  std::printf("  %-12s %8.3f ms | %6.2f GFLOP/s eff | max|diff| %.3e\n",
              name, best, gflop / (best / 1e3), ref ? max_abs : 0.0);
  HIP_OK(hipEventDestroy(e0));
  HIP_OK(hipEventDestroy(e1));
}

template <uint32_t TB>
void run_variant(const char* name, const void* d_w, const float* d_x, float* d_y,
                 uint32_t M, uint32_t K, uint32_t B, int iters,
                 const std::vector<float>& ref, double& max_abs) {
  constexpr uint32_t kWaves = 8;
  dim3 grid((M + kWaves - 1) / kWaves, (B + TB - 1) / TB);
  run_launch(name, [&]() {
    GemmLdsRocmFp4FastKernel<kWaves, TB><<<grid, kWaves * 32>>>(d_w, d_x, d_y, M, K, B);
  }, M, K, B, iters, d_y, ref.empty() ? nullptr : &ref, max_abs);
}

void bench_shape(uint32_t M, uint32_t K, uint32_t B, int iters, bool cpu_anchor) {
  const size_t wn = static_cast<size_t>(M) * (K / 32) * sizeof(quant::block_rocmfp4_fast);
  const size_t xn = static_cast<size_t>(B) * K;
  const size_t yn = static_cast<size_t>(M) * B;
  std::vector<uint8_t> hw(wn);
  std::vector<float> hx(xn);
  uint32_t seed = 777u ^ (M * 2654435761u) ^ (B * 97u);
  for (auto& b : hw) {
    seed = seed * 1664525u + 1013904223u;
    b = static_cast<uint8_t>(seed >> 16);
  }
  // Keep scale bytes finite (<= 0x7e).
  for (size_t i = 0; i < wn; ++i) {
    const size_t within = i % sizeof(quant::block_rocmfp4_fast);
    if (within == 16) hw[i] &= 0x7e;
  }
  for (auto& v : hx) v = frand(seed) * 0.5f;

  void* d_w;
  float *d_x, *d_y;
  HIP_OK(hipMalloc(&d_w, wn));
  HIP_OK(hipMalloc(&d_x, xn * sizeof(float)));
  HIP_OK(hipMalloc(&d_y, yn * sizeof(float)));
  HIP_OK(hipMemcpy(d_w, hw.data(), wn, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_x, hx.data(), xn * sizeof(float), hipMemcpyHostToDevice));

  // Reference from the production configuration T_BATCH=32.
  std::vector<float> ref32;
  {
    double d32 = 0.0;
    run_variant<32>("prod TB=32", d_w, d_x, d_y, M, K, B, iters,
                    std::vector<float>(yn, 0.0f), d32);
    ref32.resize(yn);
    HIP_OK(hipMemcpy(ref32.data(), d_y, yn * sizeof(float), hipMemcpyDeviceToHost));
  }
  double dv = 0.0;
  {
    constexpr uint32_t kWaves = 8;
    dim3 gswap((B + 31u) / 32u, (M + kWaves - 1) / kWaves);
    run_launch("swap TB=32", [&]() {
      GemmLdsRocmFp4SwapKernel<kWaves, 32><<<gswap, kWaves * 32>>>(d_w, d_x, d_y, M, K, B);
    }, M, K, B, iters, d_y, &ref32, dv);
    dim3 gd32((B + 31u) / 32u, (M + kWaves * 2u - 1u) / (kWaves * 2u));
    run_launch("dual TB=32", [&]() {
      GemmLdsRocmFp4DualRowKernel<kWaves, 32, true><<<gd32, kWaves * 32>>>(d_w, d_x, d_y, M, K, B);
    }, M, K, B, iters, d_y, &ref32, dv);
    dim3 gd16((B + 15u) / 16u, (M + kWaves * 2u - 1u) / (kWaves * 2u));
    run_launch("dual TB=16", [&]() {
      GemmLdsRocmFp4DualRowKernel<kWaves, 16, true><<<gd16, kWaves * 32>>>(d_w, d_x, d_y, M, K, B);
    }, M, K, B, iters, d_y, &ref32, dv);
    dim3 gd16n((M + kWaves * 2u - 1u) / (kWaves * 2u), (B + 15u) / 16u);
    run_launch("dual16 noswp", [&]() {
      GemmLdsRocmFp4DualRowKernel<kWaves, 16, false><<<gd16n, kWaves * 32>>>(d_w, d_x, d_y, M, K, B);
    }, M, K, B, iters, d_y, &ref32, dv);
    dim3 gp((M + kWaves - 1) / kWaves, (B + 31u) / 32u);
    run_launch("pipe TB=32", [&]() {
      GemmLdsRocmFp4PipeKernel<kWaves, 32><<<gp, kWaves * 32>>>(d_w, d_x, d_y, M, K, B);
    }, M, K, B, iters, d_y, &ref32, dv);
    dim3 gr((M + kWaves * 8u - 1u) / (kWaves * 8u), (B + 31u) / 32u);
    run_launch("rowreuse R8", [&]() {
      GemmRocmFp4RowReuseKernel<kWaves, 32, 8><<<gr, kWaves * 32>>>(d_w, d_x, d_y, M, K, B);
    }, M, K, B, iters, d_y, &ref32, dv);
    dim3 gr4((M + kWaves * 4u - 1u) / (kWaves * 4u), (B + 31u) / 32u);
    run_launch("rowreuse R4", [&]() {
      GemmRocmFp4RowReuseKernel<kWaves, 32, 4><<<gr4, kWaves * 32>>>(d_w, d_x, d_y, M, K, B);
    }, M, K, B, iters, d_y, &ref32, dv);
  }
  run_variant<64>("TB=64", d_w, d_x, d_y, M, K, B, iters, ref32, dv);

  if (cpu_anchor) {
    // fp64 CPU reference over the full matrix for a small shape.
    std::vector<double> wk(K);
    double worst = 0.0;
    for (uint32_t m = 0; m < M; ++m) {
      dequant_row(hw.data() + static_cast<size_t>(m) * (K / 32) * 17, K, wk);
      for (uint32_t t = 0; t < B; ++t) {
        double acc = 0.0;
        for (uint32_t k = 0; k < K; ++k)
          acc += wk[k] * static_cast<double>(hx[static_cast<size_t>(t) * K + k]);
        worst = std::max(worst, std::fabs(acc - static_cast<double>(ref32[static_cast<size_t>(t) * M + m])));
      }
    }
    std::printf("  fp64 CPU ref max|diff| = %.3e\n", worst);
  }

  HIP_OK(hipFree(d_w));
  HIP_OK(hipFree(d_x));
  HIP_OK(hipFree(d_y));
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t B = 2658;
  int iters = 5;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-b") && i + 1 < argc) B = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
  }
  std::printf("shape attn_q  M=8192 K=3072 B=%u\n", B);
  bench_shape(8192, 3072, B, iters, false);
  std::printf("shape attn_out/ssm_out M=3072 K=8192 B=%u\n", B);
  bench_shape(3072, 8192, B, iters, false);
  std::printf("shape ssm_qkv M=12288 K=3072 B=%u\n", B);
  bench_shape(12288, 3072, B, iters, false);
  std::printf("anchor M=64 K=64 B=16 (fp64 CPU)\n");
  bench_shape(64, 64, 16, 1, true);
  return 0;
}
