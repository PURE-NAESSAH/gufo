// WMMA (tensor core) prefill GEMM prototype for the Ornith-1.5-122B
// Q4_0_ROCMFP4_FAST projections on gfx1151.
//
// Dataflow copied from the proven HalfPrefillGemmKernel
// (src/models/qwen/hip/kernels/prefill_fp16.hip): swizzled LDS staging,
// raw __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 with activations as the
// first operand so accumulators land oriented [token][row] for coalesced
// stores into y[t*M+m].
//
// ROCmFP4 (codebook10) magnitudes {0,1,2,3,4,6,8,10} times a ue4m3 scale
// are exactly representable in FP16 ((8+m)*c <= 150 < 2^10), so the weight
// side loses nothing. Activations optionally split into hi+lo halves
// (SPLIT=true) to recover ~22 mantissa bits; SPLIT=false measures the
// single-rounded path.
//
// Build (Windows TheRock toolchain):
//   clang-cl.exe -x hip qwen35_moe_wmma_gemm_bench.hip --offload-arch=gfx1151 \
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

__device__ __forceinline__ float rocmfp4_ue4m3_to_fp32_half_finite(uint8_t x) {
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  if (exp == 0) return (float)man * (1.0f / 1024.0f);
  const uint32_t bits = ((uint32_t)exp + 119u) << 23 | ((uint32_t)man << 20);
  return __uint_as_float(bits);
}

__device__ __forceinline__ v16h WmmaHalf(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Permute 16-byte halves and the LDS row to avoid repeated bank conflicts.
__device__ __forceinline__ v16h LoadSwizzled(const short* p) {
  union {
    v16h f;
    uint4 q[2];
  } bits;
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

// Decode one 16-weight k-sub-block of a rocmfp4_fast block into exact
// scaled halves. ks selects the half of the 32-weight block (8 bytes).
struct HalfWeights {
  uint4 low, high;
};
// rocmfp4_fast stores nibble PLANES: low nibbles of qs[0..15] are k 0..15,
// high nibbles are k 16..31 (DequantizeQ4_0_ROCMFP4_FAST). ks selects the
// plane; all 16 bytes are read either way.
__device__ __forceinline__ HalfWeights DecodeRocmFp4Sub(const uint8_t* blk,
                                                        unsigned ks) {
  // 17-byte blocks: dword loads tolerate the misalignment, uint4 does not.
  const unsigned* wp = reinterpret_cast<const unsigned*>(blk);
  const unsigned words[4] = {__ldg(wp), __ldg(wp + 1), __ldg(wp + 2), __ldg(wp + 3)};
  const unsigned shift = (ks & 1) * 4;
  const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk[16]);
  union {
    v16h f;
    uint4 u[2];
  } result;
#pragma unroll
  for (unsigned wi = 0; wi < 4; ++wi) {
    const unsigned n = (words[wi] >> shift) & 0x0F0F0F0FU;
    const unsigned index = n & 0x07070707U;
    const unsigned sign_bits = (n & 0x08080808U) << 4;
    // FP16 top bytes of magnitudes {0,1,2,3,4,6,8,10}: 0x3C00, 0x4000,
    // 0x4200, 0x4400, 0x4600, 0x4800, 0x4900 (bottom byte always zero).
    // perm selector 0..3 draws from the SECOND operand on gfx1151 (probed).
    const unsigned hi = __builtin_amdgcn_perm(0x49484644U, 0x42403C00U, index) | sign_bits;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const _Float16 mag = __ushort_as_half(((hi >> (8 * j)) & 0xFFU) << 8);
      result.f[4 * wi + j] = static_cast<_Float16>(__fmul_rn(scale, __half2float(mag)));
    }
  }
  return {result.u[0], result.u[1]};
}

// ---------------------------------------------------------------------------
// Production scalar kernel (reference), T_BATCH=32.
// ---------------------------------------------------------------------------
template <uint32_t WavesPerBlock, uint32_t T_BATCH = 32>
__launch_bounds__(WavesPerBlock * 32)
__global__ void GemmLdsRocmFp4FastKernel(
    const void* __restrict__ A_data, const float* __restrict__ x,
    float* __restrict__ y, uint32_t M, uint32_t K, uint32_t B) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t_base = blockIdx.y * T_BATCH;
  if (t_base >= B) return;
  const uint32_t cur_batch = min(T_BATCH, B - t_base);
  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* base =
      m < M ? (static_cast<const uint8_t*>(A_data) + (static_cast<size_t>(m) * row_bytes))
            : nullptr;
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
        auto dec = [](uint8_t q) -> int {
          const int mag3 = q & 7;
          const int mag = mag3 <= 4 ? mag3 : (2 * mag3 - 4);
          return (q & 8) ? -mag : mag;
        };
        const float w0 = (float)dec(byte_val & 0xF) * scale;
        const float w1 = (float)dec(byte_val >> 4) * scale;
        const uint32_t offset_k = half * 32;
#pragma unroll
        for (uint32_t tb = 0; tb < T_BATCH; ++tb)
          sum[tb] += (w0 * s_x[tb][offset_k + j]) + (w1 * s_x[tb][offset_k + j + 16]);
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
        for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_xor(s, offset);
        if (lane_id == 0) y[(static_cast<size_t>(t_base + tb) * M) + m] = s;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// WMMA prototype. x must be pre-split activation planes: xh (and xl when
// SPLIT). y is [B][M].
// ---------------------------------------------------------------------------
template <int BM, int BN, int WM, int WN, bool SPLIT>
__launch_bounds__(WM *WN * 32) __global__ void WmmaRocmFp4GemmKernel(
    const void* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int batch, int m, int k) {
  constexpr int BK = 4;
  constexpr int WaveSize = 32;
  constexpr int kThreads = WM * WN * WaveSize;
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int KS = 16;
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int PFA = (BM * BK + kThreads - 1) / kThreads;
  constexpr int PFB = (BN * BK + kThreads - 1) / kThreads;

  __shared__ short s_a[BK][RS][16][KS];
  __shared__ short s_b[SPLIT ? 2 * BK : BK][TS][16][KS];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / WaveSize;
  const int lane = tid % WaveSize;
  const int sl = lane & 15;
  const int wr = wave / WN;
  const int wt = wave % WN;

  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  {
    const unsigned group_shift = 3;
    const unsigned group = 1U << group_shift;
    unsigned first = (blockIdx.y >> group_shift) << group_shift,
             within = (blockIdx.y & (group - 1)) * gridDim.x + blockIdx.x,
             rows = min(group, gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  const int r_block = row_tile * BM, t_block = token_tile * BN;
  const int ksteps = k / 16;

  v8f acc[WRS][WTS] = {};

  const std::size_t row_bytes = (k / 32) * sizeof(quant::block_rocmfp4_fast);
  const std::uint8_t* w_row[PFA];
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int physical_row = ((p * kThreads) + tid) / BK;
    const int r = r_block + physical_row;
    w_row[p] = static_cast<const std::uint8_t*>(w) +
               (static_cast<std::size_t>((r < m ? r : (m - 1))) * row_bytes);
  }
  const float* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (((p * kThreads) + tid) / BK);
    const std::size_t off = static_cast<std::size_t>((t < batch ? t : (batch - 1)) * k);
    x_row[p] = x + off;
  }

  uint4 ra[PFA][2];
  float4 rb[PFB][4];

  const auto fetch = [&](int ks0) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BM * BK && kk < ksteps) {
        const auto decoded = DecodeRocmFp4Sub(w_row[p] + (kk / 2) * 17 + 0, kk);
        // NOTE: bytes[-(ks&1)*8+16] indexing expects the scale byte at
        // offset 16 of the block; DecodeRocmFp4Sub receives the qs start.
        ra[p][0] = decoded.low;
        ra[p][1] = decoded.high;
      } else {
        ra[p][0] = make_uint4(0, 0, 0, 0);
        ra[p][1] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BN * BK && kk < ksteps) {
        const float* src = x_row[p] + static_cast<std::size_t>(kk) * 16;
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
        short* dst = &s_a[ks][placed / 16][placed % 16][0];
        StoreSwizzled(dst, ra[p][0], ra[p][1], row);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BN * BK) {
        const int placed = (row ^ (ks & 3));
        // Split fp32 activations into exact hi+lo half planes at staging.
        union { v16h f; uint4 u[2]; } hiP, loP;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const float4 v = rb[p][q4];
          const float vs[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const _Float16 h = __float2half_rn(vs[e]);
            hiP.f[q4 * 4 + e] = h;
            loP.f[q4 * 4 + e] = __float2half_rn(__fadd_rn(vs[e], -__half2float(h)));
          }
        }
        short* dst = &s_b[ks][placed / 16][placed % 16][0];
        StoreSwizzled(dst, hiP.u[0], hiP.u[1], row);
        if constexpr (SPLIT) {
          short* dst2 = &s_b[BK + ks][placed / 16][placed % 16][0];
          StoreSwizzled(dst2, loP.u[0], loP.u[1], row);
        }
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
      v16h a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i)
        a[i] = LoadSwizzled(&s_a[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const v16h b = LoadSwizzled(&s_b[ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i) acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
        if constexpr (SPLIT) {
          const v16h bl =
              LoadSwizzled(&s_b[BK + ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
          for (int i = 0; i < WRS; ++i) acc[i][j] = WmmaHalf(bl, a[i], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  const int hi = lane / 16;
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
      const int row = r_block + ((wr * WRS + i) * 16) + sl;
      const int t0 = t_block + ((wt * WTS + j) * 16);
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const int token = t0 + 2 * l + hi;
        if (row < m && token < batch)
          y[static_cast<std::size_t>(token) * m + row] = acc[i][j][l];
      }
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
    const uint8_t x = blocks[b].e;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    const double d = exp == 0 ? man / 1024.0 : std::ldexp((1.0 + man / 8.0), exp - 8);
    for (uint32_t i = 0; i < 16; ++i) {
      const uint8_t q = blocks[b].qs[i];
      auto dec = [](uint8_t c) -> double {
        const int mag3 = c & 7;
        const int mag = mag3 <= 4 ? mag3 : (2 * mag3 - 4);
        return (c & 8) ? -mag : mag;
      };
      out[b * 32 + i] = d * dec(q & 0xF);
      out[b * 32 + i + 16] = d * dec(q >> 4);
    }
  }
}

template <bool SPLIT, int BM, int BN, int WM, int WN>
double bench_cfg(const char* name, const void* d_w, const float* d_x,
                 float* d_y, float* d_ref, uint32_t M, uint32_t K, uint32_t B,
                 int iters) {
  dim3 grid((B + BN - 1) / BN, (M + BM - 1) / BM);
  auto launch = [&]() {
    WmmaRocmFp4GemmKernel<BM, BN, WM, WN, SPLIT><<<grid, WM * WN * 32>>>(
        d_w, d_x, d_y, (int)B, (int)M, (int)K);
  };
  HIP_OK(hipMemsetAsync(d_y, 0, (std::size_t)M * B * sizeof(float)));
  launch();
  if (hipGetLastError() != hipSuccess) {
    std::printf("  %-22s LAUNCH FAILED\n", name);
    return 1e30;
  }
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> h((std::size_t)M * B), r((std::size_t)M * B);
  HIP_OK(hipMemcpy(h.data(), d_y, h.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(r.data(), d_ref, r.size() * 4, hipMemcpyDeviceToHost));
  double max_abs = 0.0;
  for (size_t i = 0; i < h.size(); ++i)
    max_abs = std::max(max_abs, std::fabs((double)h[i] - (double)r[i]));
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
  std::printf("  %-22s %8.3f ms | %7.2f GFLOP/s eff | max|diff| %.3e\n", name,
              best, gflop / (best / 1e3), max_abs);
  HIP_OK(hipEventDestroy(e0));
  HIP_OK(hipEventDestroy(e1));
  return max_abs;
}

template <bool SPLIT>
double bench_wmma(const char* name, const void* d_w, const float* d_x,
                  float* d_y, float* d_ref, uint32_t M, uint32_t K,
                  uint32_t B, int iters) {
  return bench_cfg<SPLIT, 128, 64, 2, 2>(name, d_w, d_x, d_y, d_ref, M, K, B, iters);
}

void bench_shape(uint32_t M, uint32_t K, uint32_t B, int iters, bool cpu_anchor) {
  const size_t wn = (size_t)M * (K / 32) * sizeof(quant::block_rocmfp4_fast);
  const size_t xn = (size_t)B * K;
  const size_t yn = (size_t)M * B;
  std::vector<uint8_t> hw(wn);
  std::vector<float> hx(xn);
  uint32_t seed = 777u ^ (M * 2654435761u) ^ (B * 97u);
  for (auto& b : hw) {
    seed = seed * 1664525u + 1013904223u;
    b = (uint8_t)(seed >> 16);
  }
  for (size_t i = 0; i < wn; ++i)
    if (i % 17 == 16) hw[i] &= 0x7e;
  for (auto& v : hx) v = frand(seed) * 0.5f;

  void* d_w;
  float *d_x, *d_y, *d_ref;
  HIP_OK(hipMalloc(&d_w, wn));
  HIP_OK(hipMalloc(&d_x, xn * 4));
  HIP_OK(hipMalloc(&d_y, yn * 4));
  HIP_OK(hipMalloc(&d_ref, yn * 4));
  HIP_OK(hipMemcpy(d_w, hw.data(), wn, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_x, hx.data(), xn * 4, hipMemcpyHostToDevice));

  // fp32 production reference.
  {
    dim3 grid((M + 7) / 8, (B + 31) / 32);
    GemmLdsRocmFp4FastKernel<8, 32><<<grid, 256>>>(d_w, d_x, d_ref, M, K, B);
    HIP_OK(hipDeviceSynchronize());
  }
  bench_wmma<false>("wmma 128x64 s", d_w, d_x, d_y, d_ref, M, K, B, iters);
  bench_wmma<true>("wmma 128x64 split", d_w, d_x, d_y, d_ref, M, K, B, iters);
  bench_cfg<true, 256, 64, 2, 2>("wmma 256x64 split", d_w, d_x, d_y, d_ref, M, K, B, iters);
  bench_cfg<true, 128, 128, 2, 2>("wmma 128x128 split", d_w, d_x, d_y, d_ref, M, K, B, iters);
  bench_cfg<true, 128, 64, 4, 2>("wmma 128x64 w4 split", d_w, d_x, d_y, d_ref, M, K, B, iters);
  bench_cfg<true, 64, 64, 2, 2>("wmma 64x64 split", d_w, d_x, d_y, d_ref, M, K, B, iters);
  bench_cfg<true, 256, 128, 2, 2>("wmma 256x128 split", d_w, d_x, d_y, d_ref, M, K, B, iters);

  if (cpu_anchor) {
    std::vector<double> wk(K);
    std::vector<float> hr(yn);
    HIP_OK(hipMemcpy(hr.data(), d_y, yn * 4, hipMemcpyDeviceToHost));  // split result
    double worst = 0.0;
    for (uint32_t mm = 0; mm < M; ++mm) {
      dequant_row(hw.data() + (size_t)mm * (K / 32) * 17, K, wk);
      for (uint32_t t = 0; t < B; ++t) {
        double a = 0.0;
        for (uint32_t kk = 0; kk < K; ++kk)
          a += wk[kk] * (double)hx[(size_t)t * K + kk];
        worst = std::max(worst, std::fabs(a - (double)hr[(size_t)t * M + mm]));
      }
    }
    std::printf("  fp64 CPU ref (split) max|diff| = %.3e\n", worst);
  }

  HIP_OK(hipFree(d_w));
  HIP_OK(hipFree(d_x));
  HIP_OK(hipFree(d_y));
  HIP_OK(hipFree(d_ref));
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t B = 2658;
  int iters = 5;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-b") && i + 1 < argc) B = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
  }
  std::printf("attn_q M=8192 K=3072 B=%u\n", B);
  bench_shape(8192, 3072, B, iters, false);
  std::printf("out proj M=3072 K=8192 B=%u\n", B);
  bench_shape(3072, 8192, B, iters, false);
  std::printf("anchor M=64 K=64 B=16 (fp64)\n");
  bench_shape(64, 64, 16, 1, true);
  return 0;
}
