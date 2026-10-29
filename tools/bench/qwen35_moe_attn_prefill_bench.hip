// Standalone microbenchmark + numerics harness for the Ornith-1.5-122B
// (qwen35moe) full-attention causal prefill on gfx1151.
//
// Self-contained (no repository headers) so it compiles directly with the
// TheRock clang-cl HIP toolchain on Windows:
//
//   clang-cl.exe -x hip qwen35_moe_attn_prefill_bench.hip \
//     --offload-arch=gfx1151 -O3 -fp:fast -o qwen35_moe_attn_prefill_bench.exe \
//     -link <therock>/lib/amdhip64.lib
//
//   ./qwen35_moe_attn_prefill_bench [-b 2658] [-p 0] [-i 5]
//
// Shapes are the production qwen35moe full-attention layer: 32 query heads
// over 2 KV heads (GQA 16), head_dim 256, FP32 KV cache laid out
// [kv_head][position][dim], FP32 queries/output laid out
// [token][head][dim], gated output ctx * sigmoid(gate).
//
// "naive" is the historical per-token sweep (two barriers per key, KV read
// from DRAM per token). "tiled" is the production block-tiled replacement.
// Both run on device; outputs are compared against each other and, for a
// small case, against an independent FP64 CPU reference.
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

__device__ __forceinline__ float Sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// Naive baseline: one block per (head, token), full causal sweep per token.
// ---------------------------------------------------------------------------
__global__ void AttentionPrefillNaiveKernel(
    const float* __restrict__ q_buf,
    const float* __restrict__ gate_buf,
    const float* __restrict__ kv_k,
    const float* __restrict__ kv_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos0,
    float* __restrict__ context_out) {
  const uint32_t h = blockIdx.x;
  const uint32_t tok = blockIdx.y;
  if (h >= num_heads) return;

  const uint32_t group_size = num_heads / num_kv_heads;
  const uint32_t kv_h = h / group_size;
  const uint32_t pos = pos0 + tok;
  const uint32_t seq_len = pos + 1;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  const float* q_h = q_buf + (static_cast<size_t>(tok) * num_heads * head_dim) + (h * head_dim);
  const float* k_base = kv_k + (static_cast<size_t>(kv_h) * max_context * head_dim);
  const float* v_base = kv_v + (static_cast<size_t>(kv_h) * max_context * head_dim);
  const float* g_h = gate_buf + (static_cast<size_t>(tok) * num_heads * head_dim) + (h * head_dim);
  float* out_h = context_out + (static_cast<size_t>(tok) * num_heads * head_dim) + (h * head_dim);

  const uint32_t d = threadIdx.x;
  const uint32_t warp_id = d >> 5u;
  const uint32_t lane_id = d & 31u;
  const float q_val = (d < head_dim) ? q_h[d] : 0.0f;

  __shared__ float s_warp_dot[8];

  float m = -1e30f;
  float l = 0.0f;
  float acc = 0.0f;

  for (uint32_t p = 0; p < seq_len; ++p) {
    const float* k_p = k_base + (static_cast<size_t>(p) * head_dim);
    const float* v_p = v_base + (static_cast<size_t>(p) * head_dim);

    float prod = (d < head_dim) ? (q_val * k_p[d]) : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      prod += __shfl_xor(prod, offset);
    }
    if (lane_id == 0) {
      s_warp_dot[warp_id] = prod;
    }
    __syncthreads();

    if (warp_id == 0) {
      float s_dot = (lane_id < 8) ? s_warp_dot[lane_id] : 0.0f;
#pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        s_dot += __shfl_xor(s_dot, offset);
      }
      if (lane_id == 0) {
        s_warp_dot[0] = s_dot * scale;
      }
    }
    __syncthreads();

    const float score = s_warp_dot[0];
    const float m_new = fmaxf(m, score);
    const float alpha = expf(m - m_new);
    const float beta = expf(score - m_new);
    l = (l * alpha) + beta;
    if (d < head_dim) {
      acc = (acc * alpha) + (beta * v_p[d]);
    }
    m = m_new;
  }

  if (d < head_dim) {
    const float ctx = (l > 0.0f) ? (acc / l) : 0.0f;
    out_h[d] = ctx * Sigmoidf(g_h[d]);
  }
}

// ---------------------------------------------------------------------------
// Tiled candidate: production AttentionPrefillTiledKernel, copied verbatim
// from src/models/qwen35_moe/hip/kernels.hip.
// ---------------------------------------------------------------------------
__launch_bounds__(512, 2)
__global__ void AttentionPrefillTiledKernel(
    const float* __restrict__ q_buf,
    const float* __restrict__ gate_buf,
    const float* __restrict__ kv_k,
    const float* __restrict__ kv_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t max_context,
    uint32_t pos0,
    uint32_t B,
    float* __restrict__ context_out) {
  constexpr uint32_t kDim = 256;
  constexpr uint32_t kQTile = 16;             // tokens per block == warps per block
  constexpr uint32_t kKvTile = 16;            // keys per LDS stage
  constexpr uint32_t kPerLane = kDim / 32;    // eight floats per lane
  constexpr float kScale = 1.0F / 16.0F;      // 1 / sqrt(kDim)

  const uint32_t h = blockIdx.x;
  const uint32_t tok_base = blockIdx.y * kQTile;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t lane_id = tid & 31u;

  const uint32_t tok = min(tok_base + warp_id, B - 1u);
  const uint32_t pos = pos0 + tok;
  const uint32_t seq_end = pos0 + min(tok_base + kQTile, B);

  const uint32_t group_size = num_heads / num_kv_heads;
  const uint32_t kv_h = h / group_size;
  const float* k_base = kv_k + (static_cast<size_t>(kv_h) * max_context * kDim);
  const float* v_base = kv_v + (static_cast<size_t>(kv_h) * max_context * kDim);

  __shared__ float s_k[kKvTile][kDim];
  __shared__ float s_v[kKvTile][kDim];

  const size_t row_off =
      (static_cast<size_t>(tok) * num_heads + h) * kDim + lane_id * kPerLane;
  const float4 qa = reinterpret_cast<const float4*>(q_buf + row_off)[0];
  const float4 qb = reinterpret_cast<const float4*>(q_buf + row_off)[1];
  const float qf[kPerLane] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};

  float m = -1e30f;
  float l = 0.0f;
  float acc[kPerLane] = {};

  for (uint32_t kt = 0; kt < seq_end; kt += kKvTile) {
#pragma unroll
    for (uint32_t i = tid; i < kKvTile * (kDim / 4u); i += 512u) {
      const uint32_t row = i >> 6u;
      const uint32_t col = (i & 63u) * 4u;
      const uint32_t p = kt + row;
      const size_t off = static_cast<size_t>(p) * kDim + col;
      const bool live = p < seq_end;
      reinterpret_cast<float4*>(&s_k[row][col])[0] = live
          ? reinterpret_cast<const float4*>(k_base + off)[0]
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      reinterpret_cast<float4*>(&s_v[row][col])[0] = live
          ? reinterpret_cast<const float4*>(v_base + off)[0]
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    if (kt <= pos) {
#pragma unroll
      for (uint32_t j = 0; j < kKvTile; ++j) {
        if (kt + j >= seq_end) break;
        if (kt + j <= pos) {
          const float4 ka = reinterpret_cast<const float4*>(&s_k[j][lane_id * kPerLane])[0];
          const float4 kb = reinterpret_cast<const float4*>(&s_k[j][lane_id * kPerLane])[1];
          float dot = qf[0] * ka.x + qf[1] * ka.y + qf[2] * ka.z + qf[3] * ka.w +
                      qf[4] * kb.x + qf[5] * kb.y + qf[6] * kb.z + qf[7] * kb.w;
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor(dot, o);
          const float score = dot * kScale;
          const float m_new = fmaxf(m, score);
          const float alpha = expf(m - m_new);
          const float beta = expf(score - m_new);
          l = l * alpha + beta;
          const float4 va = reinterpret_cast<const float4*>(&s_v[j][lane_id * kPerLane])[0];
          const float4 vb = reinterpret_cast<const float4*>(&s_v[j][lane_id * kPerLane])[1];
          acc[0] = acc[0] * alpha + beta * va.x;
          acc[1] = acc[1] * alpha + beta * va.y;
          acc[2] = acc[2] * alpha + beta * va.z;
          acc[3] = acc[3] * alpha + beta * va.w;
          acc[4] = acc[4] * alpha + beta * vb.x;
          acc[5] = acc[5] * alpha + beta * vb.y;
          acc[6] = acc[6] * alpha + beta * vb.z;
          acc[7] = acc[7] * alpha + beta * vb.w;
          m = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (tok_base + warp_id < B) {
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    const float4 ga = reinterpret_cast<const float4*>(gate_buf + row_off)[0];
    const float4 gb = reinterpret_cast<const float4*>(gate_buf + row_off)[1];
    float4 oa, ob;
    oa.x = (acc[0] * inv_l) * Sigmoidf(ga.x);
    oa.y = (acc[1] * inv_l) * Sigmoidf(ga.y);
    oa.z = (acc[2] * inv_l) * Sigmoidf(ga.z);
    oa.w = (acc[3] * inv_l) * Sigmoidf(ga.w);
    ob.x = (acc[4] * inv_l) * Sigmoidf(gb.x);
    ob.y = (acc[5] * inv_l) * Sigmoidf(gb.y);
    ob.z = (acc[6] * inv_l) * Sigmoidf(gb.z);
    ob.w = (acc[7] * inv_l) * Sigmoidf(gb.w);
    reinterpret_cast<float4*>(context_out + row_off)[0] = oa;
    reinterpret_cast<float4*>(context_out + row_off)[1] = ob;
  }
}

// ---------------------------------------------------------------------------
// V2: score-hoisted tiled kernel. All kQTile scores of a KV stage are
// computed as an independent shuffle batch first, then the online softmax +
// PV recurrence runs with no LDS or cross-lane traffic in its chain.
// ---------------------------------------------------------------------------
__launch_bounds__(512, 2)
__global__ void AttentionPrefillTiledV2Kernel(
    const float* __restrict__ q_buf,
    const float* __restrict__ gate_buf,
    const float* __restrict__ kv_k,
    const float* __restrict__ kv_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t max_context,
    uint32_t pos0,
    uint32_t B,
    float* __restrict__ context_out) {
  constexpr uint32_t kDim = 256;
  constexpr uint32_t kQTile = 16;
  constexpr uint32_t kKvTile = 16;
  constexpr uint32_t kPerLane = kDim / 32;
  constexpr float kScale = 1.0F / 16.0F;

  const uint32_t h = blockIdx.x;
  const uint32_t tok_base = blockIdx.y * kQTile;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t lane_id = tid & 31u;

  const uint32_t tok = min(tok_base + warp_id, B - 1u);
  const uint32_t pos = pos0 + tok;
  const uint32_t seq_end = pos0 + min(tok_base + kQTile, B);

  const uint32_t group_size = num_heads / num_kv_heads;
  const uint32_t kv_h = h / group_size;
  const float* k_base = kv_k + (static_cast<size_t>(kv_h) * max_context * kDim);
  const float* v_base = kv_v + (static_cast<size_t>(kv_h) * max_context * kDim);

  __shared__ float s_k[kKvTile][kDim];
  __shared__ float s_v[kKvTile][kDim];

  const size_t row_off =
      (static_cast<size_t>(tok) * num_heads + h) * kDim + lane_id * kPerLane;
  const float4 qa = reinterpret_cast<const float4*>(q_buf + row_off)[0];
  const float4 qb = reinterpret_cast<const float4*>(q_buf + row_off)[1];
  const float qf[kPerLane] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};

  float m = -1e30f;
  float l = 0.0f;
  float acc[kPerLane] = {};

  for (uint32_t kt = 0; kt < seq_end; kt += kKvTile) {
#pragma unroll
    for (uint32_t i = tid; i < kKvTile * (kDim / 4u); i += 512u) {
      const uint32_t row = i >> 6u;
      const uint32_t col = (i & 63u) * 4u;
      const uint32_t p = kt + row;
      const size_t off = static_cast<size_t>(p) * kDim + col;
      const bool live = p < seq_end;
      reinterpret_cast<float4*>(&s_k[row][col])[0] = live
          ? reinterpret_cast<const float4*>(k_base + off)[0]
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      reinterpret_cast<float4*>(&s_v[row][col])[0] = live
          ? reinterpret_cast<const float4*>(v_base + off)[0]
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    if (kt <= pos) {
      float sc[kKvTile];
#pragma unroll
      for (uint32_t j = 0; j < kKvTile; ++j) {
        const float4 ka = reinterpret_cast<const float4*>(&s_k[j][lane_id * kPerLane])[0];
        const float4 kb = reinterpret_cast<const float4*>(&s_k[j][lane_id * kPerLane])[1];
        float dot = qf[0] * ka.x + qf[1] * ka.y + qf[2] * ka.z + qf[3] * ka.w +
                    qf[4] * kb.x + qf[5] * kb.y + qf[6] * kb.z + qf[7] * kb.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor(dot, o);
        sc[j] = dot * kScale;
      }
#pragma unroll
      for (uint32_t j = 0; j < kKvTile; ++j) {
        if (kt + j >= seq_end) break;
        if (kt + j <= pos) {
          const float score = sc[j];
          const float m_new = fmaxf(m, score);
          const float alpha = expf(m - m_new);
          const float beta = expf(score - m_new);
          l = l * alpha + beta;
          const float4 va = reinterpret_cast<const float4*>(&s_v[j][lane_id * kPerLane])[0];
          const float4 vb = reinterpret_cast<const float4*>(&s_v[j][lane_id * kPerLane])[1];
          acc[0] = acc[0] * alpha + beta * va.x;
          acc[1] = acc[1] * alpha + beta * va.y;
          acc[2] = acc[2] * alpha + beta * va.z;
          acc[3] = acc[3] * alpha + beta * va.w;
          acc[4] = acc[4] * alpha + beta * vb.x;
          acc[5] = acc[5] * alpha + beta * vb.y;
          acc[6] = acc[6] * alpha + beta * vb.z;
          acc[7] = acc[7] * alpha + beta * vb.w;
          m = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (tok_base + warp_id < B) {
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    const float4 ga = reinterpret_cast<const float4*>(gate_buf + row_off)[0];
    const float4 gb = reinterpret_cast<const float4*>(gate_buf + row_off)[1];
    float4 oa, ob;
    oa.x = (acc[0] * inv_l) * Sigmoidf(ga.x);
    oa.y = (acc[1] * inv_l) * Sigmoidf(ga.y);
    oa.z = (acc[2] * inv_l) * Sigmoidf(ga.z);
    oa.w = (acc[3] * inv_l) * Sigmoidf(ga.w);
    ob.x = (acc[4] * inv_l) * Sigmoidf(gb.x);
    ob.y = (acc[5] * inv_l) * Sigmoidf(gb.y);
    ob.z = (acc[6] * inv_l) * Sigmoidf(gb.z);
    ob.w = (acc[7] * inv_l) * Sigmoidf(gb.w);
    reinterpret_cast<float4*>(context_out + row_off)[0] = oa;
    reinterpret_cast<float4*>(context_out + row_off)[1] = ob;
  }
}

// ---------------------------------------------------------------------------
// V7: dual-row tiled kernel. One warp owns two query tokens (warp_id and
// warp_id + 16), interleaving two independent online-softmax chains so the
// shuffle/exp latency of one row overlaps the other. Each KV stage feeds 32
// tokens per block.
// ---------------------------------------------------------------------------
__launch_bounds__(512, 2)
__global__ void AttentionPrefillTiledV7Kernel(
    const float* __restrict__ q_buf,
    const float* __restrict__ gate_buf,
    const float* __restrict__ kv_k,
    const float* __restrict__ kv_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t max_context,
    uint32_t pos0,
    uint32_t B,
    float* __restrict__ context_out) {
  constexpr uint32_t kDim = 256;
  constexpr uint32_t kQTile = 32;             // two rows per warp x 16 warps
  constexpr uint32_t kKvTile = 16;
  constexpr uint32_t kPerLane = kDim / 32;
  constexpr float kScale = 1.0F / 16.0F;

  const uint32_t h = blockIdx.x;
  const uint32_t tok_base = blockIdx.y * kQTile;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t lane_id = tid & 31u;

  const uint32_t tok_a = min(tok_base + warp_id, B - 1u);
  const uint32_t tok_b = min(tok_base + warp_id + 16u, B - 1u);
  const uint32_t pos_a = pos0 + tok_a;
  const uint32_t pos_b = pos0 + tok_b;
  const uint32_t seq_end = pos0 + min(tok_base + kQTile, B);

  const uint32_t group_size = num_heads / num_kv_heads;
  const uint32_t kv_h = h / group_size;
  const float* k_base = kv_k + (static_cast<size_t>(kv_h) * max_context * kDim);
  const float* v_base = kv_v + (static_cast<size_t>(kv_h) * max_context * kDim);

  __shared__ float s_k[kKvTile][kDim];
  __shared__ float s_v[kKvTile][kDim];

  const size_t row_off_a =
      (static_cast<size_t>(tok_a) * num_heads + h) * kDim + lane_id * kPerLane;
  const size_t row_off_b =
      (static_cast<size_t>(tok_b) * num_heads + h) * kDim + lane_id * kPerLane;
  const float4 qa0 = reinterpret_cast<const float4*>(q_buf + row_off_a)[0];
  const float4 qa1 = reinterpret_cast<const float4*>(q_buf + row_off_a)[1];
  const float4 qb0 = reinterpret_cast<const float4*>(q_buf + row_off_b)[0];
  const float4 qb1 = reinterpret_cast<const float4*>(q_buf + row_off_b)[1];
  const float qfa[kPerLane] = {qa0.x, qa0.y, qa0.z, qa0.w, qa1.x, qa1.y, qa1.z, qa1.w};
  const float qfb[kPerLane] = {qb0.x, qb0.y, qb0.z, qb0.w, qb1.x, qb1.y, qb1.z, qb1.w};

  float m_a = -1e30f, l_a = 0.0f;
  float m_b = -1e30f, l_b = 0.0f;
  float acc_a[kPerLane] = {};
  float acc_b[kPerLane] = {};

  for (uint32_t kt = 0; kt < seq_end; kt += kKvTile) {
#pragma unroll
    for (uint32_t i = tid; i < kKvTile * (kDim / 4u); i += 512u) {
      const uint32_t row = i >> 6u;
      const uint32_t col = (i & 63u) * 4u;
      const uint32_t p = kt + row;
      const size_t off = static_cast<size_t>(p) * kDim + col;
      const bool live = p < seq_end;
      reinterpret_cast<float4*>(&s_k[row][col])[0] = live
          ? reinterpret_cast<const float4*>(k_base + off)[0]
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      reinterpret_cast<float4*>(&s_v[row][col])[0] = live
          ? reinterpret_cast<const float4*>(v_base + off)[0]
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    if (kt <= pos_a || kt <= pos_b) {
#pragma unroll
      for (uint32_t j = 0; j < kKvTile; ++j) {
        if (kt + j >= seq_end) break;
        const float4 ka = reinterpret_cast<const float4*>(&s_k[j][lane_id * kPerLane])[0];
        const float4 kb = reinterpret_cast<const float4*>(&s_k[j][lane_id * kPerLane])[1];
        float dot_a = qfa[0] * ka.x + qfa[1] * ka.y + qfa[2] * ka.z + qfa[3] * ka.w +
                      qfa[4] * kb.x + qfa[5] * kb.y + qfa[6] * kb.z + qfa[7] * kb.w;
        float dot_b = qfb[0] * ka.x + qfb[1] * ka.y + qfb[2] * ka.z + qfb[3] * ka.w +
                      qfb[4] * kb.x + qfb[5] * kb.y + qfb[6] * kb.z + qfb[7] * kb.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          dot_a += __shfl_xor(dot_a, o);
          dot_b += __shfl_xor(dot_b, o);
        }
        const float4 va = reinterpret_cast<const float4*>(&s_v[j][lane_id * kPerLane])[0];
        const float4 vb = reinterpret_cast<const float4*>(&s_v[j][lane_id * kPerLane])[1];
        if (kt + j <= pos_a) {
          const float score = dot_a * kScale;
          const float m_new = fmaxf(m_a, score);
          const float alpha = expf(m_a - m_new);
          const float beta = expf(score - m_new);
          l_a = l_a * alpha + beta;
          acc_a[0] = acc_a[0] * alpha + beta * va.x;
          acc_a[1] = acc_a[1] * alpha + beta * va.y;
          acc_a[2] = acc_a[2] * alpha + beta * va.z;
          acc_a[3] = acc_a[3] * alpha + beta * va.w;
          acc_a[4] = acc_a[4] * alpha + beta * vb.x;
          acc_a[5] = acc_a[5] * alpha + beta * vb.y;
          acc_a[6] = acc_a[6] * alpha + beta * vb.z;
          acc_a[7] = acc_a[7] * alpha + beta * vb.w;
          m_a = m_new;
        }
        if (kt + j <= pos_b) {
          const float score = dot_b * kScale;
          const float m_new = fmaxf(m_b, score);
          const float alpha = expf(m_b - m_new);
          const float beta = expf(score - m_new);
          l_b = l_b * alpha + beta;
          acc_b[0] = acc_b[0] * alpha + beta * va.x;
          acc_b[1] = acc_b[1] * alpha + beta * va.y;
          acc_b[2] = acc_b[2] * alpha + beta * va.z;
          acc_b[3] = acc_b[3] * alpha + beta * va.w;
          acc_b[4] = acc_b[4] * alpha + beta * vb.x;
          acc_b[5] = acc_b[5] * alpha + beta * vb.y;
          acc_b[6] = acc_b[6] * alpha + beta * vb.z;
          acc_b[7] = acc_b[7] * alpha + beta * vb.w;
          m_b = m_new;
        }
      }
    }
    __syncthreads();
  }

  const float inv_a = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
  const float inv_b = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
  if (tok_base + warp_id < B) {
    const float4 ga = reinterpret_cast<const float4*>(gate_buf + row_off_a)[0];
    const float4 gb = reinterpret_cast<const float4*>(gate_buf + row_off_a)[1];
    float4 oa, ob;
    oa.x = (acc_a[0] * inv_a) * Sigmoidf(ga.x);
    oa.y = (acc_a[1] * inv_a) * Sigmoidf(ga.y);
    oa.z = (acc_a[2] * inv_a) * Sigmoidf(ga.z);
    oa.w = (acc_a[3] * inv_a) * Sigmoidf(ga.w);
    ob.x = (acc_a[4] * inv_a) * Sigmoidf(gb.x);
    ob.y = (acc_a[5] * inv_a) * Sigmoidf(gb.y);
    ob.z = (acc_a[6] * inv_a) * Sigmoidf(gb.z);
    ob.w = (acc_a[7] * inv_a) * Sigmoidf(gb.w);
    reinterpret_cast<float4*>(context_out + row_off_a)[0] = oa;
    reinterpret_cast<float4*>(context_out + row_off_a)[1] = ob;
  }
  if (tok_base + warp_id + 16u < B) {
    const float4 ga = reinterpret_cast<const float4*>(gate_buf + row_off_b)[0];
    const float4 gb = reinterpret_cast<const float4*>(gate_buf + row_off_b)[1];
    float4 oa, ob;
    oa.x = (acc_b[0] * inv_b) * Sigmoidf(ga.x);
    oa.y = (acc_b[1] * inv_b) * Sigmoidf(ga.y);
    oa.z = (acc_b[2] * inv_b) * Sigmoidf(ga.z);
    oa.w = (acc_b[3] * inv_b) * Sigmoidf(ga.w);
    ob.x = (acc_b[4] * inv_b) * Sigmoidf(gb.x);
    ob.y = (acc_b[5] * inv_b) * Sigmoidf(gb.y);
    ob.z = (acc_b[6] * inv_b) * Sigmoidf(gb.z);
    ob.w = (acc_b[7] * inv_b) * Sigmoidf(gb.w);
    reinterpret_cast<float4*>(context_out + row_off_b)[0] = oa;
    reinterpret_cast<float4*>(context_out + row_off_b)[1] = ob;
  }
}

namespace {

constexpr uint32_t kHeads = 32;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kDim = 256;
constexpr uint32_t kMaxCtx = 4096;
constexpr int kFullLayers = 12;

float frand(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return (static_cast<float>(s >> 8) / 8388608.0f) - 1.0f;  // [-1, 1)
}

double run_case(uint32_t B, uint32_t pos0, int iters, bool small_cpu_ref) {
  const size_t qn = static_cast<size_t>(B) * kHeads * kDim;
  const size_t kn = static_cast<size_t>(kKvHeads) * kMaxCtx * kDim;

  std::vector<float> h_q(qn), h_g(qn), h_k(kn), h_v(kn);
  uint32_t seed = 12345u ^ (B * 2654435761u) ^ (pos0 * 97531u);
  for (auto& x : h_q) x = frand(seed) * 0.5f;
  for (auto& x : h_g) x = frand(seed) * 2.0f;
  for (auto& x : h_k) x = frand(seed) * 0.5f;
  for (auto& x : h_v) x = frand(seed) * 0.5f;
  // Positions before pos0 hold prior-chunk context; leave them populated so
  // continuation cases read real history.

  float *d_q, *d_g, *d_k, *d_v, *d_o_naive, *d_o_tiled;
  HIP_OK(hipMalloc(&d_q, qn * sizeof(float)));
  HIP_OK(hipMalloc(&d_g, qn * sizeof(float)));
  HIP_OK(hipMalloc(&d_k, kn * sizeof(float)));
  HIP_OK(hipMalloc(&d_v, kn * sizeof(float)));
  HIP_OK(hipMalloc(&d_o_naive, qn * sizeof(float)));
  HIP_OK(hipMalloc(&d_o_tiled, qn * sizeof(float)));
  HIP_OK(hipMemcpy(d_q, h_q.data(), qn * sizeof(float), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_g, h_g.data(), qn * sizeof(float), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_k, h_k.data(), kn * sizeof(float), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_v, h_v.data(), kn * sizeof(float), hipMemcpyHostToDevice));

  float *d_o_v2, *d_o_v7;
  HIP_OK(hipMalloc(&d_o_v2, qn * sizeof(float)));
  HIP_OK(hipMalloc(&d_o_v7, qn * sizeof(float)));

  auto launch_naive = [&]() {
    dim3 grid(kHeads, B);
    AttentionPrefillNaiveKernel<<<grid, kDim>>>(
        d_q, d_g, d_k, d_v, kHeads, kKvHeads, kDim, kMaxCtx, pos0, d_o_naive);
  };
  auto launch_tiled = [&]() {
    dim3 grid(kHeads, (B + 15u) / 16u);
    AttentionPrefillTiledKernel<<<grid, 512>>>(
        d_q, d_g, d_k, d_v, kHeads, kKvHeads, kMaxCtx, pos0, B, d_o_tiled);
  };
  auto launch_v2 = [&]() {
    dim3 grid(kHeads, (B + 15u) / 16u);
    AttentionPrefillTiledV2Kernel<<<grid, 512>>>(
        d_q, d_g, d_k, d_v, kHeads, kKvHeads, kMaxCtx, pos0, B, d_o_v2);
  };
  auto launch_v7 = [&]() {
    dim3 grid(kHeads, (B + 31u) / 32u);
    AttentionPrefillTiledV7Kernel<<<grid, 512>>>(
        d_q, d_g, d_k, d_v, kHeads, kKvHeads, kMaxCtx, pos0, B, d_o_v7);
  };

  // Correctness: every candidate against the naive baseline.
  launch_naive();
  launch_tiled();
  launch_v2();
  launch_v7();
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> o_naive(qn), o_tiled(qn), o_v2(qn), o_v7(qn);
  HIP_OK(hipMemcpy(o_naive.data(), d_o_naive, qn * sizeof(float), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(o_tiled.data(), d_o_tiled, qn * sizeof(float), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(o_v2.data(), d_o_v2, qn * sizeof(float), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(o_v7.data(), d_o_v7, qn * sizeof(float), hipMemcpyDeviceToHost));
  auto max_diff_vs = [&](const std::vector<float>& o) {
    double d = 0.0;
    for (size_t i = 0; i < qn; ++i)
      d = std::max(d, std::fabs(static_cast<double>(o_naive[i]) - static_cast<double>(o[i])));
    return d;
  };
  const double md_tiled = max_diff_vs(o_tiled);
  const double md_v2 = max_diff_vs(o_v2);
  const double md_v7 = max_diff_vs(o_v7);
  double ref_mag = 0.0;
  for (size_t i = 0; i < qn; ++i)
    ref_mag = std::max(ref_mag, std::fabs(static_cast<double>(o_naive[i])));
  const double max_abs = std::max({md_tiled, md_v2, md_v7});

  // Independent FP64 CPU reference for a small case (one head is enough to
  // catch indexing errors; sweep all heads at tiny B).
  if (small_cpu_ref) {
    double worst_ref = 0.0;
    std::vector<double> acc64(kDim);
    for (uint32_t h = 0; h < kHeads; ++h) {
      const uint32_t kv_h = h / (kHeads / kKvHeads);
      for (uint32_t tok = 0; tok < B; ++tok) {
        const uint32_t pos = pos0 + tok;
        std::fill(acc64.begin(), acc64.end(), 0.0);
        double m = -1e300, lsum = 0.0;
        for (uint32_t p = 0; p <= pos; ++p) {
          const float* qr = &h_q[(static_cast<size_t>(tok) * kHeads + h) * kDim];
          const float* kr = &h_k[(static_cast<size_t>(kv_h) * kMaxCtx + p) * kDim];
          const float* vr = &h_v[(static_cast<size_t>(kv_h) * kMaxCtx + p) * kDim];
          double dot = 0.0;
          for (uint32_t d = 0; d < kDim; ++d) dot += static_cast<double>(qr[d]) * kr[d];
          const double score = dot / 16.0;
          const double m_new = std::max(m, score);
          const double al = std::exp(m - m_new);
          const double be = std::exp(score - m_new);
          lsum = lsum * al + be;
          for (uint32_t d = 0; d < kDim; ++d)
            acc64[d] = acc64[d] * al + be * static_cast<double>(vr[d]);
          m = m_new;
        }
        for (uint32_t d = 0; d < kDim; ++d) {
          const double ctx = acc64[d] / lsum;
          const double gate = 1.0 / (1.0 + std::exp(-static_cast<double>(
              h_g[(static_cast<size_t>(tok) * kHeads + h) * kDim + d])));
          const double want = ctx * gate;
          const double got = o_tiled[(static_cast<size_t>(tok) * kHeads + h) * kDim + d];
          worst_ref = std::max(worst_ref, std::fabs(want - got));
          const double got7 = o_v7[(static_cast<size_t>(tok) * kHeads + h) * kDim + d];
          worst_ref = std::max(worst_ref, std::fabs(want - got7));
        }
      }
    }
    std::printf("  fp64 CPU ref max|diff| = %.3e\n", worst_ref);
  }

  // Timing.
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  auto bench = [&](auto launch) {
    launch();  // warmup
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
    return best;
  };
  const float t_naive = bench(launch_naive);
  const float t_tiled = bench(launch_tiled);
  const float t_v2 = bench(launch_v2);
  const float t_v7 = bench(launch_v7);

  std::printf("B=%u pos0=%u ref mag %.3g | max|diff| tiled %.3e v2 %.3e v7 %.3e\n",
              B, pos0, ref_mag, md_tiled, md_v2, md_v7);
  std::printf("  per full-attn layer ms: naive %.3f | tiled %.3f (%.1fx) | v2 %.3f (%.1fx) | v7 %.3f (%.1fx)\n",
              t_naive, t_tiled, t_naive / t_tiled, t_v2, t_naive / t_v2, t_v7, t_naive / t_v7);
  std::printf("  12-layer attention s: naive %.2f | tiled %.2f | v2 %.2f | v7 %.2f\n",
              t_naive * kFullLayers / 1e3, t_tiled * kFullLayers / 1e3,
              t_v2 * kFullLayers / 1e3, t_v7 * kFullLayers / 1e3);

  HIP_OK(hipEventDestroy(e0));
  HIP_OK(hipEventDestroy(e1));
  HIP_OK(hipFree(d_q));
  HIP_OK(hipFree(d_g));
  HIP_OK(hipFree(d_k));
  HIP_OK(hipFree(d_v));
  HIP_OK(hipFree(d_o_naive));
  HIP_OK(hipFree(d_o_tiled));
  HIP_OK(hipFree(d_o_v2));
  HIP_OK(hipFree(d_o_v7));
  return max_abs;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t B = 2658, pos0 = 0;
  int iters = 5;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-b") && i + 1 < argc) B = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) pos0 = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
  }
  run_case(B, pos0, iters, false);
  run_case(37, 13, 3, true);   // tail + continuation edge case with CPU anchor
  run_case(1, 4000, 3, true);  // decode-shaped single token at deep context
  return 0;
}
