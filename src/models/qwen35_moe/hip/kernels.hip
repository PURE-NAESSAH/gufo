#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstdint>
#include <cstddef>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/hip/rocmfp4_hip_codebook.cuh"
#include "src/core/quant/hip/rocmfp4_hip_scale.cuh"

namespace gufo::models::qwen35_moe::hip {

// Codebook10 lookup helper
__device__ __forceinline__ int8_t Rocmfp4DecodeNibble(uint8_t q) {
  constexpr int8_t kRocmfp4Codebook[16] = {
      0, 1, 2, 3, 4, 6, 8, 10,
      0, -1, -2, -3, -4, -6, -8, -10,
  };
  return kRocmfp4Codebook[q & 0x0Fu];
}

// ---------------------------------------------------------------------------
// 1. GEMV Kernels (Wave32)
// ---------------------------------------------------------------------------

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemvRocmFp4FastKernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) return;

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* base = static_cast<const uint8_t*>(A_data) +
                        (static_cast<size_t>(expert_id) * expert_stride) +
                        (static_cast<size_t>(m) * row_bytes);
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(base);

  float sum = 0.0f;
  for (uint32_t b = lane_id; b < num_blocks; b += 32) {
    const auto& blk = blocks[b];
    const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
    const float* xb = x + (b * 32);

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte_val = blk.qs[j];
      const int8_t code0 = Rocmfp4DecodeNibble(byte_val & 0x0Fu);
      const int8_t code1 = Rocmfp4DecodeNibble(byte_val >> 4u);
      sum += (static_cast<float>(code0) * scale) * xb[j];
      sum += (static_cast<float>(code1) * scale) * xb[j + 16];
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemvRocmFp4Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) return;

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4);
  const uint8_t* base = static_cast<const uint8_t*>(A_data) +
                        (static_cast<size_t>(expert_id) * expert_stride) +
                        (static_cast<size_t>(m) * row_bytes);
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4*>(base);

  float sum = 0.0f;
  for (uint32_t b = lane_id; b < num_blocks; b += 32) {
    const auto& blk = blocks[b];
    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(blk.e[0]);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(blk.e[1]);
    const float* xb = x + (b * 32);

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte_val = blk.qs[j];
      const int8_t code0 = Rocmfp4DecodeNibble(byte_val & 0x0Fu);
      const int8_t code1 = Rocmfp4DecodeNibble(byte_val >> 4u);
      sum += (static_cast<float>(code0) * d0) * xb[j];
      sum += (static_cast<float>(code1) * d1) * xb[j + 16];
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemvBF16Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) return;

  const uint16_t* row = reinterpret_cast<const uint16_t*>(
      static_cast<const uint8_t*>(A_data) +
      (static_cast<size_t>(expert_id) * expert_stride)) + (static_cast<size_t>(m) * K);

  float sum = 0.0f;
  for (uint32_t k = lane_id; k < K; k += 32) {
    const uint32_t u32 = static_cast<uint32_t>(row[k]) << 16u;
    float val = 0.0f;
    __builtin_memcpy(&val, &u32, sizeof(float));
    sum += val * x[k];
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemvF32Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) return;

  const float* row = reinterpret_cast<const float*>(
      static_cast<const uint8_t*>(A_data) +
      (static_cast<size_t>(expert_id) * expert_stride)) + (static_cast<size_t>(m) * K);

  float sum = 0.0f;
  for (uint32_t k = lane_id; k < K; k += 32) {
    sum += row[k] * x[k];
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemvF16Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) return;

  const uint16_t* row = reinterpret_cast<const uint16_t*>(
      static_cast<const uint8_t*>(A_data) +
      (static_cast<size_t>(expert_id) * expert_stride)) + (static_cast<size_t>(m) * K);

  float sum = 0.0f;
  for (uint32_t k = lane_id; k < K; k += 32) {
    const __half h = *reinterpret_cast<const __half*>(&row[k]);
    sum += __half2float(h) * x[k];
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemvQ8_0Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) return;

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_q8_0);
  const uint8_t* base = static_cast<const uint8_t*>(A_data) +
                        (static_cast<size_t>(expert_id) * expert_stride) +
                        (static_cast<size_t>(m) * row_bytes);
  const auto* blocks = reinterpret_cast<const quant::block_q8_0*>(base);

  float sum = 0.0f;
  for (uint32_t b = lane_id; b < num_blocks; b += 32) {
    const auto& blk = blocks[b];
    const __half h = *reinterpret_cast<const __half*>(&blk.d);
    const float d = __half2float(h);
    const float* xb = x + (b * 32);

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
      sum += (static_cast<float>(blk.qs[j]) * d) * xb[j];
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[m] = sum;
  }
}

void DispatchGemv(
    hipStream_t stream,
    const void* A_data,
    core::GgmlType A_type,
    uint64_t expert_stride,
    uint32_t expert_id,
    const float* x,
    float* y,
    uint32_t M,
    uint32_t K) {
  if (M == 0 || K == 0 || A_data == nullptr || x == nullptr || y == nullptr) {
    return;
  }

  constexpr uint32_t kWaves = 8;
  const uint32_t block_dim = kWaves * 32;
  const uint32_t grid_dim = (M + kWaves - 1) / kWaves;

  switch (A_type) {
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      GemvRocmFp4FastKernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      break;
    case core::GgmlType::kQ4_0_ROCMFP4:
      GemvRocmFp4Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      break;
    case core::GgmlType::kBF16:
      GemvBF16Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      break;
    case core::GgmlType::kF16:
      GemvF16Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      break;
    case core::GgmlType::kQ8_0:
      GemvQ8_0Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      break;
    case core::GgmlType::kF32:
    default:
      GemvF32Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      break;
  }
}

// ---------------------------------------------------------------------------
// 2. RMSNorm Kernel
// ---------------------------------------------------------------------------

__global__ void RMSNormKernel(
    const float* __restrict__ x,
    const void* __restrict__ weight_data,
    core::GgmlType weight_type,
    float eps,
    uint32_t size,
    float* __restrict__ out) {
  const uint32_t tid = threadIdx.x;
  float sum_sq = 0.0f;
  for (uint32_t i = tid; i < size; i += blockDim.x) {
    const float xi = x[i];
    sum_sq += xi * xi;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor(sum_sq, offset);
  }

  __shared__ float s_warp_sums[32];
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;
  if (lane == 0) {
    s_warp_sums[warp] = sum_sq;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < (blockDim.x / 32)) ? s_warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane == 0) {
      s_warp_sums[0] = rsqrtf((block_sum / static_cast<float>(size)) + eps);
    }
  }
  __syncthreads();

  const float scale = s_warp_sums[0];

  if (weight_data == nullptr) {
    for (uint32_t i = tid; i < size; i += blockDim.x) {
      out[i] = x[i] * scale;
    }
    return;
  }

  if (weight_type == core::GgmlType::kF32) {
    const auto* w = static_cast<const float*>(weight_data);
    for (uint32_t i = tid; i < size; i += blockDim.x) {
      out[i] = x[i] * scale * w[i];
    }
  } else if (weight_type == core::GgmlType::kBF16) {
    const auto* w = static_cast<const uint16_t*>(weight_data);
    for (uint32_t i = tid; i < size; i += blockDim.x) {
      const uint32_t u32 = static_cast<uint32_t>(w[i]) << 16u;
      float wf = 0.0f;
      __builtin_memcpy(&wf, &u32, sizeof(float));
      out[i] = x[i] * scale * wf;
    }
  } else if (weight_type == core::GgmlType::kF16) {
    const auto* w = static_cast<const uint16_t*>(weight_data);
    for (uint32_t i = tid; i < size; i += blockDim.x) {
      const __half h = *reinterpret_cast<const __half*>(&w[i]);
      out[i] = x[i] * scale * __half2float(h);
    }
  }
}

void DispatchRMSNorm(
    hipStream_t stream,
    const float* x,
    const void* weight_data,
    core::GgmlType weight_type,
    float eps,
    uint32_t size,
    float* out) {
  RMSNormKernel<<<1, 256, 0, stream>>>(x, weight_data, weight_type, eps, size, out);
}

// ---------------------------------------------------------------------------
// 3. Embedding Kernel
// ---------------------------------------------------------------------------

__global__ void EmbeddingKernel(
    const void* __restrict__ table,
    core::GgmlType type,
    uint32_t token_id,
    uint32_t hidden_size,
    size_t row_bytes,
    float* __restrict__ out) {
  const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= hidden_size) return;

  const uint8_t* row_base =
      static_cast<const uint8_t*>(table) + (static_cast<size_t>(token_id) * row_bytes);

  if (type == core::GgmlType::kBF16) {
    const auto* src = reinterpret_cast<const uint16_t*>(row_base);
    const uint32_t u32 = static_cast<uint32_t>(src[tid]) << 16u;
    float val = 0.0f;
    __builtin_memcpy(&val, &u32, sizeof(float));
    out[tid] = val;
  } else if (type == core::GgmlType::kF32) {
    const auto* src = reinterpret_cast<const float*>(row_base);
    out[tid] = src[tid];
  } else if (type == core::GgmlType::kF16) {
    const auto* src = reinterpret_cast<const uint16_t*>(row_base);
    const __half h = *reinterpret_cast<const __half*>(&src[tid]);
    out[tid] = __half2float(h);
  } else if (type == core::GgmlType::kQ4_0_ROCMFP4_FAST) {
    const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(row_base);
    const uint32_t b = tid / 32;
    const uint32_t j = tid % 32;
    const auto& blk = blocks[b];
    const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
    const uint8_t byte_val = blk.qs[j < 16 ? j : (j - 16)];
    const uint8_t nibble = j < 16 ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    out[tid] = static_cast<float>(Rocmfp4DecodeNibble(nibble)) * scale;
  } else if (type == core::GgmlType::kQ8_0) {
    const auto* blocks = reinterpret_cast<const quant::block_q8_0*>(row_base);
    const uint32_t b = tid / 32;
    const uint32_t j = tid % 32;
    const auto& blk = blocks[b];
    const __half h = *reinterpret_cast<const __half*>(&blk.d);
    out[tid] = static_cast<float>(blk.qs[j]) * __half2float(h);
  }
}

void DispatchEmbedding(
    hipStream_t stream,
    const void* table,
    core::GgmlType type,
    uint32_t token_id,
    uint32_t hidden_size,
    size_t row_bytes,
    float* out) {
  const uint32_t block_dim = 256;
  const uint32_t grid_dim = (hidden_size + block_dim - 1) / block_dim;
  EmbeddingKernel<<<grid_dim, block_dim, 0, stream>>>(
      table, type, token_id, hidden_size, row_bytes, out);
}

// ---------------------------------------------------------------------------
// 4. RoPE Kernel
// ---------------------------------------------------------------------------

__global__ void RoPEKernel(
    float* __restrict__ vec,
    uint32_t rotary_dim,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t pos,
    float theta_base) {
  const uint32_t h = blockIdx.x;
  if (h >= num_heads) return;
  const uint32_t pair_idx = threadIdx.x;
  if (pair_idx >= (rotary_dim / 2)) return;

  const float freq = 1.0f / powf(theta_base, static_cast<float>(2 * pair_idx) / static_cast<float>(rotary_dim));
  const float theta = static_cast<float>(pos) * freq;
  const float cos_th = cosf(theta);
  const float sin_th = sinf(theta);

  float* head_ptr = vec + (h * head_dim);
  const float x0 = head_ptr[2 * pair_idx];
  const float x1 = head_ptr[(2 * pair_idx) + 1];
  head_ptr[2 * pair_idx] = (x0 * cos_th) - (x1 * sin_th);
  head_ptr[(2 * pair_idx) + 1] = (x0 * sin_th) + (x1 * cos_th);
}

void DispatchRoPE(
    hipStream_t stream,
    float* vec,
    uint32_t rotary_dim,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t pos,
    float theta_base) {
  RoPEKernel<<<num_heads, rotary_dim / 2, 0, stream>>>(
      vec, rotary_dim, head_dim, num_heads, pos, theta_base);
}

// ---------------------------------------------------------------------------
// 5. Attention Unpack & Gate Separation Kernel
// ---------------------------------------------------------------------------

__global__ void UnpackQGKernel(
    const float* __restrict__ qg,
    uint32_t num_heads,
    uint32_t head_dim,
    float* __restrict__ q_buf,
    float* __restrict__ gate_buf) {
  const uint32_t h = blockIdx.x;
  const uint32_t d = threadIdx.x;
  if (h >= num_heads || d >= head_dim) return;

  const float* src = qg + (h * 2 * head_dim);
  q_buf[(h * head_dim) + d] = src[d];
  gate_buf[(h * head_dim) + d] = src[head_dim + d];
}

void DispatchUnpackQG(
    hipStream_t stream,
    const float* qg,
    uint32_t num_heads,
    uint32_t head_dim,
    float* q_buf,
    float* gate_buf) {
  UnpackQGKernel<<<num_heads, head_dim, 0, stream>>>(qg, num_heads, head_dim, q_buf, gate_buf);
}

// ---------------------------------------------------------------------------
// 6. Online Softmax Attention Decode Kernel
// ---------------------------------------------------------------------------

__device__ __forceinline__ float Sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void AttentionDecodeKernel(
    const float* __restrict__ q_buf,
    const float* __restrict__ gate_buf,
    const float* __restrict__ kv_k,
    const float* __restrict__ kv_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos,
    float* __restrict__ context_out) {
  const uint32_t h = blockIdx.x;
  if (h >= num_heads) return;
  const uint32_t group_size = num_heads / num_kv_heads;
  const uint32_t kv_h = h / group_size;
  const uint32_t seq_len = pos + 1;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  const float* q_h = q_buf + (h * head_dim);
  const float* k_base = kv_k + (static_cast<size_t>(kv_h) * max_context * head_dim);
  const float* v_base = kv_v + (static_cast<size_t>(kv_h) * max_context * head_dim);
  const float* g_h = gate_buf + (h * head_dim);
  float* out_h = context_out + (h * head_dim);

  const uint32_t d = threadIdx.x; // 0..255 (head_dim)
  const float q_val = (d < head_dim) ? q_h[d] : 0.0f;

  float m = -1e30f;
  float l = 0.0f;
  float acc = 0.0f;

  for (uint32_t p = 0; p < seq_len; ++p) {
    const float* kp = k_base + (static_cast<size_t>(p) * head_dim);
    const float* vp = v_base + (static_cast<size_t>(p) * head_dim);

    float prod = (d < head_dim) ? (q_val * kp[d]) : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      prod += __shfl_xor(prod, offset);
    }

    __shared__ float s_warp_dot[8];
    const uint32_t lane = d & 31u;
    const uint32_t warp = d >> 5u;
    if (lane == 0) {
      s_warp_dot[warp] = prod;
    }
    __syncthreads();

    float s_dot = 0.0f;
    if (warp == 0) {
      s_dot = (lane < 8) ? s_warp_dot[lane] : 0.0f;
      for (int offset = 4; offset > 0; offset >>= 1) {
        s_dot += __shfl_xor(s_dot, offset);
      }
      if (lane == 0) {
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
      acc = (acc * alpha) + (beta * vp[d]);
    }
    m = m_new;
  }

  if (d < head_dim) {
    const float ctx = (l > 0.0f) ? (acc / l) : 0.0f;
    out_h[d] = ctx * Sigmoidf(g_h[d]);
  }
}

void DispatchAttentionDecode(
    hipStream_t stream,
    const float* q_buf,
    const float* gate_buf,
    const float* kv_k,
    const float* kv_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos,
    float* context_out) {
  AttentionDecodeKernel<<<num_heads, head_dim, 0, stream>>>(
      q_buf, gate_buf, kv_k, kv_v, num_heads, num_kv_heads, head_dim, max_context, pos, context_out);
}

// ---------------------------------------------------------------------------
// 7. KV Cache Store Kernel
// ---------------------------------------------------------------------------

__global__ void StoreKVKernel(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos,
    float* __restrict__ kv_k,
    float* __restrict__ kv_v) {
  const uint32_t kv_h = blockIdx.x;
  const uint32_t d = threadIdx.x;
  if (kv_h >= num_kv_heads || d >= head_dim) return;

  const size_t in_offset = (kv_h * head_dim) + d;
  const size_t cache_offset = (static_cast<size_t>(kv_h) * max_context * head_dim) +
                             (static_cast<size_t>(pos) * head_dim) + d;

  kv_k[cache_offset] = k_in[in_offset];
  kv_v[cache_offset] = v_in[in_offset];
}

void DispatchStoreKV(
    hipStream_t stream,
    const float* k_in,
    const float* v_in,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos,
    float* kv_k,
    float* kv_v) {
  StoreKVKernel<<<num_kv_heads, head_dim, 0, stream>>>(
      k_in, v_in, num_kv_heads, head_dim, max_context, pos, kv_k, kv_v);
}

// ---------------------------------------------------------------------------
// 8. SSM (DeltaNet) Kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ float Siluf(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float Softplusf(float x) {
  return (x > 20.0f) ? x : log1pf(expf(x));
}

__global__ void SsmAlphaBetaKernel(
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const void* __restrict__ dt_bias,
    const void* __restrict__ a_val,
    uint32_t v_heads,
    float* __restrict__ decay_out,
    float* __restrict__ beta_sig_out) {
  const uint32_t vh = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vh >= v_heads) return;

  const float dt = dt_bias ? static_cast<const float*>(dt_bias)[vh] : 0.0f;
  const float a = a_val ? static_cast<const float*>(a_val)[vh] : 0.0f;

  beta_sig_out[vh] = Sigmoidf(beta[vh]);
  const float alpha_biased = alpha[vh] + dt;
  const float sp = Softplusf(alpha_biased);
  decay_out[vh] = expf(-expf(a) * sp);
}

__global__ void SsmConv1DKernel(
    const float* __restrict__ qkv_in,
    const void* __restrict__ conv1d_w,
    uint32_t conv_channels,
    float* __restrict__ conv_state,
    float* __restrict__ conv_out) {
  const uint32_t c = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (c >= conv_channels) return;

  const float s0 = conv_state[c];
  const float s1 = conv_state[conv_channels + c];
  const float s2 = conv_state[(2 * conv_channels) + c];
  const float s3 = qkv_in[c];

  conv_state[c] = s1;
  conv_state[conv_channels + c] = s2;
  conv_state[(2 * conv_channels) + c] = s3;

  float dot = s3;
  if (conv1d_w != nullptr) {
    const auto* w = static_cast<const float*>(conv1d_w);
    dot = (s0 * w[c]) +
          (s1 * w[conv_channels + c]) +
          (s2 * w[(2 * conv_channels) + c]) +
          (s3 * w[(3 * conv_channels) + c]);
  }
  conv_out[c] = Siluf(dot);
}

__global__ void SsmHeadNormKernel(
    const float* __restrict__ q_raw,
    const float* __restrict__ k_raw,
    uint32_t k_heads,
    uint32_t head_dim,
    float eps,
    float* __restrict__ q_normed,
    float* __restrict__ k_normed) {
  const uint32_t kh = blockIdx.x;
  const uint32_t d = threadIdx.x;
  if (kh >= k_heads || d >= head_dim) return;

  const float* qh = q_raw + (kh * head_dim);
  const float* kh_p = k_raw + (kh * head_dim);

  float q_sq = qh[d] * qh[d];
  float k_sq = kh_p[d] * kh_p[d];

  for (int offset = 16; offset > 0; offset >>= 1) {
    q_sq += __shfl_xor(q_sq, offset);
    k_sq += __shfl_xor(k_sq, offset);
  }

  __shared__ float s_q[4], s_k[4];
  const uint32_t lane = d & 31u;
  const uint32_t warp = d >> 5u;
  if (lane == 0) {
    s_q[warp] = q_sq;
    s_k[warp] = k_sq;
  }
  __syncthreads();

  if (warp == 0) {
    float sum_q = (lane < 4) ? s_q[lane] : 0.0f;
    float sum_k = (lane < 4) ? s_k[lane] : 0.0f;
    for (int offset = 2; offset > 0; offset >>= 1) {
      sum_q += __shfl_xor(sum_q, offset);
      sum_k += __shfl_xor(sum_k, offset);
    }
    if (lane == 0) {
      s_q[0] = rsqrtf(sum_q + eps);
      s_k[0] = rsqrtf(sum_k + eps);
    }
  }
  __syncthreads();

  q_normed[(kh * head_dim) + d] = qh[d] * s_q[0];
  k_normed[(kh * head_dim) + d] = kh_p[d] * s_k[0];
}

__global__ void SsmRecurrenceKernel(
    const float* __restrict__ q_normed,
    const float* __restrict__ k_normed,
    const float* __restrict__ v_raw,
    const float* __restrict__ decay,
    const float* __restrict__ beta_sig,
    const float* __restrict__ ssm_gate,
    const void* __restrict__ ssm_norm_w,
    uint32_t v_heads,
    uint32_t k_heads,
    uint32_t head_dim,
    float eps,
    float* __restrict__ deltanet_state,
    float* __restrict__ core_out) {
  const uint32_t vh = blockIdx.x;
  if (vh >= v_heads) return;
  const uint32_t heads_ratio = v_heads / k_heads;
  const uint32_t kh = vh / heads_ratio;

  const float* qh = q_normed + (kh * head_dim);
  const float* kh_p = k_normed + (kh * head_dim);
  const float* vh_p = v_raw + (vh * head_dim);
  const float d_val = decay[vh];
  const float b_val = beta_sig[vh];

  float* S_mat = deltanet_state + (static_cast<size_t>(vh) * head_dim * head_dim);
  const uint32_t c = threadIdx.x; // 0..127

  // Update column c of state matrix S: S[r, c] = S[r, c] * decay + k[r] * v[c]
  // and compute dot product with q: core[c] = beta * sum_r(S[r, c] * q[r])
  const float vc = vh_p[c];
  float readout = 0.0f;

  #pragma unroll
  for (uint32_t r = 0; r < 128; ++r) {
    float* entry = S_mat + (r * 128) + c;
    const float updated = (*entry * d_val) + (kh_p[r] * vc);
    *entry = updated;
    readout += updated * qh[r];
  }

  __shared__ float s_core[128];
  s_core[c] = readout * b_val;
  __syncthreads();

  // RMSNorm on s_core
  float sq = s_core[c] * s_core[c];
  for (int offset = 16; offset > 0; offset >>= 1) {
    sq += __shfl_xor(sq, offset);
  }
  __shared__ float s_sq_warp[4];
  const uint32_t lane = c & 31u;
  const uint32_t warp = c >> 5u;
  if (lane == 0) {
    s_sq_warp[warp] = sq;
  }
  __syncthreads();

  if (warp == 0) {
    float tot_sq = (lane < 4) ? s_sq_warp[lane] : 0.0f;
    for (int offset = 2; offset > 0; offset >>= 1) {
      tot_sq += __shfl_xor(tot_sq, offset);
    }
    if (lane == 0) {
      s_sq_warp[0] = rsqrtf((tot_sq / 128.0f) + eps);
    }
  }
  __syncthreads();

  const float norm_scale = s_sq_warp[0];
  const float nw = ssm_norm_w ? static_cast<const float*>(ssm_norm_w)[c] : 1.0f;
  const float normed_val = s_core[c] * norm_scale * nw;
  const float z_val = ssm_gate[(vh * head_dim) + c];

  core_out[(vh * head_dim) + c] = normed_val * Siluf(z_val);
}

void DispatchSsmStep(
    hipStream_t stream,
    const float* ssm_qkv,
    const float* ssm_gate,
    const float* ssm_alpha,
    const float* ssm_beta,
    const void* dt_bias,
    const void* a_val,
    const void* conv1d_w,
    const void* ssm_norm_w,
    uint32_t conv_channels,
    uint32_t v_heads,
    uint32_t k_heads,
    uint32_t head_dim,
    float eps,
    float* conv_state,
    float* deltanet_state,
    float* decay_scratch,
    float* beta_sig_scratch,
    float* conv_out_scratch,
    float* q_normed_scratch,
    float* k_normed_scratch,
    float* core_out) {
  // 1. Alpha & Beta
  SsmAlphaBetaKernel<<<1, v_heads, 0, stream>>>(
      ssm_alpha, ssm_beta, dt_bias, a_val, v_heads, decay_scratch, beta_sig_scratch);

  // 2. Conv1D
  const uint32_t conv_block = 256;
  const uint32_t conv_grid = (conv_channels + conv_block - 1) / conv_block;
  SsmConv1DKernel<<<conv_grid, conv_block, 0, stream>>>(
      ssm_qkv, conv1d_w, conv_channels, conv_state, conv_out_scratch);

  // 3. Head L2-norm
  const float* q_raw = conv_out_scratch;
  const float* k_raw = conv_out_scratch + (k_heads * head_dim);
  const float* v_raw = conv_out_scratch + (2 * k_heads * head_dim);
  SsmHeadNormKernel<<<k_heads, head_dim, 0, stream>>>(
      q_raw, k_raw, k_heads, head_dim, eps, q_normed_scratch, k_normed_scratch);

  // 4. Recurrence
  SsmRecurrenceKernel<<<v_heads, head_dim, 0, stream>>>(
      q_normed_scratch, k_normed_scratch, v_raw, decay_scratch, beta_sig_scratch,
      ssm_gate, ssm_norm_w, v_heads, k_heads, head_dim, eps, deltanet_state, core_out);
}

// ---------------------------------------------------------------------------
// 9. MoE Elementwise & Routing Kernels
// ---------------------------------------------------------------------------

__global__ void SiluMulKernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    uint32_t size,
    float* __restrict__ act) {
  const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= size) return;
  act[i] = Siluf(gate[i]) * up[i];
}

void DispatchSiluMul(hipStream_t stream, const float* gate, const float* up, uint32_t size, float* act) {
  const uint32_t block_dim = 256;
  const uint32_t grid_dim = (size + block_dim - 1) / block_dim;
  SiluMulKernel<<<grid_dim, block_dim, 0, stream>>>(gate, up, size, act);
}

__global__ void AccumWeightedKernel(
    const float* __restrict__ src,
    float weight,
    uint32_t size,
    float* __restrict__ dst) {
  const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= size) return;
  dst[i] += weight * src[i];
}

void DispatchAccumWeighted(hipStream_t stream, const float* src, float weight, uint32_t size, float* dst) {
  const uint32_t block_dim = 256;
  const uint32_t grid_dim = (size + block_dim - 1) / block_dim;
  AccumWeightedKernel<<<grid_dim, block_dim, 0, stream>>>(src, weight, size, dst);
}

__global__ void MoECombineSharedKernel(
    const float* __restrict__ routed,
    const float* __restrict__ shexp,
    float shexp_gate_dot,
    uint32_t size,
    float* __restrict__ out) {
  const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= size) return;
  out[i] = routed[i] + (Sigmoidf(shexp_gate_dot) * shexp[i]);
}

void DispatchMoECombineShared(
    hipStream_t stream,
    const float* routed,
    const float* shexp,
    float shexp_gate_dot,
    uint32_t size,
    float* out) {
  const uint32_t block_dim = 256;
  const uint32_t grid_dim = (size + block_dim - 1) / block_dim;
  MoECombineSharedKernel<<<grid_dim, block_dim, 0, stream>>>(routed, shexp, shexp_gate_dot, size, out);
}

__global__ void ResidualAddKernel(const float* __restrict__ delta, uint32_t size, float* __restrict__ hidden) {
  const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= size) return;
  hidden[i] += delta[i];
}

void DispatchResidualAdd(hipStream_t stream, const float* delta, uint32_t size, float* hidden) {
  const uint32_t block_dim = 256;
  const uint32_t grid_dim = (size + block_dim - 1) / block_dim;
  ResidualAddKernel<<<grid_dim, block_dim, 0, stream>>>(delta, size, hidden);
}

}  // namespace gufo::models::qwen35_moe::hip
#endif  // defined(ENGINE_ENABLE_HIP)
