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

// Pure ALU Codebook10 decoding (Zero memory lookup, 100% vectorizable)
__device__ __forceinline__ int8_t Rocmfp4DecodeNibble(uint8_t q) {
  const int mag3 = q & 0x07u;
  const int mag = (mag3 <= 4) ? mag3 : (2 * mag3 - 4);
  return (q & 0x08u) ? -mag : mag;
}

__device__ __forceinline__ float DecodeQ6_K(const quant::block_q6_K& block, uint32_t index) {
  const uint32_t half = index / 128;
  const uint32_t within_half = index % 128;
  const uint32_t segment = within_half / 32;
  const uint32_t lane = within_half % 32;
  const uint32_t ql_base = half * 64;
  const uint8_t qh = block.qh[(half * 32) + lane];

  uint8_t low = 0;
  uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0Fu;
      high = qh & 0x03u;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0Fu;
      high = (qh >> 2u) & 0x03u;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4u;
      high = (qh >> 4u) & 0x03u;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4u;
      high = (qh >> 6u) & 0x03u;
      break;
  }

  const uint32_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const int8_t quant = static_cast<int8_t>(static_cast<int>((high << 4u) | low) - 32);
  const float d = __half2float(*reinterpret_cast<const __half*>(&block.d));
  return d * static_cast<float>(block.scales[scale_index]) * static_cast<float>(quant);
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
  } else if (type == core::GgmlType::kQ6_K) {
    const auto* blocks = reinterpret_cast<const quant::block_q6_K*>(row_base);
    const uint32_t b = tid / 256;
    const uint32_t i = tid % 256;
    out[tid] = DecodeQ6_K(blocks[b], i);
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
  const uint32_t half_rot = rotary_dim / 2;
  if (pair_idx >= half_rot) return;

  const float freq = 1.0f / powf(theta_base, static_cast<float>(2 * pair_idx) / static_cast<float>(rotary_dim));
  const float theta = static_cast<float>(pos) * freq;
  const float cos_th = cosf(theta);
  const float sin_th = sinf(theta);

  float* head_ptr = vec + (h * head_dim);
  const float x0 = head_ptr[pair_idx];
  const float x1 = head_ptr[pair_idx + half_rot];
  head_ptr[pair_idx] = (x0 * cos_th) - (x1 * sin_th);
  head_ptr[pair_idx + half_rot] = (x0 * sin_th) + (x1 * cos_th);
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
  decay_out[vh] = expf(a * sp);
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
    const auto* wc = w + (c * 4);
    dot = (s0 * wc[0]) +
          (s1 * wc[1]) +
          (s2 * wc[2]) +
          (s3 * wc[3]);
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
      s_q[0] = 1.0f / fmaxf(sqrtf(sum_q), eps);
      s_k[0] = 1.0f / fmaxf(sqrtf(sum_k), eps);
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
  const uint32_t kh = vh % k_heads;

  const float* qh = q_normed + (kh * head_dim);
  const float* kh_p = k_normed + (kh * head_dim);
  const float* vh_p = v_raw + (vh * head_dim);
  const float d_val = decay[vh];
  const float b_val = beta_sig[vh];

  const uint32_t c = threadIdx.x; // 0..127

  __shared__ float s_q[128];
  __shared__ float s_k[128];
  s_q[c] = qh[c];
  s_k[c] = kh_p[c];
  __syncthreads();

  float* S_col = deltanet_state + (static_cast<size_t>(vh) * head_dim * head_dim) + c;

  // Step 1: kv_col = (S^T @ k)[c] = sum_r S[r, c] * k[r]
  float kv_col = 0.0f;
  #pragma unroll
  for (uint32_t r = 0; r < 128; ++r) {
    kv_col += S_col[r * 128] * s_k[r];
  }

  // Step 2: delta_col = (v[c] - decay * kv_col) * beta
  const float delta_col = (vh_p[c] - (d_val * kv_col)) * b_val;

  // Step 3: S[r, c] = decay * S[r, c] + delta_col * k[r], and readout = sum_r S[r, c] * q[r]
  float readout = 0.0f;
  #pragma unroll
  for (uint32_t r = 0; r < 128; ++r) {
    float* entry = S_col + (r * 128);
    const float updated = (*entry * d_val) + (s_k[r] * delta_col);
    *entry = updated;
    readout += updated * s_q[r];
  }

  const float scale = 0.08838834764831845f; // 1.0f / sqrtf(128.0f)
  __shared__ float s_core[128];
  s_core[c] = readout * scale;
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

// ---------------------------------------------------------------------------
// 10. Batched Prefill 2D-Tiled GEMM Kernels (Wave32 x Batch Grid)
// ---------------------------------------------------------------------------

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemmTileRocmFp4FastKernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t = blockIdx.y;
  if (m >= M || t >= B) return;

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);
  const uint8_t* base = static_cast<const uint8_t*>(A_data) +
                        (static_cast<size_t>(expert_id) * expert_stride) +
                        (static_cast<size_t>(m) * row_bytes);
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(base);
  const float* xt = x + (static_cast<size_t>(t) * K);

  float sum = 0.0f;
  for (uint32_t b = lane_id; b < num_blocks; b += 32) {
    const auto& blk = blocks[b];
    const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
    const float* xb = xt + (static_cast<size_t>(b) * 32);

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte_val = blk.qs[j];
      const float w0 = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * scale;
      const float w1 = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * scale;
      sum += (w0 * xb[j]) + (w1 * xb[j + 16]);
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[(static_cast<size_t>(t) * M) + m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemmTileRocmFp4Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t = blockIdx.y;
  if (m >= M || t >= B) return;

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4);
  const uint8_t* base = static_cast<const uint8_t*>(A_data) +
                        (static_cast<size_t>(expert_id) * expert_stride) +
                        (static_cast<size_t>(m) * row_bytes);
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4*>(base);
  const float* xt = x + (static_cast<size_t>(t) * K);

  float sum = 0.0f;
  for (uint32_t b = lane_id; b < num_blocks; b += 32) {
    const auto& blk = blocks[b];
    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(blk.e[0]);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(blk.e[1]);
    const float* xb = xt + (static_cast<size_t>(b) * 32);

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte_val = blk.qs[j];
      const float w0 = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * d0;
      const float w1 = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * d1;
      sum += (w0 * xb[j]) + (w1 * xb[j + 16]);
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[(static_cast<size_t>(t) * M) + m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemmTileBF16Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t = blockIdx.y;
  if (m >= M || t >= B) return;

  const uint8_t* base = static_cast<const uint8_t*>(A_data) +
                        (static_cast<size_t>(expert_id) * expert_stride) +
                        (static_cast<size_t>(m) * K * sizeof(hip_bfloat16));
  const auto* A_row = reinterpret_cast<const hip_bfloat16*>(base);
  const float* xt = x + (static_cast<size_t>(t) * K);

  float sum = 0.0f;
  for (uint32_t k = lane_id; k < K; k += 32) {
    sum += static_cast<float>(A_row[k]) * xt[k];
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[(static_cast<size_t>(t) * M) + m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemmTileF16Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t = blockIdx.y;
  if (m >= M || t >= B) return;

  const uint8_t* base = static_cast<const uint8_t*>(A_data) +
                        (static_cast<size_t>(expert_id) * expert_stride) +
                        (static_cast<size_t>(m) * K * sizeof(__half));
  const auto* A_row = reinterpret_cast<const __half*>(base);
  const float* xt = x + (static_cast<size_t>(t) * K);

  float sum = 0.0f;
  for (uint32_t k = lane_id; k < K; k += 32) {
    sum += __half2float(A_row[k]) * xt[k];
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

  if (lane_id == 0) {
    y[(static_cast<size_t>(t) * M) + m] = sum;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemmTileQ8_0Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t = blockIdx.y;
  if (m >= M || t >= B) return;

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_q8_0);
  const uint8_t* base = static_cast<const uint8_t*>(A_data) +
                        (static_cast<size_t>(expert_id) * expert_stride) +
                        (static_cast<size_t>(m) * row_bytes);
  const auto* blocks = reinterpret_cast<const quant::block_q8_0*>(base);
  const float* xt = x + (static_cast<size_t>(t) * K);

  float sum = 0.0f;
  for (uint32_t b = lane_id; b < num_blocks; b += 32) {
    const auto& blk = blocks[b];
    const float d = __half2float(blk.d);
    const float* xb = xt + (static_cast<size_t>(b) * 32);

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
    y[(static_cast<size_t>(t) * M) + m] = sum;
  }
}

template <uint32_t WavesPerBlock, uint32_t T_BATCH = 32>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemmLdsRocmFp4FastKernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B,
    uint64_t expert_stride,
    uint32_t expert_id) {
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
                                (static_cast<size_t>(expert_id) * expert_stride) +
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

template <uint32_t WavesPerBlock, uint32_t T_BATCH = 32>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void GemmLdsRocmFp4Kernel(
    const void* __restrict__ A_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t M,
    uint32_t K,
    uint32_t B,
    uint64_t expert_stride,
    uint32_t expert_id) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const uint32_t t_base = blockIdx.y * T_BATCH;
  if (t_base >= B) return;
  const uint32_t cur_batch = min(T_BATCH, B - t_base);

  const uint32_t num_blocks = K / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4);
  const uint8_t* base = m < M ? (static_cast<const uint8_t*>(A_data) +
                                (static_cast<size_t>(expert_id) * expert_stride) +
                                (static_cast<size_t>(m) * row_bytes)) : nullptr;
  const auto* blocks = reinterpret_cast<const quant::block_rocmfp4*>(base);

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
        const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(blk.e[0]);
        const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(blk.e[1]);
        const uint8_t byte_val = blk.qs[j];
        const float w0 = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * d0;
        const float w1 = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * d1;

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

void DispatchGemm(
    hipStream_t stream,
    const void* A_data,
    core::GgmlType A_type,
    uint64_t expert_stride,
    uint32_t expert_id,
    const float* x,
    float* y,
    uint32_t M,
    uint32_t K,
    uint32_t B) {
  if (B == 1) {
    DispatchGemv(stream, A_data, A_type, expert_stride, expert_id, x, y, M, K);
    return;
  }
  if (M == 0 || K == 0 || B == 0 || A_data == nullptr || x == nullptr || y == nullptr) {
    return;
  }

  constexpr uint32_t kWaves = 8;
  const dim3 block_dim(kWaves * 32, 1);

  switch (A_type) {
    case core::GgmlType::kQ4_0_ROCMFP4_FAST: {
      constexpr uint32_t T_BATCH = 32;
      const dim3 grid_dim_t((M + kWaves - 1) / kWaves, (B + T_BATCH - 1) / T_BATCH);
      GemmLdsRocmFp4FastKernel<kWaves, T_BATCH><<<grid_dim_t, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      break;
    }
    case core::GgmlType::kQ4_0_ROCMFP4: {
      constexpr uint32_t T_BATCH = 32;
      const dim3 grid_dim_t((M + kWaves - 1) / kWaves, (B + T_BATCH - 1) / T_BATCH);
      GemmLdsRocmFp4Kernel<kWaves, T_BATCH><<<grid_dim_t, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      break;
    }
    case core::GgmlType::kBF16: {
      const dim3 grid_dim((M + kWaves - 1) / kWaves, B);
      GemmTileBF16Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      break;
    }
    case core::GgmlType::kF16: {
      const dim3 grid_dim((M + kWaves - 1) / kWaves, B);
      GemmTileF16Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      break;
    }
    case core::GgmlType::kQ8_0: {
      const dim3 grid_dim((M + kWaves - 1) / kWaves, B);
      GemmTileQ8_0Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      break;
    }
    default:
      for (uint32_t t = 0; t < B; ++t) {
        DispatchGemv(stream, A_data, A_type, expert_stride, expert_id,
                     x + (static_cast<size_t>(t) * K),
                     y + (static_cast<size_t>(t) * M),
                     M, K);
      }
      break;
  }
}

__global__ void RMSNormBatchKernel(
    const float* __restrict__ x,
    const void* __restrict__ weight,
    core::GgmlType weight_type,
    float eps,
    uint32_t size,
    float* __restrict__ out) {
  const uint32_t tok = blockIdx.y;
  const float* x_tok = x + (static_cast<size_t>(tok) * size);
  float* out_tok = out + (static_cast<size_t>(tok) * size);

  const uint32_t tid = threadIdx.x;
  float sum_sq = 0.0f;
  for (uint32_t i = tid; i < size; i += blockDim.x) {
    const float val = x_tok[i];
    sum_sq += val * val;
  }

  // Wave reduce
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor(sum_sq, offset);
  }

  __shared__ float s_warp[8];
  const uint32_t warp_id = tid >> 5u;
  const uint32_t lane_id = tid & 31u;
  if (lane_id == 0) {
    s_warp[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float total = (lane_id < 8) ? s_warp[lane_id] : 0.0f;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      total += __shfl_xor(total, offset);
    }
    if (lane_id == 0) {
      s_warp[0] = rsqrtf((total / static_cast<float>(size)) + eps);
    }
  }
  __syncthreads();

  const float inv_rms = s_warp[0];

  if (weight_type == core::GgmlType::kF32) {
    const float* w = reinterpret_cast<const float*>(weight);
    for (uint32_t i = tid; i < size; i += blockDim.x) {
      out_tok[i] = x_tok[i] * inv_rms * w[i];
    }
  } else if (weight_type == core::GgmlType::kBF16) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(weight);
    for (uint32_t i = tid; i < size; i += blockDim.x) {
      const uint32_t u32 = static_cast<uint32_t>(w[i]) << 16u;
      float val = 0.0f;
      __builtin_memcpy(&val, &u32, sizeof(float));
      out_tok[i] = x_tok[i] * inv_rms * val;
    }
  } else if (weight_type == core::GgmlType::kF16) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(weight);
    for (uint32_t i = tid; i < size; i += blockDim.x) {
      const float val = __half2float(*reinterpret_cast<const __half*>(&w[i]));
      out_tok[i] = x_tok[i] * inv_rms * val;
    }
  }
}

void DispatchRMSNormBatch(
    hipStream_t stream,
    const float* x,
    const void* weight_data,
    core::GgmlType weight_type,
    float eps,
    uint32_t size,
    uint32_t B,
    float* out) {
  if (B == 1) {
    DispatchRMSNorm(stream, x, weight_data, weight_type, eps, size, out);
    return;
  }
  dim3 grid(1, B);
  RMSNormBatchKernel<<<grid, 256, 0, stream>>>(x, weight_data, weight_type, eps, size, out);
}

__global__ void EmbeddingBatchKernel(
    const void* __restrict__ table,
    core::GgmlType type,
    const uint32_t* __restrict__ token_ids,
    uint32_t hidden_size,
    size_t row_bytes,
    float* __restrict__ out) {
  const uint32_t tok = blockIdx.y;
  const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= hidden_size) return;

  const uint32_t token_id = token_ids[tok];
  const uint8_t* row_base = static_cast<const uint8_t*>(table) + (static_cast<size_t>(token_id) * row_bytes);
  float* out_tok = out + (static_cast<size_t>(tok) * hidden_size);

  if (type == core::GgmlType::kBF16) {
    const uint16_t* row = reinterpret_cast<const uint16_t*>(row_base);
    const uint32_t u32 = static_cast<uint32_t>(row[tid]) << 16u;
    float val = 0.0f;
    __builtin_memcpy(&val, &u32, sizeof(float));
    out_tok[tid] = val;
  } else if (type == core::GgmlType::kF16) {
    const uint16_t* row = reinterpret_cast<const uint16_t*>(row_base);
    out_tok[tid] = __half2float(*reinterpret_cast<const __half*>(&row[tid]));
  } else if (type == core::GgmlType::kF32) {
    const float* row = reinterpret_cast<const float*>(row_base);
    out_tok[tid] = row[tid];
  } else if (type == core::GgmlType::kQ6_K) {
    const auto* blocks = reinterpret_cast<const quant::block_q6_K*>(row_base);
    const uint32_t b = tid / 256;
    const uint32_t i = tid % 256;
    out_tok[tid] = DecodeQ6_K(blocks[b], i);
  } else if (type == core::GgmlType::kQ4_0_ROCMFP4_FAST) {
    const auto* blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(row_base);
    const uint32_t b = tid / 32;
    const uint32_t j = tid % 32;
    const auto& blk = blocks[b];
    const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);
    const uint8_t byte_val = blk.qs[j < 16 ? j : (j - 16)];
    const uint8_t nibble = j < 16 ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    out_tok[tid] = static_cast<float>(Rocmfp4DecodeNibble(nibble)) * scale;
  } else if (type == core::GgmlType::kQ8_0) {
    const auto* blocks = reinterpret_cast<const quant::block_q8_0*>(row_base);
    const uint32_t b = tid / 32;
    const uint32_t j = tid % 32;
    const auto& blk = blocks[b];
    const __half h = *reinterpret_cast<const __half*>(&blk.d);
    out_tok[tid] = static_cast<float>(blk.qs[j]) * __half2float(h);
  }
}

void DispatchEmbeddingBatch(
    hipStream_t stream,
    const void* table,
    core::GgmlType type,
    const uint32_t* token_ids,
    uint32_t B,
    uint32_t hidden_size,
    size_t row_bytes,
    float* out) {
  const uint32_t block_dim = 256;
  const uint32_t grid_x = (hidden_size + block_dim - 1) / block_dim;
  dim3 grid(grid_x, B);
  EmbeddingBatchKernel<<<grid, block_dim, 0, stream>>>(
      table, type, token_ids, hidden_size, row_bytes, out);
}

__global__ void RoPEBatchKernel(
    float* vec,
    uint32_t rotary_dim,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t pos0,
    float theta_base) {
  const uint32_t tok = blockIdx.z;
  const uint32_t h = blockIdx.y;
  const uint32_t pair_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint32_t half_rot = rotary_dim / 2;
  if (pair_idx >= half_rot) return;

  const uint32_t pos = pos0 + tok;
  const float freq = 1.0f / powf(theta_base, static_cast<float>(2 * pair_idx) / static_cast<float>(rotary_dim));
  const float theta = static_cast<float>(pos) * freq;
  const float cos_th = cosf(theta);
  const float sin_th = sinf(theta);

  float* head_vec = vec + (static_cast<size_t>(tok) * num_heads * head_dim) + (static_cast<size_t>(h) * head_dim);
  const float x0 = head_vec[pair_idx];
  const float x1 = head_vec[pair_idx + half_rot];

  head_vec[pair_idx] = (x0 * cos_th) - (x1 * sin_th);
  head_vec[pair_idx + half_rot] = (x0 * sin_th) + (x1 * cos_th);
}

void DispatchRoPEBatch(
    hipStream_t stream,
    float* vec,
    uint32_t rotary_dim,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t pos0,
    uint32_t B,
    float theta_base) {
  const uint32_t pairs = rotary_dim / 2;
  const uint32_t block_dim = 128;
  const uint32_t grid_x = (pairs + block_dim - 1) / block_dim;
  dim3 grid(grid_x, num_heads, B);
  RoPEBatchKernel<<<grid, block_dim, 0, stream>>>(
      vec, rotary_dim, head_dim, num_heads, pos0, theta_base);
}

__global__ void UnpackQGBatchKernel(
    const float* __restrict__ qg,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t B,
    float* __restrict__ q_buf,
    float* __restrict__ gate_buf) {
  const uint32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint32_t total = B * num_heads * head_dim;
  if (idx >= total) return;

  const uint32_t tok = idx / (num_heads * head_dim);
  const uint32_t rem = idx % (num_heads * head_dim);
  const uint32_t h = rem / head_dim;
  const uint32_t d = rem % head_dim;

  const size_t qg_head_base = (static_cast<size_t>(tok) * num_heads * 2 * head_dim) + (static_cast<size_t>(h) * 2 * head_dim);

  q_buf[idx] = qg[qg_head_base + d];
  gate_buf[idx] = qg[qg_head_base + head_dim + d];
}

void DispatchUnpackQGBatch(
    hipStream_t stream,
    const float* qg,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t B,
    float* q_buf,
    float* gate_buf) {
  const uint32_t total = B * num_heads * head_dim;
  const uint32_t block_dim = 256;
  const uint32_t grid_dim = (total + block_dim - 1) / block_dim;
  UnpackQGBatchKernel<<<grid_dim, block_dim, 0, stream>>>(
      qg, num_heads, head_dim, B, q_buf, gate_buf);
}

__global__ void StoreKVBatchKernel(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos0,
    float* __restrict__ kv_k,
    float* __restrict__ kv_v) {
  const uint32_t tok = blockIdx.y;
  const uint32_t kv_h = blockIdx.x;
  const uint32_t d = threadIdx.x;
  if (kv_h >= num_kv_heads || d >= head_dim) return;

  const uint32_t pos = pos0 + tok;
  const size_t in_offset = (static_cast<size_t>(tok) * num_kv_heads * head_dim) + (kv_h * head_dim) + d;
  const size_t cache_offset = (static_cast<size_t>(kv_h) * max_context * head_dim) +
                             (static_cast<size_t>(pos) * head_dim) + d;

  kv_k[cache_offset] = k_in[in_offset];
  kv_v[cache_offset] = v_in[in_offset];
}

void DispatchStoreKVBatch(
    hipStream_t stream,
    const float* k_in,
    const float* v_in,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos0,
    uint32_t B,
    float* kv_k,
    float* kv_v) {
  dim3 grid(num_kv_heads, B);
  StoreKVBatchKernel<<<grid, head_dim, 0, stream>>>(
      k_in, v_in, num_kv_heads, head_dim, max_context, pos0, kv_k, kv_v);
}

__global__ void AttentionPrefillKernel(
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

void DispatchAttentionPrefill(
    hipStream_t stream,
    const float* q_buf,
    const float* gate_buf,
    const float* kv_k,
    const float* kv_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos0,
    uint32_t B,
    float* context_out) {
  dim3 grid(num_heads, B);
  AttentionPrefillKernel<<<grid, head_dim, 0, stream>>>(
      q_buf, gate_buf, kv_k, kv_v, num_heads, num_kv_heads, head_dim, max_context, pos0, context_out);
}

__global__ void MoECombineSharedBatchKernel(
    const float* __restrict__ routed,
    const float* __restrict__ shexp,
    const float* __restrict__ d_shexp_gate_dot,
    uint32_t H,
    uint32_t B,
    float* __restrict__ out) {
  const uint32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint32_t total = B * H;
  if (idx >= total) return;

  const uint32_t tok = idx / H;
  const float gate_dot = d_shexp_gate_dot ? d_shexp_gate_dot[tok] : 0.0f;
  const float scale = Sigmoidf(gate_dot);

  out[idx] = routed[idx] + (scale * shexp[idx]);
}

void DispatchMoECombineSharedBatch(
    hipStream_t stream,
    const float* routed,
    const float* shexp,
    const float* d_shexp_gate_dot,
    uint32_t H,
    uint32_t B,
    float* out) {
  const uint32_t total = B * H;
  const uint32_t block_dim = 256;
  const uint32_t grid_dim = (total + block_dim - 1) / block_dim;
  MoECombineSharedBatchKernel<<<grid_dim, block_dim, 0, stream>>>(
      routed, shexp, d_shexp_gate_dot, H, B, out);
}

// ---------------------------------------------------------------------------
// 11. Batched MoE Kernels (All tokens, all experts in 2 GPU kernel launches)
// ---------------------------------------------------------------------------

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void MoeGatedSwigluFastKernel(
    const void* __restrict__ gate_data,
    const void* __restrict__ up_data,
    uint64_t expert_stride_gate,
    uint64_t expert_stride_up,
    const float* __restrict__ x,
    const uint32_t* __restrict__ expert_ids,
    uint32_t B,
    uint32_t num_experts_used,
    uint32_t expert_ff,
    uint32_t H,
    float* __restrict__ act_out) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t global_warp_id = (blockIdx.x * WavesPerBlock) + warp_id;

  const uint32_t total_rows = B * num_experts_used * expert_ff;
  if (global_warp_id >= total_rows) return;

  const uint32_t m = global_warp_id % expert_ff;
  const uint32_t slot = global_warp_id / expert_ff;
  const uint32_t t = slot / num_experts_used;
  const uint32_t expert_id = expert_ids[slot];

  const uint32_t num_blocks = H / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);

  const uint8_t* gate_base = static_cast<const uint8_t*>(gate_data) +
                             (static_cast<size_t>(expert_id) * expert_stride_gate) +
                             (static_cast<size_t>(m) * row_bytes);
  const auto* gate_blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(gate_base);

  const uint8_t* up_base = static_cast<const uint8_t*>(up_data) +
                           (static_cast<size_t>(expert_id) * expert_stride_up) +
                           (static_cast<size_t>(m) * row_bytes);
  const auto* up_blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(up_base);

  const float* x_token = x + (static_cast<size_t>(t) * H);

  float sum_gate = 0.0f;
  float sum_up = 0.0f;

  for (uint32_t b = lane_id; b < num_blocks; b += 32) {
    const float* xb = x_token + (static_cast<size_t>(b) * 32);

    // Gate
    const auto& g_blk = gate_blocks[b];
    const float g_scale = rocmfp4_ue4m3_to_fp32_half_finite(g_blk.e);

    // Up
    const auto& u_blk = up_blocks[b];
    const float u_scale = rocmfp4_ue4m3_to_fp32_half_finite(u_blk.e);

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float x0 = xb[j];
      const float x1 = xb[j + 16];

      const uint8_t g_byte = g_blk.qs[j];
      const float g_w0 = static_cast<float>(Rocmfp4DecodeNibble(g_byte & 0x0Fu)) * g_scale;
      const float g_w1 = static_cast<float>(Rocmfp4DecodeNibble(g_byte >> 4u)) * g_scale;
      sum_gate += g_w0 * x0 + g_w1 * x1;

      const uint8_t u_byte = u_blk.qs[j];
      const float u_w0 = static_cast<float>(Rocmfp4DecodeNibble(u_byte & 0x0Fu)) * u_scale;
      const float u_w1 = static_cast<float>(Rocmfp4DecodeNibble(u_byte >> 4u)) * u_scale;
      sum_up += u_w0 * x0 + u_w1 * x1;
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_gate += __shfl_xor(sum_gate, offset);
    sum_up += __shfl_xor(sum_up, offset);
  }

  if (lane_id == 0) {
    const float silu_gate = sum_gate / (1.0f + __expf(-sum_gate));
    act_out[(static_cast<size_t>(slot) * expert_ff) + m] = silu_gate * sum_up;
  }
}

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void MoeDownAccumFastKernel(
    const void* __restrict__ down_data,
    uint64_t expert_stride_down,
    const float* __restrict__ act_in,
    const float* __restrict__ expert_weights,
    const uint32_t* __restrict__ expert_ids,
    uint32_t B,
    uint32_t num_experts_used,
    uint32_t expert_ff,
    uint32_t H,
    float* __restrict__ routed_accum) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid & 31u;
  const uint32_t warp_id = tid >> 5u;
  const uint32_t global_warp_id = (blockIdx.x * WavesPerBlock) + warp_id;

  const uint32_t total_outputs = B * H;
  if (global_warp_id >= total_outputs) return;

  const uint32_t h = global_warp_id % H;
  const uint32_t t = global_warp_id / H;

  const uint32_t num_blocks = expert_ff / 32;
  const size_t row_bytes = num_blocks * sizeof(quant::block_rocmfp4_fast);

  float total_accum = 0.0f;

  for (uint32_t k = 0; k < num_experts_used; ++k) {
    const uint32_t slot = (t * num_experts_used) + k;
    const float weight = expert_weights[slot];
    if (weight == 0.0f) continue;

    const uint32_t expert_id = expert_ids[slot];
    const float* act_slot = act_in + (static_cast<size_t>(slot) * expert_ff);

    const uint8_t* down_base = static_cast<const uint8_t*>(down_data) +
                               (static_cast<size_t>(expert_id) * expert_stride_down) +
                               (static_cast<size_t>(h) * row_bytes);
    const auto* down_blocks = reinterpret_cast<const quant::block_rocmfp4_fast*>(down_base);

    float sum_k = 0.0f;
    for (uint32_t b = lane_id; b < num_blocks; b += 32) {
      const float* act_b = act_slot + (static_cast<size_t>(b) * 32);
      const auto& blk = down_blocks[b];
      const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk.e);

      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        const uint8_t byte_val = blk.qs[j];
        const float w0 = static_cast<float>(Rocmfp4DecodeNibble(byte_val & 0x0Fu)) * scale;
        const float w1 = static_cast<float>(Rocmfp4DecodeNibble(byte_val >> 4u)) * scale;
        sum_k += w0 * act_b[j] + w1 * act_b[j + 16];
      }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum_k += __shfl_xor(sum_k, offset);
    }

    total_accum += weight * sum_k;
  }

  if (lane_id == 0) {
    routed_accum[(static_cast<size_t>(t) * H) + h] = total_accum;
  }
}

void DispatchMoeGatedSwigluBatch(
    hipStream_t stream,
    const void* gate_data,
    const void* up_data,
    core::GgmlType exp_type,
    uint64_t expert_stride_gate,
    uint64_t expert_stride_up,
    const float* x,
    const uint32_t* expert_ids,
    uint32_t B,
    uint32_t num_experts_used,
    uint32_t expert_ff,
    uint32_t H,
    float* act_out) {
  constexpr uint32_t kWaves = 8;
  const uint32_t block_dim = kWaves * 32;
  const uint32_t total_rows = B * num_experts_used * expert_ff;
  const uint32_t grid_dim = (total_rows + kWaves - 1) / kWaves;

  MoeGatedSwigluFastKernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
      gate_data, up_data, expert_stride_gate, expert_stride_up,
      x, expert_ids, B, num_experts_used, expert_ff, H, act_out);
}

void DispatchMoeDownAccumBatch(
    hipStream_t stream,
    const void* down_data,
    core::GgmlType exp_type,
    uint64_t expert_stride_down,
    const float* act_in,
    const float* expert_weights,
    const uint32_t* expert_ids,
    uint32_t B,
    uint32_t num_experts_used,
    uint32_t expert_ff,
    uint32_t H,
    float* routed_accum) {
  constexpr uint32_t kWaves = 8;
  const uint32_t block_dim = kWaves * 32;
  const uint32_t total_outputs = B * H;
  const uint32_t grid_dim = (total_outputs + kWaves - 1) / kWaves;

  MoeDownAccumFastKernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
      down_data, expert_stride_down, act_in, expert_weights, expert_ids,
      B, num_experts_used, expert_ff, H, routed_accum);
}

__device__ __forceinline__ void WarpReduceMaxIdx(float& val, uint32_t& idx) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_val = __shfl_xor(val, offset);
    uint32_t other_idx = __shfl_xor(idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

__device__ __forceinline__ float WarpReduceMax(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor(val, offset));
  }
  return val;
}

__device__ __forceinline__ float WarpReduceSum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

__global__ void RouterSoftmaxTopKKernel(
    const float* __restrict__ router_logits,
    uint32_t num_experts,
    uint32_t num_experts_used,
    uint32_t B,
    uint32_t* __restrict__ expert_ids,
    float* __restrict__ expert_weights) {
  const uint32_t t = blockIdx.x;
  if (t >= B) return;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;

  __shared__ float s_warp_val[8];
  __shared__ uint32_t s_warp_idx[8];
  __shared__ float s_probs[256];
  __shared__ uint32_t s_top_ids[16];
  __shared__ float s_top_weights[16];

  const float* t_logits = router_logits + (static_cast<size_t>(t) * num_experts);
  const float logit = (tid < num_experts) ? t_logits[tid] : -1e30f;

  // 1. Find max logit across num_experts
  float max_l = logit;
  max_l = WarpReduceMax(max_l);
  if (lane == 0) {
    s_warp_val[warp] = max_l;
  }
  __syncthreads();

  if (warp == 0) {
    float m = (lane < 8) ? s_warp_val[lane] : -1e30f;
    m = WarpReduceMax(m);
    if (lane == 0) {
      s_warp_val[0] = m;
    }
  }
  __syncthreads();
  const float global_max = s_warp_val[0];

  // 2. Compute exp and sum
  const float exp_l = (tid < num_experts) ? expf(logit - global_max) : 0.0f;
  float sum_exp = exp_l;
  sum_exp = WarpReduceSum(sum_exp);
  if (lane == 0) {
    s_warp_val[warp] = sum_exp;
  }
  __syncthreads();

  if (warp == 0) {
    float s = (lane < 8) ? s_warp_val[lane] : 0.0f;
    s = WarpReduceSum(s);
    if (lane == 0) {
      s_warp_val[0] = s;
    }
  }
  __syncthreads();
  const float global_sum = s_warp_val[0];
  const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

  // 3. Store normalized probabilities in shared memory
  if (tid < num_experts) {
    s_probs[tid] = exp_l * inv_sum;
  } else if (tid < 256) {
    s_probs[tid] = -1.0f;
  }
  __syncthreads();

  // 4. Extract top-K experts iteratively
  for (uint32_t k = 0; k < num_experts_used && k < 16; ++k) {
    float val = (tid < num_experts) ? s_probs[tid] : -1.0f;
    uint32_t best_e = tid;
    WarpReduceMaxIdx(val, best_e);

    if (lane == 0) {
      s_warp_val[warp] = val;
      s_warp_idx[warp] = best_e;
    }
    __syncthreads();

    if (warp == 0) {
      float w_val = (lane < 8) ? s_warp_val[lane] : -1.0f;
      uint32_t w_idx = (lane < 8) ? s_warp_idx[lane] : 0;
      WarpReduceMaxIdx(w_val, w_idx);
      if (lane == 0) {
        s_top_ids[k] = w_idx;
        s_top_weights[k] = w_val;
      }
    }
    __syncthreads();

    // Invalidate winner so it won't be chosen in next iterations
    if (tid == s_top_ids[k]) {
      s_probs[tid] = -1.0f;
    }
    __syncthreads();
  }

  // 5. Renormalize top-K weights and write out
  if (warp == 0 && lane < num_experts_used && lane < 16) {
    float top_sum = 0.0f;
    for (uint32_t k = 0; k < num_experts_used && k < 16; ++k) {
      top_sum += s_top_weights[k];
    }
    const float inv_top_sum = (top_sum > 0.0f) ? (1.0f / top_sum) : 1.0f;

    const uint32_t k = lane;
    const size_t out_slot = (static_cast<size_t>(t) * num_experts_used) + k;
    expert_ids[out_slot] = s_top_ids[k];
    expert_weights[out_slot] = s_top_weights[k] * inv_top_sum;
  }
}

void DispatchRouterSoftmaxTopK(
    hipStream_t stream,
    const float* router_logits,
    uint32_t num_experts,
    uint32_t num_experts_used,
    uint32_t B,
    uint32_t* expert_ids,
    float* expert_weights) {
  if (B == 0 || num_experts == 0 || num_experts_used == 0) return;
  RouterSoftmaxTopKKernel<<<B, 256, 0, stream>>>(
      router_logits, num_experts, num_experts_used, B, expert_ids, expert_weights);
}

}  // namespace gufo::models::qwen35_moe::hip

#endif  // defined(ENGINE_ENABLE_HIP)

