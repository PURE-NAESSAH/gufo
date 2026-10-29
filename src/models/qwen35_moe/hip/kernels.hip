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
#include "src/models/qwen35_moe/hip/kernels.hpp"

namespace gufo::models::qwen35_moe::hip {

namespace {

struct HipErrorRecord {
  std::string message;
  bool pending = false;
};

HipErrorRecord& HipErrorSlot() {
  static thread_local HipErrorRecord record;
  return record;
}

}  // namespace

void NoteHipFailure(std::string_view what, hipError_t status) {
  if (status == hipSuccess) return;
  // The last-error value is sticky per thread; consuming it here stops
  // unrelated later calls from being blamed for this failure.
  (void)hipGetLastError();
  HipErrorRecord& record = HipErrorSlot();
  if (record.pending) return;
  record.pending = true;
  record.message.assign(what);
  record.message.append(": ");
  record.message.append(hipGetErrorString(status));
}

bool TakeHipError(std::string* out) {
  HipErrorRecord& record = HipErrorSlot();
  if (!record.pending) return false;
  record.pending = false;
  if (out != nullptr) out->swap(record.message);
  record.message.clear();
  return true;
}

void ResetHipError() noexcept {
  HipErrorRecord& record = HipErrorSlot();
  record.pending = false;
  record.message.clear();
}

bool HipErrorPending() noexcept { return HipErrorSlot().pending; }

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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
      GUFO_QWEN35_HIP_LAUNCH("gemv_rocmfp4_fast");
      break;
    case core::GgmlType::kQ4_0_ROCMFP4:
      GemvRocmFp4Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemv_rocmfp4");
      break;
    case core::GgmlType::kBF16:
      GemvBF16Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemv_bf16");
      break;
    case core::GgmlType::kF16:
      GemvF16Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemv_f16");
      break;
    case core::GgmlType::kQ8_0:
      GemvQ8_0Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemv_q8_0");
      break;
    case core::GgmlType::kF32:
    default:
      GemvF32Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemv_f32");
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
  GUFO_QWEN35_HIP_LAUNCH("rmsnorm");
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
  GUFO_QWEN35_HIP_LAUNCH("embedding");
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
  GUFO_QWEN35_HIP_LAUNCH("rope");
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
  GUFO_QWEN35_HIP_LAUNCH("unpack_qg");
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
  GUFO_QWEN35_HIP_LAUNCH("attention_decode");
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
  GUFO_QWEN35_HIP_LAUNCH("store_kv");
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
  GUFO_QWEN35_HIP_LAUNCH("ssm_alpha_beta");

  // 2. Conv1D
  const uint32_t conv_block = 256;
  const uint32_t conv_grid = (conv_channels + conv_block - 1) / conv_block;
  SsmConv1DKernel<<<conv_grid, conv_block, 0, stream>>>(
      ssm_qkv, conv1d_w, conv_channels, conv_state, conv_out_scratch);
  GUFO_QWEN35_HIP_LAUNCH("ssm_conv1d");

  // 3. Head L2-norm
  const float* q_raw = conv_out_scratch;
  const float* k_raw = conv_out_scratch + (k_heads * head_dim);
  const float* v_raw = conv_out_scratch + (2 * k_heads * head_dim);
  SsmHeadNormKernel<<<k_heads, head_dim, 0, stream>>>(
      q_raw, k_raw, k_heads, head_dim, eps, q_normed_scratch, k_normed_scratch);
  GUFO_QWEN35_HIP_LAUNCH("ssm_head_norm");

  // 4. Recurrence
  SsmRecurrenceKernel<<<v_heads, head_dim, 0, stream>>>(
      q_normed_scratch, k_normed_scratch, v_raw, decay_scratch, beta_sig_scratch,
      ssm_gate, ssm_norm_w, v_heads, k_heads, head_dim, eps, deltanet_state, core_out);
  GUFO_QWEN35_HIP_LAUNCH("ssm_recurrence");
}

__global__ void SsmConv1DChunkKernel(
    const float* __restrict__ qkv_in,
    const void* __restrict__ conv1d_w,
    uint32_t conv_channels,
    uint32_t B,
    float* __restrict__ conv_state,
    float* __restrict__ conv_out) {
  const uint32_t c = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (c >= conv_channels) return;

  float s0 = conv_state[c];
  float s1 = conv_state[conv_channels + c];
  float s2 = conv_state[(2 * conv_channels) + c];

  const float* wc = nullptr;
  if (conv1d_w != nullptr) {
    wc = static_cast<const float*>(conv1d_w) + (c * 4);
  }

  for (uint32_t t = 0; t < B; ++t) {
    const float s3 = qkv_in[(static_cast<size_t>(t) * conv_channels) + c];
    float dot = s3;
    if (wc != nullptr) {
      dot = (s0 * wc[0]) + (s1 * wc[1]) + (s2 * wc[2]) + (s3 * wc[3]);
    }
    conv_out[(static_cast<size_t>(t) * conv_channels) + c] = Siluf(dot);
    s0 = s1;
    s1 = s2;
    s2 = s3;
  }

  conv_state[c] = s0;
  conv_state[conv_channels + c] = s1;
  conv_state[(2 * conv_channels) + c] = s2;
}

__global__ void SsmAlphaBetaChunkKernel(
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const void* __restrict__ dt_bias,
    const void* __restrict__ a_val,
    uint32_t v_heads,
    uint32_t B,
    float* __restrict__ decay_out,
    float* __restrict__ beta_sig_out) {
  const uint32_t vh = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vh >= v_heads) return;

  const float dt = dt_bias ? static_cast<const float*>(dt_bias)[vh] : 0.0f;
  const float a = a_val ? static_cast<const float*>(a_val)[vh] : 0.0f;

  for (uint32_t t = 0; t < B; ++t) {
    const size_t idx = (static_cast<size_t>(t) * v_heads) + vh;
    beta_sig_out[idx] = Sigmoidf(beta[idx]);
    const float alpha_biased = alpha[idx] + dt;
    decay_out[idx] = expf(a * Softplusf(alpha_biased));
  }
}

__global__ void SsmHeadNormChunkKernel(
    const float* __restrict__ conv_out,
    uint32_t conv_channels,
    uint32_t k_heads,
    uint32_t head_dim,
    uint32_t B,
    float eps,
    float* __restrict__ q_normed,
    float* __restrict__ k_normed) {
  const uint32_t kh = blockIdx.x;
  const uint32_t t = blockIdx.y;
  const uint32_t d = threadIdx.x;
  if (kh >= k_heads || t >= B || d >= head_dim) return;

  const float* token_conv = conv_out + (static_cast<size_t>(t) * conv_channels);
  const float* qh = token_conv + (kh * head_dim);
  const float* kh_p = token_conv + (k_heads * head_dim) + (kh * head_dim);

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

  const size_t out_idx = (static_cast<size_t>(t) * k_heads * head_dim) + (kh * head_dim) + d;
  q_normed[out_idx] = qh[d] * s_q[0];
  k_normed[out_idx] = kh_p[d] * s_k[0];
}

__global__ void SsmRecurrenceChunkKernel(
    const float* __restrict__ q_normed,
    const float* __restrict__ k_normed,
    const float* __restrict__ conv_out,
    uint32_t conv_channels,
    const float* __restrict__ decay,
    const float* __restrict__ beta_sig,
    const float* __restrict__ ssm_gate,
    const void* __restrict__ ssm_norm_w,
    uint32_t v_heads,
    uint32_t k_heads,
    uint32_t head_dim,
    uint32_t B,
    float eps,
    float* __restrict__ deltanet_state,
    float* __restrict__ core_out) {
  const uint32_t vh = blockIdx.x;
  if (vh >= v_heads) return;
  const uint32_t kh = vh % k_heads;
  const uint32_t c = threadIdx.x; // 0..127
  const uint32_t lane = c & 31u;
  const uint32_t warp = c >> 5u;

  __shared__ float s_q[128];
  __shared__ float s_k[128];
  __shared__ float s_core[128];
  __shared__ float s_sq_warp[4];

  float* S_col = deltanet_state + (static_cast<size_t>(vh) * head_dim * head_dim) + c;
  const size_t v_offset = (2 * k_heads * head_dim) + (vh * head_dim);
  const float nw = ssm_norm_w ? static_cast<const float*>(ssm_norm_w)[c] : 1.0f;
  constexpr float scale = 0.08838834764831845f; // 1.0f / sqrtf(128.0f)

  for (uint32_t t = 0; t < B; ++t) {
    const size_t qk_base = (static_cast<size_t>(t) * k_heads * head_dim) + (kh * head_dim);
    s_q[c] = q_normed[qk_base + c];
    s_k[c] = k_normed[qk_base + c];
    __syncthreads();

    // Step 1: kv_col = (S^T @ k)[c] = sum_r S[r, c] * k[r]
    float kv_col = 0.0f;
    #pragma unroll
    for (uint32_t r = 0; r < 128; ++r) {
      kv_col += S_col[r * 128] * s_k[r];
    }

    const size_t t_vh_idx = (static_cast<size_t>(t) * v_heads) + vh;
    const float d_val = decay[t_vh_idx];
    const float b_val = beta_sig[t_vh_idx];
    const float v_val = conv_out[(static_cast<size_t>(t) * conv_channels) + v_offset + c];

    // Step 2: delta_col = (v[c] - decay * kv_col) * beta
    const float delta_col = (v_val - (d_val * kv_col)) * b_val;

    // Step 3: S[r, c] = decay * S[r, c] + delta_col * k[r], and readout = sum_r S[r, c] * q[r]
    float readout = 0.0f;
    #pragma unroll
    for (uint32_t r = 0; r < 128; ++r) {
      float* entry = S_col + (r * 128);
      const float updated = (*entry * d_val) + (s_k[r] * delta_col);
      *entry = updated;
      readout += updated * s_q[r];
    }

    // Step 4: RMSNorm & Gate
    s_core[c] = readout * scale;
    __syncthreads();

    float sq = s_core[c] * s_core[c];
    for (int offset = 16; offset > 0; offset >>= 1) {
      sq += __shfl_xor(sq, offset);
    }
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
    const float normed_val = s_core[c] * norm_scale * nw;
    const size_t out_idx = (static_cast<size_t>(t) * v_heads * head_dim) + (vh * head_dim) + c;
    const float z_val = ssm_gate[out_idx];

    core_out[out_idx] = normed_val * Siluf(z_val);
    __syncthreads();
  }
}

void DispatchSsmChunk(
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
    uint32_t B,
    float eps,
    float* conv_state,
    float* deltanet_state,
    float* decay_buf,
    float* beta_sig_buf,
    float* conv_out_buf,
    float* q_normed_buf,
    float* k_normed_buf,
    float* core_out) {
  // 1. Batched Conv1D for all B tokens
  const uint32_t conv_block = 256;
  const uint32_t conv_grid = (conv_channels + conv_block - 1) / conv_block;
  SsmConv1DChunkKernel<<<conv_grid, conv_block, 0, stream>>>(
      ssm_qkv, conv1d_w, conv_channels, B, conv_state, conv_out_buf);
  GUFO_QWEN35_HIP_LAUNCH("ssm_conv1d_chunk");

  // 2. Batched AlphaBeta for all B tokens
  const uint32_t ab_block = 64;
  const uint32_t ab_grid = (v_heads + ab_block - 1) / ab_block;
  SsmAlphaBetaChunkKernel<<<ab_grid, ab_block, 0, stream>>>(
      ssm_alpha, ssm_beta, dt_bias, a_val, v_heads, B, decay_buf, beta_sig_buf);
  GUFO_QWEN35_HIP_LAUNCH("ssm_alpha_beta_chunk");

  // 3. Batched HeadNorm for all B tokens
  const dim3 norm_grid(k_heads, B);
  SsmHeadNormChunkKernel<<<norm_grid, head_dim, 0, stream>>>(
      conv_out_buf, conv_channels, k_heads, head_dim, B, eps, q_normed_buf, k_normed_buf);
  GUFO_QWEN35_HIP_LAUNCH("ssm_head_norm_chunk");

  // 4. Batched Recurrence for all B tokens in one kernel
  SsmRecurrenceChunkKernel<<<v_heads, head_dim, 0, stream>>>(
      q_normed_buf, k_normed_buf, conv_out_buf, conv_channels,
      decay_buf, beta_sig_buf, ssm_gate, ssm_norm_w,
      v_heads, k_heads, head_dim, B, eps, deltanet_state, core_out);
  GUFO_QWEN35_HIP_LAUNCH("ssm_recurrence_chunk");
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
  GUFO_QWEN35_HIP_LAUNCH("silu_mul");
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
  GUFO_QWEN35_HIP_LAUNCH("accum_weighted");
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
  GUFO_QWEN35_HIP_LAUNCH("moe_combine_shared");
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
  GUFO_QWEN35_HIP_LAUNCH("residual_add");
}

// ---------------------------------------------------------------------------
// 10. Batched Prefill 2D-Tiled GEMM Kernels (Wave32 x Batch Grid)
// ---------------------------------------------------------------------------

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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

namespace {

using wmma_v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using wmma_v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ wmma_v16h WmmaHalf(wmma_v16h a, wmma_v16h b,
                                              wmma_v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Swizzled LDS fragment staging (same convention as the proven half prefill
// GEMM in src/models/qwen/hip/kernels/prefill_fp16.hip).
__device__ __forceinline__ wmma_v16h LoadSwizzled(const short* p) {
  union { wmma_v16h f; uint4 q[2]; } bits;
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

struct WmmaHalfWeights {
  uint4 low, high;
};

// One 16-weight sub-block of a rocmfp4_fast block as exact FP16 values.
// Layout is nibble planes: low nibbles of qs[0..15] hold k 0..15, high
// nibbles hold k 16..31 (matches DequantizeQ4_0_ROCMFP4_FAST). Codebook10
// magnitudes {0,1,2,3,4,6,8,10} times a ue4m3 scale fit FP16 exactly.
// Selector bytes 0..3 index the SECOND perm operand on gfx1151 (probed).
__device__ __forceinline__ WmmaHalfWeights DecodeRocmFp4SubWmma(
    const uint8_t* blk, unsigned ks) {
  // 17-byte blocks: dword loads tolerate the misalignment, uint4 does not.
  const unsigned* wp = reinterpret_cast<const unsigned*>(blk);
  const unsigned words[4] = {__ldg(wp), __ldg(wp + 1), __ldg(wp + 2), __ldg(wp + 3)};
  const unsigned shift = (ks & 1) * 4;
  const float scale = rocmfp4_ue4m3_to_fp32_half_finite(blk[16]);
  union { wmma_v16h f; uint4 u[2]; } result;
#pragma unroll
  for (unsigned wi = 0; wi < 4; ++wi) {
    const unsigned n = (words[wi] >> shift) & 0x0F0F0F0FU;
    const unsigned index = n & 0x07070707U;
    const unsigned sign_bits = (n & 0x08080808U) << 4;
    // FP16 top bytes of magnitudes {0,1,2,3,4,6,8,10}; bottom byte is 0.
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

// Tensor-core prefill GEMM for Q4_0_ROCMFP4_FAST dense projections.
// Activations arrive fp32 and split into exact hi+lo half planes during LDS
// staging, so products accumulate with ~22 mantissa bits of activation
// precision and weights are lossless; measured deviation matches fp32
// reference noise while running ~9x faster than the scalar LDS kernel.
template <int BM, int BN, int WM, int WN>
__launch_bounds__(WM *WN * 32) __global__ void GemmWmmaRocmFp4FastKernel(
    const void* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, uint32_t batch_u, uint32_t m_u, uint32_t k_u) {
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
  __shared__ short s_b[2 * BK][TS][16][KS];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / WaveSize;
  const int lane = tid % WaveSize;
  const int sl = lane & 15;
  const int wr = wave / WN;
  const int wt = wave % WN;
  const int batch = static_cast<int>(batch_u);
  const int m = static_cast<int>(m_u);
  const int k = static_cast<int>(k_u);

  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  {
    // Wider row groups improve weight locality across token tiles.
    const unsigned group_shift = 3;
    const unsigned group = 1U << group_shift;
    const unsigned first = (blockIdx.y >> group_shift) << group_shift,
                 within = (blockIdx.y & (group - 1)) * gridDim.x + blockIdx.x,
                 rows = min(group, gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  const int r_block = static_cast<int>(row_tile) * BM,
            t_block = static_cast<int>(token_tile) * BN;
  const int ksteps = k / 16;

  wmma_v8f acc[WRS][WTS] = {};

  const std::size_t row_bytes =
      (k / 32) * sizeof(quant::block_rocmfp4_fast);
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
    const std::size_t off =
        static_cast<std::size_t>((t < batch ? t : (batch - 1)) * k);
    x_row[p] = x + off;
  }

  uint4 ra[PFA][2];
  float4 rb[PFB][4];

  const auto fetch = [&](int ks0) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BM * BK && kk < ksteps) {
        const auto decoded =
            DecodeRocmFp4SubWmma(w_row[p] + (kk / 2) * 17, static_cast<unsigned>(kk));
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
        StoreSwizzled(&s_a[ks][placed / 16][placed % 16][0], ra[p][0],
                      ra[p][1], row);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BN * BK) {
        const int placed = (row ^ (ks & 3));
        // Split fp32 activations into exact hi+lo half planes at staging.
        union { wmma_v16h f; uint4 u[2]; } hiP, loP;
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
      wmma_v16h a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i)
        a[i] = LoadSwizzled(&s_a[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const wmma_v16h b =
            LoadSwizzled(&s_b[ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
        const wmma_v16h bl =
            LoadSwizzled(&s_b[BK + ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
          acc[i][j] = WmmaHalf(bl, a[i], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // Accumulators are oriented [token][row]: stores coalesce along rows of
  // y[token*M + row].
  const int hi_lane = lane / 16;
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
      const int row = r_block + ((wr * WRS + i) * 16) + sl;
      const int t0 = t_block + ((wt * WTS + j) * 16);
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const int token = t0 + 2 * l + hi_lane;
        if (row < m && token < batch)
          y[static_cast<std::size_t>(token) * m + row] = acc[i][j][l];
      }
    }
  }
}

}  // namespace

namespace {

enum class MoeWmmaMode { kGateUp, kDown };

// Tiles enumerate (expert, 64-token window) pairs; built on-device right
// before the WMMA grouped GEMM so the stride matches its BN. The scalar
// grouped kernels keep using the stride-16 table rebuilt by the sort each
// layer, so overwriting here is safe for the paired dispatch that follows.
__global__ void MoeBuildTilesWmmaKernel(
    const uint32_t* __restrict__ counts,
    uint32_t* __restrict__ tile_expert,
    uint32_t* __restrict__ tile_tok_offset,
    uint32_t* __restrict__ num_tiles_out, uint32_t num_experts) {
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

bool MoeUseWmma(uint32_t B, uint32_t expert_ff, uint32_t H) {
  return B >= 32 && (expert_ff % 32) == 0 && (H % 32) == 0 &&
         expert_ff >= 128 && H >= 128;
}

// Tensor-core grouped MoE FFN. Activations are gathered through
// sorted_slots during staging; outputs stay slot-indexed exactly like the
// scalar kernels. kGateUp reads x[token] (K=H, M=expert_ff) and stores
// swiglu results; kDown reads act_in[slot] (K=expert_ff, M=H, weights
// passed via gate_data/expert_stride_gate) and atomically accumulates
// weight-scaled contributions into routed_accum[token*H+row].
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
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
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

  wmma_v8f acc[WRS][WTS] = {};
  wmma_v8f acc2[WRS][WTS] = {};

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
            DecodeRocmFp4SubWmma(w_row[p] + (kk / 2) * 17, static_cast<unsigned>(kk));
        ra[p][0] = decoded.low;
        ra[p][1] = decoded.high;
        if constexpr (kGateUp) {
          const auto decoded2 = DecodeRocmFp4SubWmma(w_row2[p] + (kk / 2) * 17,
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
        union { wmma_v16h f; uint4 u[2]; } hiP, loP;
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
      wmma_v16h a[WRS], a2[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        a[i] = LoadSwizzled(&s_a[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
        if constexpr (kGateUp)
          a2[i] = LoadSwizzled(&s_a2[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const wmma_v16h b =
            LoadSwizzled(&s_b[ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
        const wmma_v16h bl =
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

}  // namespace

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
      // Tensor-core path wins from B=32 upward (measured 4.3x at B=32,
      // 9.4x at B=2658 on gfx1151). Needs whole 32-weight quant blocks and
      // at least one full 128-row tile to avoid clamped-row waste.
      if (B >= 32 && K % 32 == 0 && M >= 128 && expert_stride == 0 &&
          expert_id == 0) {
        constexpr int BM = 128, BN = 64, WM = 2, WN = 2;
        const dim3 grid((B + BN - 1) / BN, (M + BM - 1) / BM);
        GemmWmmaRocmFp4FastKernel<BM, BN, WM, WN><<<grid, WM * WN * 32, 0,
                                                    stream>>>(A_data, x, y, B,
                                                               M, K);
        GUFO_QWEN35_HIP_LAUNCH("gemm_wmma_rocmfp4_fast");
        break;
      }
      constexpr uint32_t T_BATCH = 32;
      const dim3 grid_dim_t((M + kWaves - 1) / kWaves, (B + T_BATCH - 1) / T_BATCH);
      GemmLdsRocmFp4FastKernel<kWaves, T_BATCH><<<grid_dim_t, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemm_lds_rocmfp4_fast");
      break;
    }
    case core::GgmlType::kQ4_0_ROCMFP4: {
      constexpr uint32_t T_BATCH = 32;
      const dim3 grid_dim_t((M + kWaves - 1) / kWaves, (B + T_BATCH - 1) / T_BATCH);
      GemmLdsRocmFp4Kernel<kWaves, T_BATCH><<<grid_dim_t, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemm_lds_rocmfp4");
      break;
    }
    case core::GgmlType::kBF16: {
      const dim3 grid_dim((M + kWaves - 1) / kWaves, B);
      GemmTileBF16Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemm_tile_bf16");
      break;
    }
    case core::GgmlType::kF16: {
      const dim3 grid_dim((M + kWaves - 1) / kWaves, B);
      GemmTileF16Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemm_tile_f16");
      break;
    }
    case core::GgmlType::kQ8_0: {
      const dim3 grid_dim((M + kWaves - 1) / kWaves, B);
      GemmTileQ8_0Kernel<kWaves><<<grid_dim, block_dim, 0, stream>>>(
          A_data, x, y, M, K, B, expert_stride, expert_id);
      GUFO_QWEN35_HIP_LAUNCH("gemm_tile_q8_0");
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
  GUFO_QWEN35_HIP_LAUNCH("rmsnorm_batch");
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
  GUFO_QWEN35_HIP_LAUNCH("embedding_batch");
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
  GUFO_QWEN35_HIP_LAUNCH("rope_batch");
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
  GUFO_QWEN35_HIP_LAUNCH("unpack_qg_batch");
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
  GUFO_QWEN35_HIP_LAUNCH("store_kv_batch");
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

// Block-tiled causal FlashAttention prefill for head_dim 256. One block per
// (head, 32-token group); each warp owns two query tokens, interleaving two
// independent online-softmax chains so one row's shuffle/exp latency hides
// behind the other. Every 16-key KV tile is staged into LDS once and feeds
// all 32 rows, replacing the naive sweep's per-token DRAM re-reads and two
// barriers per key. Rows keep ascending key order so each chain matches the
// scalar reference up to fp32 rounding.
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
  constexpr uint32_t kQTile = 32;             // two rows per warp x 16 warps
  constexpr uint32_t kKvTile = 16;            // keys per LDS stage
  constexpr uint32_t kPerLane = kDim / 32;    // eight floats per lane
  constexpr float kScale = 1.0F / 16.0F;      // 1 / sqrt(kDim)

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
  if (head_dim == 256 && B >= 32) {
    dim3 grid(num_heads, (B + 31u) / 32u);
    AttentionPrefillTiledKernel<<<grid, 512, 0, stream>>>(
        q_buf, gate_buf, kv_k, kv_v, num_heads, num_kv_heads,
        max_context, pos0, B, context_out);
    GUFO_QWEN35_HIP_LAUNCH("attention_prefill_tiled");
    return;
  }
  dim3 grid(num_heads, B);
  AttentionPrefillKernel<<<grid, head_dim, 0, stream>>>(
      q_buf, gate_buf, kv_k, kv_v, num_heads, num_kv_heads, head_dim, max_context, pos0, context_out);
  GUFO_QWEN35_HIP_LAUNCH("attention_prefill");
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
  GUFO_QWEN35_HIP_LAUNCH("moe_combine_shared_batch");
}

// ---------------------------------------------------------------------------
// 11. Batched MoE Kernels (All tokens, all experts in 2 GPU kernel launches)
// ---------------------------------------------------------------------------

template <uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32)
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
__launch_bounds__(WavesPerBlock * 32)
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
  GUFO_QWEN35_HIP_LAUNCH("moe_gated_swiglu_batch");
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
  GUFO_QWEN35_HIP_LAUNCH("moe_down_accum_batch");
}

__global__ void MoeCountExpertsKernel(
    const uint32_t* __restrict__ expert_ids,
    uint32_t* __restrict__ counts,
    uint32_t total_slots) {
  const uint32_t slot = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (slot < total_slots) {
    const uint32_t e = expert_ids[slot];
    atomicAdd(&counts[e], 1u);
  }
}

__global__ void MoePrefixSumOffsetsKernel(
    const uint32_t* __restrict__ counts,
    uint32_t* __restrict__ offsets,
    uint32_t num_experts) {
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    uint32_t sum = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
      offsets[e] = sum;
      sum += counts[e];
    }
    offsets[num_experts] = sum;
  }
}

__global__ void MoeScatterSlotsKernel(
    const uint32_t* __restrict__ expert_ids,
    const uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ cursors,
    uint32_t* __restrict__ sorted_slots,
    uint32_t total_slots) {
  const uint32_t slot = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (slot < total_slots) {
    const uint32_t e = expert_ids[slot];
    const uint32_t pos = offsets[e] + atomicAdd(&cursors[e], 1u);
    sorted_slots[pos] = slot;
  }
}

__global__ void MoeBuildTilesKernel(
    const uint32_t* __restrict__ counts,
    uint32_t* __restrict__ tile_expert,
    uint32_t* __restrict__ tile_tok_offset,
    uint32_t* __restrict__ num_tiles_out,
    uint32_t num_experts) {
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    uint32_t tile_idx = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
      const uint32_t c = counts[e];
      for (uint32_t tok = 0; tok < c; tok += 16) {
        tile_expert[tile_idx] = e;
        tile_tok_offset[tile_idx] = tok;
        tile_idx++;
      }
    }
    *num_tiles_out = tile_idx;
  }
}

void DispatchSortMoESlots(
    hipStream_t stream,
    const uint32_t* expert_ids,
    uint32_t total_slots,
    uint32_t num_experts,
    uint32_t* counts,
    uint32_t* offsets,
    uint32_t* cursors,
    uint32_t* sorted_slots,
    uint32_t* tile_expert,
    uint32_t* tile_tok_offset,
    uint32_t* num_tiles_out) {
  (void)hipMemsetAsync(counts, 0, num_experts * sizeof(uint32_t), stream);
  (void)hipMemsetAsync(cursors, 0, num_experts * sizeof(uint32_t), stream);

  const uint32_t block_dim = 256;
  const uint32_t grid_dim = (total_slots + block_dim - 1) / block_dim;

  MoeCountExpertsKernel<<<grid_dim, block_dim, 0, stream>>>(expert_ids, counts, total_slots);
  GUFO_QWEN35_HIP_LAUNCH("moe_count_experts");
  MoePrefixSumOffsetsKernel<<<1, 32, 0, stream>>>(counts, offsets, num_experts);
  GUFO_QWEN35_HIP_LAUNCH("moe_prefix_sum_offsets");
  MoeScatterSlotsKernel<<<grid_dim, block_dim, 0, stream>>>(expert_ids, offsets, cursors, sorted_slots, total_slots);
  GUFO_QWEN35_HIP_LAUNCH("moe_scatter_slots");
  MoeBuildTilesKernel<<<1, 32, 0, stream>>>(counts, tile_expert, tile_tok_offset, num_tiles_out, num_experts);
  GUFO_QWEN35_HIP_LAUNCH("moe_build_tiles");
}

template <uint32_t WavesPerBlock = 8, uint32_t T_BATCH = 16>
__launch_bounds__(WavesPerBlock * 32)
__global__ void MoeGatedSwigluFastGroupedKernel(
    const void* __restrict__ gate_data,
    const void* __restrict__ up_data,
    uint64_t expert_stride_gate,
    uint64_t expert_stride_up,
    const float* __restrict__ x,
    const uint32_t* __restrict__ counts,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ sorted_slots,
    const uint32_t* __restrict__ tile_expert,
    const uint32_t* __restrict__ tile_tok_offset,
    const uint32_t* __restrict__ num_tiles_out,
    uint32_t B,
    uint32_t num_experts,
    uint32_t num_experts_used,
    uint32_t expert_ff,
    uint32_t H,
    float* __restrict__ act_out) {
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
        float sg = sum_gate[tb];
        float su = sum_up[tb];
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
    const void* __restrict__ down_data,
    uint64_t expert_stride_down,
    const float* __restrict__ act_in,
    const float* __restrict__ expert_weights,
    const uint32_t* __restrict__ counts,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ sorted_slots,
    const uint32_t* __restrict__ tile_expert,
    const uint32_t* __restrict__ tile_tok_offset,
    const uint32_t* __restrict__ num_tiles_out,
    uint32_t B,
    uint32_t num_experts,
    uint32_t num_experts_used,
    uint32_t expert_ff,
    uint32_t H,
    float* __restrict__ routed_accum) {
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
        for (uint32_t tb = 0; tb < T_BATCH; ++tb) {
          sum[tb] += (w0 * s_act[tb][offset_k + j]) + (w1 * s_act[tb][offset_k + j + 16]);
        }
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
        for (int offset = 16; offset > 0; offset >>= 1) {
          s += __shfl_xor(s, offset);
        }
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

void DispatchMoeGatedSwigluGrouped(
    hipStream_t stream,
    const void* gate_data,
    const void* up_data,
    core::GgmlType exp_type,
    uint64_t expert_stride_gate,
    uint64_t expert_stride_up,
    const float* x,
    const uint32_t* counts,
    const uint32_t* offsets,
    const uint32_t* sorted_slots,
    const uint32_t* tile_expert,
    const uint32_t* tile_tok_offset,
    const uint32_t* num_tiles_out,
    uint32_t B,
    uint32_t num_experts,
    uint32_t num_experts_used,
    uint32_t expert_ff,
    uint32_t H,
    float* act_out) {
  (void)exp_type;
  if (MoeUseWmma(B, expert_ff, H)) {
    constexpr int BM = 64, BN = 64, WM = 2, WN = 2;
    // The caller-owned tile buffers are mutable; the dispatch API just
    // exposes them as const because the scalar kernels only read them.
    MoeBuildTilesWmmaKernel<<<1, 32, 0, stream>>>(
        counts, const_cast<uint32_t*>(tile_expert),
        const_cast<uint32_t*>(tile_tok_offset),
        const_cast<uint32_t*>(num_tiles_out), num_experts);
    GUFO_QWEN35_HIP_LAUNCH("moe_build_tiles_wmma");
    const uint32_t max_tiles = std::min(
        8192u, (B * num_experts_used + BN - 1u) / BN + num_experts);
    const dim3 grid_dim((expert_ff + BM - 1) / BM, max_tiles);
    MoeGemmWmmaKernel<BM, BN, WM, WN, MoeWmmaMode::kGateUp>
        <<<grid_dim, WM * WN * 32, 0, stream>>>(
            gate_data, up_data, expert_stride_gate, expert_stride_up, x,
            nullptr, counts, offsets, sorted_slots, tile_expert,
            tile_tok_offset, num_tiles_out, num_experts_used, expert_ff, H,
            act_out);
    GUFO_QWEN35_HIP_LAUNCH("moe_gemm_wmma_gateup");
    return;
  }
  constexpr uint32_t kWaves = 8;
  constexpr uint32_t T_BATCH = 16;
  const dim3 block_dim(kWaves * 32);
  const uint32_t max_tiles = std::min(8192u, (B * num_experts_used + T_BATCH - 1u) / T_BATCH + num_experts);
  const dim3 grid_dim((expert_ff + kWaves - 1) / kWaves, max_tiles);

  MoeGatedSwigluFastGroupedKernel<kWaves, T_BATCH><<<grid_dim, block_dim, 0, stream>>>(
      gate_data, up_data, expert_stride_gate, expert_stride_up,
      x, counts, offsets, sorted_slots,
      tile_expert, tile_tok_offset, num_tiles_out,
      B, num_experts, num_experts_used, expert_ff, H, act_out);
  GUFO_QWEN35_HIP_LAUNCH("moe_gated_swiglu_grouped");
}

void DispatchMoeDownAccumGrouped(
    hipStream_t stream,
    const void* down_data,
    core::GgmlType exp_type,
    uint64_t expert_stride_down,
    const float* act_in,
    const float* expert_weights,
    const uint32_t* counts,
    const uint32_t* offsets,
    const uint32_t* sorted_slots,
    const uint32_t* tile_expert,
    const uint32_t* tile_tok_offset,
    const uint32_t* num_tiles_out,
    uint32_t B,
    uint32_t num_experts,
    uint32_t num_experts_used,
    uint32_t expert_ff,
    uint32_t H,
    float* routed_accum) {
  (void)exp_type;
  if (MoeUseWmma(B, expert_ff, H)) {
    // Tiles were rebuilt with WMMA stride by the paired gate/up dispatch.
    constexpr int BM = 128, BN = 64, WM = 2, WN = 2;
    const uint32_t max_tiles = std::min(
        8192u, (B * num_experts_used + BN - 1u) / BN + num_experts);
    const dim3 grid_dim((H + BM - 1) / BM, max_tiles);
    MoeGemmWmmaKernel<BM, BN, WM, WN, MoeWmmaMode::kDown>
        <<<grid_dim, WM * WN * 32, 0, stream>>>(
            down_data, down_data, expert_stride_down, 0, act_in,
            expert_weights, counts, offsets, sorted_slots, tile_expert,
            tile_tok_offset, num_tiles_out, num_experts_used, expert_ff, H,
            routed_accum);
    GUFO_QWEN35_HIP_LAUNCH("moe_gemm_wmma_down");
    return;
  }
  constexpr uint32_t kWaves = 8;
  constexpr uint32_t T_BATCH = 16;
  const dim3 block_dim(kWaves * 32);
  const uint32_t max_tiles = std::min(8192u, (B * num_experts_used + T_BATCH - 1u) / T_BATCH + num_experts);
  const dim3 grid_dim((H + kWaves - 1) / kWaves, max_tiles);

  MoeDownAccumFastGroupedKernel<kWaves, T_BATCH><<<grid_dim, block_dim, 0, stream>>>(
      down_data, expert_stride_down, act_in, expert_weights,
      counts, offsets, sorted_slots,
      tile_expert, tile_tok_offset, num_tiles_out,
      B, num_experts, num_experts_used, expert_ff, H, routed_accum);
  GUFO_QWEN35_HIP_LAUNCH("moe_down_accum_grouped");
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
  GUFO_QWEN35_HIP_LAUNCH("router_softmax_topk");
}

}  // namespace gufo::models::qwen35_moe::hip

#endif  // defined(ENGINE_ENABLE_HIP)

