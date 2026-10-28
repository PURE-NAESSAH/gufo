#ifndef GUFO_MODELS_QWEN35_MOE_HIP_KERNELS_HPP_
#define GUFO_MODELS_QWEN35_MOE_HIP_KERNELS_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen35_moe::hip {

void DispatchGemv(
    hipStream_t stream,
    const void* A_data,
    core::GgmlType A_type,
    uint64_t expert_stride,
    uint32_t expert_id,
    const float* x,
    float* y,
    uint32_t M,
    uint32_t K);

void DispatchRMSNorm(
    hipStream_t stream,
    const float* x,
    const void* weight_data,
    core::GgmlType weight_type,
    float eps,
    uint32_t size,
    float* out);

void DispatchEmbedding(
    hipStream_t stream,
    const void* table,
    core::GgmlType type,
    uint32_t token_id,
    uint32_t hidden_size,
    size_t row_bytes,
    float* out);

void DispatchRoPE(
    hipStream_t stream,
    float* vec,
    uint32_t rotary_dim,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t pos,
    float theta_base);

void DispatchUnpackQG(
    hipStream_t stream,
    const float* qg,
    uint32_t num_heads,
    uint32_t head_dim,
    float* q_buf,
    float* gate_buf);

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
    float* context_out);

void DispatchStoreKV(
    hipStream_t stream,
    const float* k_in,
    const float* v_in,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_context,
    uint32_t pos,
    float* kv_k,
    float* kv_v);

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
    float* core_out);

void DispatchSiluMul(
    hipStream_t stream,
    const float* gate,
    const float* up,
    uint32_t size,
    float* act);

void DispatchAccumWeighted(
    hipStream_t stream,
    const float* src,
    float weight,
    uint32_t size,
    float* dst);

void DispatchMoECombineShared(
    hipStream_t stream,
    const float* routed,
    const float* shexp,
    float shexp_gate_dot,
    uint32_t size,
    float* out);

void DispatchResidualAdd(
    hipStream_t stream,
    const float* delta,
    uint32_t size,
    float* hidden);

}  // namespace gufo::models::qwen35_moe::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN35_MOE_HIP_KERNELS_HPP_
