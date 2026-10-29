#ifndef GUFO_MODELS_QWEN35_MOE_HIP_KERNELS_HPP_
#define GUFO_MODELS_QWEN35_MOE_HIP_KERNELS_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen35_moe::hip {

// Thread-local record of the first HIP failure seen since the last
// TakeHipError()/ResetHipError(). Every HIP result in this model funnels
// through here so a dead GPU context cannot look like fast inference.
void NoteHipFailure(std::string_view what, hipError_t status);

// Moves out the pending message and clears it; false when nothing failed.
[[nodiscard]] bool TakeHipError(std::string* out);

void ResetHipError() noexcept;

// True while a recorded failure is still waiting for its caller.
[[nodiscard]] bool HipErrorPending() noexcept;

#ifndef GUFO_QWEN35_HIP
#define GUFO_QWEN35_HIP(expr) NoteHipFailure(#expr, (expr))
#endif

#ifndef GUFO_QWEN35_HIP_LAUNCH
#define GUFO_QWEN35_HIP_LAUNCH(name) NoteHipFailure(name " launch", hipGetLastError())
#endif

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
    uint32_t B);

void DispatchRMSNormBatch(
    hipStream_t stream,
    const float* x,
    const void* weight_data,
    core::GgmlType weight_type,
    float eps,
    uint32_t size,
    uint32_t B,
    float* out);

void DispatchEmbeddingBatch(
    hipStream_t stream,
    const void* table,
    core::GgmlType type,
    const uint32_t* token_ids,
    uint32_t B,
    uint32_t hidden_size,
    size_t row_bytes,
    float* out);

void DispatchRoPEBatch(
    hipStream_t stream,
    float* vec,
    uint32_t rotary_dim,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t pos0,
    uint32_t B,
    float theta_base);

void DispatchUnpackQGBatch(
    hipStream_t stream,
    const float* qg,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t B,
    float* q_buf,
    float* gate_buf);

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
    float* context_out);

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
    float* kv_v);

void DispatchMoECombineSharedBatch(
    hipStream_t stream,
    const float* routed,
    const float* shexp,
    const float* d_shexp_gate_dot,
    uint32_t H,
    uint32_t B,
    float* out);


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
    float* act_out);

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
    float* routed_accum);

void DispatchRouterSoftmaxTopK(
    hipStream_t stream,
    const float* router_logits,
    uint32_t num_experts,
    uint32_t num_experts_used,
    uint32_t B,
    uint32_t* expert_ids,
    float* expert_weights);

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
    uint32_t* num_tiles_out);

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
    float* act_out);

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
    float* routed_accum);

}  // namespace gufo::models::qwen35_moe::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN35_MOE_HIP_KERNELS_HPP_
