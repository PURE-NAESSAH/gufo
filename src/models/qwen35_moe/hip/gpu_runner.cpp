#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen35_moe/hip/gpu_runner.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "src/models/qwen35_moe/hip/kernels.hpp"

namespace gufo::models::qwen35_moe::hip {

namespace {

void* AllocBytes(size_t bytes, std::vector<void*>& record) {
  void* ptr = nullptr;
  hipError_t status = hipMalloc(&ptr, bytes);
  if (status != hipSuccess) {
    status = hipMallocManaged(&ptr, bytes, hipMemAttachGlobal);
  }
  if (status == hipSuccess && ptr != nullptr) {
    record.push_back(ptr);
  }
  return ptr;
}

}  // namespace

void GpuState::Allocate(const Config& cfg, uint32_t max_ctx) {
  max_context = max_ctx;
  Free();

  const uint32_t H = cfg.hidden_size;
  const uint32_t num_heads = cfg.num_heads;
  const uint32_t num_kv_heads = cfg.num_kv_heads;
  const uint32_t head_dim = cfg.head_dim;
  const uint32_t conv_channels = cfg.SsmConvChannels();
  const uint32_t val_dim = cfg.SsmValueDim();
  const uint32_t v_heads = cfg.ssm_num_v_heads;
  const uint32_t ssm_head_dim = cfg.ssm_head_dim;
  const uint32_t expert_ff = cfg.expert_ff;
  const uint32_t shared_expert_ff = cfg.shared_expert_ff;
  const uint32_t full_layers = cfg.TotalFullAttentionLayers();
  const uint32_t ssm_layers = cfg.TotalSsmLayers();

  d_hidden = static_cast<float*>(AllocBytes(H * sizeof(float), allocations));
  d_x_normed = static_cast<float*>(AllocBytes(H * sizeof(float), allocations));
  d_attn_out = static_cast<float*>(AllocBytes(H * sizeof(float), allocations));

  // Full attention
  d_qg_buf = static_cast<float*>(AllocBytes(2 * num_heads * head_dim * sizeof(float), allocations));
  d_q_buf = static_cast<float*>(AllocBytes(num_heads * head_dim * sizeof(float), allocations));
  d_gate_buf = static_cast<float*>(AllocBytes(num_heads * head_dim * sizeof(float), allocations));
  d_k_buf = static_cast<float*>(AllocBytes(num_kv_heads * head_dim * sizeof(float), allocations));
  d_v_buf = static_cast<float*>(AllocBytes(num_kv_heads * head_dim * sizeof(float), allocations));
  d_context = static_cast<float*>(AllocBytes(num_heads * head_dim * sizeof(float), allocations));

  const size_t kv_elements = static_cast<size_t>(full_layers) * num_kv_heads * max_context * head_dim;
  d_kv_k = static_cast<float*>(AllocBytes(kv_elements * sizeof(float), allocations));
  d_kv_v = static_cast<float*>(AllocBytes(kv_elements * sizeof(float), allocations));

  // SSM
  d_ssm_qkv = static_cast<float*>(AllocBytes(conv_channels * sizeof(float), allocations));
  d_ssm_gate = static_cast<float*>(AllocBytes(val_dim * sizeof(float), allocations));
  d_ssm_alpha = static_cast<float*>(AllocBytes(v_heads * sizeof(float), allocations));
  d_ssm_beta = static_cast<float*>(AllocBytes(v_heads * sizeof(float), allocations));
  d_decay = static_cast<float*>(AllocBytes(v_heads * sizeof(float), allocations));
  d_beta_sig = static_cast<float*>(AllocBytes(v_heads * sizeof(float), allocations));
  d_ssm_conv_out = static_cast<float*>(AllocBytes(conv_channels * sizeof(float), allocations));
  d_ssm_q_normed = static_cast<float*>(AllocBytes(cfg.SsmKeyDim() * sizeof(float), allocations));
  d_ssm_k_normed = static_cast<float*>(AllocBytes(cfg.SsmKeyDim() * sizeof(float), allocations));
  d_ssm_core_out = static_cast<float*>(AllocBytes(val_dim * sizeof(float), allocations));

  const size_t conv_elements = static_cast<size_t>(ssm_layers) * 3 * conv_channels;
  d_conv_state = static_cast<float*>(AllocBytes(conv_elements * sizeof(float), allocations));

  const size_t deltanet_elements = static_cast<size_t>(ssm_layers) * v_heads * ssm_head_dim * ssm_head_dim;
  d_deltanet_state = static_cast<float*>(AllocBytes(deltanet_elements * sizeof(float), allocations));

  // MoE
  d_router_logits = static_cast<float*>(AllocBytes(cfg.num_experts * sizeof(float), allocations));
  d_expert_gate = static_cast<float*>(AllocBytes(expert_ff * sizeof(float), allocations));
  d_expert_up = static_cast<float*>(AllocBytes(expert_ff * sizeof(float), allocations));
  d_expert_act = static_cast<float*>(AllocBytes(expert_ff * sizeof(float), allocations));
  d_expert_down = static_cast<float*>(AllocBytes(H * sizeof(float), allocations));
  d_routed_accum = static_cast<float*>(AllocBytes(H * sizeof(float), allocations));
  d_shexp_gate = static_cast<float*>(AllocBytes(shared_expert_ff * sizeof(float), allocations));
  d_shexp_up = static_cast<float*>(AllocBytes(shared_expert_ff * sizeof(float), allocations));
  d_shexp_act = static_cast<float*>(AllocBytes(shared_expert_ff * sizeof(float), allocations));
  d_shexp_down = static_cast<float*>(AllocBytes(H * sizeof(float), allocations));
  d_shexp_dot = static_cast<float*>(AllocBytes(sizeof(float), allocations));
  d_moe_out = static_cast<float*>(AllocBytes(H * sizeof(float), allocations));

  // Logits
  d_logits = static_cast<float*>(AllocBytes(cfg.vocab_size * sizeof(float), allocations));

  h_router_logits.resize(cfg.num_experts, 0.0f);
  h_probs.resize(cfg.num_experts);

  Reset(cfg, max_ctx);
}

void GpuState::Free() {
  for (void* p : allocations) {
    if (p != nullptr) {
      (void)hipFree(p);
    }
  }
  allocations.clear();
  d_hidden = nullptr;
  d_x_normed = nullptr;
  d_attn_out = nullptr;
  d_qg_buf = nullptr;
  d_q_buf = nullptr;
  d_gate_buf = nullptr;
  d_k_buf = nullptr;
  d_v_buf = nullptr;
  d_context = nullptr;
  d_kv_k = nullptr;
  d_kv_v = nullptr;
  d_ssm_qkv = nullptr;
  d_ssm_gate = nullptr;
  d_ssm_alpha = nullptr;
  d_ssm_beta = nullptr;
  d_decay = nullptr;
  d_beta_sig = nullptr;
  d_ssm_conv_out = nullptr;
  d_ssm_q_normed = nullptr;
  d_ssm_k_normed = nullptr;
  d_ssm_core_out = nullptr;
  d_conv_state = nullptr;
  d_deltanet_state = nullptr;
  d_router_logits = nullptr;
  d_expert_gate = nullptr;
  d_expert_up = nullptr;
  d_expert_act = nullptr;
  d_expert_down = nullptr;
  d_routed_accum = nullptr;
  d_shexp_gate = nullptr;
  d_shexp_up = nullptr;
  d_shexp_act = nullptr;
  d_shexp_down = nullptr;
  d_shexp_dot = nullptr;
  d_moe_out = nullptr;
  d_logits = nullptr;
}

void GpuState::Reset(const Config& cfg, uint32_t max_ctx) {
  (void)max_ctx;
  const uint32_t full_layers = cfg.TotalFullAttentionLayers();
  const uint32_t ssm_layers = cfg.TotalSsmLayers();
  const uint32_t conv_channels = cfg.SsmConvChannels();
  const uint32_t v_heads = cfg.ssm_num_v_heads;
  const uint32_t ssm_head_dim = cfg.ssm_head_dim;

  if (d_conv_state != nullptr) {
    const size_t conv_elements = static_cast<size_t>(ssm_layers) * 3 * conv_channels;
    (void)hipMemsetAsync(d_conv_state, 0, conv_elements * sizeof(float), nullptr);
  }
  if (d_deltanet_state != nullptr) {
    const size_t deltanet_elements = static_cast<size_t>(ssm_layers) * v_heads * ssm_head_dim * ssm_head_dim;
    (void)hipMemsetAsync(d_deltanet_state, 0, deltanet_elements * sizeof(float), nullptr);
  }
  if (d_kv_k != nullptr && d_kv_v != nullptr) {
    const size_t kv_elements = static_cast<size_t>(full_layers) * cfg.num_kv_heads * max_context * cfg.head_dim;
    (void)hipMemsetAsync(d_kv_k, 0, kv_elements * sizeof(float), nullptr);
    (void)hipMemsetAsync(d_kv_v, 0, kv_elements * sizeof(float), nullptr);
  }
  (void)hipDeviceSynchronize();
}

void ForwardModelGpu(
    uint32_t token_id,
    uint32_t pos,
    const ModelWeights& weights,
    GpuState& state,
    std::span<float> out_logits) {
  hipStream_t stream = nullptr;
  const auto& cfg = weights.config;
  const uint32_t H = cfg.hidden_size;

  // 1. Embedding lookup -> state.d_hidden
  DispatchEmbedding(stream, weights.token_embd.data, weights.token_embd.type,
                    token_id, H, weights.token_embd.RowBytes(), state.d_hidden);

  uint32_t full_attn_idx = 0;
  uint32_t ssm_layer_idx = 0;

  for (uint32_t l = 0; l < cfg.num_layers; ++l) {
    const auto& layer = weights.layers[l];

    // 2. Pre-attention RMSNorm: d_hidden -> d_x_normed
    DispatchRMSNorm(stream, state.d_hidden, layer.attn_norm.data, layer.attn_norm.type,
                    cfg.rms_eps, H, state.d_x_normed);

    if (layer.is_full_attention) {
      // Joint Q + Gate projection: [H -> 2 * num_heads * head_dim]
      DispatchGemv(stream, layer.attn_q.data, layer.attn_q.type, 0, 0,
                   state.d_x_normed, state.d_qg_buf, 2 * cfg.num_heads * cfg.head_dim, H);

      // K and V projections: [H -> num_kv_heads * head_dim]
      DispatchGemv(stream, layer.attn_k.data, layer.attn_k.type, 0, 0,
                   state.d_x_normed, state.d_k_buf, cfg.num_kv_heads * cfg.head_dim, H);
      DispatchGemv(stream, layer.attn_v.data, layer.attn_v.type, 0, 0,
                   state.d_x_normed, state.d_v_buf, cfg.num_kv_heads * cfg.head_dim, H);

      // Unpack Q and Gate
      DispatchUnpackQG(stream, state.d_qg_buf, cfg.num_heads, cfg.head_dim,
                       state.d_q_buf, state.d_gate_buf);

      // Q-norm and RoPE
      DispatchRMSNorm(stream, state.d_q_buf, layer.attn_q_norm.data, layer.attn_q_norm.type,
                      cfg.rms_eps, cfg.num_heads * cfg.head_dim, state.d_q_buf);
      DispatchRoPE(stream, state.d_q_buf, cfg.rotary_dim, cfg.head_dim, cfg.num_heads, pos, cfg.rope_theta);

      // K-norm and RoPE
      DispatchRMSNorm(stream, state.d_k_buf, layer.attn_k_norm.data, layer.attn_k_norm.type,
                      cfg.rms_eps, cfg.num_kv_heads * cfg.head_dim, state.d_k_buf);
      DispatchRoPE(stream, state.d_k_buf, cfg.rotary_dim, cfg.head_dim, cfg.num_kv_heads, pos, cfg.rope_theta);

      // Store in KV cache
      float* layer_kv_k = state.d_kv_k +
                          (static_cast<size_t>(full_attn_idx) * cfg.num_kv_heads * state.max_context * cfg.head_dim);
      float* layer_kv_v = state.d_kv_v +
                          (static_cast<size_t>(full_attn_idx) * cfg.num_kv_heads * state.max_context * cfg.head_dim);
      DispatchStoreKV(stream, state.d_k_buf, state.d_v_buf, cfg.num_kv_heads, cfg.head_dim,
                      state.max_context, pos, layer_kv_k, layer_kv_v);

      // Attention decode
      DispatchAttentionDecode(stream, state.d_q_buf, state.d_gate_buf, layer_kv_k, layer_kv_v,
                              cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, state.max_context, pos,
                              state.d_context);

      // Output projection: [num_heads * head_dim -> H]
      DispatchGemv(stream, layer.attn_output.data, layer.attn_output.type, 0, 0,
                   state.d_context, state.d_attn_out, H, cfg.num_heads * cfg.head_dim);

      ++full_attn_idx;
    } else {
      // Linear Attention (SSM)
      const uint32_t val_dim = cfg.SsmValueDim();
      const uint32_t conv_channels = cfg.SsmConvChannels();
      const uint32_t v_heads = cfg.ssm_num_v_heads;
      const uint32_t k_heads = cfg.ssm_num_k_heads;
      const uint32_t ssm_head_dim = cfg.ssm_head_dim;

      DispatchGemv(stream, layer.ssm_qkv.data, layer.ssm_qkv.type, 0, 0,
                   state.d_x_normed, state.d_ssm_qkv, conv_channels, H);
      DispatchGemv(stream, layer.ssm_gate.data, layer.ssm_gate.type, 0, 0,
                   state.d_x_normed, state.d_ssm_gate, val_dim, H);
      DispatchGemv(stream, layer.ssm_alpha.data, layer.ssm_alpha.type, 0, 0,
                   state.d_x_normed, state.d_ssm_alpha, v_heads, H);
      DispatchGemv(stream, layer.ssm_beta.data, layer.ssm_beta.type, 0, 0,
                   state.d_x_normed, state.d_ssm_beta, v_heads, H);

      float* layer_conv_state = state.d_conv_state + (static_cast<size_t>(ssm_layer_idx) * 3 * conv_channels);
      float* layer_deltanet_state =
          state.d_deltanet_state + (static_cast<size_t>(ssm_layer_idx) * v_heads * ssm_head_dim * ssm_head_dim);

      DispatchSsmStep(
          stream, state.d_ssm_qkv, state.d_ssm_gate, state.d_ssm_alpha, state.d_ssm_beta,
          layer.ssm_dt.data, layer.ssm_a.data, layer.ssm_conv1d.data, layer.ssm_norm.data,
          conv_channels, v_heads, k_heads, ssm_head_dim, cfg.rms_eps,
          layer_conv_state, layer_deltanet_state,
          state.d_decay, state.d_beta_sig, state.d_ssm_conv_out,
          state.d_ssm_q_normed, state.d_ssm_k_normed, state.d_ssm_core_out);

      DispatchGemv(stream, layer.ssm_out.data, layer.ssm_out.type, 0, 0,
                   state.d_ssm_core_out, state.d_attn_out, H, val_dim);

      ++ssm_layer_idx;
    }

    // Residual add: d_hidden += d_attn_out
    DispatchResidualAdd(stream, state.d_attn_out, H, state.d_hidden);

    // Post-attention RMSNorm: d_hidden -> d_x_normed
    DispatchRMSNorm(stream, state.d_hidden, layer.post_attention_norm.data, layer.post_attention_norm.type,
                    cfg.rms_eps, H, state.d_x_normed);

    // MoE Routing & FFN
    // 1. Router GEMV: [H -> num_experts]
    DispatchGemv(stream, layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, 0, 0,
                 state.d_x_normed, state.d_router_logits, cfg.num_experts, H);

    // Copy 256 router logits to host to do top-k
    hipMemcpyDtoHAsync(state.h_router_logits.data(), state.d_router_logits,
                       cfg.num_experts * sizeof(float), stream);
    hipStreamSynchronize(stream);

    // Host selects top 8 experts
    float max_l = -1e30f;
    for (uint32_t e = 0; e < cfg.num_experts; ++e) {
      max_l = std::max(max_l, state.h_router_logits[e]);
    }
    float sum_exp = 0.0f;
    for (uint32_t e = 0; e < cfg.num_experts; ++e) {
      const float val = std::exp(state.h_router_logits[e] - max_l);
      state.h_probs[e] = {val, e};
      sum_exp += val;
    }
    const float inv_sum = 1.0f / sum_exp;
    for (uint32_t e = 0; e < cfg.num_experts; ++e) {
      state.h_probs[e].first *= inv_sum;
    }
    std::partial_sort(state.h_probs.begin(), state.h_probs.begin() + cfg.num_experts_used, state.h_probs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    float top_sum = 0.0f;
    for (uint32_t k = 0; k < cfg.num_experts_used; ++k) {
      top_sum += state.h_probs[k].first;
    }
    const float inv_top_sum = 1.0f / (top_sum > 0.0f ? top_sum : 1.0f);

    // Zero routed accum buffer
    hipMemsetAsync(state.d_routed_accum, 0, H * sizeof(float), stream);

    // Execute selected experts
    const uint64_t expert_stride_gate = layer.ffn_gate_exps.RowBytes() * cfg.expert_ff;
    const uint64_t expert_stride_up = layer.ffn_up_exps.RowBytes() * cfg.expert_ff;
    const uint64_t expert_stride_down = layer.ffn_down_exps.RowBytes() * H;

    for (uint32_t k = 0; k < cfg.num_experts_used; ++k) {
      const uint32_t expert_id = state.h_probs[k].second;
      const float weight = state.h_probs[k].first * inv_top_sum;

      DispatchGemv(stream, layer.ffn_gate_exps.data, layer.ffn_gate_exps.type,
                   expert_stride_gate, expert_id, state.d_x_normed, state.d_expert_gate, cfg.expert_ff, H);
      DispatchGemv(stream, layer.ffn_up_exps.data, layer.ffn_up_exps.type,
                   expert_stride_up, expert_id, state.d_x_normed, state.d_expert_up, cfg.expert_ff, H);

      DispatchSiluMul(stream, state.d_expert_gate, state.d_expert_up, cfg.expert_ff, state.d_expert_act);

      DispatchGemv(stream, layer.ffn_down_exps.data, layer.ffn_down_exps.type,
                   expert_stride_down, expert_id, state.d_expert_act, state.d_expert_down, H, cfg.expert_ff);

      DispatchAccumWeighted(stream, state.d_expert_down, weight, H, state.d_routed_accum);
    }

    // Shared expert
    DispatchGemv(stream, layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type, 0, 0,
                 state.d_x_normed, state.d_shexp_gate, cfg.shared_expert_ff, H);
    DispatchGemv(stream, layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, 0, 0,
                 state.d_x_normed, state.d_shexp_up, cfg.shared_expert_ff, H);

    DispatchSiluMul(stream, state.d_shexp_gate, state.d_shexp_up, cfg.shared_expert_ff, state.d_shexp_act);

    DispatchGemv(stream, layer.ffn_down_shexp.data, layer.ffn_down_shexp.type, 0, 0,
                 state.d_shexp_act, state.d_shexp_down, H, cfg.shared_expert_ff);

    // Shared expert gating dot product
    float shexp_gate_dot = 0.0f;
    if (!layer.ffn_gate_inp_shexp.empty()) {
      DispatchGemv(stream, layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_dot, 1, H);
      hipMemcpyDtoHAsync(&shexp_gate_dot, state.d_shexp_dot, sizeof(float), stream);
      hipStreamSynchronize(stream);
    }

    // Combine routed + shared expert
    DispatchMoECombineShared(stream, state.d_routed_accum, state.d_shexp_down, shexp_gate_dot, H, state.d_moe_out);

    // Residual add: d_hidden += d_moe_out
    DispatchResidualAdd(stream, state.d_moe_out, H, state.d_hidden);
  }

  // Final Output Norm & LM Head (if out_logits requested)
  if (!out_logits.empty()) {
    DispatchRMSNorm(stream, state.d_hidden, weights.output_norm.data, weights.output_norm.type,
                    cfg.rms_eps, H, state.d_x_normed);

    DispatchGemv(stream, weights.output.data, weights.output.type, 0, 0,
                 state.d_x_normed, state.d_logits, cfg.vocab_size, H);

    hipMemcpyDtoHAsync(out_logits.data(), state.d_logits, cfg.vocab_size * sizeof(float), stream);
    hipStreamSynchronize(stream);
  } else {
    hipStreamSynchronize(stream);
  }
}

}  // namespace gufo::models::qwen35_moe::hip
#endif  // defined(ENGINE_ENABLE_HIP)
