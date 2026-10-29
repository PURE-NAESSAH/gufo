#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen35_moe/hip/gpu_runner.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "src/core/hip/managed_alloc.hpp"
#include "src/models/qwen35_moe/hip/kernels.hpp"

namespace gufo::models::qwen35_moe::hip {

namespace {

void* AllocBytes(std::string_view name, size_t bytes, std::vector<void*>& record) {
  void* ptr = nullptr;
  hipError_t status = gufo::hip::AllocateDevice(&ptr, bytes);
  if (status == hipSuccess && ptr != nullptr) {
    record.push_back(ptr);
  } else {
    std::fprintf(stderr, "[GpuState] %.*s allocation failed (%zu bytes): %s\n",
                 static_cast<int>(name.size()), name.data(), bytes,
                 hipGetErrorString(status));
    NoteHipFailure(name, status != hipSuccess ? status : hipErrorMemoryAllocation);
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

  d_hidden = static_cast<float*>(AllocBytes("d_hidden", kMaxChunk * H * sizeof(float), allocations));
  d_x_normed = static_cast<float*>(AllocBytes("d_x_normed", kMaxChunk * H * sizeof(float), allocations));
  d_attn_out = static_cast<float*>(AllocBytes("d_attn_out", kMaxChunk * H * sizeof(float), allocations));
  d_token_ids = static_cast<uint32_t*>(AllocBytes("d_token_ids", kMaxChunk * sizeof(uint32_t), allocations));

  // Full attention
  d_qg_buf = static_cast<float*>(AllocBytes("d_qg_buf", kMaxChunk * 2 * num_heads * head_dim * sizeof(float), allocations));
  d_q_buf = static_cast<float*>(AllocBytes("d_q_buf", kMaxChunk * num_heads * head_dim * sizeof(float), allocations));
  d_gate_buf = static_cast<float*>(AllocBytes("d_gate_buf", kMaxChunk * num_heads * head_dim * sizeof(float), allocations));
  d_k_buf = static_cast<float*>(AllocBytes("d_k_buf", kMaxChunk * num_kv_heads * head_dim * sizeof(float), allocations));
  d_v_buf = static_cast<float*>(AllocBytes("d_v_buf", kMaxChunk * num_kv_heads * head_dim * sizeof(float), allocations));
  d_context = static_cast<float*>(AllocBytes("d_context", kMaxChunk * num_heads * head_dim * sizeof(float), allocations));

  const size_t kv_elements = static_cast<size_t>(full_layers) * num_kv_heads * max_context * head_dim;
  d_kv_k = static_cast<float*>(AllocBytes("d_kv_k", kv_elements * sizeof(float), allocations));
  d_kv_v = static_cast<float*>(AllocBytes("d_kv_v", kv_elements * sizeof(float), allocations));

  // SSM
  d_ssm_qkv = static_cast<float*>(AllocBytes("d_ssm_qkv", kMaxChunk * conv_channels * sizeof(float), allocations));
  d_ssm_gate = static_cast<float*>(AllocBytes("d_ssm_gate", kMaxChunk * val_dim * sizeof(float), allocations));
  d_ssm_alpha = static_cast<float*>(AllocBytes("d_ssm_alpha", kMaxChunk * v_heads * sizeof(float), allocations));
  d_ssm_beta = static_cast<float*>(AllocBytes("d_ssm_beta", kMaxChunk * v_heads * sizeof(float), allocations));
  d_decay = static_cast<float*>(AllocBytes("d_decay", kMaxChunk * v_heads * sizeof(float), allocations));
  d_beta_sig = static_cast<float*>(AllocBytes("d_beta_sig", kMaxChunk * v_heads * sizeof(float), allocations));
  d_ssm_conv_out = static_cast<float*>(AllocBytes("d_ssm_conv_out", kMaxChunk * conv_channels * sizeof(float), allocations));
  d_ssm_q_normed = static_cast<float*>(AllocBytes("d_ssm_q_normed", kMaxChunk * cfg.SsmKeyDim() * sizeof(float), allocations));
  d_ssm_k_normed = static_cast<float*>(AllocBytes("d_ssm_k_normed", kMaxChunk * cfg.SsmKeyDim() * sizeof(float), allocations));
  d_ssm_core_out = static_cast<float*>(AllocBytes("d_ssm_core_out", kMaxChunk * val_dim * sizeof(float), allocations));

  const size_t conv_elements = static_cast<size_t>(ssm_layers) * 3 * conv_channels;
  d_conv_state = static_cast<float*>(AllocBytes("d_conv_state", conv_elements * sizeof(float), allocations));

  const size_t deltanet_elements = static_cast<size_t>(ssm_layers) * v_heads * ssm_head_dim * ssm_head_dim;
  d_deltanet_state = static_cast<float*>(AllocBytes("d_deltanet_state", deltanet_elements * sizeof(float), allocations));

  // MoE
  d_router_logits = static_cast<float*>(AllocBytes("d_router_logits", kMaxChunk * cfg.num_experts * sizeof(float), allocations));
  d_expert_gate = static_cast<float*>(AllocBytes("d_expert_gate", expert_ff * sizeof(float), allocations));
  d_expert_up = static_cast<float*>(AllocBytes("d_expert_up", expert_ff * sizeof(float), allocations));
  d_expert_act = static_cast<float*>(AllocBytes("d_expert_act", kMaxChunk * cfg.num_experts_used * expert_ff * sizeof(float), allocations));
  d_expert_down = static_cast<float*>(AllocBytes("d_expert_down", H * sizeof(float), allocations));
  d_routed_accum = static_cast<float*>(AllocBytes("d_routed_accum", kMaxChunk * H * sizeof(float), allocations));
  d_expert_ids = static_cast<uint32_t*>(AllocBytes("d_expert_ids", kMaxChunk * cfg.num_experts_used * sizeof(uint32_t), allocations));
  d_expert_weights = static_cast<float*>(AllocBytes("d_expert_weights", kMaxChunk * cfg.num_experts_used * sizeof(float), allocations));
  d_expert_counts = static_cast<uint32_t*>(AllocBytes("d_expert_counts", cfg.num_experts * sizeof(uint32_t), allocations));
  d_expert_offsets = static_cast<uint32_t*>(AllocBytes("d_expert_offsets", (cfg.num_experts + 1) * sizeof(uint32_t), allocations));
  d_expert_cursors = static_cast<uint32_t*>(AllocBytes("d_expert_cursors", cfg.num_experts * sizeof(uint32_t), allocations));
  d_sorted_slots = static_cast<uint32_t*>(AllocBytes("d_sorted_slots", kMaxChunk * cfg.num_experts_used * sizeof(uint32_t), allocations));
  d_tok_tile_expert = static_cast<uint32_t*>(AllocBytes("d_tok_tile_expert", 8192 * sizeof(uint32_t), allocations));
  d_tok_tile_offset = static_cast<uint32_t*>(AllocBytes("d_tok_tile_offset", 8192 * sizeof(uint32_t), allocations));
  d_num_tiles = static_cast<uint32_t*>(AllocBytes("d_num_tiles", sizeof(uint32_t), allocations));
  d_shexp_gate = static_cast<float*>(AllocBytes("d_shexp_gate", kMaxChunk * shared_expert_ff * sizeof(float), allocations));
  d_shexp_up = static_cast<float*>(AllocBytes("d_shexp_up", kMaxChunk * shared_expert_ff * sizeof(float), allocations));
  d_shexp_act = static_cast<float*>(AllocBytes("d_shexp_act", kMaxChunk * shared_expert_ff * sizeof(float), allocations));
  d_shexp_down = static_cast<float*>(AllocBytes("d_shexp_down", kMaxChunk * H * sizeof(float), allocations));
  d_shexp_dot = static_cast<float*>(AllocBytes("d_shexp_dot", kMaxChunk * sizeof(float), allocations));
  d_moe_out = static_cast<float*>(AllocBytes("d_moe_out", kMaxChunk * H * sizeof(float), allocations));

  // Logits
  d_logits = static_cast<float*>(AllocBytes("d_logits", cfg.vocab_size * sizeof(float), allocations));

  h_router_logits.resize(kMaxChunk * cfg.num_experts, 0.0f);
  h_probs.resize(cfg.num_experts);
  h_expert_ids.resize(kMaxChunk * cfg.num_experts_used, 0);
  h_expert_weights.resize(kMaxChunk * cfg.num_experts_used, 0.0f);

  if (main_stream == nullptr) {
    hipError_t err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
    if (err != hipSuccess) {
      std::fprintf(stderr, "[GpuState] hipStreamCreateWithFlags(main_stream) failed: %s\n", hipGetErrorString(err));
      main_stream = nullptr;
    }
  }
  if (side_stream == nullptr) {
    hipError_t err = hipStreamCreateWithFlags(&side_stream, hipStreamNonBlocking);
    if (err != hipSuccess) {
      std::fprintf(stderr, "[GpuState] hipStreamCreateWithFlags(side_stream) failed: %s\n", hipGetErrorString(err));
      side_stream = nullptr;
    }
  }
  if (side_start == nullptr) {
    GUFO_QWEN35_HIP(hipEventCreateWithFlags(&side_start, hipEventDisableTiming));
  }
  if (side_done == nullptr) {
    GUFO_QWEN35_HIP(hipEventCreateWithFlags(&side_done, hipEventDisableTiming));
  }

  Reset(cfg, max_ctx);
}

void GpuState::Free() {
  if (main_stream != nullptr) {
    (void)hipStreamSynchronize(main_stream);
    (void)hipStreamDestroy(main_stream);
    main_stream = nullptr;
  }
  if (side_stream != nullptr) {
    (void)hipStreamSynchronize(side_stream);
    (void)hipStreamDestroy(side_stream);
    side_stream = nullptr;
  }
  if (side_start != nullptr) {
    (void)hipEventDestroy(side_start);
    side_start = nullptr;
  }
  if (side_done != nullptr) {
    (void)hipEventDestroy(side_done);
    side_done = nullptr;
  }

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
  d_expert_ids = nullptr;
  d_expert_weights = nullptr;
  d_expert_counts = nullptr;
  d_expert_offsets = nullptr;
  d_expert_cursors = nullptr;
  d_sorted_slots = nullptr;
  d_tok_tile_expert = nullptr;
  d_tok_tile_offset = nullptr;
  d_num_tiles = nullptr;
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
    GUFO_QWEN35_HIP(hipMemsetAsync(d_conv_state, 0, conv_elements * sizeof(float), main_stream));
  }
  if (d_deltanet_state != nullptr) {
    const size_t deltanet_elements = static_cast<size_t>(ssm_layers) * v_heads * ssm_head_dim * ssm_head_dim;
    GUFO_QWEN35_HIP(hipMemsetAsync(d_deltanet_state, 0, deltanet_elements * sizeof(float), main_stream));
  }
  if (d_kv_k != nullptr && d_kv_v != nullptr) {
    const size_t kv_elements = static_cast<size_t>(full_layers) * cfg.num_kv_heads * max_context * cfg.head_dim;
    GUFO_QWEN35_HIP(hipMemsetAsync(d_kv_k, 0, kv_elements * sizeof(float), main_stream));
    GUFO_QWEN35_HIP(hipMemsetAsync(d_kv_v, 0, kv_elements * sizeof(float), main_stream));
  }
  if (main_stream != nullptr) {
    GUFO_QWEN35_HIP(hipStreamSynchronize(main_stream));
  } else {
    GUFO_QWEN35_HIP(hipDeviceSynchronize());
  }
}

void ForwardModelGpu(
    uint32_t token_id,
    uint32_t pos,
    const ModelWeights& weights,
    GpuState& state,
    std::span<float> out_logits) {
  ResetHipError();
  const hipStream_t main = state.main_stream;
  const hipStream_t side = (state.side_stream != nullptr) ? state.side_stream : main;
  const bool has_side = (state.side_stream != nullptr && state.side_start != nullptr && state.side_done != nullptr);
  const auto& cfg = weights.config;
  const uint32_t H = cfg.hidden_size;

  // 1. Embedding lookup -> state.d_hidden
  DispatchEmbedding(main, weights.token_embd.data, weights.token_embd.type,
                    token_id, H, weights.token_embd.RowBytes(), state.d_hidden);

  uint32_t full_attn_idx = 0;
  uint32_t ssm_layer_idx = 0;

  for (uint32_t l = 0; l < cfg.num_layers; ++l) {
    const auto& layer = weights.layers[l];

    // 2. Pre-attention RMSNorm: d_hidden -> d_x_normed
    DispatchRMSNorm(main, state.d_hidden, layer.attn_norm.data, layer.attn_norm.type,
                    cfg.rms_eps, H, state.d_x_normed);

    if (layer.is_full_attention) {
      if (has_side) {
        GUFO_QWEN35_HIP(hipEventRecord(state.side_start, main));
        GUFO_QWEN35_HIP(hipStreamWaitEvent(side, state.side_start, 0));

        // K and V projections on side stream
        DispatchGemv(side, layer.attn_k.data, layer.attn_k.type, 0, 0,
                     state.d_x_normed, state.d_k_buf, cfg.num_kv_heads * cfg.head_dim, H);
        DispatchGemv(side, layer.attn_v.data, layer.attn_v.type, 0, 0,
                     state.d_x_normed, state.d_v_buf, cfg.num_kv_heads * cfg.head_dim, H);
        GUFO_QWEN35_HIP(hipEventRecord(state.side_done, side));
      } else {
        DispatchGemv(main, layer.attn_k.data, layer.attn_k.type, 0, 0,
                     state.d_x_normed, state.d_k_buf, cfg.num_kv_heads * cfg.head_dim, H);
        DispatchGemv(main, layer.attn_v.data, layer.attn_v.type, 0, 0,
                     state.d_x_normed, state.d_v_buf, cfg.num_kv_heads * cfg.head_dim, H);
      }

      // Joint Q + Gate projection on main stream
      DispatchGemv(main, layer.attn_q.data, layer.attn_q.type, 0, 0,
                   state.d_x_normed, state.d_qg_buf, 2 * cfg.num_heads * cfg.head_dim, H);

      // Unpack Q and Gate
      DispatchUnpackQG(main, state.d_qg_buf, cfg.num_heads, cfg.head_dim,
                       state.d_q_buf, state.d_gate_buf);

      // Q-norm and RoPE
      DispatchRMSNormBatch(main, state.d_q_buf, layer.attn_q_norm.data, layer.attn_q_norm.type,
                           cfg.rms_eps, cfg.head_dim, cfg.num_heads, state.d_q_buf);
      DispatchRoPE(main, state.d_q_buf, cfg.rotary_dim, cfg.head_dim, cfg.num_heads, pos, cfg.rope_theta);

      if (has_side) {
        GUFO_QWEN35_HIP(hipStreamWaitEvent(main, state.side_done, 0));
      }

      // K-norm and RoPE
      DispatchRMSNormBatch(main, state.d_k_buf, layer.attn_k_norm.data, layer.attn_k_norm.type,
                           cfg.rms_eps, cfg.head_dim, cfg.num_kv_heads, state.d_k_buf);
      DispatchRoPE(main, state.d_k_buf, cfg.rotary_dim, cfg.head_dim, cfg.num_kv_heads, pos, cfg.rope_theta);

      // Store in KV cache
      float* layer_kv_k = state.d_kv_k +
                          (static_cast<size_t>(full_attn_idx) * cfg.num_kv_heads * state.max_context * cfg.head_dim);
      float* layer_kv_v = state.d_kv_v +
                          (static_cast<size_t>(full_attn_idx) * cfg.num_kv_heads * state.max_context * cfg.head_dim);
      DispatchStoreKV(main, state.d_k_buf, state.d_v_buf, cfg.num_kv_heads, cfg.head_dim,
                      state.max_context, pos, layer_kv_k, layer_kv_v);

      // Attention decode
      DispatchAttentionDecode(main, state.d_q_buf, state.d_gate_buf, layer_kv_k, layer_kv_v,
                              cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, state.max_context, pos,
                              state.d_context);

      // Output projection: [num_heads * head_dim -> H]
      DispatchGemv(main, layer.attn_output.data, layer.attn_output.type, 0, 0,
                   state.d_context, state.d_attn_out, H, cfg.num_heads * cfg.head_dim);

      ++full_attn_idx;
    } else {
      // Linear Attention (SSM)
      const uint32_t val_dim = cfg.SsmValueDim();
      const uint32_t conv_channels = cfg.SsmConvChannels();
      const uint32_t v_heads = cfg.ssm_num_v_heads;
      const uint32_t k_heads = cfg.ssm_num_k_heads;
      const uint32_t ssm_head_dim = cfg.ssm_head_dim;

      if (has_side) {
        GUFO_QWEN35_HIP(hipEventRecord(state.side_start, main));
        GUFO_QWEN35_HIP(hipStreamWaitEvent(side, state.side_start, 0));

        // Side stream computes gate, alpha, beta in parallel
        DispatchGemv(side, layer.ssm_gate.data, layer.ssm_gate.type, 0, 0,
                     state.d_x_normed, state.d_ssm_gate, val_dim, H);
        DispatchGemv(side, layer.ssm_alpha.data, layer.ssm_alpha.type, 0, 0,
                     state.d_x_normed, state.d_ssm_alpha, v_heads, H);
        DispatchGemv(side, layer.ssm_beta.data, layer.ssm_beta.type, 0, 0,
                     state.d_x_normed, state.d_ssm_beta, v_heads, H);
        GUFO_QWEN35_HIP(hipEventRecord(state.side_done, side));
      } else {
        DispatchGemv(main, layer.ssm_gate.data, layer.ssm_gate.type, 0, 0,
                     state.d_x_normed, state.d_ssm_gate, val_dim, H);
        DispatchGemv(main, layer.ssm_alpha.data, layer.ssm_alpha.type, 0, 0,
                     state.d_x_normed, state.d_ssm_alpha, v_heads, H);
        DispatchGemv(main, layer.ssm_beta.data, layer.ssm_beta.type, 0, 0,
                     state.d_x_normed, state.d_ssm_beta, v_heads, H);
      }

      // Main stream computes ssm_qkv
      DispatchGemv(main, layer.ssm_qkv.data, layer.ssm_qkv.type, 0, 0,
                   state.d_x_normed, state.d_ssm_qkv, conv_channels, H);

      if (has_side) {
        GUFO_QWEN35_HIP(hipStreamWaitEvent(main, state.side_done, 0));
      }

      float* layer_conv_state = state.d_conv_state + (static_cast<size_t>(ssm_layer_idx) * 3 * conv_channels);
      float* layer_deltanet_state =
          state.d_deltanet_state + (static_cast<size_t>(ssm_layer_idx) * v_heads * ssm_head_dim * ssm_head_dim);

      DispatchSsmStep(
          main, state.d_ssm_qkv, state.d_ssm_gate, state.d_ssm_alpha, state.d_ssm_beta,
          layer.ssm_dt.data, layer.ssm_a.data, layer.ssm_conv1d.data, layer.ssm_norm.data,
          conv_channels, v_heads, k_heads, ssm_head_dim, cfg.rms_eps,
          layer_conv_state, layer_deltanet_state,
          state.d_decay, state.d_beta_sig, state.d_ssm_conv_out,
          state.d_ssm_q_normed, state.d_ssm_k_normed, state.d_ssm_core_out);

      DispatchGemv(main, layer.ssm_out.data, layer.ssm_out.type, 0, 0,
                   state.d_ssm_core_out, state.d_attn_out, H, val_dim);

      ++ssm_layer_idx;
    }

    // Residual add: d_hidden += d_attn_out
    DispatchResidualAdd(main, state.d_attn_out, H, state.d_hidden);

    // Post-attention RMSNorm: d_hidden -> d_x_normed
    DispatchRMSNorm(main, state.d_hidden, layer.post_attention_norm.data, layer.post_attention_norm.type,
                    cfg.rms_eps, H, state.d_x_normed);

    // MoE Routing & FFN with Shared Expert Overlap
    if (has_side) {
      GUFO_QWEN35_HIP(hipEventRecord(state.side_start, main));
      GUFO_QWEN35_HIP(hipStreamWaitEvent(side, state.side_start, 0));

      // Shared expert evaluated in parallel on side stream
      DispatchGemv(side, layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_gate, cfg.shared_expert_ff, H);
      DispatchGemv(side, layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_up, cfg.shared_expert_ff, H);

      DispatchSiluMul(side, state.d_shexp_gate, state.d_shexp_up, cfg.shared_expert_ff, state.d_shexp_act);

      DispatchGemv(side, layer.ffn_down_shexp.data, layer.ffn_down_shexp.type, 0, 0,
                   state.d_shexp_act, state.d_shexp_down, H, cfg.shared_expert_ff);

      // Shared expert gating dot product
      if (!layer.ffn_gate_inp_shexp.empty()) {
        DispatchGemv(side, layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type, 0, 0,
                     state.d_x_normed, state.d_shexp_dot, 1, H);
      } else {
        GUFO_QWEN35_HIP(hipMemsetAsync(state.d_shexp_dot, 0, sizeof(float), side));
      }
      GUFO_QWEN35_HIP(hipEventRecord(state.side_done, side));
    } else {
      DispatchGemv(main, layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_gate, cfg.shared_expert_ff, H);
      DispatchGemv(main, layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_up, cfg.shared_expert_ff, H);

      DispatchSiluMul(main, state.d_shexp_gate, state.d_shexp_up, cfg.shared_expert_ff, state.d_shexp_act);

      DispatchGemv(main, layer.ffn_down_shexp.data, layer.ffn_down_shexp.type, 0, 0,
                   state.d_shexp_act, state.d_shexp_down, H, cfg.shared_expert_ff);

      if (!layer.ffn_gate_inp_shexp.empty()) {
        DispatchGemv(main, layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type, 0, 0,
                     state.d_x_normed, state.d_shexp_dot, 1, H);
      } else {
        GUFO_QWEN35_HIP(hipMemsetAsync(state.d_shexp_dot, 0, sizeof(float), main));
      }
    }

    // Router GEMV on main stream: [H -> num_experts]
    DispatchGemv(main, layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, 0, 0,
                 state.d_x_normed, state.d_router_logits, cfg.num_experts, H);

    // On-device Top-K selection (zero host sync!)
    DispatchRouterSoftmaxTopK(main, state.d_router_logits, cfg.num_experts,
                              cfg.num_experts_used, 1, state.d_expert_ids, state.d_expert_weights);

    const uint64_t expert_stride_gate = layer.ffn_gate_exps.RowBytes() * cfg.expert_ff;
    const uint64_t expert_stride_up = layer.ffn_up_exps.RowBytes() * cfg.expert_ff;
    const uint64_t expert_stride_down = layer.ffn_down_exps.RowBytes() * H;

    // Batched MoE: all experts in parallel on main stream
    DispatchMoeGatedSwigluBatch(
        main, layer.ffn_gate_exps.data, layer.ffn_up_exps.data, layer.ffn_gate_exps.type,
        expert_stride_gate, expert_stride_up, state.d_x_normed, state.d_expert_ids,
        1, cfg.num_experts_used, cfg.expert_ff, H, state.d_expert_act);

    DispatchMoeDownAccumBatch(
        main, layer.ffn_down_exps.data, layer.ffn_down_exps.type,
        expert_stride_down, state.d_expert_act, state.d_expert_weights, state.d_expert_ids,
        1, cfg.num_experts_used, cfg.expert_ff, H, state.d_routed_accum);

    if (has_side) {
      GUFO_QWEN35_HIP(hipStreamWaitEvent(main, state.side_done, 0));
    }

    // Combine routed + shared expert
    DispatchMoECombineSharedBatch(main, state.d_routed_accum, state.d_shexp_down, state.d_shexp_dot, H, 1, state.d_moe_out);

    // Residual add: d_hidden += d_moe_out
    DispatchResidualAdd(main, state.d_moe_out, H, state.d_hidden);
    if (HipErrorPending()) {
      break;
    }
  }

  // Final Output Norm & LM Head (if out_logits requested)
  if (!out_logits.empty()) {
    DispatchRMSNorm(main, state.d_hidden, weights.output_norm.data, weights.output_norm.type,
                    cfg.rms_eps, H, state.d_x_normed);

    DispatchGemv(main, weights.output.data, weights.output.type, 0, 0,
                 state.d_x_normed, state.d_logits, cfg.vocab_size, H);

    GUFO_QWEN35_HIP(hipMemcpyDtoHAsync(out_logits.data(), state.d_logits, cfg.vocab_size * sizeof(float), main));
    GUFO_QWEN35_HIP(hipStreamSynchronize(main));
  } else {
    GUFO_QWEN35_HIP(hipStreamSynchronize(main));
  }
}


void ForwardModelGpuChunk(
    std::span<const uint32_t> tokens,
    uint32_t pos0,
    const ModelWeights& weights,
    GpuState& state,
    std::span<float> out_logits) {
  if (tokens.empty()) return;

  const uint32_t B = static_cast<uint32_t>(tokens.size());
  if (B == 1) {
    ForwardModelGpu(tokens[0], pos0, weights, state, out_logits);
    return;
  }
  ResetHipError();
  auto t_chunk_start = std::chrono::high_resolution_clock::now();
  const auto& cfg = weights.config;
  const uint32_t H = cfg.hidden_size;
  const hipStream_t main = state.main_stream;
  const hipStream_t side = (state.side_stream != nullptr) ? state.side_stream : main;
  const bool has_side = (state.side_stream != nullptr && state.side_start != nullptr && state.side_done != nullptr);

  // 1. Copy tokens to device and dispatch batched embedding
  GUFO_QWEN35_HIP(hipMemcpyHtoDAsync(state.d_token_ids, tokens.data(), B * sizeof(uint32_t), main));
  const size_t embd_row_bytes = weights.token_embd.RowBytes();
  DispatchEmbeddingBatch(main, weights.token_embd.data, weights.token_embd.type,
                         state.d_token_ids, B, H, embd_row_bytes, state.d_hidden);

  uint32_t full_attn_idx = 0;
  uint32_t ssm_layer_idx = 0;

  for (uint32_t l = 0; l < cfg.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    // Pre-attention RMSNorm: [B, H] -> d_x_normed
    DispatchRMSNormBatch(main, state.d_hidden, layer.attn_norm.data, layer.attn_norm.type,
                         cfg.rms_eps, H, B, state.d_x_normed);

    if (layer.is_full_attention) {
      if (has_side) {
        GUFO_QWEN35_HIP(hipEventRecord(state.side_start, main));
        GUFO_QWEN35_HIP(hipStreamWaitEvent(side, state.side_start, 0));

        // K and V projections on side stream
        DispatchGemm(side, layer.attn_k.data, layer.attn_k.type, 0, 0,
                     state.d_x_normed, state.d_k_buf, cfg.num_kv_heads * cfg.head_dim, H, B);
        DispatchGemm(side, layer.attn_v.data, layer.attn_v.type, 0, 0,
                     state.d_x_normed, state.d_v_buf, cfg.num_kv_heads * cfg.head_dim, H, B);
        GUFO_QWEN35_HIP(hipEventRecord(state.side_done, side));
      } else {
        DispatchGemm(main, layer.attn_k.data, layer.attn_k.type, 0, 0,
                     state.d_x_normed, state.d_k_buf, cfg.num_kv_heads * cfg.head_dim, H, B);
        DispatchGemm(main, layer.attn_v.data, layer.attn_v.type, 0, 0,
                     state.d_x_normed, state.d_v_buf, cfg.num_kv_heads * cfg.head_dim, H, B);
      }

      // Full Attention Q projection on main stream
      DispatchGemm(main, layer.attn_q.data, layer.attn_q.type, 0, 0,
                   state.d_x_normed, state.d_qg_buf, 2 * cfg.num_heads * cfg.head_dim, H, B);

      // Unpack Q and Gate
      DispatchUnpackQGBatch(main, state.d_qg_buf, cfg.num_heads, cfg.head_dim, B,
                            state.d_q_buf, state.d_gate_buf);

      // Q-norm and RoPE
      DispatchRMSNormBatch(main, state.d_q_buf, layer.attn_q_norm.data, layer.attn_q_norm.type,
                           cfg.rms_eps, cfg.head_dim, B * cfg.num_heads, state.d_q_buf);
      DispatchRoPEBatch(main, state.d_q_buf, cfg.rotary_dim, cfg.head_dim, cfg.num_heads,
                        pos0, B, cfg.rope_theta);

      if (has_side) {
        GUFO_QWEN35_HIP(hipStreamWaitEvent(main, state.side_done, 0));
      }

      // K-norm and RoPE
      DispatchRMSNormBatch(main, state.d_k_buf, layer.attn_k_norm.data, layer.attn_k_norm.type,
                           cfg.rms_eps, cfg.head_dim, B * cfg.num_kv_heads, state.d_k_buf);
      DispatchRoPEBatch(main, state.d_k_buf, cfg.rotary_dim, cfg.head_dim, cfg.num_kv_heads,
                        pos0, B, cfg.rope_theta);

      // Store in KV cache
      float* layer_kv_k = state.d_kv_k +
                          (static_cast<size_t>(full_attn_idx) * cfg.num_kv_heads * state.max_context * cfg.head_dim);
      float* layer_kv_v = state.d_kv_v +
                          (static_cast<size_t>(full_attn_idx) * cfg.num_kv_heads * state.max_context * cfg.head_dim);
      DispatchStoreKVBatch(main, state.d_k_buf, state.d_v_buf, cfg.num_kv_heads, cfg.head_dim,
                           state.max_context, pos0, B, layer_kv_k, layer_kv_v);

      // Causal Attention Prefill
      DispatchAttentionPrefill(main, state.d_q_buf, state.d_gate_buf, layer_kv_k, layer_kv_v,
                               cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, state.max_context,
                               pos0, B, state.d_context);

      // Output projection: [num_heads * head_dim -> H]
      DispatchGemm(main, layer.attn_output.data, layer.attn_output.type, 0, 0,
                   state.d_context, state.d_attn_out, H, cfg.num_heads * cfg.head_dim, B);

      ++full_attn_idx;
    } else {
      // Linear Attention (SSM)
      const uint32_t val_dim = cfg.SsmValueDim();
      const uint32_t conv_channels = cfg.SsmConvChannels();
      const uint32_t v_heads = cfg.ssm_num_v_heads;
      const uint32_t k_heads = cfg.ssm_num_k_heads;
      const uint32_t ssm_head_dim = cfg.ssm_head_dim;

      if (has_side) {
        GUFO_QWEN35_HIP(hipEventRecord(state.side_start, main));
        GUFO_QWEN35_HIP(hipStreamWaitEvent(side, state.side_start, 0));

        // Side stream computes gate, alpha, beta GEMMs
        DispatchGemm(side, layer.ssm_gate.data, layer.ssm_gate.type, 0, 0,
                     state.d_x_normed, state.d_ssm_gate, val_dim, H, B);
        DispatchGemm(side, layer.ssm_alpha.data, layer.ssm_alpha.type, 0, 0,
                     state.d_x_normed, state.d_ssm_alpha, v_heads, H, B);
        DispatchGemm(side, layer.ssm_beta.data, layer.ssm_beta.type, 0, 0,
                     state.d_x_normed, state.d_ssm_beta, v_heads, H, B);
        GUFO_QWEN35_HIP(hipEventRecord(state.side_done, side));
      } else {
        DispatchGemm(main, layer.ssm_gate.data, layer.ssm_gate.type, 0, 0,
                     state.d_x_normed, state.d_ssm_gate, val_dim, H, B);
        DispatchGemm(main, layer.ssm_alpha.data, layer.ssm_alpha.type, 0, 0,
                     state.d_x_normed, state.d_ssm_alpha, v_heads, H, B);
        DispatchGemm(main, layer.ssm_beta.data, layer.ssm_beta.type, 0, 0,
                     state.d_x_normed, state.d_ssm_beta, v_heads, H, B);
      }

      // Main stream computes ssm_qkv GEMM
      DispatchGemm(main, layer.ssm_qkv.data, layer.ssm_qkv.type, 0, 0,
                   state.d_x_normed, state.d_ssm_qkv, conv_channels, H, B);

      if (has_side) {
        GUFO_QWEN35_HIP(hipStreamWaitEvent(main, state.side_done, 0));
      }

      float* layer_conv_state = state.d_conv_state + (static_cast<size_t>(ssm_layer_idx) * 3 * conv_channels);
      float* layer_deltanet_state =
          state.d_deltanet_state + (static_cast<size_t>(ssm_layer_idx) * v_heads * ssm_head_dim * ssm_head_dim);

      DispatchSsmChunk(
          main, state.d_ssm_qkv, state.d_ssm_gate, state.d_ssm_alpha, state.d_ssm_beta,
          layer.ssm_dt.data, layer.ssm_a.data, layer.ssm_conv1d.data, layer.ssm_norm.data,
          conv_channels, v_heads, k_heads, ssm_head_dim, B, cfg.rms_eps,
          layer_conv_state, layer_deltanet_state,
          state.d_decay, state.d_beta_sig, state.d_ssm_conv_out,
          state.d_ssm_q_normed, state.d_ssm_k_normed, state.d_ssm_core_out);

      DispatchGemm(main, layer.ssm_out.data, layer.ssm_out.type, 0, 0,
                   state.d_ssm_core_out, state.d_attn_out, H, val_dim, B);

      ++ssm_layer_idx;
    }

    // Residual add: d_hidden += d_attn_out across all B tokens
    DispatchResidualAdd(main, state.d_attn_out, B * H, state.d_hidden);

    // Post-attention RMSNorm: d_hidden -> d_x_normed
    DispatchRMSNormBatch(main, state.d_hidden, layer.post_attention_norm.data, layer.post_attention_norm.type,
                         cfg.rms_eps, H, B, state.d_x_normed);

    // MoE Routing & FFN with Shared Expert Overlap
    if (has_side) {
      GUFO_QWEN35_HIP(hipEventRecord(state.side_start, main));
      GUFO_QWEN35_HIP(hipStreamWaitEvent(side, state.side_start, 0));

      // Shared expert evaluated for all B tokens in one batched GEMM on side stream
      DispatchGemm(side, layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_gate, cfg.shared_expert_ff, H, B);
      DispatchGemm(side, layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_up, cfg.shared_expert_ff, H, B);

      DispatchSiluMul(side, state.d_shexp_gate, state.d_shexp_up, B * cfg.shared_expert_ff, state.d_shexp_act);

      DispatchGemm(side, layer.ffn_down_shexp.data, layer.ffn_down_shexp.type, 0, 0,
                   state.d_shexp_act, state.d_shexp_down, H, cfg.shared_expert_ff, B);

      // Shared expert gating dot product
      if (!layer.ffn_gate_inp_shexp.empty()) {
        DispatchGemm(side, layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type, 0, 0,
                     state.d_x_normed, state.d_shexp_dot, 1, H, B);
      } else {
        GUFO_QWEN35_HIP(hipMemsetAsync(state.d_shexp_dot, 0, B * sizeof(float), side));
      }
      GUFO_QWEN35_HIP(hipEventRecord(state.side_done, side));
    } else {
      DispatchGemm(main, layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_gate, cfg.shared_expert_ff, H, B);
      DispatchGemm(main, layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, 0, 0,
                   state.d_x_normed, state.d_shexp_up, cfg.shared_expert_ff, H, B);

      DispatchSiluMul(main, state.d_shexp_gate, state.d_shexp_up, B * cfg.shared_expert_ff, state.d_shexp_act);

      DispatchGemm(main, layer.ffn_down_shexp.data, layer.ffn_down_shexp.type, 0, 0,
                   state.d_shexp_act, state.d_shexp_down, H, cfg.shared_expert_ff, B);

      if (!layer.ffn_gate_inp_shexp.empty()) {
        DispatchGemm(main, layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type, 0, 0,
                     state.d_x_normed, state.d_shexp_dot, 1, H, B);
      } else {
        GUFO_QWEN35_HIP(hipMemsetAsync(state.d_shexp_dot, 0, B * sizeof(float), main));
      }
    }

    // Router GEMM on main stream
    DispatchGemm(main, layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, 0, 0,
                 state.d_x_normed, state.d_router_logits, cfg.num_experts, H, B);

    // On-device Top-K selection (zero host sync, zero roundtrips!)
    DispatchRouterSoftmaxTopK(main, state.d_router_logits, cfg.num_experts,
                              cfg.num_experts_used, B, state.d_expert_ids, state.d_expert_weights);

    const uint64_t expert_stride_gate = layer.ffn_gate_exps.RowBytes() * cfg.expert_ff;
    const uint64_t expert_stride_up = layer.ffn_up_exps.RowBytes() * cfg.expert_ff;
    const uint64_t expert_stride_down = layer.ffn_down_exps.RowBytes() * H;

    if (B > 1) {
      GUFO_QWEN35_HIP(hipMemsetAsync(state.d_routed_accum, 0, static_cast<size_t>(B) * H * sizeof(float), main));
      DispatchSortMoESlots(
          main, state.d_expert_ids, B * cfg.num_experts_used, cfg.num_experts,
          state.d_expert_counts, state.d_expert_offsets, state.d_expert_cursors, state.d_sorted_slots,
          state.d_tok_tile_expert, state.d_tok_tile_offset, state.d_num_tiles);

      DispatchMoeGatedSwigluGrouped(
          main, layer.ffn_gate_exps.data, layer.ffn_up_exps.data, layer.ffn_gate_exps.type,
          expert_stride_gate, expert_stride_up, state.d_x_normed, state.d_expert_counts,
          state.d_expert_offsets, state.d_sorted_slots,
          state.d_tok_tile_expert, state.d_tok_tile_offset, state.d_num_tiles,
          B, cfg.num_experts, cfg.num_experts_used, cfg.expert_ff, H, state.d_expert_act);

      DispatchMoeDownAccumGrouped(
          main, layer.ffn_down_exps.data, layer.ffn_down_exps.type,
          expert_stride_down, state.d_expert_act, state.d_expert_weights,
          state.d_expert_counts, state.d_expert_offsets, state.d_sorted_slots,
          state.d_tok_tile_expert, state.d_tok_tile_offset, state.d_num_tiles,
          B, cfg.num_experts, cfg.num_experts_used, cfg.expert_ff, H, state.d_routed_accum);
    } else {
      DispatchMoeGatedSwigluBatch(
          main, layer.ffn_gate_exps.data, layer.ffn_up_exps.data, layer.ffn_gate_exps.type,
          expert_stride_gate, expert_stride_up, state.d_x_normed, state.d_expert_ids,
          B, cfg.num_experts_used, cfg.expert_ff, H, state.d_expert_act);

      DispatchMoeDownAccumBatch(
          main, layer.ffn_down_exps.data, layer.ffn_down_exps.type,
          expert_stride_down, state.d_expert_act, state.d_expert_weights, state.d_expert_ids,
          B, cfg.num_experts_used, cfg.expert_ff, H, state.d_routed_accum);
    }

    if (has_side) {
      GUFO_QWEN35_HIP(hipStreamWaitEvent(main, state.side_done, 0));
    }

    // Combine routed + shared expert (directly on device, zero CPU sync!)
    DispatchMoECombineSharedBatch(main, state.d_routed_accum, state.d_shexp_down, state.d_shexp_dot, H, B, state.d_moe_out);

    // Residual add: d_hidden += d_moe_out
    DispatchResidualAdd(main, state.d_moe_out, B * H, state.d_hidden);
    if (HipErrorPending()) {
      break;
    }
  }

  // Final Output Norm & LM Head (for the last token of chunk, if requested)
  if (!out_logits.empty()) {
    const float* last_hidden = state.d_hidden + (static_cast<size_t>(B - 1) * H);
    DispatchRMSNorm(main, last_hidden, weights.output_norm.data, weights.output_norm.type,
                    cfg.rms_eps, H, state.d_x_normed);

    DispatchGemv(main, weights.output.data, weights.output.type, 0, 0,
                 state.d_x_normed, state.d_logits, cfg.vocab_size, H);

    GUFO_QWEN35_HIP(hipMemcpyDtoHAsync(out_logits.data(), state.d_logits, cfg.vocab_size * sizeof(float), main));
    GUFO_QWEN35_HIP(hipStreamSynchronize(main));
  } else {
    GUFO_QWEN35_HIP(hipStreamSynchronize(main));
  }

  auto t_chunk_end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t_chunk_end - t_chunk_start).count();
  if (HipErrorPending()) {
    std::fprintf(stderr, "[ForwardModelGpuChunk] B=%u pos0=%u FAILED: GPU error pending\n", B, pos0);
  } else {
    std::fprintf(stderr, "[ForwardModelGpuChunk] B=%u pos0=%u: %.2f ms (%.1f tok/s)\n", B, pos0, ms, (B * 1000.0) / ms);
  }
  std::fflush(stderr);

}

}  // namespace gufo::models::qwen35_moe::hip
#endif  // defined(ENGINE_ENABLE_HIP)
