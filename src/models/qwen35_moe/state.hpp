#ifndef GUFO_MODELS_QWEN35_MOE_STATE_HPP_
#define GUFO_MODELS_QWEN35_MOE_STATE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/models/qwen35_moe/config.hpp"

namespace gufo::models::qwen35_moe {

/// Non-linear KV cache for the full attention layers in qwen35_moe.
class KvCache {
public:
  KvCache() = default;
  KvCache(std::uint32_t num_layers, std::uint32_t num_kv_heads,
          std::uint32_t head_dim, std::uint32_t max_context)
      : num_layers_(num_layers),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        max_context_(max_context) {
    const std::size_t layer_slice = static_cast<std::size_t>(num_kv_heads_) *
                                    head_dim_ * max_context_;
    k_data_.assign(num_layers_ * layer_slice, 0.0F);
    v_data_.assign(num_layers_ * layer_slice, 0.0F);
  }
  KvCache(const Config& cfg, std::uint32_t max_context)
      : KvCache(cfg.TotalFullAttentionLayers(), cfg.num_kv_heads, cfg.head_dim, max_context) {}

  void Reset() noexcept {
    std::ranges::fill(k_data_, 0.0F);
    std::ranges::fill(v_data_, 0.0F);
  }

  [[nodiscard]] std::span<float> KeySlice(std::uint32_t full_attn_layer_idx,
                                          std::uint32_t kv_head,
                                          std::uint32_t pos) noexcept {
    const std::size_t offset =
        (static_cast<std::size_t>(full_attn_layer_idx) * num_kv_heads_ * max_context_ +
         static_cast<std::size_t>(kv_head) * max_context_ + pos) *
        head_dim_;
    return std::span<float>(k_data_.data() + offset, head_dim_);
  }

  [[nodiscard]] std::span<float> ValueSlice(std::uint32_t full_attn_layer_idx,
                                            std::uint32_t kv_head,
                                            std::uint32_t pos) noexcept {
    const std::size_t offset =
        (static_cast<std::size_t>(full_attn_layer_idx) * num_kv_heads_ * max_context_ +
         static_cast<std::size_t>(kv_head) * max_context_ + pos) *
        head_dim_;
    return std::span<float>(v_data_.data() + offset, head_dim_);
  }

  [[nodiscard]] std::uint32_t HeadDim() const noexcept { return head_dim_; }
  [[nodiscard]] std::uint32_t NumKvHeads() const noexcept { return num_kv_heads_; }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept { return max_context_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept {
    return (k_data_.capacity() + v_data_.capacity()) * sizeof(float);
  }

private:
  std::uint32_t num_layers_{0};
  std::uint32_t num_kv_heads_{0};
  std::uint32_t head_dim_{0};
  std::uint32_t max_context_{0};
  std::vector<float> k_data_;
  std::vector<float> v_data_;
};

/// Recurrent Gated DeltaNet state cache across all linear attention layers.
class SsmState {
public:
  SsmState() = default;
  SsmState(std::uint32_t num_ssm_layers, std::uint32_t conv_channels,
           std::uint32_t conv_kernel, std::uint32_t v_heads,
           std::uint32_t head_k_dim, std::uint32_t head_v_dim)
      : num_ssm_layers_(num_ssm_layers),
        conv_channels_(conv_channels),
        conv_history_len_(conv_kernel > 1 ? conv_kernel - 1 : 1),
        v_heads_(v_heads),
        head_k_dim_(head_k_dim),
        head_v_dim_(head_v_dim) {
    const std::size_t conv_layer_size = conv_history_len_ * conv_channels_;
    const std::size_t ssm_layer_size =
        static_cast<std::size_t>(v_heads_) * head_k_dim_ * head_v_dim_;
    conv_state_.assign(num_ssm_layers_ * conv_layer_size, 0.0F);
    deltanet_state_.assign(num_ssm_layers_ * ssm_layer_size, 0.0F);
  }
  explicit SsmState(const Config& cfg)
      : SsmState(cfg.TotalSsmLayers(), cfg.SsmConvChannels(), cfg.ssm_conv_kernel,
                 cfg.ssm_num_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim) {}

  void Reset() noexcept {
    std::ranges::fill(conv_state_, 0.0F);
    std::ranges::fill(deltanet_state_, 0.0F);
  }

  [[nodiscard]] std::span<float> ConvState(std::uint32_t ssm_layer_idx) noexcept {
    const std::size_t size = conv_history_len_ * conv_channels_;
    return std::span<float>(conv_state_.data() + (ssm_layer_idx * size), size);
  }

  [[nodiscard]] std::span<float> DeltaNetState(std::uint32_t ssm_layer_idx,
                                               std::uint32_t v_head) noexcept {
    const std::size_t matrix_size = head_k_dim_ * head_v_dim_;
    const std::size_t offset =
        (static_cast<std::size_t>(ssm_layer_idx) * v_heads_ + v_head) *
        matrix_size;
    return std::span<float>(deltanet_state_.data() + offset, matrix_size);
  }

  [[nodiscard]] std::uint32_t ConvChannels() const noexcept { return conv_channels_; }
  [[nodiscard]] std::uint32_t ConvHistoryLen() const noexcept { return conv_history_len_; }
  [[nodiscard]] std::uint32_t VHeads() const noexcept { return v_heads_; }
  [[nodiscard]] std::uint32_t HeadKDim() const noexcept { return head_k_dim_; }
  [[nodiscard]] std::uint32_t HeadVDim() const noexcept { return head_v_dim_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept {
    return (conv_state_.capacity() + deltanet_state_.capacity()) * sizeof(float);
  }

private:
  std::uint32_t num_ssm_layers_{0};
  std::uint32_t conv_channels_{0};
  std::uint32_t conv_history_len_{0};
  std::uint32_t v_heads_{0};
  std::uint32_t head_k_dim_{0};
  std::uint32_t head_v_dim_{0};
  std::vector<float> conv_state_;
  std::vector<float> deltanet_state_;
};

struct ScratchArena {
  std::vector<float> hidden;
  std::vector<float> normed;
  std::vector<float> attn_out;

  // Full attention scratch
  std::vector<float> q_buf;
  std::vector<float> k_buf;
  std::vector<float> v_buf;
  std::vector<float> gate_buf;
  std::vector<float> scores;
  std::vector<float> context;

  // Linear attention (SSM) scratch
  std::vector<float> ssm_qkv;
  std::vector<float> ssm_gate;
  std::vector<float> ssm_alpha;
  std::vector<float> ssm_beta;
  std::vector<float> ssm_conv_out;
  std::vector<float> ssm_core_out;
  std::vector<float> ssm_out;

  // MoE scratch
  std::vector<float> router_logits;
  std::vector<float> router_weights;
  std::vector<std::uint32_t> top_experts;
  std::vector<float> expert_gate;
  std::vector<float> expert_up;
  std::vector<float> expert_act;
  std::vector<float> expert_down;
  std::vector<float> routed_accum;
  std::vector<float> shexp_gate;
  std::vector<float> shexp_up;
  std::vector<float> shexp_act;
  std::vector<float> shexp_down;

  ScratchArena() = default;
  ScratchArena(const Config& cfg, std::uint32_t max_context) {
    Allocate(cfg, max_context);
  }

  void Allocate(const Config& cfg, std::uint32_t max_context) {
    hidden.assign(cfg.hidden_size, 0.0F);
    normed.assign(cfg.hidden_size, 0.0F);
    attn_out.assign(cfg.hidden_size, 0.0F);

    q_buf.assign(cfg.AttentionQDim(), 0.0F);
    k_buf.assign(cfg.AttentionKvDim(), 0.0F);
    v_buf.assign(cfg.AttentionKvDim(), 0.0F);
    gate_buf.assign(cfg.AttentionQDim(), 0.0F);
    scores.assign(max_context, 0.0F);
    context.assign(cfg.AttentionQDim(), 0.0F);

    ssm_qkv.assign(cfg.SsmConvChannels(), 0.0F);
    ssm_gate.assign(cfg.SsmValueDim(), 0.0F);
    ssm_alpha.assign(cfg.ssm_num_v_heads, 0.0F);
    ssm_beta.assign(cfg.ssm_num_v_heads, 0.0F);
    ssm_conv_out.assign(cfg.SsmConvChannels(), 0.0F);
    ssm_core_out.assign(cfg.SsmValueDim(), 0.0F);
    ssm_out.assign(cfg.hidden_size, 0.0F);

    router_logits.assign(cfg.num_experts, 0.0F);
    router_weights.assign(cfg.num_experts_used, 0.0F);
    top_experts.assign(cfg.num_experts_used, 0);

    expert_gate.assign(cfg.expert_ff, 0.0F);
    expert_up.assign(cfg.expert_ff, 0.0F);
    expert_act.assign(cfg.expert_ff, 0.0F);
    expert_down.assign(cfg.hidden_size, 0.0F);
    routed_accum.assign(cfg.hidden_size, 0.0F);

    shexp_gate.assign(cfg.shared_expert_ff, 0.0F);
    shexp_up.assign(cfg.shared_expert_ff, 0.0F);
    shexp_act.assign(cfg.shared_expert_ff, 0.0F);
    shexp_down.assign(cfg.hidden_size, 0.0F);
  }

  [[nodiscard]] std::size_t AllocatedBytes() const noexcept {
    return (hidden.capacity() + normed.capacity() + attn_out.capacity() +
            q_buf.capacity() + k_buf.capacity() + v_buf.capacity() +
            gate_buf.capacity() + scores.capacity() + context.capacity() +
            ssm_qkv.capacity() + ssm_gate.capacity() + ssm_alpha.capacity() +
            ssm_beta.capacity() + ssm_conv_out.capacity() +
            ssm_core_out.capacity() + ssm_out.capacity() +
            router_logits.capacity() + router_weights.capacity() +
            expert_gate.capacity() + expert_up.capacity() + expert_act.capacity() +
            expert_down.capacity() + routed_accum.capacity() +
            shexp_gate.capacity() + shexp_up.capacity() + shexp_act.capacity() +
            shexp_down.capacity()) * sizeof(float) +
           top_experts.capacity() * sizeof(std::uint32_t);
  }
};

}  // namespace gufo::models::qwen35_moe

#endif  // GUFO_MODELS_QWEN35_MOE_STATE_HPP_
