#ifndef GUFO_MODELS_QWEN35_MOE_CONFIG_HPP_
#define GUFO_MODELS_QWEN35_MOE_CONFIG_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen35_moe {

/// Architecture parameters of a `qwen35moe` GGUF artifact.
/// Specifically configured for Ornith-1.5-122B (122B-A10B, 48 layers).
struct Config {
  std::string architecture{"qwen35moe"};
  std::uint32_t num_layers{48};
  std::uint32_t hidden_size{3072};
  std::uint32_t vocab_size{248320};
  std::uint32_t context_length{262144};
  float rms_eps{1e-6F};

  // Full Attention (Gated GQA)
  std::uint32_t full_attention_interval{4};
  std::uint32_t num_heads{32};
  std::uint32_t num_kv_heads{2};
  std::uint32_t head_dim{256};
  std::uint32_t rotary_dim{64};
  float rope_theta{10000000.0F};
  std::array<std::uint32_t, 4> rope_sections{11, 11, 10, 0};

  // Linear Attention (Gated DeltaNet SSM)
  std::uint32_t ssm_conv_kernel{4};
  std::uint32_t ssm_head_dim{128};      ///< state_size
  std::uint32_t ssm_num_k_heads{16};    ///< group_count
  std::uint32_t ssm_num_v_heads{64};    ///< time_step_rank
  std::uint32_t ssm_inner_size{8192};

  // Mixture of Experts
  std::uint32_t num_experts{256};
  std::uint32_t num_experts_used{8};
  std::uint32_t expert_ff{1024};
  std::uint32_t shared_expert_ff{1024};

  [[nodiscard]] constexpr std::uint32_t SsmKeyDim() const noexcept {
    return ssm_num_k_heads * ssm_head_dim;  // 2048
  }
  [[nodiscard]] constexpr std::uint32_t SsmValueDim() const noexcept {
    return ssm_num_v_heads * ssm_head_dim;  // 8192
  }
  [[nodiscard]] constexpr std::uint32_t SsmConvChannels() const noexcept {
    return (2 * SsmKeyDim()) + SsmValueDim();  // 12288
  }
  [[nodiscard]] constexpr std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;  // 8192
  }
  [[nodiscard]] constexpr std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;  // 512
  }
  [[nodiscard]] constexpr bool IsFullAttentionLayer(std::uint32_t layer) const noexcept {
    return full_attention_interval != 0 && ((layer + 1) % full_attention_interval) == 0;
  }
  [[nodiscard]] constexpr bool IsLinearLayer(std::uint32_t layer) const noexcept {
    return !IsFullAttentionLayer(layer);
  }
  [[nodiscard]] constexpr std::uint32_t TotalFullAttentionLayers() const noexcept {
    return full_attention_interval != 0 ? num_layers / full_attention_interval : 0;
  }
  [[nodiscard]] constexpr std::uint32_t TotalSsmLayers() const noexcept {
    return num_layers - TotalFullAttentionLayers();
  }

  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen35_moe

#endif  // GUFO_MODELS_QWEN35_MOE_CONFIG_HPP_
