#ifndef GUFO_MODELS_QWEN35_MOE_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN35_MOE_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen35_moe/config.hpp"

namespace gufo::models::qwen35_moe {

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + (RowBytes() * rows * e);
  }
};

struct LayerWeights {
  bool is_full_attention{false};

  // Pre-attention and post-attention RMS norms
  TensorRef attn_norm;            ///< [hidden_size]
  TensorRef post_attention_norm;  ///< [hidden_size]

  // Full Attention (layers where (layer + 1) % 4 == 0)
  TensorRef attn_q;       ///< [hidden_size, 2 * num_heads * head_dim] (joint Q + Gate)
  TensorRef attn_k;       ///< [hidden_size, num_kv_heads * head_dim]
  TensorRef attn_v;       ///< [hidden_size, num_kv_heads * head_dim]
  TensorRef attn_output;  ///< [num_heads * head_dim, hidden_size]
  TensorRef attn_q_norm;  ///< [head_dim]
  TensorRef attn_k_norm;  ///< [head_dim]

  // Linear Attention (SSM, layers where (layer + 1) % 4 != 0)
  TensorRef ssm_qkv;     ///< [hidden_size, 2 * key_dim + value_dim]
  TensorRef ssm_gate;    ///< [hidden_size, value_dim] (z)
  TensorRef ssm_conv1d;  ///< [conv_kernel, conv_channels]
  TensorRef ssm_dt;      ///< [v_heads]
  TensorRef ssm_a;       ///< [v_heads]
  TensorRef ssm_beta;    ///< [hidden_size, v_heads]
  TensorRef ssm_alpha;   ///< [hidden_size, v_heads]
  TensorRef ssm_norm;    ///< [ssm_head_dim]
  TensorRef ssm_out;     ///< [value_dim, hidden_size]

  // Routed Experts MoE FFN (all layers)
  TensorRef ffn_gate_inp;   ///< [hidden_size, num_experts] (router logits)
  TensorRef ffn_gate_exps;  ///< [hidden_size, expert_ff, num_experts]
  TensorRef ffn_up_exps;    ///< [hidden_size, expert_ff, num_experts]
  TensorRef ffn_down_exps;  ///< [expert_ff, hidden_size, num_experts]

  // Shared Expert
  TensorRef ffn_gate_shexp;      ///< [hidden_size, shared_expert_ff]
  TensorRef ffn_up_shexp;        ///< [hidden_size, shared_expert_ff]
  TensorRef ffn_down_shexp;      ///< [shared_expert_ff, hidden_size]
  TensorRef ffn_gate_inp_shexp;  ///< [hidden_size] (shared expert gate vector)
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden_size, vocab_size]
  TensorRef output_norm;  ///< [hidden_size]
  TensorRef output;       ///< [hidden_size, vocab_size]
  std::vector<LayerWeights> layers;

  [[nodiscard]] static std::unique_ptr<ModelWeights> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen35_moe

#endif  // GUFO_MODELS_QWEN35_MOE_WEIGHTS_HPP_
