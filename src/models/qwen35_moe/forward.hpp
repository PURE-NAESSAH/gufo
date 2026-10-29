#ifndef GUFO_MODELS_QWEN35_MOE_FORWARD_HPP_
#define GUFO_MODELS_QWEN35_MOE_FORWARD_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen35_moe/state.hpp"
#include "src/models/qwen35_moe/weights.hpp"

namespace gufo::models::qwen35_moe {

/// Matrix-vector multiplication for any supported quant or float format.
void TensorGEMV(const TensorRef& A, std::span<const float> x, std::size_t M,
                std::size_t K, std::span<float> y) noexcept;

/// Matrix-vector multiplication for a stacked 3D expert tensor.
void TensorExpertGEMV(const TensorRef& A, std::size_t expert,
                      std::span<const float> x, std::size_t M, std::size_t K,
                      std::span<float> y) noexcept;

/// Standard RMSNorm layer with epsilon.
void ForwardRMSNorm(std::span<const float> x, const TensorRef& weight, float eps,
                    std::span<float> out) noexcept;

/// Single token forward pass across all layers of Qwen3.5-MoE (Ornith-1.5-122B).
void ForwardModel(std::uint32_t token_id, std::uint32_t pos,
                  const ModelWeights& weights, KvCache& kv_cache,
                  SsmState& ssm_state, ScratchArena& arena,
                  std::span<float> logits_out) noexcept;

/// Greedy argmax over logits.
[[nodiscard]] std::uint32_t GreedyArgmax(std::span<const float> logits) noexcept;

}  // namespace gufo::models::qwen35_moe

#endif  // GUFO_MODELS_QWEN35_MOE_FORWARD_HPP_
