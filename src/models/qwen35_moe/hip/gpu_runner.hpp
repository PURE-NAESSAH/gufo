#ifndef GUFO_MODELS_QWEN35_MOE_HIP_GPU_RUNNER_HPP_
#define GUFO_MODELS_QWEN35_MOE_HIP_GPU_RUNNER_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "src/models/qwen35_moe/config.hpp"
#include "src/models/qwen35_moe/weights.hpp"

namespace gufo::models::qwen35_moe::hip {

struct GpuState {
  uint32_t max_context{0};
  static constexpr uint32_t kMaxChunk{64};
  uint32_t* d_token_ids{nullptr};

  // Residual and activations
  float* d_hidden{nullptr};
  float* d_x_normed{nullptr};
  float* d_attn_out{nullptr};

  // Full attention
  float* d_qg_buf{nullptr};
  float* d_q_buf{nullptr};
  float* d_gate_buf{nullptr};
  float* d_k_buf{nullptr};
  float* d_v_buf{nullptr};
  float* d_context{nullptr};
  float* d_kv_k{nullptr};
  float* d_kv_v{nullptr};

  // SSM (DeltaNet)
  float* d_ssm_qkv{nullptr};
  float* d_ssm_gate{nullptr};
  float* d_ssm_alpha{nullptr};
  float* d_ssm_beta{nullptr};
  float* d_decay{nullptr};
  float* d_beta_sig{nullptr};
  float* d_ssm_conv_out{nullptr};
  float* d_ssm_q_normed{nullptr};
  float* d_ssm_k_normed{nullptr};
  float* d_ssm_core_out{nullptr};
  float* d_conv_state{nullptr};
  float* d_deltanet_state{nullptr};

  // MoE
  float* d_router_logits{nullptr};
  float* d_expert_gate{nullptr};
  float* d_expert_up{nullptr};
  float* d_expert_act{nullptr};
  float* d_expert_down{nullptr};
  float* d_routed_accum{nullptr};
  uint32_t* d_expert_ids{nullptr};
  float* d_expert_weights{nullptr};
  std::vector<uint32_t> h_expert_ids;
  std::vector<float> h_expert_weights;
  float* d_shexp_gate{nullptr};
  float* d_shexp_up{nullptr};
  float* d_shexp_act{nullptr};
  float* d_shexp_down{nullptr};
  float* d_shexp_dot{nullptr};
  float* d_moe_out{nullptr};

  // Logits
  float* d_logits{nullptr};

  // Host scratch for router top-k
  std::vector<float> h_router_logits;
  std::vector<std::pair<float, uint32_t>> h_probs;

  std::vector<void*> allocations;

  void Allocate(const Config& cfg, uint32_t max_ctx);
  void Free();
  void Reset(const Config& cfg, uint32_t max_ctx);
  ~GpuState() { Free(); }
};

void ForwardModelGpu(
    uint32_t token_id,
    uint32_t pos,
    const ModelWeights& weights,
    GpuState& state,
    std::span<float> out_logits);

void ForwardModelGpuChunk(
    std::span<const uint32_t> tokens,
    uint32_t pos0,
    const ModelWeights& weights,
    GpuState& state,
    std::span<float> out_logits);

}  // namespace gufo::models::qwen35_moe::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN35_MOE_HIP_GPU_RUNNER_HPP_
