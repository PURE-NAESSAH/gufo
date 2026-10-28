#include "src/models/qwen35_moe/config.hpp"

#include <string>

namespace gufo::models::qwen35_moe {

std::optional<Config> Config::FromGguf(const core::GgufReader& reader,
                                       std::string* error_msg) {
  const auto arch = reader.GetMetadataString("general.architecture").value_or("");
  if (arch != "qwen35moe") {
    if (error_msg != nullptr) {
      *error_msg = std::string("Expected architecture qwen35moe, got '") + std::string(arch) + "'";
    }
    return std::nullopt;
  }

  Config cfg;
  cfg.architecture = std::string(arch);

  cfg.num_layers = reader.GetMetadataUint32("qwen35moe.block_count").value_or(0);
  if (cfg.num_layers == 0) {
    if (error_msg != nullptr) {
      *error_msg = "Missing or invalid qwen35moe.block_count";
    }
    return std::nullopt;
  }

  cfg.hidden_size = reader.GetMetadataUint32("qwen35moe.embedding_length").value_or(0);
  if (cfg.hidden_size == 0) {
    if (error_msg != nullptr) {
      *error_msg = "Missing or invalid qwen35moe.embedding_length";
    }
    return std::nullopt;
  }

  cfg.context_length = reader.GetMetadataUint32("qwen35moe.context_length").value_or(262144);
  cfg.rms_eps = reader.GetMetadataFloat32("qwen35moe.attention.layer_norm_rms_epsilon").value_or(1e-6F);

  // Full Attention
  cfg.full_attention_interval = reader.GetMetadataUint32("qwen35moe.full_attention_interval").value_or(4);
  cfg.num_heads = reader.GetMetadataUint32("qwen35moe.attention.head_count").value_or(32);
  cfg.num_kv_heads = reader.GetMetadataUint32("qwen35moe.attention.head_count_kv").value_or(2);
  cfg.head_dim = reader.GetMetadataUint32("qwen35moe.attention.key_length").value_or(256);
  cfg.rotary_dim = reader.GetMetadataUint32("qwen35moe.rope.dimension_count").value_or(64);
  cfg.rope_theta = reader.GetMetadataFloat32("qwen35moe.rope.freq_base").value_or(10000000.0F);

  cfg.rope_sections = {11, 11, 10, 0};
  if (const auto* meta = reader.FindMetadata("qwen35moe.rope.dimension_sections")) {
    if (std::holds_alternative<std::vector<std::uint64_t>>(meta->value)) {
      const auto& arr = std::get<std::vector<std::uint64_t>>(meta->value);
      if (arr.size() >= 4) {
        cfg.rope_sections = {
            static_cast<std::uint32_t>(arr[0]),
            static_cast<std::uint32_t>(arr[1]),
            static_cast<std::uint32_t>(arr[2]),
            static_cast<std::uint32_t>(arr[3])};
      }
    }
  }

  // Linear Attention (SSM)
  cfg.ssm_conv_kernel = reader.GetMetadataUint32("qwen35moe.ssm.conv_kernel").value_or(4);
  cfg.ssm_head_dim = reader.GetMetadataUint32("qwen35moe.ssm.state_size").value_or(128);
  cfg.ssm_num_k_heads = reader.GetMetadataUint32("qwen35moe.ssm.group_count").value_or(16);
  cfg.ssm_num_v_heads = reader.GetMetadataUint32("qwen35moe.ssm.time_step_rank").value_or(64);
  cfg.ssm_inner_size = reader.GetMetadataUint32("qwen35moe.ssm.inner_size").value_or(8192);

  // MoE
  cfg.num_experts = reader.GetMetadataUint32("qwen35moe.expert_count").value_or(256);
  cfg.num_experts_used = reader.GetMetadataUint32("qwen35moe.expert_used_count").value_or(8);
  cfg.expert_ff = reader.GetMetadataUint32("qwen35moe.expert_feed_forward_length").value_or(1024);
  cfg.shared_expert_ff = reader.GetMetadataUint32("qwen35moe.expert_shared_feed_forward_length").value_or(1024);

  // Vocabulary size from token_embd.weight dimensions
  const auto* embd_tensor = reader.FindTensor("token_embd.weight");
  if (embd_tensor != nullptr && embd_tensor->dimensions.size() >= 2) {
    cfg.vocab_size = static_cast<std::uint32_t>(embd_tensor->dimensions[1]);
  } else {
    cfg.vocab_size = reader.GetMetadataUint32("qwen35moe.vocab_size").value_or(248320);
  }

  return cfg;
}

}  // namespace gufo::models::qwen35_moe
