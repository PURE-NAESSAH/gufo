#include "src/models/qwen35_moe/config.hpp"
#include "src/models/qwen35_moe/weights.hpp"
#include "src/models/qwen35_moe/state.hpp"
#include "src/models/qwen35_moe/forward.hpp"
#include "src/core/gguf_reader.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace moe = gufo::models::qwen35_moe;

int main() {
  const std::string model_path =
      "D:/AI_Stack/Models/Harsha001/Ornith-1.5-122B-ROCmFPX/Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf";

  if (!std::filesystem::exists(model_path)) {
    std::cout << "Model file not found at " << model_path << ", skipping live test.\n";
    return 0;
  }

  std::cout << "[Test] Opening Ornith-1.5-122B multi-shard GGUF...\n";
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::cerr << "Failed to open GGUF: " << error << "\n";
    return 1;
  }

  std::cout << "[Test] Reading Config from GGUF...\n";
  auto cfg = moe::Config::FromGguf(*reader, &error);
  if (!cfg) {
    std::cerr << "Failed to parse Config: " << error << "\n";
    return 1;
  }

  std::cout << "Architecture: " << cfg->architecture << "\n";
  std::cout << "Layers: " << cfg->num_layers << "\n";
  std::cout << "Hidden: " << cfg->hidden_size << "\n";
  std::cout << "Vocab: " << cfg->vocab_size << "\n";
  std::cout << "Full attention interval: " << cfg->full_attention_interval << "\n";
  std::cout << "Total full attention layers: " << cfg->TotalFullAttentionLayers() << "\n";
  std::cout << "Total SSM layers: " << cfg->TotalSsmLayers() << "\n";

  assert(cfg->architecture == "qwen35moe");
  assert(cfg->num_layers == 48);
  assert(cfg->hidden_size == 3072);
  assert(cfg->vocab_size == 248320);
  assert(cfg->full_attention_interval == 4);
  assert(cfg->TotalFullAttentionLayers() == 12);
  assert(cfg->TotalSsmLayers() == 36);
  assert(cfg->num_experts == 256);
  assert(cfg->num_experts_used == 8);
  assert(cfg->expert_ff == 1024);
  assert(cfg->shared_expert_ff == 1024);

  std::cout << "[Test] Loading ModelWeights...\n";
  auto weights = moe::ModelWeights::FromGguf(*reader, &error);
  if (!weights) {
    std::cerr << "Failed to load ModelWeights: " << error << "\n";
    return 1;
  }

  assert(weights->token_embd.data != nullptr);
  assert(weights->output.data != nullptr);
  assert(weights->output_norm.data != nullptr);
  std::cout << "Token embd type: " << static_cast<int>(weights->token_embd.type) << "\n";
  std::cout << "Output type: " << static_cast<int>(weights->output.type) << "\n";

  // Check layer 0 (SSM layer)
  const auto& l0 = weights->layers[0];
  assert(!l0.is_full_attention);
  assert(l0.ssm_qkv.data != nullptr);
  assert(l0.ssm_gate.data != nullptr);
  assert(l0.ssm_conv1d.data != nullptr);
  assert(l0.ssm_a.data != nullptr);
  assert(l0.ssm_alpha.data != nullptr);
  assert(l0.ssm_beta.data != nullptr);
  assert(l0.ssm_norm.data != nullptr);
  assert(l0.ssm_out.data != nullptr);
  assert(l0.ffn_gate_inp.data != nullptr);
  assert(l0.ffn_gate_exps.data != nullptr);
  assert(l0.ffn_up_exps.data != nullptr);
  assert(l0.ffn_down_exps.data != nullptr);
  assert(l0.ffn_gate_shexp.data != nullptr);
  assert(l0.ffn_up_shexp.data != nullptr);
  assert(l0.ffn_down_shexp.data != nullptr);
  assert(l0.ffn_gate_inp_shexp.data != nullptr);

  // Check layer 3 (Full attention layer)
  const auto& l3 = weights->layers[3];
  assert(l3.is_full_attention);
  assert(l3.attn_q.data != nullptr);
  assert(l3.attn_k.data != nullptr);
  assert(l3.attn_v.data != nullptr);
  assert(l3.attn_output.data != nullptr);
  assert(l3.ffn_gate_inp.data != nullptr);
  assert(l3.ffn_gate_exps.data != nullptr);

  std::cout << "[Test] All weights successfully bound from both GGUF shards!\n";

  std::cout << "[Test] Initializing states (KvCache, SsmState, ScratchArena)...\n";
  const std::uint32_t max_context = 256;
  moe::KvCache kv_cache(*cfg, max_context);
  moe::SsmState ssm_state(*cfg);
  moe::ScratchArena scratch(*cfg, max_context);
  std::vector<float> logits(cfg->vocab_size, 0.0f);

  std::cout << "KvCache bytes: " << kv_cache.AllocatedBytes() << "\n";
  std::cout << "SsmState bytes: " << ssm_state.AllocatedBytes() << "\n";
  std::cout << "Scratch bytes: " << scratch.AllocatedBytes() << "\n";

  std::cout << "[Test] Running ForwardModel on token 151644 at position 0...\n";
  moe::ForwardModel(151644, 0, *weights, kv_cache, ssm_state, scratch,
                    std::span<float>(logits));

  // Check logits are non-zero, finite
  bool has_nonzero = false;
  bool all_finite = true;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (logits[i] != 0.0f) {
      has_nonzero = true;
    }
    if (!std::isfinite(logits[i])) {
      all_finite = false;
      std::cerr << "Non-finite logit at index " << i << ": " << logits[i] << "\n";
      break;
    }
  }
  assert(has_nonzero);
  assert(all_finite);
  std::cout << "Pos 0 Logits[0..5]: " << logits[0] << ", " << logits[1] << ", "
            << logits[2] << ", " << logits[3] << ", " << logits[4] << "\n";

  std::cout << "[Test] Running ForwardModel on token 872 at position 1...\n";
  moe::ForwardModel(872, 1, *weights, kv_cache, ssm_state, scratch,
                    std::span<float>(logits));

  has_nonzero = false;
  all_finite = true;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (logits[i] != 0.0f) {
      has_nonzero = true;
    }
    if (!std::isfinite(logits[i])) {
      all_finite = false;
      std::cerr << "Non-finite logit at index " << i << ": " << logits[i] << "\n";
      break;
    }
  }
  assert(has_nonzero);
  assert(all_finite);
  std::cout << "Pos 1 Logits[0..5]: " << logits[0] << ", " << logits[1] << ", "
            << logits[2] << ", " << logits[3] << ", " << logits[4] << "\n";

  std::cout << "[Test] All tests PASSED successfully!\n";
  return 0;
}
