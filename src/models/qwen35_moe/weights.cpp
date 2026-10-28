#include "src/models/qwen35_moe/weights.hpp"

#include <initializer_list>
#include <string>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen35_moe {

std::size_t TensorRef::RowBytes() const noexcept {
  return quant::EncodedSizeBytes(type, static_cast<std::size_t>(cols));
}

namespace {

using core::GgmlType;

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor: " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts) {
      Fail("tensor " + name + " shape mismatch: got [" + std::to_string(got_cols) +
           ", " + std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " + std::to_string(rows) +
           ", " + std::to_string(experts) + "]");
      return t;
    }

    bool type_ok = false;
    for (auto expected_type : types) {
      if (info->type == expected_type) {
        type_ok = true;
        break;
      }
    }
    if (!type_ok) {
      Fail("tensor " + name + " has unexpected type " +
           std::string(core::ToString(info->type)));
      return t;
    }

    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    return t;
  }
};

}  // namespace

std::unique_ptr<ModelWeights> ModelWeights::FromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto cfg = Config::FromGguf(reader, error_msg);
  if (!cfg.has_value()) {
    return nullptr;
  }

  auto w = std::make_unique<ModelWeights>();
  w->config = *cfg;
  w->layers.resize(w->config.num_layers);

  Binder b{reader, error_msg, true};

  // Standard acceptable types
  const auto kAnyQuant = {
      GgmlType::kQ4_0_ROCMFP4_FAST, GgmlType::kQ4_0_ROCMFP4,
      GgmlType::kQ8_0,             GgmlType::kBF16,
      GgmlType::kF16,              GgmlType::kF32,
      GgmlType::kQ2_0_ROCMFPX,     GgmlType::kQ3_0_ROCMFPX,
      GgmlType::kQ6_0_ROCMFPX,     GgmlType::kQ8_0_ROCMFPX,
      GgmlType::kQ4_0_ROCMI4,      GgmlType::kQ6_K,
      GgmlType::kQ4_K,             GgmlType::kQ5_K,
      GgmlType::kQ4_0,             GgmlType::kQ8_K,
  };
  const auto kNormTypes = {GgmlType::kF32, GgmlType::kBF16, GgmlType::kF16};
  const auto kEmbdTypes = {GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32,
                           GgmlType::kQ8_0, GgmlType::kQ4_0_ROCMFP4_FAST,
                           GgmlType::kQ6_K, GgmlType::kQ4_K, GgmlType::kQ5_K};

  w->token_embd = b.Get("token_embd.weight", w->config.hidden_size,
                        w->config.vocab_size, 1, kEmbdTypes);
  w->output_norm =
      b.Get("output_norm.weight", w->config.hidden_size, 1, 1, kNormTypes);
  w->output = b.Get("output.weight", w->config.hidden_size,
                    w->config.vocab_size, 1, kAnyQuant, false);
  if (w->output.empty()) {
    w->output = w->token_embd;
  }

  const std::uint64_t H = w->config.hidden_size;
  const std::uint64_t num_heads = w->config.num_heads;
  const std::uint64_t num_kv_heads = w->config.num_kv_heads;
  const std::uint64_t head_dim = w->config.head_dim;
  const std::uint64_t key_dim = w->config.SsmKeyDim();
  const std::uint64_t val_dim = w->config.SsmValueDim();
  const std::uint64_t conv_channels = w->config.SsmConvChannels();
  const std::uint64_t conv_kernel = w->config.ssm_conv_kernel;
  const std::uint64_t v_heads = w->config.ssm_num_v_heads;
  const std::uint64_t ssm_head_dim = w->config.ssm_head_dim;
  const std::uint64_t num_experts = w->config.num_experts;
  const std::uint64_t expert_ff = w->config.expert_ff;
  const std::uint64_t shared_expert_ff = w->config.shared_expert_ff;

  for (std::uint32_t l = 0; l < w->config.num_layers; ++l) {
    auto& layer = w->layers[l];
    const std::string prefix = "blk." + std::to_string(l) + ".";

    layer.is_full_attention = w->config.IsFullAttentionLayer(l);

    // Norms
    layer.attn_norm =
        b.Get(prefix + "attn_norm.weight", H, 1, 1, kNormTypes);
    layer.post_attention_norm =
        b.Get(prefix + "post_attention_norm.weight", H, 1, 1, kNormTypes);

    if (layer.is_full_attention) {
      // Joint Q + Gate: [H, 2 * num_heads * head_dim]
      layer.attn_q = b.Get(prefix + "attn_q.weight", H,
                           2 * num_heads * head_dim, 1, kAnyQuant);
      layer.attn_k =
          b.Get(prefix + "attn_k.weight", H, num_kv_heads * head_dim, 1, kAnyQuant);
      layer.attn_v =
          b.Get(prefix + "attn_v.weight", H, num_kv_heads * head_dim, 1, kAnyQuant);
      layer.attn_output =
          b.Get(prefix + "attn_output.weight", num_heads * head_dim, H, 1, kAnyQuant);
      layer.attn_q_norm =
          b.Get(prefix + "attn_q_norm.weight", head_dim, 1, 1, kNormTypes);
      layer.attn_k_norm =
          b.Get(prefix + "attn_k_norm.weight", head_dim, 1, 1, kNormTypes);
    } else {
      // Linear Attention (SSM)
      layer.ssm_qkv =
          b.Get(prefix + "attn_qkv.weight", H, 2 * key_dim + val_dim, 1, kAnyQuant);
      layer.ssm_gate =
          b.Get(prefix + "attn_gate.weight", H, val_dim, 1, kAnyQuant);
      layer.ssm_conv1d =
          b.Get(prefix + "ssm_conv1d.weight", conv_kernel, conv_channels, 1, kNormTypes);
      layer.ssm_dt =
          b.Get(prefix + "ssm_dt.bias", v_heads, 1, 1, kNormTypes);
      layer.ssm_a =
          b.Get(prefix + "ssm_a", v_heads, 1, 1, kNormTypes);
      layer.ssm_beta =
          b.Get(prefix + "ssm_beta.weight", H, v_heads, 1, kAnyQuant);
      layer.ssm_alpha =
          b.Get(prefix + "ssm_alpha.weight", H, v_heads, 1, kAnyQuant);
      layer.ssm_norm =
          b.Get(prefix + "ssm_norm.weight", ssm_head_dim, 1, 1, kNormTypes);
      layer.ssm_out =
          b.Get(prefix + "ssm_out.weight", val_dim, H, 1, kAnyQuant);
    }

    // MoE FFN (all layers)
    layer.ffn_gate_inp =
        b.Get(prefix + "ffn_gate_inp.weight", H, num_experts, 1, kNormTypes);
    layer.ffn_gate_exps =
        b.Get(prefix + "ffn_gate_exps.weight", H, expert_ff, num_experts, kAnyQuant);
    layer.ffn_up_exps =
        b.Get(prefix + "ffn_up_exps.weight", H, expert_ff, num_experts, kAnyQuant);
    layer.ffn_down_exps =
        b.Get(prefix + "ffn_down_exps.weight", expert_ff, H, num_experts, kAnyQuant);

    // Shared expert
    layer.ffn_gate_shexp =
        b.Get(prefix + "ffn_gate_shexp.weight", H, shared_expert_ff, 1, kAnyQuant);
    layer.ffn_up_shexp =
        b.Get(prefix + "ffn_up_shexp.weight", H, shared_expert_ff, 1, kAnyQuant);
    layer.ffn_down_shexp =
        b.Get(prefix + "ffn_down_shexp.weight", shared_expert_ff, H, 1, kAnyQuant);
    layer.ffn_gate_inp_shexp =
        b.Get(prefix + "ffn_gate_inp_shexp.weight", H, 1, 1, kNormTypes);

    if (!b.ok) {
      return nullptr;
    }
  }

  return w;
}

}  // namespace gufo::models::qwen35_moe
