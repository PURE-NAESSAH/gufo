#include "src/models/qwen35_moe/forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen35_moe {

namespace {

inline float Silu(float x) noexcept {
  return x / (1.0F + std::exp(-x));
}

inline float Sigmoid(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}

inline float Softplus(float x) noexcept {
  return (x > 20.0F) ? x : std::log1p(std::exp(x));
}

const void* QuantizedRow(const TensorRef& tensor, std::size_t row,
                         std::size_t columns) noexcept {
  const std::size_t row_bytes = quant::QuantizedRowBytes(tensor.type, columns);
  if (row_bytes == 0) {
    return nullptr;
  }
  return static_cast<const std::uint8_t*>(tensor.data) + (row * row_bytes);
}

float DotRow(core::GgmlType type, const void* row, std::span<const float> x,
             std::size_t K) noexcept {
  switch (type) {
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      return quant::DotProductQ4_0_ROCMFP4_FAST(row, x, K);
    case core::GgmlType::kQ4_0_ROCMFP4:
      return quant::DotProductQ4_0_ROCMFP4(row, x, K);
    case core::GgmlType::kQ2_0_ROCMFPX:
      return quant::DotProductQ2_0_ROCMFPX(row, x, K);
    case core::GgmlType::kQ3_0_ROCMFPX:
      return quant::DotProductQ3_0_ROCMFPX(row, x, K);
    case core::GgmlType::kQ6_0_ROCMFPX:
      return quant::DotProductQ6_0_ROCMFPX(row, x, K);
    case core::GgmlType::kQ8_0_ROCMFPX:
      return quant::DotProductQ8_0_ROCMFPX(row, x, K);
    case core::GgmlType::kQ4_0_ROCMI4:
      return quant::DotProductQ4_0_ROCMI4(row, x, K);
    case core::GgmlType::kQ8_0:
      return quant::DotProductQ8_0(row, x, K);
    case core::GgmlType::kQ6_K:
      return quant::DotProductQ6_K(row, x, K);
    case core::GgmlType::kQ4_K:
      return quant::DotProductQ4_K(row, x, K);
    case core::GgmlType::kQ5_K:
      return quant::DotProductQ5_K(row, x, K);
    case core::GgmlType::kF32: {
      const auto* ptr = static_cast<const float*>(row);
      float sum = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        sum += ptr[k] * x[k];
      }
      return sum;
    }
    case core::GgmlType::kBF16: {
      const auto* ptr = static_cast<const std::uint16_t*>(row);
      float sum = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        const std::uint32_t u32 = static_cast<std::uint32_t>(ptr[k]) << 16U;
        float val = 0.0F;
        std::memcpy(&val, &u32, sizeof(float));
        sum += val * x[k];
      }
      return sum;
    }
    case core::GgmlType::kF16: {
      const auto* ptr = static_cast<const std::uint16_t*>(row);
      float sum = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        sum += quant::Fp16ToFloat(ptr[k]) * x[k];
      }
      return sum;
    }
    default:
      return 0.0F;
  }
}

}  // namespace

void TensorGEMV(const TensorRef& A, std::span<const float> x, std::size_t M,
                std::size_t K, std::span<float> y) noexcept {
  if (A.empty() || x.size() < K || y.size() < M) {
    return;
  }
  const std::size_t row_bytes = A.RowBytes();
  const auto* base = static_cast<const std::uint8_t*>(A.data);

#pragma omp parallel for schedule(static)
  for (std::int64_t m = 0; m < static_cast<std::int64_t>(M); ++m) {
    const void* row = base + (static_cast<std::size_t>(m) * row_bytes);
    y[static_cast<std::size_t>(m)] = DotRow(A.type, row, x, K);
  }
}

void TensorExpertGEMV(const TensorRef& A, std::size_t expert,
                      std::span<const float> x, std::size_t M, std::size_t K,
                      std::span<float> y) noexcept {
  if (A.empty() || expert >= A.experts || x.size() < K || y.size() < M) {
    return;
  }
  const std::size_t row_bytes = A.RowBytes();
  const auto* expert_base = A.Expert(expert);

#pragma omp parallel for schedule(static)
  for (std::int64_t m = 0; m < static_cast<std::int64_t>(M); ++m) {
    const void* row = expert_base + (static_cast<std::size_t>(m) * row_bytes);
    y[static_cast<std::size_t>(m)] = DotRow(A.type, row, x, K);
  }
}

void ForwardRMSNorm(std::span<const float> x, const TensorRef& weight, float eps,
                    std::span<float> out) noexcept {
  const std::size_t size = x.size();
  if (size == 0 || out.size() < size) {
    return;
  }

  float sum_sq = 0.0F;
  for (std::size_t i = 0; i < size; ++i) {
    sum_sq += x[i] * x[i];
  }
  const float scale = 1.0F / std::sqrt((sum_sq / static_cast<float>(size)) + eps);

  if (weight.empty()) {
    for (std::size_t i = 0; i < size; ++i) {
      out[i] = x[i] * scale;
    }
    return;
  }

  if (weight.type == core::GgmlType::kF32) {
    const auto* w = static_cast<const float*>(weight.data);
    for (std::size_t i = 0; i < size; ++i) {
      out[i] = x[i] * scale * w[i];
    }
  } else if (weight.type == core::GgmlType::kBF16) {
    const auto* w = static_cast<const std::uint16_t*>(weight.data);
    for (std::size_t i = 0; i < size; ++i) {
      const std::uint32_t u32 = static_cast<std::uint32_t>(w[i]) << 16U;
      float wf = 0.0F;
      std::memcpy(&wf, &u32, sizeof(float));
      out[i] = x[i] * scale * wf;
    }
  } else {
    for (std::size_t i = 0; i < size; ++i) {
      out[i] = x[i] * scale;
    }
  }
}

namespace {

void ForwardEmbedding(std::uint32_t token_id, const TensorRef& token_embd,
                      std::size_t hidden_size, std::span<float> hidden_out) noexcept {
  if (token_embd.empty() || hidden_out.size() < hidden_size) {
    return;
  }
  const std::size_t row_bytes = token_embd.RowBytes();
  const auto* row_ptr = static_cast<const std::uint8_t*>(token_embd.data) +
                        (static_cast<std::size_t>(token_id) * row_bytes);

  if (token_embd.type == core::GgmlType::kBF16) {
    const auto* src = reinterpret_cast<const std::uint16_t*>(row_ptr);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      const std::uint32_t u32 = static_cast<std::uint32_t>(src[i]) << 16U;
      float f = 0.0F;
      std::memcpy(&f, &u32, sizeof(float));
      hidden_out[i] = f;
    }
  } else if (token_embd.type == core::GgmlType::kF32) {
    const auto* src = reinterpret_cast<const float*>(row_ptr);
    std::copy_n(src, hidden_size, hidden_out.data());
  } else if (token_embd.type == core::GgmlType::kQ4_0_ROCMFP4_FAST) {
    quant::DequantizeQ4_0_ROCMFP4_FAST(row_ptr, hidden_out.data(), hidden_size);
  } else if (token_embd.type == core::GgmlType::kQ4_0_ROCMFP4) {
    quant::DequantizeQ4_0_ROCMFP4(row_ptr, hidden_out.data(), hidden_size);
  } else if (token_embd.type == core::GgmlType::kQ6_K) {
    quant::DequantizeQ6_K(row_ptr, hidden_out.data(), hidden_size);
  } else if (token_embd.type == core::GgmlType::kQ8_0) {
    quant::DequantizeQ8_0(row_ptr, hidden_out.data(), hidden_size);
  } else if (token_embd.type == core::GgmlType::kQ4_K) {
    quant::DequantizeQ4_K(row_ptr, hidden_out.data(), hidden_size);
  } else if (token_embd.type == core::GgmlType::kQ5_K) {
    quant::DequantizeQ5_K(row_ptr, hidden_out.data(), hidden_size);
  } else if (token_embd.type == core::GgmlType::kF16) {
    const auto* src = reinterpret_cast<const std::uint16_t*>(row_ptr);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      hidden_out[i] = quant::Fp16ToFloat(src[i]);
    }
  }
}

void ApplyRope(std::span<float> vec, std::uint32_t rotary_dim,
               std::uint32_t pos, float theta_base) noexcept {
  const std::size_t half_rot = rotary_dim / 2;
  for (std::size_t i = 0; i < half_rot; ++i) {
    const float freq = 1.0F / std::pow(theta_base, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
    const float theta = static_cast<float>(pos) * freq;
    const float cos_th = std::cos(theta);
    const float sin_th = std::sin(theta);

    const float x0 = vec[i];
    const float x1 = vec[i + half_rot];
    vec[i] = (x0 * cos_th) - (x1 * sin_th);
    vec[i + half_rot] = (x0 * sin_th) + (x1 * cos_th);
  }
}

void ForwardFullAttention(const LayerWeights& layer, const Config& config,
                          std::span<const float> x_normed, KvCache& kv_cache,
                          std::uint32_t full_attn_idx, std::uint32_t pos,
                          ScratchArena& arena) noexcept {
  const std::size_t H = config.hidden_size;
  const std::size_t num_heads = config.num_heads;
  const std::size_t num_kv_heads = config.num_kv_heads;
  const std::size_t head_dim = config.head_dim;
  const std::size_t group_size = num_heads / num_kv_heads;

  // Joint QG projection: [H -> 2 * num_heads * head_dim]
  // Stride per head is 2 * head_dim (512 elements): first 256 is Q, next 256 is Gate
  std::vector<float> qg(2 * num_heads * head_dim, 0.0F);
  TensorGEMV(layer.attn_q, x_normed, 2 * num_heads * head_dim, H, qg);

  // K and V projections: [H -> num_kv_heads * head_dim]
  TensorGEMV(layer.attn_k, x_normed, num_kv_heads * head_dim, H, arena.k_buf);
  TensorGEMV(layer.attn_v, x_normed, num_kv_heads * head_dim, H, arena.v_buf);

  // Unpack Q and Gate, apply Q-norm and RoPE
  for (std::size_t h = 0; h < num_heads; ++h) {
    const float* head_qg = qg.data() + (h * 2 * head_dim);
    auto q_head = std::span<float>(arena.q_buf.data() + (h * head_dim), head_dim);
    auto gate_head = std::span<float>(arena.gate_buf.data() + (h * head_dim), head_dim);

    std::copy_n(head_qg, head_dim, q_head.data());
    std::copy_n(head_qg + head_dim, head_dim, gate_head.data());

    // Q RMSNorm
    ForwardRMSNorm(q_head, layer.attn_q_norm, config.rms_eps, q_head);
    // M-RoPE
    ApplyRope(q_head, config.rotary_dim, pos, config.rope_theta);
  }

  // Apply K-norm and RoPE, store in KV cache
  for (std::size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
    auto k_head = std::span<float>(arena.k_buf.data() + (kv_h * head_dim), head_dim);
    ForwardRMSNorm(k_head, layer.attn_k_norm, config.rms_eps, k_head);
    ApplyRope(k_head, config.rotary_dim, pos, config.rope_theta);

    auto k_slot = kv_cache.KeySlice(full_attn_idx, static_cast<std::uint32_t>(kv_h), pos);
    auto v_slot = kv_cache.ValueSlice(full_attn_idx, static_cast<std::uint32_t>(kv_h), pos);

    std::ranges::copy(k_head, k_slot.begin());
    std::copy_n(arena.v_buf.data() + (kv_h * head_dim), head_dim, v_slot.begin());
  }

  // Attention computation
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;

  std::ranges::fill(arena.context, 0.0F);

  for (std::size_t h = 0; h < num_heads; ++h) {
    const std::size_t kv_h = h / group_size;
    const auto q_head = std::span<const float>(arena.q_buf.data() + (h * head_dim), head_dim);

    float max_score = -1e30F;
    for (std::size_t p = 0; p < seq_len; ++p) {
      const auto k_p = kv_cache.KeySlice(full_attn_idx, static_cast<std::uint32_t>(kv_h), static_cast<std::uint32_t>(p));
      float dot = 0.0F;
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot += q_head[d] * k_p[d];
      }
      const float score = dot * scale;
      arena.scores[p] = score;
      max_score = std::max(max_score, score);
    }

    float sum_exp = 0.0F;
    for (std::size_t p = 0; p < seq_len; ++p) {
      const float exp_val = std::exp(arena.scores[p] - max_score);
      arena.scores[p] = exp_val;
      sum_exp += exp_val;
    }
    const float inv_sum = 1.0F / sum_exp;
    for (std::size_t p = 0; p < seq_len; ++p) {
      arena.scores[p] *= inv_sum;
    }

    auto ctx_head = std::span<float>(arena.context.data() + (h * head_dim), head_dim);
    for (std::size_t p = 0; p < seq_len; ++p) {
      const auto v_p = kv_cache.ValueSlice(full_attn_idx, static_cast<std::uint32_t>(kv_h), static_cast<std::uint32_t>(p));
      const float weight = arena.scores[p];
      for (std::size_t d = 0; d < head_dim; ++d) {
        ctx_head[d] += weight * v_p[d];
      }
    }

    // Apply sigmoid attention gating
    const auto gate_head = std::span<const float>(arena.gate_buf.data() + (h * head_dim), head_dim);
    for (std::size_t d = 0; d < head_dim; ++d) {
      ctx_head[d] *= Sigmoid(gate_head[d]);
    }
  }

  // Output projection: [num_heads * head_dim -> H]
  TensorGEMV(layer.attn_output, arena.context, H, num_heads * head_dim, arena.attn_out);
}

void ForwardLinearAttention(const LayerWeights& layer, const Config& config,
                            std::span<const float> x_normed, SsmState& ssm_state,
                            std::uint32_t ssm_layer_idx, ScratchArena& arena) noexcept {
  const std::size_t H = config.hidden_size;
  const std::size_t key_dim = config.SsmKeyDim();       // 2048
  const std::size_t val_dim = config.SsmValueDim();     // 8192
  const std::size_t conv_channels = config.SsmConvChannels(); // 12288
  const std::size_t v_heads = config.ssm_num_v_heads;   // 64
  const std::size_t k_heads = config.ssm_num_k_heads;   // 16
  const std::size_t ssm_head_dim = config.ssm_head_dim; // 128
  const std::size_t heads_ratio = v_heads / k_heads;    // 4

  // 1. Projections
  TensorGEMV(layer.ssm_qkv, x_normed, conv_channels, H, arena.ssm_qkv);
  TensorGEMV(layer.ssm_gate, x_normed, val_dim, H, arena.ssm_gate);
  TensorGEMV(layer.ssm_alpha, x_normed, v_heads, H, arena.ssm_alpha);
  TensorGEMV(layer.ssm_beta, x_normed, v_heads, H, arena.ssm_beta);

  // 2. Alpha (decay) and Beta (output gate)
  std::vector<float> decay(v_heads);
  std::vector<float> beta_sig(v_heads);
  const auto* dt_ptr = static_cast<const float*>(layer.ssm_dt.data);
  const auto* a_ptr = static_cast<const float*>(layer.ssm_a.data);

  for (std::size_t vh = 0; vh < v_heads; ++vh) {
    beta_sig[vh] = Sigmoid(arena.ssm_beta[vh]);
    const float alpha_biased = arena.ssm_alpha[vh] + (dt_ptr ? dt_ptr[vh] : 0.0F);
    const float sp = Softplus(alpha_biased);
    const float a_val = a_ptr ? a_ptr[vh] : 0.0F;
    decay[vh] = std::exp(a_val * sp);
  }

  // 3. Conv1D causal update
  auto conv_state = ssm_state.ConvState(ssm_layer_idx);
  const float* conv1d_w = static_cast<const float*>(layer.ssm_conv1d.data);

  for (std::size_t c = 0; c < conv_channels; ++c) {
    const float s0 = conv_state[c];
    const float s1 = conv_state[conv_channels + c];
    const float s2 = conv_state[(2 * conv_channels) + c];
    const float s3 = arena.ssm_qkv[c];

    conv_state[c] = s1;
    conv_state[conv_channels + c] = s2;
    conv_state[(2 * conv_channels) + c] = s3;

    float dot = s3;
    if (conv1d_w != nullptr) {
      const float* wc = conv1d_w + (c * 4);
      dot = (s0 * wc[0]) +
            (s1 * wc[1]) +
            (s2 * wc[2]) +
            (s3 * wc[3]);
    }
    arena.ssm_conv_out[c] = Silu(dot);
  }

  // 4. Extract Q (2048), K (2048), V (8192)
  const float* q_ptr = arena.ssm_conv_out.data();
  const float* k_ptr = arena.ssm_conv_out.data() + key_dim;
  const float* v_ptr = arena.ssm_conv_out.data() + (2 * key_dim);

  std::vector<float> q_normed(key_dim);
  std::vector<float> k_normed(key_dim);

  // L2-norm on Q and K heads
  for (std::size_t kh = 0; kh < k_heads; ++kh) {
    const float* qh = q_ptr + (kh * ssm_head_dim);
    const float* kh_p = k_ptr + (kh * ssm_head_dim);

    float sum_q = 0.0F;
    float sum_k = 0.0F;
    for (std::size_t d = 0; d < ssm_head_dim; ++d) {
      sum_q += qh[d] * qh[d];
      sum_k += kh_p[d] * kh_p[d];
    }
    const float inv_q = 1.0F / std::max(std::sqrt(sum_q), config.rms_eps);
    const float inv_k = 1.0F / std::max(std::sqrt(sum_k), config.rms_eps);

    for (std::size_t d = 0; d < ssm_head_dim; ++d) {
      q_normed[(kh * ssm_head_dim) + d] = qh[d] * inv_q;
      k_normed[(kh * ssm_head_dim) + d] = kh_p[d] * inv_k;
    }
  }

  // 5. DeltaNet recurrent state update and readout
  const float scale = 1.0F / std::sqrt(static_cast<float>(ssm_head_dim));
  for (std::size_t vh = 0; vh < v_heads; ++vh) {
    const std::size_t kh = vh % config.ssm_num_k_heads;
    const float* q_head = q_normed.data() + (kh * ssm_head_dim);
    const float* k_head = k_normed.data() + (kh * ssm_head_dim);
    const float* v_head = v_ptr + (vh * ssm_head_dim);
    const float d_val = decay[vh];
    const float b_val = beta_sig[vh];

    auto S = ssm_state.DeltaNetState(ssm_layer_idx, static_cast<std::uint32_t>(vh));

    // Step 1: kv_col = (S^T @ k)[col] = sum_r S[r, col] * k[r]
    std::vector<float> kv_col(ssm_head_dim, 0.0F);
    for (std::size_t col = 0; col < ssm_head_dim; ++col) {
      float dot = 0.0F;
      for (std::size_t r = 0; r < ssm_head_dim; ++r) {
        dot += S[(r * ssm_head_dim) + col] * k_head[r];
      }
      kv_col[col] = dot;
    }

    // Step 2 & 3: delta_col = (v[col] - decay * kv_col) * beta
    // S[r, col] = decay * S[r, col] + delta_col * k[r]
    // Readout: core = (S^T @ q) * scale
    float* core_h = arena.ssm_core_out.data() + (vh * ssm_head_dim);
    for (std::size_t col = 0; col < ssm_head_dim; ++col) {
      const float delta_col = (v_head[col] - (d_val * kv_col[col])) * b_val;
      float readout = 0.0F;
      for (std::size_t r = 0; r < ssm_head_dim; ++r) {
        float& s_entry = S[(r * ssm_head_dim) + col];
        s_entry = (s_entry * d_val) + (k_head[r] * delta_col);
        readout += s_entry * q_head[r];
      }
      core_h[col] = readout * scale;
    }

    // Gated RMSNorm: rms_norm(core_h, layer.ssm_norm) * silu(z)
    ForwardRMSNorm(std::span<const float>(core_h, ssm_head_dim), layer.ssm_norm,
                   config.rms_eps, std::span<float>(core_h, ssm_head_dim));
    const float* z_h = arena.ssm_gate.data() + (vh * ssm_head_dim);
    for (std::size_t col = 0; col < ssm_head_dim; ++col) {
      core_h[col] *= Silu(z_h[col]);
    }
  }

  // 6. Output projection: [val_dim -> H]
  TensorGEMV(layer.ssm_out, arena.ssm_core_out, H, val_dim, arena.attn_out);
}

void ForwardMoE(const LayerWeights& layer, const Config& config,
                std::span<const float> x_normed, ScratchArena& arena,
                std::span<float> moe_out) noexcept {
  const std::size_t H = config.hidden_size;
  const std::size_t num_experts = config.num_experts;
  const std::size_t num_experts_used = config.num_experts_used;
  const std::size_t expert_ff = config.expert_ff;
  const std::size_t shared_expert_ff = config.shared_expert_ff;

  // 1. Router logits: [H -> num_experts]
  TensorGEMV(layer.ffn_gate_inp, x_normed, num_experts, H, arena.router_logits);

  // Softmax over router logits
  float max_logit = -1e30F;
  for (std::size_t e = 0; e < num_experts; ++e) {
    max_logit = std::max(max_logit, arena.router_logits[e]);
  }

  std::vector<std::pair<float, std::uint32_t>> probs(num_experts);
  float sum_exp = 0.0F;
  for (std::size_t e = 0; e < num_experts; ++e) {
    const float val = std::exp(arena.router_logits[e] - max_logit);
    probs[e] = {val, static_cast<std::uint32_t>(e)};
    sum_exp += val;
  }
  const float inv_sum = 1.0F / sum_exp;
  for (std::size_t e = 0; e < num_experts; ++e) {
    probs[e].first *= inv_sum;
  }

  // Top-k selection
  std::partial_sort(probs.begin(), probs.begin() + num_experts_used, probs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  float top_sum = 0.0F;
  for (std::size_t k = 0; k < num_experts_used; ++k) {
    top_sum += probs[k].first;
  }
  const float inv_top_sum = 1.0F / (top_sum > 0.0F ? top_sum : 1.0F);

  std::ranges::fill(arena.routed_accum, 0.0F);

  // 2. Execute selected routed experts
  for (std::size_t k = 0; k < num_experts_used; ++k) {
    const std::uint32_t expert_id = probs[k].second;
    const float expert_weight = probs[k].first * inv_top_sum;

    TensorExpertGEMV(layer.ffn_gate_exps, expert_id, x_normed, expert_ff, H, arena.expert_gate);
    TensorExpertGEMV(layer.ffn_up_exps, expert_id, x_normed, expert_ff, H, arena.expert_up);

    for (std::size_t i = 0; i < expert_ff; ++i) {
      arena.expert_act[i] = Silu(arena.expert_gate[i]) * arena.expert_up[i];
    }

    TensorExpertGEMV(layer.ffn_down_exps, expert_id, arena.expert_act, H, expert_ff, arena.expert_down);

    for (std::size_t h = 0; h < H; ++h) {
      arena.routed_accum[h] += expert_weight * arena.expert_down[h];
    }
  }

  // 3. Execute shared expert
  TensorGEMV(layer.ffn_gate_shexp, x_normed, shared_expert_ff, H, arena.shexp_gate);
  TensorGEMV(layer.ffn_up_shexp, x_normed, shared_expert_ff, H, arena.shexp_up);

  for (std::size_t i = 0; i < shared_expert_ff; ++i) {
    arena.shexp_act[i] = Silu(arena.shexp_gate[i]) * arena.shexp_up[i];
  }

  TensorGEMV(layer.ffn_down_shexp, arena.shexp_act, H, shared_expert_ff, arena.shexp_down);

  // Shared expert gating: sigmoid(dot(x_normed, ffn_gate_inp_shexp))
  float shexp_gate_dot = 0.0F;
  if (!layer.ffn_gate_inp_shexp.empty()) {
    if (layer.ffn_gate_inp_shexp.type == core::GgmlType::kF32) {
      const auto* gw = static_cast<const float*>(layer.ffn_gate_inp_shexp.data);
      for (std::size_t h = 0; h < H; ++h) {
        shexp_gate_dot += x_normed[h] * gw[h];
      }
    } else if (layer.ffn_gate_inp_shexp.type == core::GgmlType::kBF16) {
      const auto* gw = static_cast<const std::uint16_t*>(layer.ffn_gate_inp_shexp.data);
      for (std::size_t h = 0; h < H; ++h) {
        const std::uint32_t u32 = static_cast<std::uint32_t>(gw[h]) << 16U;
        float val = 0.0F;
        std::memcpy(&val, &u32, sizeof(float));
        shexp_gate_dot += x_normed[h] * val;
      }
    }
  }
  const float shexp_gate_val = Sigmoid(shexp_gate_dot);

  // 4. Combine routed and shared expert
  for (std::size_t h = 0; h < H; ++h) {
    moe_out[h] = arena.routed_accum[h] + (shexp_gate_val * arena.shexp_down[h]);
  }
}

}  // namespace

void ForwardModel(std::uint32_t token_id, std::uint32_t pos,
                  const ModelWeights& weights, KvCache& kv_cache,
                  SsmState& ssm_state, ScratchArena& arena,
                  std::span<float> logits_out) noexcept {
  const auto& cfg = weights.config;
  const std::size_t H = cfg.hidden_size;

  // 1. Token embedding
  ForwardEmbedding(token_id, weights.token_embd, H, arena.hidden);

  std::uint32_t full_attn_idx = 0;
  std::uint32_t ssm_layer_idx = 0;
  std::vector<float> moe_out(H, 0.0F);

  // 2. Transformer layers
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    const auto& layer = weights.layers[l];

    // Pre-RMSNorm
    ForwardRMSNorm(arena.hidden, layer.attn_norm, cfg.rms_eps, arena.normed);

    // Attention / SSM
    if (layer.is_full_attention) {
      ForwardFullAttention(layer, cfg, arena.normed, kv_cache, full_attn_idx++, pos, arena);
    } else {
      ForwardLinearAttention(layer, cfg, arena.normed, ssm_state, ssm_layer_idx++, arena);
    }

    // Residual Add
    for (std::size_t h = 0; h < H; ++h) {
      arena.hidden[h] += arena.attn_out[h];
    }

    // Post-RMSNorm
    ForwardRMSNorm(arena.hidden, layer.post_attention_norm, cfg.rms_eps, arena.normed);

    // MoE FFN
    ForwardMoE(layer, cfg, arena.normed, arena, moe_out);

    // Residual Add
    for (std::size_t h = 0; h < H; ++h) {
      arena.hidden[h] += moe_out[h];
    }
  }

  // 3. Output Norm
  ForwardRMSNorm(arena.hidden, weights.output_norm, cfg.rms_eps, arena.normed);

  // 4. Output projection to logits
  if (!logits_out.empty() && !weights.output.empty()) {
    TensorGEMV(weights.output, arena.normed, cfg.vocab_size, H, logits_out);
  }
}

std::uint32_t GreedyArgmax(std::span<const float> logits) noexcept {
  if (logits.empty()) {
    return 0;
  }
  std::size_t best_idx = 0;
  float best_val = logits[0];
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  return static_cast<std::uint32_t>(best_idx);
}

}  // namespace gufo::models::qwen35_moe
