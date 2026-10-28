#ifndef GUFO_MODELS_QWEN35_MOE_ENGINE_HPP_
#define GUFO_MODELS_QWEN35_MOE_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen35_moe/config.hpp"
#include "src/models/qwen35_moe/forward.hpp"
#include "src/models/qwen35_moe/state.hpp"
#include "src/models/qwen35_moe/weights.hpp"

namespace gufo::models::qwen35_moe {

struct ModelOptions {
  std::uint32_t max_context = 4096;
  std::string vision_model_path;
  std::uint32_t decode_concurrency = 1;
};

class Session;

class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const ModelWeights& weights() const noexcept { return *weights_; }
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] const std::shared_ptr<qwen::vision::Encoder>& VisionEncoder()
      const noexcept {
    return vision_;
  }

private:
  Model() = default;

  ModelOptions options_;
  Config config_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::shared_ptr<qwen::vision::Encoder> vision_;

  friend class Session;
};

class Session final {
public:
  Session(std::shared_ptr<const Model> model, std::uint32_t max_context);
  ~Session() = default;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);

  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };

  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);

  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept {
    return static_cast<std::uint32_t>(tokens_.size());
  }
  [[nodiscard]] std::uint32_t ContextSize() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  void SetCancellationCheck(std::function<bool()> check) {
    cancellation_check_ = std::move(check);
  }
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt);

private:
  std::shared_ptr<const Model> model_;
  std::uint32_t max_context_{4096};
  bool valid_{false};
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  KvCache kv_cache_;
  SsmState ssm_state_;
  ScratchArena scratch_;
  std::function<bool()> cancellation_check_;
  std::shared_ptr<const qwen::vision::Prompt> vision_prompt_;
};

}  // namespace gufo::models::qwen35_moe

#endif  // GUFO_MODELS_QWEN35_MOE_ENGINE_HPP_
