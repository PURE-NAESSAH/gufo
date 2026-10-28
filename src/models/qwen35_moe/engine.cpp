#include "src/models/qwen35_moe/engine.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace gufo::models::qwen35_moe {

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  auto model = std::shared_ptr<Model>(new Model());
  model->options_ = options;

  model->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!model->reader_) {
    return nullptr;
  }

  auto cfg_opt = Config::FromGguf(*model->reader_, error_msg);
  if (!cfg_opt) {
    return nullptr;
  }
  model->config_ = *cfg_opt;

  model->weights_ = ModelWeights::FromGguf(*model->reader_, error_msg);
  if (!model->weights_) {
    return nullptr;
  }

  model->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*model->reader_, error_msg);
  if (!model->tokenizer_) {
    return nullptr;
  }

  if (!options.vision_model_path.empty()) {
#if defined(ENGINE_ENABLE_HIP)
    try {
      model->vision_ = qwen::vision::Encoder::Open(
          model_path, options.vision_model_path, model->config_.hidden_size);
    } catch (const std::exception& e) {
      if (error_msg != nullptr) {
        *error_msg = e.what();
      }
      return nullptr;
    }
#else
    if (error_msg != nullptr) {
      *error_msg = "Vision encoder requires HIP";
    }
    return nullptr;
#endif
  }

  return model;
}

std::unique_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  (void)mode;
  (void)error_msg;
  const auto effective_context =
      max_context > 0 ? std::min(max_context, options_.max_context)
                      : options_.max_context;
  return std::make_unique<Session>(shared_from_this(), effective_context);
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  if (!tokenizer_) {
    return {};
  }
  const auto unsigned_tokens = tokenizer_->Encode(text);
  std::vector<std::int32_t> result;
  result.reserve(unsigned_tokens.size());
  for (const auto t : unsigned_tokens) {
    result.push_back(static_cast<std::int32_t>(t));
  }
  return result;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  if (!tokenizer_) {
    return {};
  }
  std::vector<tokenization::TokenId> unsigned_tokens;
  unsigned_tokens.reserve(tokens.size());
  for (const auto t : tokens) {
    unsigned_tokens.push_back(static_cast<tokenization::TokenId>(t));
  }
  return tokenizer_->Decode(unsigned_tokens);
}

std::string Model::TokenText(std::int32_t token) const {
  if (!tokenizer_) {
    return {};
  }
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return tokenizer_ ? static_cast<std::int32_t>(tokenizer_->GetEosTokenId())
                    : 151645;
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_ &&
         tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return config_.vocab_size;
}

std::string Model::ModelName() const {
  if (reader_) {
    if (auto name = reader_->GetMetadataString("general.name")) {
      return std::string(*name);
    }
  }
  return "Ornith-1.5-122B";
}

std::size_t Model::ResidentBytes() const noexcept {
  return reader_ ? reader_->GetSize() : 0;
}

Session::Session(std::shared_ptr<const Model> model, std::uint32_t max_context)
    : model_(std::move(model)),
      max_context_(max_context),
      kv_cache_(model_->config(), max_context_),
      ssm_state_(model_->config()),
      scratch_(model_->config(), max_context_) {
  logits_.resize(model_->config().vocab_size, 0.0f);
}

void Session::Reset() {
  tokens_.clear();
  valid_ = false;
  kv_cache_.Reset();
  ssm_state_.Reset();
  vision_prompt_.reset();
}

std::size_t Session::AllocatedBytes() const noexcept {
  std::size_t bytes = 0;
  bytes += kv_cache_.AllocatedBytes();
  bytes += ssm_state_.AllocatedBytes();
  bytes += scratch_.AllocatedBytes();
  bytes += logits_.capacity() * sizeof(float);
  bytes += tokens_.capacity() * sizeof(std::int32_t);
  return bytes;
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt) {
  vision_prompt_ = std::move(prompt);
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    Reset();
    return true;
  }

  if (prompt.size() > max_context_) {
    if (error_msg) {
      *error_msg = "Prompt length (" + std::to_string(prompt.size()) +
                   ") exceeds maximum context (" + std::to_string(max_context_) +
                   ")";
    }
    return false;
  }

  std::size_t common_prefix = 0;
  while (common_prefix < tokens_.size() && common_prefix < prompt.size() &&
         tokens_[common_prefix] == prompt[common_prefix]) {
    ++common_prefix;
  }

  if (common_prefix < tokens_.size()) {
    Reset();
    common_prefix = 0;
  }

  tokens_.resize(common_prefix);

  for (std::size_t i = common_prefix; i < prompt.size(); ++i) {
    if (cancellation_check_ && cancellation_check_()) {
      if (error_msg) {
        *error_msg = "Execution cancelled";
      }
      return false;
    }

    const auto token = prompt[i];
    const auto pos = static_cast<std::uint32_t>(i);
    const bool is_last = (i + 1 == prompt.size());
    std::span<float> out_logits =
        is_last ? std::span<float>(logits_) : std::span<float>{};

    ForwardModel(static_cast<std::uint32_t>(token), pos, model_->weights(),
                 kv_cache_, ssm_state_, scratch_, out_logits);
    tokens_.push_back(token);
  }

  valid_ = true;
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  const auto pos = static_cast<std::uint32_t>(tokens_.size());
  if (pos >= max_context_) {
    if (error_msg) {
      *error_msg = "Context window exceeded";
    }
    return false;
  }

  ForwardModel(static_cast<std::uint32_t>(token), pos, model_->weights(),
               kv_cache_, ssm_state_, scratch_, std::span<float>(logits_));

  tokens_.push_back(token);
  valid_ = true;
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler,
                         DecodeResult* result,
                         std::string* error_msg,
                         bool stop_at_eos) {
  if (!valid_ || logits_.empty()) {
    if (error_msg) {
      *error_msg = "Session has not been synchronized with a prompt";
    }
    return false;
  }

  if (result == nullptr) {
    if (error_msg) {
      *error_msg = "DecodeResult output pointer is null";
    }
    return false;
  }

  result->tokens.clear();
  result->stop = false;

  for (std::size_t step = 0; step < max_tokens; ++step) {
    if (cancellation_check_ && cancellation_check_()) {
      if (error_msg) {
        *error_msg = "Execution cancelled";
      }
      return false;
    }

    const auto sampled_token =
        static_cast<std::int32_t>(sampler.Sample(logits_));
    result->tokens.push_back(sampled_token);

    if (stop_at_eos && model_->IsStopToken(sampled_token)) {
      result->stop = true;
      break;
    }

    if (step + 1 < max_tokens) {
      if (!Evaluate(sampled_token, error_msg)) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace gufo::models::qwen35_moe
