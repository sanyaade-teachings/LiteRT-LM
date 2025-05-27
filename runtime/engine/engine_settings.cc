#include "runtime/engine/engine_settings.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

absl::StatusOr<EngineSettings> EngineSettings::CreateDefault(
    const ModelAssets& model_assets, Backend backend) {
  ASSIGN_OR_RETURN(auto executor_settings,  // NOLINT
                   LlmExecutorSettings::CreateDefault(model_assets, backend));
  return EngineSettings(executor_settings, std::nullopt);
}

absl::Status EngineSettings::MaybeUpdateAndValidate(
    std::shared_ptr<Tokenizer> tokenizer) {
  // TODO(b/413793273): Load the metadata from the model assets.
  // Currently pretending the values are hard-coded.
  proto::LlmMetadata& metadata = GetMutableLlmMetadata();
  metadata.mutable_stop_tokens()->Add()->set_token_str("<eos>");
  metadata.mutable_stop_tokens()->Add()->set_token_str("<end_of_turn>");
  metadata.mutable_stop_tokens()->Add()->set_token_str("<ctrl100>");
  metadata.mutable_start_token()->mutable_token_ids()->add_ids(2);

  // Convert the start/stop tokens from string to token ids.
  for (auto& stop_token : *metadata.mutable_stop_tokens()) {
    if (stop_token.has_token_str()) {
      auto stop_token_ids = tokenizer->TextToTokenIds(stop_token.token_str());
      if (stop_token_ids.ok()) {
        stop_token.mutable_token_ids()->mutable_ids()->Add(
            stop_token_ids->begin(), stop_token_ids->end());
      }
    }
  }
  if (metadata.start_token().has_token_str()) {
    auto start_token_ids =
        tokenizer->TextToTokenIds(metadata.start_token().token_str());
    if (start_token_ids.ok()) {
      metadata.mutable_start_token()
          ->mutable_token_ids()
          ->mutable_ids()
          ->Add(start_token_ids->begin(), start_token_ids->end());
    }
  }
  // Load the max num tokens from the model file.
  main_executor_settings_.SetMaxNumTokens(160);

  // Set the default values for the sampler params.
  if (!metadata.has_sampler_params()) {
    proto::SamplerParameters& sampler_params =
        *metadata.mutable_sampler_params();
    Backend backend = main_executor_settings_.GetBackend();
    if (backend == Backend::QNN) {
      sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
    } else if (backend == Backend::CPU || backend == Backend::GPU) {
      sampler_params.set_type(proto::SamplerParameters::TOP_P);
      sampler_params.set_k(1);
      sampler_params.set_p(0.95f);
      sampler_params.set_temperature(1.0f);
      sampler_params.set_seed(0);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Not recognized backend: ", backend));
    }
  }
  return absl::OkStatus();
}

EngineSettings::EngineSettings(
    const LlmExecutorSettings& executor_settings,
    std::optional<proto::BenchmarkParams> benchmark_params)
    : main_executor_settings_(executor_settings),
      benchmark_params_(benchmark_params) {}

const LlmExecutorSettings& EngineSettings::GetMainExecutorSettings() const {
  return main_executor_settings_;
}
LlmExecutorSettings& EngineSettings::GetMutableMainExecutorSettings() {
  return main_executor_settings_;
}

// Benchmark parameters:
// Returns true if the benchmark is enabled.
bool EngineSettings::IsBenchmarkEnabled() const {
  return benchmark_params_.has_value();
}
// Returns the benchmark parameters.
const std::optional<proto::BenchmarkParams>&
EngineSettings::GetBenchmarkParams() const {
  return benchmark_params_;
}
// Returns the mutable benchmark parameters.
proto::BenchmarkParams& EngineSettings::GetMutableBenchmarkParams() {
  if (!benchmark_params_.has_value()) {
    benchmark_params_ = proto::BenchmarkParams();
  }
  return benchmark_params_.value();
}

const std::optional<proto::LlmMetadata>& EngineSettings::GetLlmMetadata()
    const {
  return metadata_;
}

proto::LlmMetadata& EngineSettings::GetMutableLlmMetadata() {
  if (!metadata_.has_value()) {
    metadata_ = proto::LlmMetadata();
  }
  return metadata_.value();
}

SessionConfig SessionConfig::CreateDefault() {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
  auto config = SessionConfig(sampler_params);
  config.SetNumOutputCandidates(1);
  // Default to -1 to indicate the start token is not set. This is to be
  // overridden by the EngineSettings.
  config.SetStartTokenId(-1);
  return config;
}

absl::Status SessionConfig::MaybeUpdateAndValidate(
    const EngineSettings& engine_settings) {
  if ((start_token_id_ == -1 || stop_token_ids_.empty()) &&
      !engine_settings.GetLlmMetadata().has_value()) {
    return absl::InvalidArgumentError(
        "Start token and stop tokens are required. Either set the start token "
        "id or provide a valid start token in the model file/engine settings.");
  }
  // Update the parameters from the engine settings when the LlmMetadata is
  // present.
  if (engine_settings.GetLlmMetadata().has_value()) {
    const auto llm_metadata = engine_settings.GetLlmMetadata().value();
    proto::SamplerParameters& sampler_params = GetMutableSamplerParams();
    // Update the sampler params if the session config does not have a sampler
    // params and the engine settings has a sampler params (probably read from
    // the model file).
    if ((sampler_params.type() == proto::SamplerParameters::TYPE_UNSPECIFIED)) {
      if (llm_metadata.has_sampler_params()) {
        sampler_params = engine_settings.GetLlmMetadata()->sampler_params();
      }
    }

    // Set and validate the start token.
    if (start_token_id_ == -1) {
      if (llm_metadata.has_start_token() &&
          llm_metadata.start_token().token_ids().ids_size() == 1) {
        start_token_id_ = llm_metadata.start_token().token_ids().ids(0);
      }
    }

    // Set and validate the stop tokens.
    if (stop_token_ids_.empty()) {
      for (const auto& stop_token : llm_metadata.stop_tokens()) {
        if (stop_token.has_token_ids() &&
            stop_token.token_ids().ids_size() > 0) {
          std::vector<int> stop_token_ids(
              stop_token.token_ids().ids().begin(),
              stop_token.token_ids().ids().end());
          stop_token_ids_.push_back(stop_token_ids);
        }
      }
    }
  }

  // Validating the required fields are set correctly.
  if (start_token_id_ == -1) {
    return absl::InvalidArgumentError(
        "Start token is required. Either set the start token id or provide "
        "a valid start token in the model file/engine settings.");
  }
  if (stop_token_ids_.empty()) {
    return absl::InvalidArgumentError(
        "Stop tokens are required. Either set the stop token ids or "
        "provide "
        "a valid stop token in the model file/engine settings.");
  }
  if (num_output_candidates_ < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of output candidates need to be at least 1, but got: ",
        num_output_candidates_));
  }
  return absl::OkStatus();
}

SessionConfig::SessionConfig(const proto::SamplerParameters& sampler_params)
    : sampler_params_(sampler_params) {}

const proto::SamplerParameters& SessionConfig::GetSamplerParams() const {
  return sampler_params_;
}

proto::SamplerParameters& SessionConfig::GetMutableSamplerParams() {
  return sampler_params_;
}

const std::vector<std::vector<int>>& SessionConfig::GetStopTokenIds() const {
  return stop_token_ids_;
}

std::vector<std::vector<int>>& SessionConfig::GetMutableStopTokenIds() {
  return stop_token_ids_;
}
int SessionConfig::GetStartTokenId() const { return start_token_id_; }

void SessionConfig::SetStartTokenId(int start_token_id) {
  start_token_id_ = start_token_id;
}

int SessionConfig::GetNumOutputCandidates() const {
  return num_output_candidates_;
}
void SessionConfig::SetNumOutputCandidates(int num_output_candidates) {
  num_output_candidates_ = num_output_candidates;
}

}  // namespace litert::lm
