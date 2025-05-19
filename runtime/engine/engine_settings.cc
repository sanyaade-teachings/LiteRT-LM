#include "runtime/engine/engine_settings.h"

#include <optional>
#include <vector>

#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

EngineSettings::EngineSettings(
    const LlmExecutorSettings& executor_settings,
    std::optional<proto::BenchmarkParams> benchmark_params)
    : main_executor_settings_(executor_settings),
      benchmark_params_(benchmark_params) {}

const LlmExecutorSettings& EngineSettings::GetMainExecutorSettings() const {
  return main_executor_settings_;
}

// Benchmark parameters:
// Returns true if the benchmark is enabled.
bool EngineSettings::IsBenchmarkEnabled() const {
  return benchmark_params_.has_value();
}
// Returns the benchmark parameters.
std::optional<proto::BenchmarkParams> EngineSettings::GetBenchmarkParams()
    const {
  return benchmark_params_;
}
// Sets the benchmark parameters.
void EngineSettings::SetBenchmarkParams(
    const proto::BenchmarkParams& benchmark_params) {
  benchmark_params_ = benchmark_params;
}

SessionConfig SessionConfig::CreateDefault() {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.95f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  auto config = SessionConfig(sampler_params);
  config.SetNumOutputCandidates(1);
  return config;
}

SessionConfig::SessionConfig(const proto::SamplerParameters& sampler_params)
    : sampler_params_(sampler_params) {}

const proto::SamplerParameters& SessionConfig::GetSamplerParams() const {
  return sampler_params_;
}

proto::SamplerParameters& SessionConfig::GetMutableSamplerParams() {
  return sampler_params_;
}

void SessionConfig::SetSamplerParams(
    const proto::SamplerParameters& sampler_params) {
  sampler_params_ = sampler_params;
}

const std::vector<std::vector<int>>& SessionConfig::GetStopTokenIds() const {
  return stop_token_ids_;
}

std::vector<std::vector<int>>& SessionConfig::GetMutableStopTokenIds() {
  return stop_token_ids_;
}

void SessionConfig::SetStopTokenIds(
    const std::vector<std::vector<int>>& stop_token_ids) {
  stop_token_ids_ = stop_token_ids;
}

int SessionConfig::GetNumOutputCandidates() const {
  return num_output_candidates_;
}
void SessionConfig::SetNumOutputCandidates(int num_output_candidates) {
  num_output_candidates_ = num_output_candidates;
}

}  // namespace litert::lm
