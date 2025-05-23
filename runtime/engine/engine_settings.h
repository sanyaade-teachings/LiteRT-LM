#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_

#include <optional>
#include <ostream>
#include <vector>

#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Settings used for initializing LiteRT LM.
// This class encapsulates the model-specific settings that are used for
// initializing the LiteRT LM. These settings are typically fixed for a given
// model and are not expected to change during the inference process.
// The model assets are required to initialize the LiteRT LM.
// TODO(b/397975034) Add overloading << operator for debugging.
class EngineSettings {
 public:
  explicit EngineSettings(
      const LlmExecutorSettings& executor_settings,
      std::optional<proto::BenchmarkParams> benchmark_params = std::nullopt);

  const LlmExecutorSettings& GetMainExecutorSettings() const;

  // Benchmark parameters:
  // Returns true if the benchmark is enabled.
  bool IsBenchmarkEnabled() const;
  // Returns the benchmark parameters.
  std::optional<proto::BenchmarkParams> GetBenchmarkParams() const;
  // Sets the benchmark parameters.
  void SetBenchmarkParams(const proto::BenchmarkParams& benchmark_params);

 private:
  // Settings for the main executor.
  LlmExecutorSettings main_executor_settings_;

  // Parameters used to configure the benchmarking process.
  std::optional<proto::BenchmarkParams> benchmark_params_;
};
std::ostream& operator<<(std::ostream& os, const EngineSettings& settings);

// Configurations used for the session.
// This class encapsulates the session-specific configurations that are used for
// creating a LiteRT LM session.
class SessionConfig {
 public:
  // Creates a default SessionConfig.
  static SessionConfig CreateDefault();

  // Sampler parameters:
  // Getters for the sampler parameters.
  const proto::SamplerParameters& GetSamplerParams() const;
  proto::SamplerParameters& GetMutableSamplerParams();
  // Sets the sampler parameters.
  void SetSamplerParams(const proto::SamplerParameters& sampler_params);

  // Stop token ids:
  // Getters for the stop token ids.
  const std::vector<std::vector<int>>& GetStopTokenIds() const;
  std::vector<std::vector<int>>& GetMutableStopTokenIds();
  // Sets the stop token ids.
  void SetStopTokenIds(const std::vector<std::vector<int>>& stop_token_ids);

  // Number of output candidates:
  // Getters for the number of output candidates.
  int GetNumOutputCandidates() const;
  void SetNumOutputCandidates(int num_output_candidates);

 private:
  // Private constructor for the SessionConfig. The user should use the
  // CreateDefault() method to create a SessionConfig.
  explicit SessionConfig(const proto::SamplerParameters& sampler_params);

  // Parameters used to configure the sampling process.
  proto::SamplerParameters sampler_params_;

  // Stop token ids for the session. Note that the stop token could be a
  // sequence of token ids (as opposed to a single token id). The first
  // dimension is the index of the stop token in the session, and the second
  // dimension is the sequence of token ids that constitutes the stop token.
  std::vector<std::vector<int>> stop_token_ids_;

  // The number of output candidates to generate. Default value is 1 and setting
  // it to a value greater than 1 will require the model to support batching.
  int num_output_candidates_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
