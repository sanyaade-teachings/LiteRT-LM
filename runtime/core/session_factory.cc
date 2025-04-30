#include "runtime/core/session_factory.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_basic.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<Engine::Session>> InitializeSession(
    std::shared_ptr<LlmExecutor> executor, std::shared_ptr<Tokenizer> tokenizer,
    const std::vector<int>& stop_token_ids,
    const SessionConfig& session_config) {
  return SessionBasic::Create(executor, tokenizer, stop_token_ids,
                              session_config);
}

}  // namespace litert::lm
