// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {

// A container to host the model responses.
class Responses {
 public:
  explicit Responses(int num_output_candidates);

  // Returns the number of output candidates.
  int GetNumOutputCandidates() const { return num_output_candidates_; }

  // Returns the response text at the given index. Returns error if the index is
  // out of range.
  absl::StatusOr<absl::string_view> GetResponseTextAt(int index) const;

  // Returns the score at the given index. Returns error if the index is out of
  // range or if scores are not included.
  // Note that the "score" is calculated as the sum of the log probabilities of
  // the whole decoded sequence normalized by the total number of tokens.
  absl::StatusOr<float> GetScoreAt(int index) const;

  // Returns the mutable response texts vector.
  std::vector<std::string>& GetMutableResponseTexts();

  // Returns the mutable scores vector. If it is the first time calling this
  // function, the scores vector will be allocated to the size of
  // num_output_candidates_ and initialized to the default value of -Inf
  // (= log(0.0f)).
  std::vector<float>& GetMutableScores();

 private:
  // The number of output candidates.
  const int num_output_candidates_;

  // The output vector of response tokens (as strings).
  std::vector<std::string> response_texts_;

  // The output vector of scores for each response text. The "score" is pulled
  // from the probability of the last token in the response text.
  std::vector<float> scores_;
};
std::ostream& operator<<(std::ostream& os, const Responses& responses);

// Class to store the data for a single turn of the benchmark. A "turn" is
// defined as a single RunPrefill or RunDecode call.
struct BenchmarkTurnData {
  absl::Duration duration;  // Duration of this entire operation/turn.
  uint64_t num_tokens;      // The number of tokens processed in this turn.
  BenchmarkTurnData(uint64_t tokens, absl::Duration dur);
};

// Class to store and manage comprehensive performance benchmark information for
// LLMs.
class BenchmarkInfo {
 public:
  BenchmarkInfo() = default;

  // --- Methods to record data ---
  void AddInitPhase(const std::string& phase_name, absl::Duration duration);
  void AddPrefillTurn(uint64_t num_tokens, absl::Duration duration);
  void AddDecodeTurn(uint64_t num_generated_tokens, absl::Duration duration);

  // --- Getters for raw data ---
  const std::map<std::string, absl::Duration>& GetInitPhases() const;

  // --- Calculated Metrics for Prefill ---
  uint64_t GetTotalPrefillTurns() const;
  double GetPrefillTokensPerSec(int turn_index) const;

  // --- Calculated Metrics for Decode ---
  uint64_t GetTotalDecodeTurns() const;
  double GetDecodeTokensPerSec(int turn_index) const;

 private:
  std::map<std::string, absl::Duration> init_phases_;
  std::vector<BenchmarkTurnData> prefill_turns_;
  std::vector<BenchmarkTurnData> decode_turns_;
};
std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
