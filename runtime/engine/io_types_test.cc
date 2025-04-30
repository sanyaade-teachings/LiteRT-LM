#include "runtime/engine/io_types.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

std::string FloatToString(float val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

TEST(ResponsesTest, GetResponseTextAt) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableResponseTexts()[0] = "Hello World!";
  responses.GetMutableResponseTexts()[1] = "How's it going?";
  EXPECT_THAT(responses.GetResponseTextAt(0), IsOkAndHolds("Hello World!"));
  EXPECT_THAT(responses.GetResponseTextAt(1), IsOkAndHolds("How's it going?"));
  EXPECT_THAT(responses.GetResponseTextAt(2),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, GetScoreAt) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
  EXPECT_THAT(responses.GetScoreAt(0), IsOkAndHolds(0.1));
  EXPECT_THAT(responses.GetScoreAt(1), IsOkAndHolds(0.2));
  EXPECT_THAT(responses.GetScoreAt(2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, GetMutableScores) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
  EXPECT_THAT(responses.GetScoreAt(0), IsOkAndHolds(0.1));
  EXPECT_THAT(responses.GetScoreAt(1), IsOkAndHolds(0.2));
  EXPECT_THAT(responses.GetScoreAt(2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, HasScores) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
}

TEST(ResponsesTest, GetMutableResponseTexts) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableResponseTexts()[0] = "Hello World!";
  responses.GetMutableResponseTexts()[1] = "How's it going?";
  EXPECT_THAT(responses.GetMutableResponseTexts()[0], "Hello World!");
  EXPECT_THAT(responses.GetMutableResponseTexts()[1], "How's it going?");
}

TEST(ResponsesTest, HandlesMultipleCandidatesWithTextAndScores) {
  litert::lm::Responses responses(2);
  responses.GetMutableResponseTexts()[0] = "Hello";
  responses.GetMutableResponseTexts()[1] = "World";
  responses.GetMutableScores()[0] = 0.9f;
  responses.GetMutableScores()[1] = -0.5f;  // Test with a negative score

  std::stringstream ss;
  ss << responses;

  const std::string expected_output =
      "Total candidates: 2:\n"
      "  Candidate 0 (score: " +
      FloatToString(0.9f) +
      "):\n"
      "    Text: \"Hello\"\n"
      "  Candidate 1 (score: " +
      FloatToString(-0.5f) +
      "):\n"
      "    Text: \"World\"\n";
  EXPECT_EQ(ss.str(), expected_output);
}

TEST(ResponsesTest, HandlesMultipleCandidatesWithTextAndNoScores) {
  litert::lm::Responses responses(2);
  responses.GetMutableResponseTexts()[0] = "Hello";
  responses.GetMutableResponseTexts()[1] = "World";

  std::stringstream ss;
  ss << responses;

  const std::string expected_output =
      "Total candidates: 2:\n"
      "  Candidate 0 (score: N/A):\n"
      "    Text: \"Hello\"\n"
      "  Candidate 1 (score: N/A):\n"
      "    Text: \"World\"\n";
  EXPECT_EQ(ss.str(), expected_output);
}

// --- Test Init Phases ---
TEST(BenchmarkInfoTests, AddAndGetInitPhases) {
  BenchmarkInfo benchmark_info;
  benchmark_info.AddInitPhase("Tokenizer Load", absl::Milliseconds(50.5));
  benchmark_info.AddInitPhase("Model Load", absl::Milliseconds(1200.75));

  const auto& phases = benchmark_info.GetInitPhases();
  ASSERT_EQ(phases.size(), 2);
  EXPECT_EQ(phases.at("Tokenizer Load"), absl::Milliseconds(50.5));
  EXPECT_EQ(phases.at("Model Load"), absl::Milliseconds(1200.75));
}

// --- Test Prefill Turns ---
TEST(BenchmarkInfoTests, AddAndGetPrefillTurns) {
  BenchmarkInfo benchmark_info;
  benchmark_info.AddPrefillTurn(100, absl::Milliseconds(50));
  benchmark_info.AddPrefillTurn(200, absl::Milliseconds(100));

  EXPECT_EQ(benchmark_info.GetTotalPrefillTurns(), 2);
  double expected_actual_tps =
      100.0 / absl::ToDoubleSeconds(absl::Milliseconds(50));
  EXPECT_EQ(benchmark_info.GetPrefillTokensPerSec(0), expected_actual_tps);
  EXPECT_EQ(benchmark_info.GetPrefillTokensPerSec(1), expected_actual_tps);
}

// --- Test Decode Turns ---
TEST(BenchmarkInfoTests, AddAndGetDecodeTurns) {
  BenchmarkInfo benchmark_info;
  benchmark_info.AddDecodeTurn(10, absl::Milliseconds(50));

  EXPECT_EQ(benchmark_info.GetTotalDecodeTurns(), 1);

  double expected_decode_turns_ps =
      10.0 / absl::ToDoubleSeconds(absl::Milliseconds(50));
  EXPECT_EQ(benchmark_info.GetDecodeTokensPerSec(0), expected_decode_turns_ps);
}

TEST(BenchmarkInfoTests, OperatorOutputWithData) {
  BenchmarkInfo benchmark_info;
  benchmark_info.AddInitPhase("Load Model", absl::Milliseconds(1000.50));
  benchmark_info.AddInitPhase("Load Tokenizer", absl::Milliseconds(50.25));

  benchmark_info.AddPrefillTurn(10, absl::Milliseconds(20));
  benchmark_info.AddPrefillTurn(20, absl::Milliseconds(10));

  benchmark_info.AddDecodeTurn(5, absl::Milliseconds(25));

  std::stringstream ss;
  ss << benchmark_info;
  const std::string expected_output = R"(BenchmarkInfo:
  Init Phases (2):
    - Load Model: 1000.50 ms
    - Load Tokenizer: 50.25 ms
    Total init time: 1050.75 ms
--------------------------------------------------
  Prefill Turns (Total: 2):
    Prefill Turn 1:
      Prefill Speed: 500.00 tokens/sec.
    Prefill Turn 2:
      Prefill Speed: 2000.00 tokens/sec.
--------------------------------------------------
  Decode Turns (Total: 1):
    Decode Turn 1:
      Decode Speed: 200.00 tokens/sec.
--------------------------------------------------
)";
  EXPECT_EQ(ss.str(), expected_output);
}

}  // namespace
}  // namespace litert::lm
