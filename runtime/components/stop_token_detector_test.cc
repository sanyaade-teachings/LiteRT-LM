#include "runtime/components/stop_token_detector.h"

#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

TEST(StopTokenDetectorTest, AddStopSequence) {
  StopTokenDetector detector(1);
  EXPECT_TRUE(detector.AddStopTokenSequence({1, 2, 3}).ok());

  // Adding an empty sequence should fail
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            detector.AddStopTokenSequence({}).code());

  // Adding a repeated sequence should fail
  EXPECT_EQ(absl::StatusCode::kAlreadyExists,
            detector.AddStopTokenSequence({1, 2, 3}).code());

  EXPECT_TRUE(detector.AddStopTokenSequence({9}).ok());
}

TEST(StopTokenDetectorTest, ProcessTokensSingleStopToken) {
  StopTokenDetector detector(2);  // Batch size 2
  EXPECT_OK(detector.AddStopTokenSequence({5}));

  std::vector<int> tokens_item0 = {3, 4, 5, 6, 7};
  std::vector<int> tokens_item1 = {1, 0, 6, 5, 99};

  // Simulate processing token by token
  size_t i;
  for (i = 0; i < tokens_item0.size(); ++i) {
    std::vector<int> current_batch_tokens = {tokens_item0[i], tokens_item1[i]};
    EXPECT_TRUE(
        detector.ProcessTokens(absl::MakeSpan(current_batch_tokens)).ok());
    if (detector.AllDone().value()) {
      break;
    }
  }
  // Stop token, 5, is found for all batch items at step 3.
  EXPECT_EQ(i, 3);

  const auto& steps_before_stop_tokens = detector.GetStepsBeforeStopTokens();
  EXPECT_EQ(2, steps_before_stop_tokens.size());
  // Batch item 0: stop token found at step 2, the current step is 3. So the
  // steps before stop token is 2 = (3 - 2 + 1(# stop tokens)).
  EXPECT_EQ(2, steps_before_stop_tokens[0]);

  // Batch item 1: stop token found at step 3, the current step is 3. So the
  // steps before stop token is  = (3 - 3 + 1(# stop tokens)).
  EXPECT_EQ(1, steps_before_stop_tokens[1]);
}


TEST(StopTokenDetectorTest, ProcessTokensMultipleStopTokens) {
  StopTokenDetector detector(2);  // Batch size 2
  EXPECT_OK(detector.AddStopTokenSequence({5}));
  EXPECT_OK(detector.AddStopTokenSequence({7, 8, 9}));

  std::vector<int> tokens_item0 = {3, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> tokens_item1 = {1, 0, 0, 0, 0, 6, 5, 99};

  // Simulate processing token by token
  size_t i;
  for (i = 0; i < tokens_item0.size(); ++i) {
    std::vector<int> current_batch_tokens = {tokens_item0[i], tokens_item1[i]};
    EXPECT_TRUE(
        detector.ProcessTokens(absl::MakeSpan(current_batch_tokens)).ok());
    if (detector.AllDone().value()) {
      break;
    }
  }
  // Stop tokens are found for all batch items at step 6.
  EXPECT_EQ(i, 6);

  const auto& steps_before_stop_tokens = detector.GetStepsBeforeStopTokens();
  EXPECT_EQ(2, steps_before_stop_tokens.size());

  // Batch item 0: stop token found at step 5, the current step is 6. So the
  // steps before stop token is 4 = (6 - 5 + 3(# stop tokens)).
  EXPECT_EQ(5, steps_before_stop_tokens[0]);

  // Batch item 1: stop token found at step 6, the current step is 6. So the
  // steps before stop token is 1 = (6 - 6 + 1(# stop tokens)).
  EXPECT_EQ(1, steps_before_stop_tokens[1]);
}

TEST(StopTokenDetectorTest, ResetBatch) {
  StopTokenDetector detector(1);
  EXPECT_OK(detector.AddStopTokenSequence({1}));
  std::vector<int> tokens1 = {0, 2, 3, 1, 5};
  size_t i;
  for (i = 0; i < tokens1.size(); ++i) {
    std::vector<int> current_batch_tokens = {tokens1[i]};
    EXPECT_TRUE(
        detector.ProcessTokens(absl::MakeSpan(current_batch_tokens)).ok());
    if (detector.AllDone().value()) {
      break;
    }
  }
  EXPECT_EQ(i, 3);
  detector.ResetBatch();
  // Batch is not done after reset.
  EXPECT_FALSE(detector.AllDone().value());
  EXPECT_EQ(0, detector.GetStepsBeforeStopTokens()[0]);
}

TEST(StopTokenDetectorTest, ProcessTokenStrsSingleStopToken) {
  StopTokenDetector detector(2);  // Batch size 2
  EXPECT_OK(detector.AddStopTokenSequenceStr({"stop"}));

  std::vector<std::string> tokens_item0 = {"a", "b", "stop", "c", "d"};
  std::vector<std::string> tokens_item1 = {"x", "y", "z", "stop", "end"};

  // Simulate processing token by token
  size_t i;
  for (i = 0; i < tokens_item0.size(); ++i) {
    std::vector<std::string> current_batch_tokens = {tokens_item0[i],
                                                     tokens_item1[i]};
    EXPECT_TRUE(detector.ProcessTokenStrs(current_batch_tokens).ok());
    if (detector.AllDone().value()) {
      break;
    }
  }
  // Stop token, "stop", is found for all batch items at step 3.
  EXPECT_EQ(i, 3);

  // steps_before_stop_tokens is not updated in ProcessTokenStrs.
  const auto& steps_before_stop_tokens = detector.GetStepsBeforeStopTokens();
  EXPECT_EQ(2, steps_before_stop_tokens.size());
  EXPECT_EQ(0, steps_before_stop_tokens[0]);
  EXPECT_EQ(0, steps_before_stop_tokens[1]);
}

TEST(StopTokenDetectorTest, ProcessTokenStrsMultipleStopTokens) {
  StopTokenDetector detector(2);  // Batch size 2
  EXPECT_OK(detector.AddStopTokenSequenceStr({"stop"}));
  EXPECT_OK(detector.AddStopTokenSequenceStr({"end"}));

  std::vector<std::string> tokens_item0 = {"a",    "b", "end", "of",
                                           "text", "c", "d"};
  std::vector<std::string> tokens_item1 = {"x", "y", "z", "stop", "end"};

  // Simulate processing token by token
  size_t i;
  for (i = 0; i < tokens_item0.size(); ++i) {
    std::vector<std::string> current_batch_tokens = {tokens_item0[i],
                                                     tokens_item1[i]};
    EXPECT_TRUE(detector.ProcessTokenStrs(current_batch_tokens).ok());
    if (detector.AllDone().value()) {
      break;
    }
  }
  // Stop tokens are found for all batch items at step 4.
  EXPECT_EQ(i, 3);

  // steps_before_stop_tokens is not updated in ProcessTokenStrs.
  const auto& steps_before_stop_tokens = detector.GetStepsBeforeStopTokens();
  EXPECT_EQ(2, steps_before_stop_tokens.size());
  EXPECT_EQ(0, steps_before_stop_tokens[0]);
  EXPECT_EQ(0, steps_before_stop_tokens[1]);
}

TEST(StopTokenDetectorTest, ProcessTokenStrsNoStopToken) {
  StopTokenDetector detector(2);  // Batch size 2
  EXPECT_OK(detector.AddStopTokenSequenceStr({"stop"}));

  std::vector<std::string> tokens_item0 = {"a", "b", "c", "d", "e"};
  std::vector<std::string> tokens_item1 = {"x", "y", "z", "p", "q"};

  // Simulate processing token by token
  size_t i;
  for (i = 0; i < tokens_item0.size(); ++i) {
    std::vector<std::string> current_batch_tokens = {tokens_item0[i],
                                                     tokens_item1[i]};
    EXPECT_TRUE(detector.ProcessTokenStrs(current_batch_tokens).ok());
    EXPECT_FALSE(detector.AllDone().value());
  }

  // steps_before_stop_tokens is not updated in ProcessTokenStrs.
  const auto& steps_before_stop_tokens = detector.GetStepsBeforeStopTokens();
  EXPECT_EQ(2, steps_before_stop_tokens.size());
  EXPECT_EQ(0, steps_before_stop_tokens[0]);
  EXPECT_EQ(0, steps_before_stop_tokens[1]);
}

}  // namespace
}  // namespace litert::lm
