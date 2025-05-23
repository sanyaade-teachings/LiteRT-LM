#include "runtime/engine/engine_settings.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::litert::lm::EngineSettings;
using ::litert::lm::LlmExecutorSettings;
using ::testing::ElementsAre;
using ::testing::Eq;

TEST(EngineSettingsTest, GetModelPath) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  EngineSettings settings(*executor_settings);

  auto model_path =
      settings.GetMainExecutorSettings().GetModelAssets().GetPath();
  ASSERT_OK(model_path);
  EXPECT_EQ(*model_path, "test_model_path_1");
}

TEST(EngineSettingsTest, SetAndGetCacheDir) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir("test_cache_dir");
  EngineSettings settings(*executor_settings);
  EXPECT_EQ(settings.GetMainExecutorSettings().GetCacheDir(), "test_cache_dir");
}

TEST(EngineSettingsTest, SetAndGetMaxNumTokens) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);

  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetMaxNumTokens(128);
  EngineSettings settings(*executor_settings);
  EXPECT_EQ(settings.GetMainExecutorSettings().GetMaxNumTokens(), 128);
}

TEST(EngineSettingsTest, SetAndGetExecutorBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);

  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetBackend(Backend::GPU);
  EngineSettings settings(*executor_settings);
  EXPECT_THAT(settings.GetMainExecutorSettings().GetBackend(),
              Eq(Backend::GPU));
}

TEST(EngineSettingsTest, DefaultExecutorBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  EngineSettings settings(*executor_settings);
  EXPECT_THAT(settings.GetMainExecutorSettings().GetBackend(),
              Eq(Backend::CPU));
}

TEST(EngineSettingsTest, BenchmarkParams) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);

  EngineSettings settings(*executor_settings);
  EXPECT_FALSE(settings.IsBenchmarkEnabled());

  proto::BenchmarkParams benchmark_params;
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  settings.SetBenchmarkParams(benchmark_params);
  EXPECT_TRUE(settings.IsBenchmarkEnabled());
  EXPECT_EQ(settings.GetBenchmarkParams()->num_decode_tokens(), 100);
  EXPECT_EQ(settings.GetBenchmarkParams()->num_prefill_tokens(), 100);
}

TEST(SessionConfigTest, CreateDefault) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TOP_P);
  EXPECT_EQ(session_config.GetSamplerParams().k(), 1);
  EXPECT_EQ(session_config.GetSamplerParams().p(), 0.95f);
  EXPECT_EQ(session_config.GetSamplerParams().temperature(), 1.0f);
  EXPECT_EQ(session_config.GetSamplerParams().seed(), 0);
}

TEST(SessionConfigTest, SetAndGetSamplerParams) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_K);
  sampler_params.set_k(10);
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetSamplerParams(sampler_params);
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TOP_K);
  EXPECT_EQ(session_config.GetSamplerParams().k(), 10);

  // Mutable sampler params.
  session_config.GetMutableSamplerParams().set_type(
      proto::SamplerParameters::TYPE_UNSPECIFIED);
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TYPE_UNSPECIFIED);
}

TEST(SessionConfigTest, SetAndGetStopTokenIds) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  std::vector<std::vector<int>> stop_token_ids = {{0}, {1, 2}};
  session_config.SetStopTokenIds(stop_token_ids);
  EXPECT_EQ(session_config.GetStopTokenIds().size(), 2);
  EXPECT_THAT(session_config.GetStopTokenIds()[0], ElementsAre(0));
  EXPECT_THAT(session_config.GetStopTokenIds()[1], ElementsAre(1, 2));
}

TEST(SessionConfigTest, SetAndGetNumOutputCandidates) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetNumOutputCandidates(), 1);
  session_config.SetNumOutputCandidates(2);
  EXPECT_EQ(session_config.GetNumOutputCandidates(), 2);
}

}  // namespace
}  // namespace litert::lm
