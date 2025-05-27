#include "schema/core/litertlm_section.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl

namespace litert {
namespace litertlm {
namespace schema {
namespace {

TEST(LiteRTLMSectionTest, TestFileBackedSectionStream) {
  // Get the file path
  const std::string file_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/attention.tflite";

  // Define an output file path
  std::ofstream output_file("/tmp/attention_copy.tflite", std::ios::binary);
  EXPECT_TRUE(output_file.is_open());

  // Create the file-backed Section stream object
  FileBackedSectionStream fbss(file_path);
  absl::Status result = fbss.Prepare();
  ASSERT_TRUE(result.ok());

  size_t fbss_size = fbss.BufferSize();
  auto& fbss_stream = fbss.GetStream();
  output_file << fbss_stream.rdbuf();
  EXPECT_EQ(output_file.tellp(), fbss_size);

  output_file.close();

  // Read the file back and check contents.
  std::ifstream input_file("/tmp/attention_copy.tflite", std::ios::binary);
  ASSERT_TRUE(input_file.is_open());

  // Read the original file into a buffer
  std::ifstream original_file(file_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(original_file.is_open());
  size_t original_size = original_file.tellg();
  original_file.seekg(0, std::ios::beg);
  std::vector<char> original_buffer(original_size);
  original_file.read(original_buffer.data(), original_size);
  original_file.close();

  // Read the copied file into a buffer
  std::vector<char> copied_buffer(fbss_size);
  input_file.read(copied_buffer.data(), fbss_size);
  input_file.close();

  // Compare the buffers
  ASSERT_EQ(original_size, fbss_size);
  EXPECT_TRUE(std::equal(original_buffer.begin(), original_buffer.end(),
                         copied_buffer.begin(), copied_buffer.end()));
}

TEST(LiteRTLMSectionTest, TestProtoSectionStream) {
  using odml::infra::proto::LlmParameters;
  using odml::infra::proto::PromptTemplate;

  // Constants for the Token Generation Data
  const std::string start_token = "<start>";
  const std::vector<std::string> stop_tokens = {"<stop>", "<eos>"};

  bool enable_bytes_to_unicode_mapping = true;

  const std::string system_prompt = "You are a helpful AI assistant.";
  const std::string prompt_prefix = "User: ";
  const std::string prompt_suffix = "Assistant: ";
  const std::string output_filename = "/tmp/llm_params.pb";

  // Create an LlmParameters protocol buffer
  LlmParameters params;

  // Set the start_token
  params.set_start_token(start_token);

  // Set the stop_tokens
  for (const std::string& stop_token : stop_tokens) {
    params.add_stop_tokens(stop_token);
  }

  // Set the input_output_normalizations based on the config flag
  if (enable_bytes_to_unicode_mapping) {
    params.add_input_output_normalizations(
        LlmParameters::INPUT_OUTPUT_NORMALIZATION_BYTES_TO_UNICODE);
  }

  // Set the prompt template fields
  PromptTemplate* prompt_template = params.mutable_prompt_template();
  prompt_template->set_session_prefix(system_prompt);
  prompt_template->set_prompt_prefix(prompt_prefix);
  prompt_template->set_prompt_suffix(prompt_suffix);

  // ** Write the file using typical protobuf serialization **
  std::string serialized_params = params.SerializeAsString();
  // Convert the serialized string to a vector of unsigned chars
  std::vector<unsigned char> buffer(serialized_params.begin(),
                                    serialized_params.end());
  // Write the buffer to a file
  std::ofstream output_file(output_filename, std::ios::binary);
  ASSERT_TRUE(output_file.is_open());
  output_file.write(reinterpret_cast<const char*>(buffer.data()),
                    buffer.size());
  std::ofstream::pos_type bytes_written = output_file.tellp();
  ASSERT_GT(bytes_written, 0);
  output_file.close();

  // ** Write the file using SectionStream interface **
  ProtoBufSectionStream<LlmParameters> pbss(params);
  absl::Status result = pbss.Prepare();
  ASSERT_TRUE(result.ok());

  size_t pbss_size = pbss.BufferSize();
  auto& pbss_stream = pbss.GetStream();
  const std::string output_filename_streamed = "/tmp/llm_params_streamed.pb";
  std::ofstream output_streamed(output_filename_streamed, std::ios::binary);
  ASSERT_TRUE(output_streamed.is_open());

  output_streamed << pbss_stream.rdbuf();
  EXPECT_EQ(output_streamed.tellp(), pbss_size);

  output_file.close();

  // ** Read the file back in and check the contents **
  std::ifstream input_streamed(output_filename_streamed, std::ios::binary);
  ASSERT_TRUE(input_streamed.is_open());
  std::stringstream ss;
  ss << input_streamed.rdbuf();  // Read the entire file into a stringstream
  std::string serialized_read_back = ss.str();
  input_streamed.close();

  LlmParameters params_read_back;
  ASSERT_TRUE(params_read_back.ParseFromString(serialized_read_back));

  // Compare the fields.
  EXPECT_EQ(params.start_token(), params_read_back.start_token());
  EXPECT_EQ(params.stop_tokens().size(), params_read_back.stop_tokens().size());
  for (int i = 0; i < params.stop_tokens().size(); ++i) {
    EXPECT_EQ(params.stop_tokens(i), params_read_back.stop_tokens(i));
  }
  EXPECT_EQ(params.input_output_normalizations().size(),
            params_read_back.input_output_normalizations().size());
  for (int i = 0; i < params.input_output_normalizations().size(); ++i) {
    EXPECT_EQ(params.input_output_normalizations(i),
              params_read_back.input_output_normalizations(i));
  }

  const PromptTemplate& prompt_template_read_back =
      params_read_back.prompt_template();
  EXPECT_EQ(params.prompt_template().session_prefix(),
            prompt_template_read_back.session_prefix());
  EXPECT_EQ(params.prompt_template().prompt_prefix(),
            prompt_template_read_back.prompt_prefix());
  EXPECT_EQ(params.prompt_template().prompt_suffix(),
            prompt_template_read_back.prompt_suffix());
}

}  // namespace
}  // namespace schema
}  // namespace litertlm
}  // namespace litert
