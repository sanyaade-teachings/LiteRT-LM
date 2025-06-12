#include "runtime/components/huggingface_tokenizer.h"

#include <fcntl.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "runtime/util/test_utils.h"     // NOLINT

namespace litert::lm {
namespace {

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

std::string GetHuggingFaceModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          "tokenizer.json")
      .string();
}

absl::StatusOr<std::string> GetContents(absl::string_view path) {
  ASSIGN_OR_RETURN(auto file, ScopedFile::Open(path));  // NOLINT
  int64_t contents_length = lseek(file.file(), 0, SEEK_END);
  if (contents_length < 0) {
    return absl::InternalError(absl::StrCat("Failed to get length: ", path));
  }

  std::string contents(contents_length, '\0');
  lseek(file.file(), 0, SEEK_SET);
  char* contents_ptr = contents.data();
  while (contents_length > 0) {
    int read_bytes = read(file.file(), contents_ptr, contents_length);
    if (read_bytes < 0) {
      return absl::InternalError(absl::StrCat("Failed to read: ", path));
    } else if (read_bytes == 0) {
      return absl::InternalError(absl::StrCat("File is empty: ", path));
    }
    contents_ptr += read_bytes;
    contents_length -= read_bytes;
  }

  return std::move(contents);
}

TEST(HuggingFaceTokenizerTtest, CreateFromFile) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(HuggingFaceTokenizerTtest, CreateFromBuffer) {
  ASSERT_OK_AND_ASSIGN(auto json, GetContents(GetHuggingFaceModelPath()));
  auto tokenizer_or = HuggingFaceTokenizer::CreateFromJson(json);
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(HuggingFaceTokenizerTtest, Create) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(HuggingFaceTokenizerTest, TextToTokenIds) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  ASSERT_OK(tokenizer_or);
  auto tokenizer = std::move(tokenizer_or.value());

  absl::string_view text = "How's it going?";
  auto ids_or = tokenizer->TextToTokenIds(text);
  ASSERT_OK(ids_or);

  EXPECT_THAT(ids_or.value(), ::testing::ElementsAre(2020, 506, 357, 2045, 47));
}

TEST(HuggingFaceTokenizerTest, TokenIdsToText) {
  auto tokenizer_or =
      HuggingFaceTokenizer::CreateFromFile(GetHuggingFaceModelPath());
  ASSERT_OK(tokenizer_or);
  auto tokenizer = std::move(tokenizer_or.value());

  const std::vector<int> ids = {2020, 506, 357, 2045, 47};
  auto text_or = tokenizer->TokenIdsToText(ids);
  ASSERT_OK(text_or);

  EXPECT_EQ(text_or.value(), "How's it going?");
}

}  // namespace
}  // namespace litert::lm
