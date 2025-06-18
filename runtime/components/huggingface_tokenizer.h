#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_HUGGING_FACE_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_HUGGING_FACE_TOKENIZER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "include/tokenizers_cpp.h"  // from @tokenizers_cpp

namespace litert::lm {

// A Tokenizer implementation using HuggingFace.
class HuggingFaceTokenizer : public Tokenizer {
 public:
  // Creates a HuggingFaceTokenizer from the JSON file
  static absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>> CreateFromFile(
      absl::string_view json_path);

  // Creates a HuggingFaceTokenizer from a JSON string.
  static absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>> CreateFromJson(
      std::string json);

  // Encodes the given text into a sequence of token ids.
  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override;

  // Decodes the given sequence of token ids into a string.
  // Returns absl::DataLossError if any of the tokens are part of an incomplete
  // BPE sequence.
  absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) override;

 private:
  // Constructor.
  explicit HuggingFaceTokenizer(
      std::unique_ptr<tokenizers::Tokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {};

  // HuggingFace processor.
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_HUGGING_FACE_TOKENIZER_H_
