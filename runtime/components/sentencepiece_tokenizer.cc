#include "runtime/components/sentencepiece_tokenizer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::lm {

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromFile(absl::string_view model_path) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->Load(model_path);
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromBuffer(absl::string_view model_buffer) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->LoadFromSerializedProto(model_buffer);
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

// Encodes the given text into a TensorBuffer of token ids.
absl::StatusOr<std::vector<int>> SentencePieceTokenizer::TextToTokenIds(
    absl::string_view text) {
  std::vector<int> ids;
  auto status = processor_->Encode(text, &ids);
  if (!status.ok()) {
    return status;
  }
  return ids;
}

// Decodes the given TensorBuffer of token ids into a string.
absl::StatusOr<std::string> SentencePieceTokenizer::TokenIdsToText(
    const std::vector<int>& token_ids) {
  std::string text = "";
  for (const auto& token_id : token_ids) {
    text += processor_->IdToPiece(token_id);
  }
  return text;
}

// Returns BOS id.
absl::StatusOr<int> SentencePieceTokenizer::BosId() const {
  auto id = processor_->bos_id();
  // Some converted SPM would mark BOS as EOS or UNK. Reuse EosId().
  if (id == -1) {
    return EosId();
  }
  return id;
};

// Returns EOS id.
absl::StatusOr<int> SentencePieceTokenizer::EosId() const {
  auto id = processor_->eos_id();
  // Some converted SPM would mark EOS as UNK.
  if (id == -1) {
    id = processor_->unk_id();
  }
  if (id == -1) {
    return absl::InternalError("EOS id is not found.");
  }
  return id;
}

}  // namespace litert::lm
