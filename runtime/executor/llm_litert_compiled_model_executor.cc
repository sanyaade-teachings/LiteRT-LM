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

#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_runtime_options.h"  // from @litert
#include "runtime/components/embedding_lookup_text.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_util.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::absl::Span;
using ::litert::Expected;
using ::litert::GpuOptions;
using ::litert::TensorBuffer;

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr char kPrefillSignatureRunner[] = "prefill";
constexpr char kDecodeSignatureRunner[] = "decode";

absl::Status GetCacheRootNames(std::vector<absl::string_view> input_names,
                               std::string& k_root_name,
                               std::string& v_root_name) {
  for (auto input_name : input_names) {
    if (input_name == "kv_cache_k_0") {
      k_root_name = "kv_cache_k_";
      v_root_name = "kv_cache_v_";
      return absl::OkStatus();
    } else if (input_name == "k_cache_0") {
      k_root_name = "k_cache_";
      v_root_name = "v_cache_";
      return absl::OkStatus();
    }
  }
  return absl::FailedPreconditionError("No KV cache inputs found.");
}

bool IsCalculationPrecisionF16() { return true; }

}  // namespace

absl::Status LlmLiteRtCompiledModelExecutor::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {
  LITERT_ASSIGN_OR_RETURN_ABSL(auto tensor_type,
                               (*inputs.GetTextTokenIdsPtr())->TensorType());
  // Only accept batch size 1 for now.
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_ASSIGN_OR_RETURN_ABSL(auto ids, ReferTensorBufferAsSpan<int32_t>(
                                             *(*inputs.GetTextTokenIdsPtr())));

  ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                         prefill_signature_map_, ids.size()));
  for (const auto& [prefill_signature, prefill_length] : work_groups) {
    // Create input_token, positions and attn_mask buffers after determining
    // the prefill length.
    if (!signatures_.input_tokens.empty()) {
      auto tokens_buffer = compiled_model_.CreateInputBuffer(
          prefill_signature, signatures_.input_tokens);
      prefill_input_buffers_[signatures_.input_tokens] =
          std::move(*tokens_buffer);
    } else {
      // If input_tokens is empty, we must have input_embeddings.
      if (!signatures_.input_embeddings.has_value()) {
        return absl::FailedPreconditionError(
            "Input tokens or embeddings must be provided.");
      }
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Input embeddings required by signature but embedding lookup "
            "model is not initialized.");
      }
      auto embeddings_buffer = compiled_model_.CreateInputBuffer(
          prefill_signature, signatures_.input_embeddings.value());
      prefill_input_buffers_[signatures_.input_embeddings.value()] =
          std::move(*embeddings_buffer);

      // We may have per layer embedding as well.
      if (signatures_.input_per_layer_embeddings.has_value()) {
        if (embedding_lookup_ == nullptr) {
          return absl::FailedPreconditionError(
              "Input per layer embeddings required by signature but embedding "
              "lookup model is not initialized.");
        }
        auto per_layer_embeddings_buffer = compiled_model_.CreateInputBuffer(
            prefill_signature, signatures_.input_per_layer_embeddings.value());
        prefill_input_buffers_[signatures_.input_per_layer_embeddings.value()] =
            std::move(*per_layer_embeddings_buffer);
      }
    }
    auto positions_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_positions);
    prefill_input_buffers_[signatures_.input_positions] =
        std::move(*positions_buffer);

    if (signatures_.input_attn_mask.has_value()) {
      auto attn_mask_buffer = compiled_model_.CreateInputBuffer(
          prefill_signature, signatures_.input_attn_mask.value());
      prefill_input_buffers_[signatures_.input_attn_mask.value()] =
          std::move(*attn_mask_buffer);
    }
    RETURN_IF_ERROR(PrefillInternal(prefill_signature,
                                    ids.subspan(/*pos=*/0, prefill_length)));
    ids = ids.subspan(/*pos=*/prefill_length);
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::PrefillInternal(
    absl::string_view prefill_signature, Span<const int> ids) {
  {
    // Fill the input buffers with scoped locks.
    auto& prefill_input_pos =
        prefill_input_buffers_[signatures_.input_positions];
    LITERT_ASSIGN_OR_RETURN_ABSL(auto prefill_input_pos_size,
                                 prefill_input_pos.PackedSize());
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto prefill_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            prefill_input_pos, TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr.second);
    bool has_input_attn_mask = signatures_.input_attn_mask.has_value();

    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    if (has_input_attn_mask) {
      RET_CHECK(signatures_.input_attn_mask_data_type.has_value())
          << "Attention mask data type is not provided.";
      RETURN_IF_ERROR(InitializeAttentionMask(
          prefill_input_buffers_[signatures_.input_attn_mask.value()],
          signatures_.input_attn_mask_data_type.value(),
          IsCalculationPrecisionF16()));
    }
    // We will not fill the last token of the current input into the
    // interpreter now. It will be stored in next_input_token_id_ and used in
    // the next prefill or decode.
    int start_step = current_step_;
    std::vector<int> tokens_to_lookup;
    // TODO(b/425396146): Add the unit tests for checking the prefill length.
    int input_idx = 0;
    int prefill_length = ids.size();
    if (prefill_length > 1) {
      // If the prefill length is larger than 1, we will not use the last token
      // of the current input. Last token will be used as input in the next
      // prefill.
      prefill_length = ids.size() - 1;
    }
    for (int i = 0; i < prefill_length; input_idx++, current_step_++) {
      if (next_input_token_id_ != -1) {
        // Use next_input_token_id_ if it is valid.
        // Currently we use -1 to indicate that next_input_token_id_ is
        // invalid.
        tokens_to_lookup.push_back(next_input_token_id_);
        // next_input_token_id_ should only be used once at the beginning of
        // the loop.
        next_input_token_id_ = -1;
      } else {
        tokens_to_lookup.push_back(ids[input_idx]);
        // Only increase i if we used the token inside ids.
        i++;
      }
      prefill_input_pos_ptr[input_idx] = current_step_;
    }
    if (!signatures_.input_tokens.empty()) {
      auto& prefill_input_buffer =
          prefill_input_buffers_[signatures_.input_tokens];
      LITERT_ASSIGN_OR_RETURN_ABSL(auto prefill_input_size,
                                   prefill_input_buffer.PackedSize());
      LITERT_ASSIGN_OR_RETURN_ABSL(
          auto prefill_input_lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              prefill_input_buffer, TensorBuffer::LockMode::kWrite));
      int32_t* prefill_input_ptr =
          static_cast<int32_t*>(prefill_input_lock_and_addr.second);
      memset(prefill_input_ptr, 0, prefill_input_size);
      memcpy(prefill_input_ptr, tokens_to_lookup.data(),
             tokens_to_lookup.size() * sizeof(int32_t));
    } else {
      // If input_tokens is empty, we must have input_embeddings. There is no
      // need to create input_embeddings_ptr because TensorBuffer locking and
      // filling is handled by the embedding lookup.
      TensorBuffer* prefill_input_embeddings_buffer =
          &(prefill_input_buffers_[signatures_.input_embeddings.value()]);
      RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          tokens_to_lookup, prefill_input_embeddings_buffer, 0));

      // We may have per layer embedding as well.
      if (signatures_.input_per_layer_embeddings.has_value()) {
        TensorBuffer* prefill_input_per_layer_embeddings_buffer =
            &(prefill_input_buffers_[signatures_.input_per_layer_embeddings
                                         .value()]);
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            tokens_to_lookup, prefill_input_per_layer_embeddings_buffer, 0));
      }
    }
    if (has_input_attn_mask) {
      RETURN_IF_ERROR(FillAttentionMask(
          prefill_input_buffers_[signatures_.input_attn_mask.value()],
          start_step,
          /*steps=*/current_step_ - start_step,
          signatures_.input_attn_mask_data_type.value()));
    }
  }
  next_input_token_id_ = ids[ids.size() - 1];

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  for (const auto& [input_name, input_buffer] : prefill_input_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    prefill_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    prefill_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  for (const auto& [output_name, output_buffer] : prefill_output_buffers_) {
    auto duplicated_output_buffer = output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    prefill_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    auto duplicated_output_buffer = output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    prefill_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }

  auto res = compiled_model_.Run(prefill_signature, prefill_input_buffers,
                                 prefill_output_buffers);
  RET_CHECK(res) << "Failed to run compiled model." << res.Error().Message();
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::Decode(
    ::litert::TensorBuffer& output_tokens) {
  ASSIGN_OR_RETURN(decoded_logits_, DecodeLogits(ExecutorInputs()));
  LITERT_ASSIGN_OR_RETURN_ABSL(auto size, decoded_logits_.PackedSize());
  if (decoded_logits_vector_.empty()) {
    decoded_logits_vector_ = std::vector<float>(size / sizeof(float));
  }
  RETURN_IF_ERROR(SampleLogits(decoded_logits_, output_tokens));

  // Read the first output token for the next input token id.
  bool reset_output_token = false;
  {
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto lock_and_addr, ::litert::TensorBufferScopedLock::Create(
                                output_tokens, TensorBuffer::LockMode::kRead));
    auto output_tokens_ptr = static_cast<int32_t*>(lock_and_addr.second);
    reset_output_token = output_tokens_ptr[0] < 0;
    next_input_token_id_ = reset_output_token ? 0 : output_tokens_ptr[0];
  }

  // If the first output token is invalid, reset it to 0 to avoid crash.
  if (reset_output_token) {
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto lock_and_addr, ::litert::TensorBufferScopedLock::Create(
                                output_tokens, TensorBuffer::LockMode::kWrite));
    ABSL_LOG(WARNING) << "Invalid decode and sample result. The sampled token "
                         "is casted to 0 to avoid crash.";
    auto output_tokens_ptr = static_cast<int32_t*>(lock_and_addr.second);
    output_tokens_ptr[0] = 0;
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::Decode(
    const ExecutorInputs& inputs, ::litert::TensorBuffer& output_logits) {
  int id = next_input_token_id_;

  if (inputs.GetTextDataPtr().ok()) {
    auto input_tensor_size = (*inputs.GetTextTokenIdsPtr())->PackedSize();
    if (input_tensor_size && *input_tensor_size != 0) {
      // Input token ids provided, so use it regardless of whether next input
      // token id is set. Only accept batch size 1 and a single token for now.
      RET_CHECK_EQ(*input_tensor_size, 1 * sizeof(int32_t));
      LITERT_ASSIGN_OR_RETURN_ABSL(
          auto ids,
          ReferTensorBufferAsSpan<int32_t>(*(*inputs.GetTextTokenIdsPtr())));
      id = ids[0];
    }
  }
  if (id == -1) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  // Invalidate the previous next_input_token_id_, regardless of whether it is
  // used.
  next_input_token_id_ = -1;

  {
    // Fill the input buffers with scoped locks.
    if (!signatures_.input_tokens.empty()) {
      auto& decode_input_buffer =
          decode_input_buffers_[signatures_.input_tokens];
      auto decode_input_lock_and_addr =
          ::litert::TensorBufferScopedLock::Create(
              decode_input_buffer, TensorBuffer::LockMode::kWrite);
      RET_CHECK(decode_input_lock_and_addr)
          << "Failed to lock decode input buffer.";
      int32_t* decode_input_ptr =
          static_cast<int32_t*>(decode_input_lock_and_addr->second);
      decode_input_ptr[0] = id;
    } else {
      if (!signatures_.input_embeddings.has_value()) {
        return absl::InvalidArgumentError(
            "Input tokens or embeddings must be provided.");
      }
      auto& decode_input_embeddings_buffer =
          decode_input_buffers_[signatures_.input_embeddings.value()];
      RETURN_IF_ERROR(
          embedding_lookup_->LookupDecode(id, &decode_input_embeddings_buffer));

      if (signatures_.input_per_layer_embeddings.has_value()) {
        auto& decode_input_per_layer_embeddings_buffer =
            decode_input_buffers_[signatures_.input_per_layer_embeddings
                                      .value()];
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupDecode(
            id, &decode_input_per_layer_embeddings_buffer));
      }
    }
    auto& decode_input_pos_buffer =
        decode_input_buffers_[signatures_.input_positions];
    auto decode_input_pos_lock_and_addr =
        ::litert::TensorBufferScopedLock::Create(
            decode_input_pos_buffer, TensorBuffer::LockMode::kWrite);
    RET_CHECK(decode_input_pos_lock_and_addr)
        << "Failed to lock decode input position buffer.";
    auto* decode_input_pos_ptr =
        static_cast<int32_t*>(decode_input_pos_lock_and_addr->second);
    bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
    if (has_input_attn_mask) {
      RET_CHECK(signatures_.input_attn_mask_data_type.has_value())
          << "Attention mask data type is not provided.";
      RETURN_IF_ERROR(InitializeAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask.value()],
          signatures_.input_attn_mask_data_type.value(),
          IsCalculationPrecisionF16()));
      RETURN_IF_ERROR(FillAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask.value()],
          current_step_, /*steps=*/1,
          signatures_.input_attn_mask_data_type.value()));
    }
    decode_input_pos_ptr[0] = current_step_;
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    decode_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    decode_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  for (const auto& [output_name, output_buffer] : decode_output_buffers_) {
    auto duplicated_output_buffer = output_name == signatures_.output_logits
                                        ? output_logits.Duplicate()
                                        : output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    decode_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    auto duplicated_output_buffer = output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    decode_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }

  auto res = compiled_model_.Run(kDecodeSignatureRunner, decode_input_buffers,
                                 decode_output_buffers);
  RET_CHECK(res) << "Failed to run compiled model: " << res.Error().Message();
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);

  ++current_step_;
  return absl::OkStatus();
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtCompiledModelExecutor::DecodeLogits(const ExecutorInputs& inputs) {
  int id = next_input_token_id_;

  if (inputs.GetTextDataPtr().ok()) {
    auto input_tensor_size = (*inputs.GetTextTokenIdsPtr())->PackedSize();
    if (input_tensor_size && *input_tensor_size != 0) {
      // Input token ids provided, so use it regardless of whether next input
      // token id is set. Only accept batch size 1 and a single token for now.
      RET_CHECK_EQ(*input_tensor_size, 1 * sizeof(int32_t));
      LITERT_ASSIGN_OR_RETURN_ABSL(
          auto ids,
          ReferTensorBufferAsSpan<int32_t>(*(*inputs.GetTextTokenIdsPtr())));
      id = ids[0];
    }
  }
  if (id == -1) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  // Invalidate the previous next_input_token_id_, regardless of whether it is
  // used.
  next_input_token_id_ = -1;

  {
    // Fill the input buffers with scoped locks.
    if (!signatures_.input_tokens.empty()) {
      auto& decode_input_buffer =
          decode_input_buffers_[signatures_.input_tokens];
      auto decode_input_lock_and_addr =
          ::litert::TensorBufferScopedLock::Create(
              decode_input_buffer, TensorBuffer::LockMode::kWrite);
      RET_CHECK(decode_input_lock_and_addr)
          << "Failed to lock decode input buffer.";
      int32_t* decode_input_ptr =
          static_cast<int32_t*>(decode_input_lock_and_addr->second);
      decode_input_ptr[0] = id;
    } else {
      if (!signatures_.input_embeddings.has_value()) {
        return absl::InvalidArgumentError(
            "Input tokens or embeddings must be provided.");
      }
      auto& decode_embeddings_buffer =
          decode_input_buffers_[signatures_.input_embeddings.value()];
      RETURN_IF_ERROR(
          embedding_lookup_->LookupDecode(id, &decode_embeddings_buffer));

      if (signatures_.input_per_layer_embeddings.has_value()) {
        auto& decode_input_per_layer_embeddings_buffer =
            decode_input_buffers_[signatures_.input_per_layer_embeddings
                                      .value()];
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupDecode(
            id, &decode_input_per_layer_embeddings_buffer));
      }
    }

    auto& decode_input_pos_buffer =
        decode_input_buffers_[signatures_.input_positions];
    auto decode_input_pos_lock_and_addr =
        ::litert::TensorBufferScopedLock::Create(
            decode_input_pos_buffer, TensorBuffer::LockMode::kWrite);
    RET_CHECK(decode_input_pos_lock_and_addr)
        << "Failed to lock decode input position buffer.";
    auto* decode_input_pos_ptr =
        static_cast<int32_t*>(decode_input_pos_lock_and_addr->second);
    bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
    if (has_input_attn_mask) {
      RET_CHECK(signatures_.input_attn_mask_data_type.has_value())
          << "Attention mask data type is not provided.";
      RETURN_IF_ERROR(InitializeAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask.value()],
          signatures_.input_attn_mask_data_type.value(),
          IsCalculationPrecisionF16()));
      RETURN_IF_ERROR(FillAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask.value()],
          current_step_, /*steps=*/1,
          signatures_.input_attn_mask_data_type.value()));
    }
    decode_input_pos_ptr[0] = current_step_;
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    decode_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    decode_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  for (const auto& [output_name, output_buffer] : decode_output_buffers_) {
    auto duplicated_output_buffer = output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    decode_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    auto duplicated_output_buffer = output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    decode_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }

  auto res = compiled_model_.Run(kDecodeSignatureRunner, decode_input_buffers,
                                 decode_output_buffers);
  RET_CHECK(res) << "Failed to run compiled model: " << res.Error().Message();
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);

  ++current_step_;
  auto output_logits =
      decode_output_buffers[signatures_.output_logits].Duplicate();
  if (!output_logits.HasValue()) {
    return absl::InternalError(output_logits.Error().Message());
  }
  return std::move(*output_logits);
}

absl::Status LlmLiteRtCompiledModelExecutor::SampleLogits(
    const TensorBuffer& logits, TensorBuffer& ids_tensor) {
  ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());

  if (sampler_ == nullptr) {
    LITERT_ASSIGN_OR_RETURN_ABSL(auto env_options, env_.GetOptions());
    Backend sampler_backend = Backend::CPU;
    if (env_options.GetOption(kLiteRtEnvOptionTagOpenClDeviceId)) {
      sampler_backend = Backend::GPU;
    }
    LITERT_ASSIGN_OR_RETURN_ABSL(const auto decoded_logits_tensor_type,
                                 logits.TensorType());
    proto::SamplerParameters sampler_params;
    sampler_params.set_type(proto::SamplerParameters::TOP_P);
    sampler_params.set_k(1);
    sampler_params.set_p(0.0f);
    sampler_params.set_temperature(1.0f);
    sampler_params.set_seed(0);
    ASSIGN_OR_RETURN(
        sampler_,
        CreateSampler(
            sampler_backend,
            /*batch_size=*/decoded_logits_tensor_type.Layout().Dimensions()[0],
            std::move(sampler_params), env_.Get(), vocab_size,
            logits_data_type_));
  }

  RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
      logits, ids_tensor, /*scores_tensor=*/nullptr));
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::Reset() {
  current_step_ = 0;
  next_input_token_id_ = -1;
  processed_tokens_.clear();
  sampler_.reset();
  return absl::OkStatus();
}

absl::StatusOr<int> LlmLiteRtCompiledModelExecutor::GetVocabSize() {
  if (!decode_output_buffers_.contains(signatures_.output_logits)) {
    return absl::NotFoundError("Output logits info not found.");
  }

  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto logits_tensor_type,
      decode_output_buffers_[signatures_.output_logits].TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  return logits_tensor_type.Layout().Dimensions()[2];
}

// static
// Creates a LlmLiteRtCompiledModelExecutor from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutor>>
LlmLiteRtCompiledModelExecutor::Create(LlmExecutorSettings executor_settings,
                                       ModelResources& resources) {
  ASSIGN_OR_RETURN(auto litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  // For the LlmLiteRtCompiledModelExecutor, ML_DRIFT backend is used by
  // default.
  // TODO(b/405424188): - Add support for NPU backends.
  auto compilation_options = ::litert::Options::Create();
  std::string weight_cache_path = executor_settings.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  // TODO(b/433590109): Some GPUs do not support FP16, so we need to check the
  // capabilities of the GPU and set the activation data type accordingly.
  if (executor_settings.GetActivationDataType().has_value()) {
    activation_data_type = executor_settings.GetActivationDataType().value();
  }
  const Backend backend = executor_settings.GetBackend();
  switch (backend) {
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN_ABSL(GpuOptions gpu_compilation_options,
                                   GpuOptions::Create());
      gpu_compilation_options.EnableConstantTensorSharing(true);
      gpu_compilation_options.EnableInfiniteFloatCapping(true);
      gpu_compilation_options.EnableAllowSrcQuantizedFcConvOps(true);
      if (activation_data_type == ActivationDataType::FLOAT32) {
        gpu_compilation_options.SetDelegatePrecision(
            LiteRtDelegatePrecision::kLiteRtDelegatePrecisionFp32);
      } else {
        gpu_compilation_options.SetDelegatePrecision(
            LiteRtDelegatePrecision::kLiteRtDelegatePrecisionFp16);
      }
      gpu_compilation_options.SetPreferTextureWeights(true);
      if (weight_cache_path != ":nocache") {
        ASSIGN_OR_RETURN(auto model_path,
                         executor_settings.GetModelAssets().GetPath());
        if (weight_cache_path.empty()) {
          weight_cache_path = Dirname(model_path);
        }
        gpu_compilation_options.SetSerializationDir(weight_cache_path.c_str());
        absl::string_view model_name = Basename(model_path);
        gpu_compilation_options.SetModelCacheKey(model_name.data());
        gpu_compilation_options.SetSerializeProgramCache(true);
        gpu_compilation_options.SetSerializeExternalTensors(true);
      }
      gpu_compilation_options.EnableNoImmutableExternalTensorsMode(true);
      // This option prevents KVCache handling from being affected by
      // NoExternalTensorsMode.
      gpu_compilation_options.AddExternalTensorPattern("kv_cache_");
      gpu_compilation_options.AddExternalTensorPattern("logits");
      compilation_options->AddOpaqueOptions(std::move(gpu_compilation_options));
      compilation_options->SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
      break;
    }
    case Backend::CPU: {
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      Expected<CpuOptions> cpu_compilation_options = CpuOptions::Create();
      const uint32_t num_threads =
          executor_settings.GetBackendConfig<CpuConfig>()->number_of_threads;
      cpu_compilation_options->SetNumThreads(num_threads);
      if (weight_cache_path != ":nocache") {
        ASSIGN_OR_RETURN(auto model_path,
                         executor_settings.GetModelAssets().GetPath());
        if (weight_cache_path.empty()) {
          weight_cache_path = absl::StrCat(model_path, ".xnnpack_cache");
        } else {
          ASSIGN_OR_RETURN(weight_cache_path,
                           JoinPath(weight_cache_path, Basename(model_path)));
        }
        cpu_compilation_options->SetXNNPackWeightCachePath(
            weight_cache_path.c_str());
      }
      LITERT_ASSIGN_OR_RETURN_ABSL(RuntimeOptions runtime_options,
                                   RuntimeOptions::Create());
      runtime_options.SetShloCompositeInlining(true);
      compilation_options->AddOpaqueOptions(std::move(runtime_options));
      compilation_options->AddOpaqueOptions(
          std::move(*cpu_compilation_options));
      compilation_options->SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported backend: ", executor_settings.GetBackend()));
  }

  auto lrt_env = ::litert::Environment::Create({});
  if (!lrt_env) {
    return absl::InternalError(absl::StrCat(
        "Failed to create litert environment: ", lrt_env.Error().Message()));
  }

  if (!litert_model || !*litert_model) {
    return absl::InternalError("Failed to build LiteRt model");
  }
  auto compiled_model = ::litert::CompiledModel::Create(
      *lrt_env, *litert_model, std::move(*compilation_options));
  if (!compiled_model) {
    return absl::InternalError(absl::StrCat("Failed to create compiled model: ",
                                            compiled_model.Error().Message()));
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_kv_cache_buffers;

  absl::string_view prefill_signature_key = "";
  for (int i = 0; i < litert_model->GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN_ABSL(auto sig, litert_model->GetSignature(i));
    absl::string_view key = sig.Key();
    if (absl::StartsWith(key, kPrefillSignatureRunner)) {
      prefill_signature_key = key;
      break;
    }
  }
  auto prefill_signature = litert_model->FindSignature(prefill_signature_key);
  RET_CHECK(prefill_signature) << "Prefill signature not found.";
  std::string kv_cache_k_root_name;
  std::string kv_cache_v_root_name;
  RETURN_IF_ERROR(GetCacheRootNames(prefill_signature->InputNames(),
                                    kv_cache_k_root_name,
                                    kv_cache_v_root_name));
  auto decode_signature = litert_model->FindSignature(kDecodeSignatureRunner);
  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature->InputNames(),
                                             decode_signature->OutputNames()));

  for (auto input_name : prefill_signature->InputNames()) {
    // Skip creating buffers for the input tokens, positions and attn mask. Move
    // into prefill function to create them based on the ids size.
    if (input_name == signatures.input_tokens ||
        input_name == signatures.input_positions ||
        input_name == signatures.input_attn_mask) {
      continue;
    }
    auto input_buffer =
        compiled_model->CreateInputBuffer(prefill_signature_key, input_name);
    if (!input_buffer) {
      return absl::InternalError(
          absl::StrCat("Failed to create prefill input buffer for ", "'",
                       input_name, "' : ", input_buffer.Error().Message()));
    }
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      if (backend == Backend::CPU) {
        auto output_buffer = input_buffer->Duplicate();
        RET_CHECK(output_buffer) << "Failed to duplicate input buffer.";
        output_kv_cache_buffers[input_name] = std::move(*output_buffer);
      }
      input_kv_cache_buffers[input_name] = std::move(*input_buffer);
    } else {
      prefill_input_buffers[input_name] = std::move(*input_buffer);
    }
  }
  for (auto output_name : prefill_signature->OutputNames()) {
    auto output_buffer =
        compiled_model->CreateOutputBuffer(prefill_signature_key, output_name);
    if (!output_buffer) {
      return absl::InternalError(
          absl::StrCat("Failed to create prefill output buffer for ", "'",
                       output_name, "' : ", output_buffer.Error().Message()));
    }
    if (absl::StartsWith(output_name, kv_cache_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_v_root_name)) {
      if (backend == Backend::GPU) {
        output_kv_cache_buffers[output_name] = std::move(*output_buffer);
      }
      // For CPU, we will use single buffer for kv cache input and output to
      // improve performance and memory usage.
    } else {
      prefill_output_buffers[output_name] = std::move(*output_buffer);
    }
  }

  for (auto input_name : decode_signature->InputNames()) {
    if (!absl::StartsWith(input_name, kv_cache_k_root_name) &&
        !absl::StartsWith(input_name, kv_cache_v_root_name)) {
      auto input_buffer =
          compiled_model->CreateInputBuffer(kDecodeSignatureRunner, input_name);
      if (!input_buffer) {
        return absl::InternalError(
            absl::StrCat("Failed to create decode input buffer for ", "'",
                         input_name, "' : ", input_buffer.Error().Message()));
      }
      decode_input_buffers[input_name] = std::move(*input_buffer);
    }
  }
  for (auto output_name : decode_signature->OutputNames()) {
    if (!absl::StartsWith(output_name, kv_cache_k_root_name) &&
        !absl::StartsWith(output_name, kv_cache_v_root_name)) {
      auto output_buffer = compiled_model->CreateOutputBuffer(
          kDecodeSignatureRunner, output_name);
      if (!output_buffer) {
        return absl::InternalError(
            absl::StrCat("Failed to create decode output buffer for ", "'",
                         output_name, "' : ", output_buffer.Error().Message()));
      }
      decode_output_buffers[output_name] = std::move(*output_buffer);
    }
  }

  auto output_logits_buffer =
      decode_output_buffers[signatures.output_logits].Duplicate();
  RET_CHECK(output_logits_buffer)
      << "Failed to duplicate output logits buffer.";
  LITERT_ASSIGN_OR_RETURN_ABSL(auto output_logits_buffer_tensor_type,
                               output_logits_buffer->TensorType());
  RET_CHECK(output_logits_buffer_tensor_type.Layout().Dimensions().size() == 3)
      << "Output logits must be (batch, seq, vocab)";
  RET_CHECK(output_logits_buffer_tensor_type.Layout().Dimensions()[0] == 1)
      << "Only support batch size 1 for now.";
  int batch_size = output_logits_buffer_tensor_type.Layout().Dimensions()[0];

  ASSIGN_OR_RETURN(auto prefill_runner_set,
                   GetPrefillRunnerSetFromModel(
                       *litert_model, kPrefillSignatureRunner,
                       /*input_positions_name=*/signatures.input_positions));
  RET_CHECK(!prefill_runner_set.empty()) << "No prefill runner available.";

  // Create embedding lookups from the resources.
  std::unique_ptr<EmbeddingLookupText> embedding_lookup;
  auto embedder_model = resources.GetTFLiteModel(ModelType::kTfLiteEmbedder);
  if (embedder_model.ok()) {
    ASSIGN_OR_RETURN(embedding_lookup,  // NOLINT
                     EmbeddingLookupText::Create(*embedder_model));
  }

  // Create per layer embedding lookups from the resources.
  std::unique_ptr<EmbeddingLookupText> per_layer_embedding_lookup;
  auto per_layer_embedder_model =
      resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder);
  if (per_layer_embedder_model.ok()) {
    ASSIGN_OR_RETURN(per_layer_embedding_lookup,  // NOLINT
                     EmbeddingLookupText::Create(*per_layer_embedder_model));
  }

  return absl::WrapUnique(new LlmLiteRtCompiledModelExecutor(
      std::move(executor_settings), std::move(*lrt_env), litert_model,
      std::move(*compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(input_kv_cache_buffers),
      std::move(output_kv_cache_buffers), std::move(prefill_runner_set),
      signatures, batch_size, weight_cache_path, std::move(embedding_lookup),
      std::move(per_layer_embedding_lookup), activation_data_type));
}

}  // namespace litert::lm
