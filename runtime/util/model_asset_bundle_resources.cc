/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "runtime/util/model_asset_bundle_resources.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/external_file_handler.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/zip_utils.h"

namespace litert::lm {

class ScopedFileModelAssetBundleResources : public ModelAssetBundleResources {
 public:
  ScopedFileModelAssetBundleResources(
      const std::string& tag,
      std::shared_ptr<ScopedFile> model_asset_bundle_file)
      : ModelAssetBundleResources(tag),
        model_asset_bundle_file_(std::move(model_asset_bundle_file)) {}

  ~ScopedFileModelAssetBundleResources() override = default;

  absl::StatusOr<absl::string_view> GetFileData() override {
    if (!mapped_model_asset_bundle_file_) {
      ASSIGN_OR_RETURN(
          mapped_model_asset_bundle_file_,
          MemoryMappedFile::Create(model_asset_bundle_file_->file()));
    }

    return absl::string_view(
        reinterpret_cast<const char*>(mapped_model_asset_bundle_file_->data()),
        mapped_model_asset_bundle_file_->length());
  }

 private:
  // The model asset bundle file to be memory mapped.
  const std::shared_ptr<ScopedFile> model_asset_bundle_file_;

  // This owns the memory backing `files_`.
  std::unique_ptr<MemoryMappedFile> mapped_model_asset_bundle_file_;
};

class ExternalFileModelAssetBundleResources : public ModelAssetBundleResources {
 public:
  ExternalFileModelAssetBundleResources(
      const std::string& tag,
      std::unique_ptr<proto::ExternalFile> model_asset_bundle_file)
      : ModelAssetBundleResources(tag),
        model_asset_bundle_file_(std::move(model_asset_bundle_file)) {}

  ~ExternalFileModelAssetBundleResources() override = default;

  absl::StatusOr<absl::string_view> GetFileData() override {
    if (!model_asset_bundle_file_handler_) {
      ASSIGN_OR_RETURN(model_asset_bundle_file_handler_,
                       ExternalFileHandler::CreateFromExternalFile(
                           model_asset_bundle_file_.get()));
    }

    return model_asset_bundle_file_handler_->GetFileContent();
  }

 private:
  // The model asset bundle file.
  const std::unique_ptr<proto::ExternalFile> model_asset_bundle_file_;

  // The ExternalFileHandler for the model asset bundle. This owns the memory
  // backing `files_`.
  std::unique_ptr<ExternalFileHandler> model_asset_bundle_file_handler_;
};

/* static */
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(
    const std::string& tag,
    std::unique_ptr<proto::ExternalFile> model_asset_bundle_file) {
  if (model_asset_bundle_file == nullptr) {
    return absl::InvalidArgumentError(
        "The model asset bundle file proto cannot be nullptr.");
  }
  auto model_bundle_resources =
      std::make_unique<ExternalFileModelAssetBundleResources>(
          tag, std::move(model_asset_bundle_file));
  RETURN_IF_ERROR(model_bundle_resources->ExtractFiles());
  return model_bundle_resources;
}

/* static */
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(
    const std::string& tag,
    std::shared_ptr<ScopedFile> model_asset_bundle_file){
  if (!model_asset_bundle_file->IsValid()) {
    return absl::InvalidArgumentError(
        "The model asset bundle file is not valid.");
  }
  auto model_bundle_resources =
      std::make_unique<ScopedFileModelAssetBundleResources>(
          tag, std::move(model_asset_bundle_file));
  RETURN_IF_ERROR(model_bundle_resources->ExtractFiles());
  return model_bundle_resources;
}

ModelAssetBundleResources::ModelAssetBundleResources(std::string tag)
    : tag_(std::move(tag)) {}

absl::StatusOr<absl::string_view> ModelAssetBundleResources::GetFile(
    const std::string& filename) const {
  auto it = files_.find(filename);
  if (it != files_.end()) {
    return it->second;
  }

  auto files = ListFiles();
  std::string all_files = absl::StrJoin(files, ", ");

  return absl::NotFoundError(
      absl::StrFormat("No file with name: %s. All files in the model asset "
                      "bundle are: %s.",
                      filename, all_files));
}

std::vector<std::string> ModelAssetBundleResources::ListFiles() const {
  std::vector<std::string> file_names;
  file_names.reserve(files_.size());
  for (const auto& [file_name, _] : files_) {
    file_names.push_back(file_name);
  }
  return file_names;
}

absl::Status ModelAssetBundleResources::ExtractFiles() {
  ASSIGN_OR_RETURN(absl::string_view data, GetFileData());
  return ExtractFilesfromZipFile(data.data(), data.size(), &files_);
}

}  // namespace litert::lm
