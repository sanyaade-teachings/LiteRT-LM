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

#include "runtime/util/scoped_file.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

void WriteFile(absl::string_view path, absl::string_view contents) {
  std::ofstream ofstr(std::string(path), std::ios::out);
  ofstr << contents;
}

TEST(ScopedFile, FailsOpeningNonExistentFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "bad.txt";
  ASSERT_FALSE(ScopedFile::Open(path.string()).ok());
}

TEST(ScopedFile, GetSize) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto file = ScopedFile::Open(path.string());
  ASSERT_OK(file);
  EXPECT_TRUE(file->IsValid());

  EXPECT_THAT(file->GetSize(), IsOkAndHolds(7));
  EXPECT_THAT(ScopedFile::GetSize(file->file()), IsOkAndHolds(7));
}

TEST(ScopedFile, GetSizeOfWritableFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto file = ScopedFile::OpenWritable(path.string());
  ASSERT_OK(file);
  EXPECT_TRUE(file->IsValid());

  EXPECT_THAT(file->GetSize(), IsOkAndHolds(7));
  EXPECT_THAT(ScopedFile::GetSize(file->file()), IsOkAndHolds(7));
}

TEST(ScopedFile, MoveInvalidatesFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  absl::StatusOr<ScopedFile> file = ScopedFile::Open(path.string());
  ASSERT_OK(file);
  EXPECT_TRUE(file->IsValid());

  ScopedFile other_file = std::move(*file);
  EXPECT_TRUE(other_file.IsValid());
  EXPECT_FALSE(file->IsValid());  // NOLINT: use after move is intended to check
                                  // the state.
}

TEST(ScopedFile, GetSizeOfInvalidFile) {
  ScopedFile uninitialized_file;
  EXPECT_THAT(uninitialized_file.GetSize(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(ScopedFile::GetSize(uninitialized_file.file()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace litert::lm
