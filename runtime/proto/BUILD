# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

proto_library(
    name = "engine_proto",
    srcs = ["engine.proto"],
)

cc_proto_library(
    name = "engine_cc_proto",
    deps = [":engine_proto"],
)

proto_library(
    name = "sampler_params_proto",
    srcs = ["sampler_params.proto"],
)

cc_proto_library(
    name = "sampler_params_cc_proto",
    deps = [":sampler_params_proto"],
)

py_proto_library(
    name = "sampler_params_py_pb2",
    deps = [":sampler_params_proto"],
)

proto_library(
    name = "llm_metadata_proto",
    srcs = ["llm_metadata.proto"],
    deps = [":sampler_params_proto"],
)

cc_proto_library(
    name = "llm_metadata_cc_proto",
    deps = [":llm_metadata_proto"],
)

py_proto_library(
    name = "llm_metadata_py_pb2",
    deps = [":llm_metadata_proto"],
)
