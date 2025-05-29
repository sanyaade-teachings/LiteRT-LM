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

#ifndef THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_WORKER_THREAD_H_
#define THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_WORKER_THREAD_H_

#include <memory>
#include <string>

#include "runtime/framework/threadpool.h"

namespace litert::lm {

class WorkerThread {
 public:
  // Creates and starts a thread that runs pool->RunWorker().
  static std::unique_ptr<WorkerThread> Create(ThreadPool& pool,
                                              const std::string& name_prefix);

  // REQUIRES: Join() must have been called.
  virtual ~WorkerThread() = default;

  // Joins with the running thread.
  virtual void Join() = 0;

 protected:
  WorkerThread(ThreadPool& pool, const std::string& name_prefix)
      : pool_(pool), name_prefix_(name_prefix) {}

  // For the visibility from WorkerThread subclasses.
  void RunWorker() { pool_.RunWorker(); }

  ThreadPool& pool_;
  const std::string name_prefix_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_WORKER_THREAD_H_
