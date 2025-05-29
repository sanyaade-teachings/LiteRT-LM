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

#include <atomic>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "runtime/framework/threadpool.h"
#include "runtime/framework/worker_thread.h"

namespace litert::lm {
namespace {

class WorkerThreadStdThread : public WorkerThread {
 public:
  WorkerThreadStdThread(ThreadPool& pool, const std::string& name_prefix);

  ~WorkerThreadStdThread() override;

  void Join() override;

 private:
  static void* ThreadBody(void* arg);

  std::thread thread_;
  // Track if this thread is joined.
  std::atomic<bool> joined_;
};

WorkerThreadStdThread::WorkerThreadStdThread(ThreadPool& pool,
                                             const std::string& name_prefix)
    : WorkerThread(pool, name_prefix), joined_(false) {
  thread_ = std::thread(ThreadBody, this);
}

WorkerThreadStdThread::~WorkerThreadStdThread() {
  if (joined_) {
    return;
  }

  ABSL_LOG(WARNING)
      << "WorkerThread for pool " << name_prefix_
      << " destroyed without Join(). Potential resource leak or race.";
}

void WorkerThreadStdThread::Join() {
  if (joined_) {
    return;
  }

  thread_.join();
  joined_ = true;
}

void* WorkerThreadStdThread::ThreadBody(void* arg) {
  auto thread = reinterpret_cast<WorkerThreadStdThread*>(arg);
  thread->RunWorker();
  return nullptr;
}

}  // namespace

std::unique_ptr<WorkerThread> WorkerThread::Create(
    ThreadPool& pool, const std::string& name_prefix) {
  return std::make_unique<WorkerThreadStdThread>(pool, name_prefix);
}

}  // namespace litert::lm
