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

#include "runtime/framework/threadpool.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/framework/thread_options.h"
#include "runtime/framework/worker_thread.h"

namespace litert::lm {

ThreadPool::ThreadPool(const ThreadOptions& thread_options,
                       const std::string& name_prefix, size_t num_threads)
    : thread_options_(thread_options),
      name_prefix_(name_prefix),
      num_threads_(num_threads == 0 ? 1 : num_threads) {
  ABSL_LOG(INFO) << "ThreadPool: Created with " << num_threads_ << " threads.";
}

ThreadPool::~ThreadPool() {
  ABSL_LOG(INFO) << "ThreadPool '" << name_prefix_ << "': Shutting down...";

  std::vector<std::unique_ptr<WorkerThread>> threads_to_join;
  {
    absl::MutexLock lock(&mutex_);
    stopped_ = true;
    threads_to_join.swap(threads_);
  }

  for (auto& thread_ptr : threads_to_join) {
    // Wait for each worker thread to finish.
    thread_ptr->Join();
  }

  // Log num_active_tasks_ state at shutdown. Should be 0 if all workers exited
  // cleanly.
  int final_num_active_tasks = 0;
  {
    absl::MutexLock lock(&mutex_);
    final_num_active_tasks = num_active_tasks_;
  }
  ABSL_LOG(INFO) << "ThreadPool '" << name_prefix_ << "': Shutdown complete. "
                 << final_num_active_tasks
                 << " active tasks recorded at the end (should ideally be 0).";
}

void ThreadPool::StartWorkers() {
  absl::MutexLock lock(&mutex_);
  if (!threads_.empty() || stopped_) {
    ABSL_LOG(WARNING)
        << "ThreadPool '" << name_prefix_
        << "': StartWorkers called on an already started or stopped pool.";
    return;
  }

  threads_.reserve(num_threads_);
  for (int i = 0; i < num_threads_; ++i) {
    threads_.push_back(WorkerThread::Create(*this, name_prefix_));
  }
  ABSL_LOG(INFO) << "ThreadPool '" << name_prefix_ << "': Started "
                 << num_threads_ << " workers.";
}

void ThreadPool::Schedule(absl::AnyInvocable<void() &&> callback) {
  absl::MutexLock lock(&mutex_);
  if (stopped_) {
    ABSL_LOG(WARNING) << "ThreadPool '" << name_prefix_
                      << "': Schedule called on a stopped pool. Task for pool '"
                      << name_prefix_ << "' ignored.";
    return;
  }

  tasks_.push_back(std::move(callback));
}

absl::Status ThreadPool::WaitUntilIdle(absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);
  absl::Time deadline = absl::Now() + timeout;
  // Wait until tasks_ is empty OR the deadline is reached.
  auto is_tasks_empty = [this]() {
    mutex_.AssertHeld();
    return tasks_.empty();
  };
  if (mutex_.AwaitWithDeadline(absl::Condition(&is_tasks_empty), deadline)) {
    return absl::OkStatus();
  }
  return absl::DeadlineExceededError(
      absl::StrCat("Timeout waiting for task queue to become idle in pool '",
                   name_prefix_, "'. Tasks still in queue: ", tasks_.size()));
}

absl::Status ThreadPool::WaitUntilDone(absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);
  absl::Time deadline = absl::Now() + timeout;
  // Wait until tasks_ is empty OR the deadline is reached.
  auto is_done = [this]() {
    mutex_.AssertHeld();
    return tasks_.empty() && num_active_tasks_ == 0;
  };
  if (mutex_.AwaitWithDeadline(absl::Condition(&is_done), deadline)) {
    return absl::OkStatus();
  }
  return absl::DeadlineExceededError(
      absl::StrCat("Timeout waiting for all tasks to be done in pool '",
                   name_prefix_, "'. Tasks still in queue: ", tasks_.size(),
                   ", Active tasks: ", num_active_tasks_));
}

void ThreadPool::RunWorker() {
  absl::MutexLock lock(&mutex_);
  while (true) {
    // Wait until a task is available OR the pool is stopped.
    auto is_task_available = [this]() {
      mutex_.AssertHeld();
      return !tasks_.empty() || stopped_;
    };
    mutex_.Await(absl::Condition(&is_task_available));

    if (stopped_ && tasks_.empty()) {
      return;
    }
    ABSL_CHECK(!tasks_.empty());

    auto task_to_run = std::move(tasks_.front());
    tasks_.pop_front();
    ++num_active_tasks_;

    // Release mutex before executing task.
    mutex_.Unlock();
    // Execute the task.
    std::move(task_to_run)();
    // Task finished. Re-acquire mutex and decrement active tasks.
    mutex_.Lock();

    --num_active_tasks_;
  }
}

}  // namespace litert::lm
