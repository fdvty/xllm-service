/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <atomic>
#include <memory>

namespace xllm_service {

// Single-writer / multi-reader holder for an immutable snapshot published by
// the control plane (ClusterStateActor) and read lock-free by the data plane
// (Router).
//
// Concurrency model:
//   - Exactly one writer (the actor thread) calls publish().
//   - Any number of reader threads call load() concurrently.
//   - A reader that holds the returned shared_ptr<const T> sees an immutable
//     value for the whole lifetime of that pointer; a concurrent publish()
//     only affects subsequent load() calls.
//
// Implementation note: C++20 offers std::atomic<std::shared_ptr<T>>, but this
// project is built with C++17 (see top-level CMakeLists.txt), so we use the
// C++17 free-function atomics on shared_ptr. They are deprecated in C++20 but
// correct and available in C++17. This type is the ONLY place that dependency
// lives, so migrating to std::atomic<std::shared_ptr<T>> (or
// folly::atomic_shared_ptr) later is a single-file change with no impact on
// callers.
template <typename T>
class SnapshotHolder {
 public:
  SnapshotHolder() = default;
  explicit SnapshotHolder(std::shared_ptr<const T> initial) {
    store(std::move(initial));
  }

  // Reader side: lock-free acquire of the current snapshot.
  std::shared_ptr<const T> load() const {
    return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
  }

  // Writer side: atomically replace the published snapshot. Intended to be
  // called only by the single control-plane thread.
  void store(std::shared_ptr<const T> next) {
    std::atomic_store_explicit(
        &snapshot_, std::move(next), std::memory_order_release);
  }

 private:
  std::shared_ptr<const T> snapshot_;
};

}  // namespace xllm_service
