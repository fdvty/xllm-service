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

#include <absl/synchronization/mutex.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace xllm_service {

// A cheap read-only copy of the in-flight counts at one instant, handed to
// scorers so they can correct the heartbeat blind spot with service-local
// predictive load. Immutable once returned.
class InflightView {
 public:
  explicit InflightView(std::unordered_map<std::string, int64_t> counts)
      : counts_(std::move(counts)) {}
  InflightView() = default;

  int64_t get(const std::string& instance) const {
    auto it = counts_.find(instance);
    return it == counts_.end() ? 0 : it->second;
  }

  const std::unordered_map<std::string, int64_t>& counts() const {
    return counts_;
  }

 private:
  std::unordered_map<std::string, int64_t> counts_;
};

// Per-instance in-flight request accounting, local to this service, zero extra
// network traffic. dispatch() does +1, the finished/failed/cancel callbacks do
// -1. Deliberately kept OUT of the ClusterStateActor queue: it changes far more
// frequently than cluster topology and must not be serialized behind control
// events. Correctness only needs monotonic per-key counters, so a sharded
// atomic map is enough.
//
// Structural changes (first sight of an instance, or dropping a dead one) take
// a short write lock; the hot inc/dec path takes a read lock and mutates an
// atomic, so concurrent dispatches never block each other.
class InflightTable {
 public:
  InflightTable() = default;

  // dispatch: +1 for the target instance, creating the counter on first use.
  void inc(const std::string& instance);

  // finished / dispatch-failed / cancelled / stream-error / disconnect: -1.
  // Never drops below zero (guards against double-decrement on error paths).
  void dec(const std::string& instance);

  // Current count for one instance (0 if unknown).
  int64_t get(const std::string& instance) const;

  // Sum across all instances (for the aggregate bvar / metrics).
  int64_t total() const;

  // Immutable snapshot for scorers.
  InflightView view() const;

  // Drop an instance that has left the cluster, so its counter does not leak.
  void erase(const std::string& instance);

 private:
  struct Counter {
    std::atomic<int64_t> value{0};
  };

  mutable absl::Mutex mutex_;
  // shared_ptr<Counter> so a reader holding a read lock can keep touching the
  // atomic even if a concurrent erase() removes the map entry.
  std::unordered_map<std::string, std::shared_ptr<Counter>> counters_
      ABSL_GUARDED_BY(mutex_);

  std::shared_ptr<Counter> get_or_create(const std::string& instance);
};

}  // namespace xllm_service
