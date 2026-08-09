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

#include "core/inflight_table.h"

namespace xllm_service {

std::shared_ptr<InflightTable::Counter> InflightTable::get_or_create(
    const std::string& instance) {
  {
    absl::ReaderMutexLock lock(&mutex_);
    auto it = counters_.find(instance);
    if (it != counters_.end()) {
      return it->second;
    }
  }
  absl::WriterMutexLock lock(&mutex_);
  auto it = counters_.find(instance);
  if (it != counters_.end()) {
    return it->second;
  }
  auto counter = std::make_shared<Counter>();
  counters_.emplace(instance, counter);
  return counter;
}

void InflightTable::inc(const std::string& instance) {
  auto counter = get_or_create(instance);
  counter->value.fetch_add(1, std::memory_order_relaxed);
}

void InflightTable::dec(const std::string& instance) {
  std::shared_ptr<Counter> counter;
  {
    absl::ReaderMutexLock lock(&mutex_);
    auto it = counters_.find(instance);
    if (it == counters_.end()) {
      return;
    }
    counter = it->second;
  }
  // Clamp at zero: error paths (dispatch failure + later cancel) can otherwise
  // double-decrement and leak a negative count into scoring.
  int64_t prev = counter->value.load(std::memory_order_relaxed);
  while (prev > 0 && !counter->value.compare_exchange_weak(
                         prev, prev - 1, std::memory_order_relaxed)) {
  }
}

int64_t InflightTable::get(const std::string& instance) const {
  absl::ReaderMutexLock lock(&mutex_);
  auto it = counters_.find(instance);
  return it == counters_.end()
             ? 0
             : it->second->value.load(std::memory_order_relaxed);
}

int64_t InflightTable::total() const {
  absl::ReaderMutexLock lock(&mutex_);
  int64_t sum = 0;
  for (const auto& [name, counter] : counters_) {
    sum += counter->value.load(std::memory_order_relaxed);
  }
  return sum;
}

InflightView InflightTable::view() const {
  absl::ReaderMutexLock lock(&mutex_);
  std::unordered_map<std::string, int64_t> snapshot;
  snapshot.reserve(counters_.size());
  for (const auto& [name, counter] : counters_) {
    snapshot.emplace(name, counter->value.load(std::memory_order_relaxed));
  }
  return InflightView(std::move(snapshot));
}

void InflightTable::erase(const std::string& instance) {
  absl::WriterMutexLock lock(&mutex_);
  counters_.erase(instance);
}

}  // namespace xllm_service
