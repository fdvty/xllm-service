/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace xllm_service {

enum class RegistrationLookupStatus {
  FOUND = 0,
  MISSING = 1,
  UNAVAILABLE = 2,
};

struct RegistrationLookupResult {
  RegistrationLookupStatus status = RegistrationLookupStatus::UNAVAILABLE;
  std::string value;
  int64_t lease_id = 0;
  std::string error;
};

struct RegistrationCreateResult {
  bool created = false;
  int64_t lease_id = 0;
  std::string error;
};

class ServiceRegistrationManager final {
 public:
  using Lookup = std::function<RegistrationLookupResult()>;
  using Create = std::function<RegistrationCreateResult()>;

  struct Options {
    std::chrono::milliseconds healthy_interval{1000};
    std::chrono::milliseconds initial_retry_delay{1000};
    std::chrono::milliseconds max_retry_delay{8000};
  };

  ServiceRegistrationManager(std::string expected_value,
                             Lookup lookup,
                             Create create);
  ServiceRegistrationManager(std::string expected_value,
                             Lookup lookup,
                             Create create,
                             Options options);
  ~ServiceRegistrationManager();

  bool start();
  void notify_registration_may_be_missing();
  void stop();

  bool healthy() const { return healthy_.load(std::memory_order_acquire); }
  int64_t owned_lease_id() const {
    return owned_lease_id_.load(std::memory_order_acquire);
  }

 private:
  bool reconcile_once();
  void run();
  void set_healthy(bool healthy);
  std::chrono::milliseconds retry_delay(size_t consecutive_failures) const;

  const std::string expected_value_;
  const Lookup lookup_;
  const Create create_;
  const Options options_;

  std::atomic<bool> healthy_{false};
  std::atomic<int64_t> owned_lease_id_{0};
  bool ever_registered_ = false;

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool running_ = false;
  bool stopping_ = false;
  bool wake_requested_ = false;
  std::thread worker_;
};

}  // namespace xllm_service
