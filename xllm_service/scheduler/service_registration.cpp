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

#include "scheduler/service_registration.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <glog/logging.h>

#include "common/metrics.h"

namespace xllm_service {

ServiceRegistrationManager::ServiceRegistrationManager(
    std::string expected_value,
    Lookup lookup,
    Create create)
    : ServiceRegistrationManager(std::move(expected_value),
                                 std::move(lookup),
                                 std::move(create),
                                 Options{}) {}

ServiceRegistrationManager::ServiceRegistrationManager(
    std::string expected_value,
    Lookup lookup,
    Create create,
    Options options)
    : expected_value_(std::move(expected_value)),
      lookup_(std::move(lookup)),
      create_(std::move(create)),
      options_(options) {
  if (expected_value_.empty() || !lookup_ || !create_) {
    throw std::invalid_argument(
        "service registration requires value, lookup and create callbacks");
  }
  if (options_.healthy_interval.count() <= 0 ||
      options_.initial_retry_delay.count() <= 0 ||
      options_.max_retry_delay < options_.initial_retry_delay) {
    throw std::invalid_argument("invalid service registration intervals");
  }
}

ServiceRegistrationManager::~ServiceRegistrationManager() { stop(); }

bool ServiceRegistrationManager::start() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) {
      return healthy();
    }
  }
  if (!reconcile_once()) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = false;
    wake_requested_ = false;
    running_ = true;
  }
  worker_ = std::thread(&ServiceRegistrationManager::run, this);
  return true;
}

void ServiceRegistrationManager::notify_registration_may_be_missing() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_ || stopping_) {
    return;
  }
  wake_requested_ = true;
  condition_.notify_one();
}

void ServiceRegistrationManager::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    stopping_ = true;
    condition_.notify_one();
  }
  if (worker_.joinable()) {
    worker_.join();
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
  }
  set_healthy(false);
}

bool ServiceRegistrationManager::reconcile_once() {
  COUNTER_INC(xservice_registration_reconcile_total);
  const RegistrationLookupResult lookup = lookup_();
  if (lookup.status == RegistrationLookupStatus::UNAVAILABLE) {
    set_healthy(false);
    COUNTER_INC(xservice_registration_failure_total);
    LOG(WARNING) << "Cannot verify current xllm_service registration: "
                 << lookup.error;
    return false;
  }

  const int64_t owned_lease = owned_lease_id();
  if (lookup.status == RegistrationLookupStatus::FOUND) {
    if (owned_lease > 0 && lookup.value == expected_value_ &&
        lookup.lease_id == owned_lease) {
      set_healthy(true);
      return true;
    }
    set_healthy(false);
    COUNTER_INC(xservice_registration_failure_total);
    LOG(ERROR) << "Service registration is owned by a different lease, value: "
               << lookup.value << ", lease: " << lookup.lease_id
               << ", expected lease: " << owned_lease;
    return false;
  }

  COUNTER_INC(xservice_registration_attempt_total);
  const RegistrationCreateResult created = create_();
  if (!created.created || created.lease_id <= 0) {
    set_healthy(false);
    COUNTER_INC(xservice_registration_failure_total);
    LOG(WARNING) << "Failed to create current xllm_service registration: "
                 << created.error;
    return false;
  }

  const bool recovered = ever_registered_;
  ever_registered_ = true;
  owned_lease_id_.store(created.lease_id, std::memory_order_release);
  set_healthy(true);
  COUNTER_INC(xservice_registration_success_total);
  if (recovered) {
    COUNTER_INC(xservice_registration_recovery_total);
  }
  LOG(INFO) << (recovered ? "Recovered" : "Created")
            << " current xllm_service registration, lease: "
            << created.lease_id;
  return true;
}

void ServiceRegistrationManager::run() {
  size_t consecutive_failures = 0;
  std::chrono::milliseconds delay = options_.healthy_interval;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait_for(lock, delay, [this]() {
        return stopping_ || wake_requested_;
      });
      if (stopping_) {
        return;
      }
      wake_requested_ = false;
    }

    if (reconcile_once()) {
      consecutive_failures = 0;
      delay = options_.healthy_interval;
      continue;
    }
    ++consecutive_failures;
    delay = retry_delay(consecutive_failures);
  }
}

void ServiceRegistrationManager::set_healthy(bool healthy) {
  healthy_.store(healthy, std::memory_order_release);
  GAUGE_SET(xservice_registration_healthy, healthy ? 1.0 : 0.0);
}

std::chrono::milliseconds ServiceRegistrationManager::retry_delay(
    size_t consecutive_failures) const {
  int64_t delay = options_.initial_retry_delay.count();
  for (size_t attempt = 1; attempt < consecutive_failures; ++attempt) {
    delay = std::min(delay * 2, options_.max_retry_delay.count());
  }
  const size_t jitter_percent =
      (std::hash<std::string>{}(expected_value_) + consecutive_failures) % 21;
  delay += delay * static_cast<int64_t>(jitter_percent) / 100;
  return std::chrono::milliseconds(delay);
}

}  // namespace xllm_service
