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

#include <chrono>
#include <functional>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

using namespace std::chrono_literals;

struct FakeRegistrationStore {
  RegistrationLookupResult lookup() {
    std::lock_guard<std::mutex> lock(mutex);
    ++lookup_count;
    if (!available) {
      return {RegistrationLookupStatus::UNAVAILABLE, {}, 0, "unavailable"};
    }
    if (!found) {
      return {RegistrationLookupStatus::MISSING, {}, 0, {}};
    }
    return {RegistrationLookupStatus::FOUND, value, lease_id, {}};
  }

  RegistrationCreateResult create() {
    std::lock_guard<std::mutex> lock(mutex);
    ++create_count;
    if (!available || found) {
      return {false, 0, "create failed"};
    }
    found = true;
    value = "service-a";
    lease_id = next_lease_id++;
    return {true, lease_id, {}};
  }

  std::mutex mutex;
  bool available = true;
  bool found = false;
  std::string value;
  int64_t lease_id = 0;
  int64_t next_lease_id = 10;
  int lookup_count = 0;
  int create_count = 0;
};

ServiceRegistrationManager::Options fast_options() {
  return {5ms, 5ms, 20ms};
}

bool wait_for(const std::function<bool()>& predicate) {
  const auto deadline = std::chrono::steady_clock::now() + 1s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

TEST(ServiceRegistrationManagerTest, RegistersMissingKeyAtStartup) {
  FakeRegistrationStore store;
  ServiceRegistrationManager manager(
      "service-a",
      [&store]() { return store.lookup(); },
      [&store]() { return store.create(); },
      fast_options());

  ASSERT_TRUE(manager.start());
  EXPECT_TRUE(manager.healthy());
  EXPECT_EQ(manager.owned_lease_id(), 10);
  {
    std::lock_guard<std::mutex> lock(store.mutex);
    EXPECT_EQ(store.create_count, 1);
  }
  manager.stop();
}

TEST(ServiceRegistrationManagerTest, RecreatesDeletedRegistration) {
  FakeRegistrationStore store;
  ServiceRegistrationManager manager(
      "service-a",
      [&store]() { return store.lookup(); },
      [&store]() { return store.create(); },
      fast_options());
  ASSERT_TRUE(manager.start());

  {
    std::lock_guard<std::mutex> lock(store.mutex);
    store.found = false;
    store.lease_id = 0;
  }
  manager.notify_registration_may_be_missing();

  ASSERT_TRUE(wait_for([&manager]() {
    return manager.healthy() && manager.owned_lease_id() == 11;
  }));
  {
    std::lock_guard<std::mutex> lock(store.mutex);
    EXPECT_EQ(store.create_count, 2);
  }
  manager.stop();
}

TEST(ServiceRegistrationManagerTest, RecoversAfterStoreUnavailability) {
  FakeRegistrationStore store;
  ServiceRegistrationManager manager(
      "service-a",
      [&store]() { return store.lookup(); },
      [&store]() { return store.create(); },
      fast_options());
  ASSERT_TRUE(manager.start());

  {
    std::lock_guard<std::mutex> lock(store.mutex);
    store.available = false;
  }
  manager.notify_registration_may_be_missing();
  ASSERT_TRUE(wait_for([&manager]() { return !manager.healthy(); }));

  {
    std::lock_guard<std::mutex> lock(store.mutex);
    store.available = true;
    store.found = false;
    store.lease_id = 0;
  }
  manager.notify_registration_may_be_missing();
  ASSERT_TRUE(wait_for([&manager]() {
    return manager.healthy() && manager.owned_lease_id() == 11;
  }));
  manager.stop();
}

TEST(ServiceRegistrationManagerTest, DoesNotOverwriteDifferentLease) {
  FakeRegistrationStore store;
  ServiceRegistrationManager manager(
      "service-a",
      [&store]() { return store.lookup(); },
      [&store]() { return store.create(); },
      fast_options());
  ASSERT_TRUE(manager.start());

  {
    std::lock_guard<std::mutex> lock(store.mutex);
    store.lease_id = 999;
  }
  manager.notify_registration_may_be_missing();
  ASSERT_TRUE(wait_for([&manager]() { return !manager.healthy(); }));
  std::this_thread::sleep_for(30ms);
  {
    std::lock_guard<std::mutex> lock(store.mutex);
    EXPECT_EQ(store.create_count, 1);
  }
  manager.stop();
}

}  // namespace
}  // namespace xllm_service
