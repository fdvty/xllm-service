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

#include "core/snapshot_holder.h"

#include <gtest/gtest.h>

#include <memory>
#include <thread>
#include <vector>

namespace xllm_service {

namespace {
struct Value {
  int x;
};
}  // namespace

TEST(SnapshotHolderTest, LoadReturnsNullWhenEmpty) {
  SnapshotHolder<Value> holder;
  EXPECT_EQ(holder.load(), nullptr);
}

TEST(SnapshotHolderTest, StoreThenLoad) {
  SnapshotHolder<Value> holder;
  holder.store(std::make_shared<const Value>(Value{42}));
  auto v = holder.load();
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->x, 42);
}

TEST(SnapshotHolderTest, HeldSnapshotSurvivesReplacement) {
  SnapshotHolder<Value> holder;
  holder.store(std::make_shared<const Value>(Value{1}));
  auto first = holder.load();
  holder.store(std::make_shared<const Value>(Value{2}));
  // The pointer taken earlier must still observe the old immutable value.
  EXPECT_EQ(first->x, 1);
  EXPECT_EQ(holder.load()->x, 2);
}

TEST(SnapshotHolderTest, ConcurrentReadersDuringWrites) {
  SnapshotHolder<Value> holder;
  holder.store(std::make_shared<const Value>(Value{0}));
  std::atomic<bool> stop{false};
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&holder, &stop]() {
      while (!stop.load()) {
        auto v = holder.load();
        // Value must always be readable and internally consistent.
        ASSERT_NE(v, nullptr);
        volatile int sink = v->x;
        (void)sink;
      }
    });
  }
  for (int w = 1; w <= 5000; ++w) {
    holder.store(std::make_shared<const Value>(Value{w}));
  }
  stop.store(true);
  for (auto& r : readers) {
    r.join();
  }
  EXPECT_EQ(holder.load()->x, 5000);
}

}  // namespace xllm_service
