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

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace xllm_service {

TEST(InflightTableTest, IncDecBasic) {
  InflightTable table;
  EXPECT_EQ(table.get("a"), 0);
  table.inc("a");
  table.inc("a");
  EXPECT_EQ(table.get("a"), 2);
  table.dec("a");
  EXPECT_EQ(table.get("a"), 1);
  EXPECT_EQ(table.total(), 1);
}

TEST(InflightTableTest, DecNeverGoesNegative) {
  InflightTable table;
  table.dec("missing");  // unknown key: no-op
  EXPECT_EQ(table.get("missing"), 0);
  table.inc("a");
  table.dec("a");
  table.dec("a");  // double-decrement on error path
  EXPECT_EQ(table.get("a"), 0);
}

TEST(InflightTableTest, ViewIsImmutableSnapshot) {
  InflightTable table;
  table.inc("a");
  table.inc("b");
  table.inc("b");
  InflightView view = table.view();
  EXPECT_EQ(view.get("a"), 1);
  EXPECT_EQ(view.get("b"), 2);
  EXPECT_EQ(view.get("c"), 0);
  // Mutating the table afterwards must not change the taken view.
  table.inc("a");
  EXPECT_EQ(view.get("a"), 1);
  EXPECT_EQ(table.get("a"), 2);
}

TEST(InflightTableTest, EraseDropsCounter) {
  InflightTable table;
  table.inc("a");
  table.erase("a");
  EXPECT_EQ(table.get("a"), 0);
  EXPECT_EQ(table.total(), 0);
}

TEST(InflightTableTest, ConcurrentIncDecBalancesToZero) {
  InflightTable table;
  constexpr int kThreads = 8;
  constexpr int kPerThread = 10000;
  std::vector<std::thread> workers;
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&table]() {
      for (int i = 0; i < kPerThread; ++i) {
        table.inc("hot");
        table.dec("hot");
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  EXPECT_EQ(table.get("hot"), 0);
}

}  // namespace xllm_service
