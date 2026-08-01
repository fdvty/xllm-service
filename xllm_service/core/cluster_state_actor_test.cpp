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

#include "core/cluster_state_actor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <memory>
#include <thread>

#include "common/types.h"
#include "core/event.h"

namespace xllm_service {

namespace {

// Events apply asynchronously on a bthread queue, so tests wait for a predicate
// on the published snapshot rather than asserting synchronously.
bool wait_for(ClusterStateActor& actor,
              const std::function<bool(const ClusterSnapshot&)>& pred,
              int timeout_ms = 2000) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    auto snap = actor.snapshot();
    if (snap && pred(*snap)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

std::shared_ptr<InstanceMetaInfo> make_meta(const std::string& name,
                                            const std::string& incarnation,
                                            InstanceType type) {
  auto m = std::make_shared<InstanceMetaInfo>(name, name + ":addr", type);
  m->incarnation_id = incarnation;
  return m;
}

}  // namespace

TEST(ClusterStateActorTest, SnapshotNeverNullAfterStart) {
  ClusterStateActor actor;
  ASSERT_TRUE(actor.start());
  ASSERT_NE(actor.snapshot(), nullptr);
  EXPECT_EQ(actor.snapshot()->instance_count(), 0u);
  actor.stop();
}

TEST(ClusterStateActorTest, InstancePutAppearsInSnapshot) {
  ClusterStateActor actor;
  ASSERT_TRUE(actor.start());
  actor.offer(Event::instance_put(make_meta("i0", "inc1", InstanceType::MIX)));
  EXPECT_TRUE(wait_for(actor, [](const ClusterSnapshot& s) {
    return s.find("i0") != nullptr;
  }));
  auto* v = actor.snapshot()->find("i0");
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->incarnation_id, "inc1");
  EXPECT_EQ(v->rpc_address, "i0:addr");
  actor.stop();
}

TEST(ClusterStateActorTest, InstanceDeleteRemovesIt) {
  ClusterStateActor actor;
  ASSERT_TRUE(actor.start());
  actor.offer(Event::instance_put(make_meta("i0", "inc1", InstanceType::MIX)));
  ASSERT_TRUE(wait_for(
      actor, [](const ClusterSnapshot& s) { return s.find("i0"); }));
  actor.offer(Event::instance_delete("i0", "inc1"));
  EXPECT_TRUE(wait_for(actor, [](const ClusterSnapshot& s) {
    return s.find("i0") == nullptr;
  }));
  actor.stop();
}

TEST(ClusterStateActorTest, HeartbeatUpdatesLoadMetrics) {
  ClusterStateActor actor;
  ASSERT_TRUE(actor.start());
  actor.offer(Event::instance_put(make_meta("i0", "inc1", InstanceType::MIX)));
  ASSERT_TRUE(wait_for(
      actor, [](const ClusterSnapshot& s) { return s.find("i0"); }));

  Event hb;
  hb.kind = Event::Kind::kHeartbeat;
  hb.instance_name = "i0";
  hb.incarnation_id = "inc1";
  hb.has_load = true;
  hb.load = LoadMetrics(/*waiting=*/7, /*usage=*/0.5f);
  actor.offer(std::move(hb));

  EXPECT_TRUE(wait_for(actor, [](const ClusterSnapshot& s) {
    auto* v = s.find("i0");
    return v && v->load.waiting_requests_num == 7;
  }));
  actor.stop();
}

TEST(ClusterStateActorTest, StaleIncarnationHeartbeatDropped) {
  ClusterStateActor actor;
  ASSERT_TRUE(actor.start());
  actor.offer(Event::instance_put(make_meta("i0", "inc2", InstanceType::MIX)));
  ASSERT_TRUE(wait_for(
      actor, [](const ClusterSnapshot& s) { return s.find("i0"); }));

  // Heartbeat from an older incarnation must be ignored.
  Event hb;
  hb.kind = Event::Kind::kHeartbeat;
  hb.instance_name = "i0";
  hb.incarnation_id = "inc1";  // stale
  hb.has_load = true;
  hb.load = LoadMetrics(/*waiting=*/99, /*usage=*/0.9f);
  actor.offer(std::move(hb));

  // Give the actor time to process, then assert the stale value did NOT land.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto* v = actor.snapshot()->find("i0");
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->load.waiting_requests_num, 0u);
  EXPECT_GE(actor.stats().stale_dropped.load(), 1u);
  actor.stop();
}

TEST(ClusterStateActorTest, IncarnationChangeClearsInheritedMetrics) {
  ClusterStateActor actor;
  ASSERT_TRUE(actor.start());
  actor.offer(Event::instance_put(make_meta("i0", "inc1", InstanceType::MIX)));
  ASSERT_TRUE(wait_for(
      actor, [](const ClusterSnapshot& s) { return s.find("i0"); }));
  Event hb;
  hb.kind = Event::Kind::kHeartbeat;
  hb.instance_name = "i0";
  hb.incarnation_id = "inc1";
  hb.has_load = true;
  hb.load = LoadMetrics(5, 0.5f);
  actor.offer(std::move(hb));
  ASSERT_TRUE(wait_for(actor, [](const ClusterSnapshot& s) {
    auto* v = s.find("i0");
    return v && v->load.waiting_requests_num == 5;
  }));

  // Restart: same name, new incarnation -> inherited load must reset to 0.
  actor.offer(Event::instance_put(make_meta("i0", "inc2", InstanceType::MIX)));
  EXPECT_TRUE(wait_for(actor, [](const ClusterSnapshot& s) {
    auto* v = s.find("i0");
    return v && v->incarnation_id == "inc2" &&
           v->load.waiting_requests_num == 0;
  }));
  actor.stop();
}

TEST(ClusterStateActorTest, VersionIncreasesOnChange) {
  ClusterStateActor actor;
  ASSERT_TRUE(actor.start());
  uint64_t v0 = actor.snapshot()->version();
  actor.offer(Event::instance_put(make_meta("i0", "inc1", InstanceType::MIX)));
  ASSERT_TRUE(wait_for(
      actor, [](const ClusterSnapshot& s) { return s.find("i0"); }));
  EXPECT_GT(actor.snapshot()->version(), v0);
  actor.stop();
}

TEST(ClusterStateActorTest, SnapshotOrderIsStableByInstanceName) {
  ClusterStateActor actor;
  ASSERT_TRUE(actor.start());
  actor.offer(Event::instance_put(make_meta("i1", "inc1", InstanceType::MIX)));
  actor.offer(Event::instance_put(make_meta("i0", "inc1", InstanceType::MIX)));
  ASSERT_TRUE(wait_for(
      actor, [](const ClusterSnapshot& s) { return s.instance_count() == 2; }));
  auto snapshot = actor.snapshot();
  ASSERT_EQ(snapshot->instance_count(), 2u);
  EXPECT_EQ(snapshot->instances()[0].name, "i0");
  EXPECT_EQ(snapshot->instances()[1].name, "i1");
  actor.stop();
}

TEST(InstanceViewTest, DefaultInstanceSupportsAggregatedExecution) {
  InstanceView view;
  view.type = InstanceType::DEFAULT;
  EXPECT_TRUE(view.is_prefill_capable());
  EXPECT_TRUE(view.is_decode_capable());
}

}  // namespace xllm_service
