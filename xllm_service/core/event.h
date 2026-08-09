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

#include <cstdint>
#include <memory>
#include <string>

#include "common/types.h"

namespace xllm_service {

// A control-plane event. Every state mutation in the peer service -- instance
// registration/removal, heartbeat load/latency, KV cache delta/snapshot -- is
// funneled through this single tagged type into the ClusterStateActor queue and
// applied by exactly one thread. This is what removes the cross-module locks
// and the InstanceMgr -> Scheduler back-pointer from the old design.
//
// The type is intentionally a flat struct rather than std::variant: it must be
// trivially movable through bthread::ExecutionQueue, and different kinds share
// most fields (instance name + incarnation). Unused fields stay default.
struct Event {
  enum class Kind : int8_t {
    // Instance metainfo appeared/updated in etcd (PUT on XLLM:<TYPE>:<name>).
    kInstancePut = 0,
    // Instance metainfo lease expired / deleted (DELETE) -> authoritative down.
    kInstanceDelete = 1,
    // Heartbeat carrying load/latency metrics (and, in ZMQ-off fallback, a
    // cache delta -- carried separately as kCacheDelta by the ingest layer).
    kHeartbeat = 2,
    // Incremental KV cache event (heartbeat fallback or ZMQ subscriber).
    kCacheDelta = 3,
    // Full KV cache snapshot for periodic index correction (ZMQ).
    kCacheSnapshot = 4,
    // A peer service appeared/disappeared (observability only in peer mode).
    kServiceChanged = 5,
  };

  Kind kind = Kind::kHeartbeat;

  // Identity. incarnation_id lets the actor drop stale events from a restarted
  // instance without any cross-thread coordination.
  std::string instance_name;
  std::string incarnation_id;

  // kInstancePut payload (full metainfo). Held by shared_ptr to keep the event
  // cheap to move through the queue.
  std::shared_ptr<InstanceMetaInfo> metainfo;

  // kHeartbeat payload.
  LoadMetrics load;
  LatencyMetrics latency;
  bool has_load = false;
  bool has_latency = false;

  // kCacheDelta / kCacheSnapshot payload. Serialized protobuf bytes of the
  // KvCacheEvent, decoded inside the actor to keep ingest threads cheap. seq_no
  // drives gap detection; is_snapshot distinguishes replace-vs-merge semantics.
  std::shared_ptr<std::string> cache_event_bytes;
  uint64_t seq_no = 0;
  bool is_snapshot = false;

  static Event instance_put(std::shared_ptr<InstanceMetaInfo> info) {
    Event e;
    e.kind = Kind::kInstancePut;
    e.instance_name = info->name;
    e.incarnation_id = info->incarnation_id;
    e.metainfo = std::move(info);
    return e;
  }

  static Event instance_delete(std::string name, std::string incarnation) {
    Event e;
    e.kind = Kind::kInstanceDelete;
    e.instance_name = std::move(name);
    e.incarnation_id = std::move(incarnation);
    return e;
  }
};

}  // namespace xllm_service
