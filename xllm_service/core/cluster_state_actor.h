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

#include <bthread/execution_queue.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/types.h"
#include "core/cluster_snapshot.h"
#include "core/event.h"
#include "core/snapshot_holder.h"

namespace xllm_service {

// ClusterStateActor is the single writer of cluster state.
//
// Every ingest source (RPC heartbeat handler, etcd watcher, ZMQ subscriber)
// calls offer(Event) from its own thread and returns immediately. A single
// bthread::ExecutionQueue consumer drains events in batches and applies them to
// the private ClusterModel -- with no locks, because only this one consumer
// ever touches the model. After each batch it constructs a fresh immutable
// ClusterSnapshot and publishes it atomically through SnapshotHolder, where the
// lock-free router reads it.
//
// This is the structural fix for the old design's cross-module locks and the
// InstanceMgr -> Scheduler back-pointer: sources no longer share mutable state,
// and stale/incarnation/seq logic all lives in one serial place.
class ClusterStateActor {
 public:
  struct Stats {
    // Monotonic count of events applied.
    std::atomic<uint64_t> events_applied{0};
    // Count of events dropped as stale (older incarnation).
    std::atomic<uint64_t> stale_dropped{0};
    // Snapshots published (== batches with an effective change).
    std::atomic<uint64_t> snapshots_published{0};
  };

  ClusterStateActor() = default;
  ~ClusterStateActor();

  ClusterStateActor(const ClusterStateActor&) = delete;
  ClusterStateActor& operator=(const ClusterStateActor&) = delete;

  // Start the consumer queue. Returns false if the queue failed to start.
  bool start();

  // Stop and join the consumer. Safe to call multiple times.
  void stop();

  // Thread-safe: enqueue an event from any ingest thread. Returns false if the
  // queue is not running.
  bool offer(Event event);

  // Lock-free read of the current published snapshot. Never null after start().
  std::shared_ptr<const ClusterSnapshot> snapshot() const {
    return published_.load();
  }

  const Stats& stats() const { return stats_; }

 private:
  // Authoritative mutable state, touched ONLY by the consumer thread.
  struct ModelEntry {
    InstanceMetaInfo meta;
    LoadMetrics load;
    LatencyMetrics latency;
  };
  struct ClusterModel {
    std::unordered_map<std::string, ModelEntry> instances;
  };

  // ExecutionQueue trampoline -> apply_batch.
  static int consume(void* meta, bthread::TaskIterator<Event>& iter);

  // Apply one batch of events, then republish once if anything changed.
  void apply_batch(bthread::TaskIterator<Event>& iter);

  // Apply a single event to model_. Returns true if it changed visible state.
  bool apply_one(const Event& event);

  // Build + atomically publish a ClusterSnapshot from the current model_.
  void republish();

  ClusterModel model_;  // consumer-thread-only, no lock needed
  SnapshotHolder<ClusterSnapshot> published_;
  std::atomic<uint64_t> next_version_{1};

  bthread::ExecutionQueueId<Event> queue_id_{0};
  std::atomic<bool> running_{false};

  Stats stats_;
};

}  // namespace xllm_service
