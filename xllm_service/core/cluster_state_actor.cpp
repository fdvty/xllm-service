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

#include <glog/logging.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace xllm_service {

ClusterStateActor::~ClusterStateActor() { stop(); }

bool ClusterStateActor::start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return true;  // already running
  }
  // Publish an empty snapshot up front so readers never see null after start().
  republish();

  bthread::ExecutionQueueOptions options;
  if (bthread::execution_queue_start(&queue_id_, &options, &consume, this) !=
      0) {
    running_.store(false);
    LOG(ERROR) << "ClusterStateActor: failed to start execution queue";
    return false;
  }
  return true;
}

void ClusterStateActor::stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;  // not running
  }
  bthread::execution_queue_stop(queue_id_);
  bthread::execution_queue_join(queue_id_);
}

bool ClusterStateActor::offer(Event event) {
  if (!running_.load()) {
    return false;
  }
  if (bthread::execution_queue_execute(queue_id_, std::move(event)) != 0) {
    LOG(WARNING) << "ClusterStateActor: failed to enqueue event";
    return false;
  }
  return true;
}

int ClusterStateActor::consume(void* meta,
                               bthread::TaskIterator<Event>& iter) {
  auto* self = static_cast<ClusterStateActor*>(meta);
  if (iter.is_queue_stopped()) {
    return 0;
  }
  self->apply_batch(iter);
  return 0;
}

void ClusterStateActor::apply_batch(bthread::TaskIterator<Event>& iter) {
  bool changed = false;
  for (; iter; ++iter) {
    if (apply_one(*iter)) {
      changed = true;
    }
    stats_.events_applied.fetch_add(1, std::memory_order_relaxed);
  }
  // Coalesce: one snapshot per drained batch, not per event. Under load the
  // queue hands us many events at once and readers only ever need the latest.
  if (changed) {
    republish();
  }
}

bool ClusterStateActor::apply_one(const Event& event) {
  switch (event.kind) {
    case Event::Kind::kInstancePut: {
      if (!event.metainfo) {
        return false;
      }
      auto& entry = model_.instances[event.instance_name];
      // Incarnation change means the instance restarted: drop inherited
      // metrics so the new process does not present the old one's load.
      if (entry.meta.incarnation_id != event.metainfo->incarnation_id) {
        entry.load = LoadMetrics();
        entry.latency = LatencyMetrics();
      }
      entry.meta = *event.metainfo;
      return true;
    }
    case Event::Kind::kInstanceDelete: {
      auto it = model_.instances.find(event.instance_name);
      if (it == model_.instances.end()) {
        return false;
      }
      // Only honor the delete if it targets the current (or unknown)
      // incarnation; a delete for an already-replaced incarnation is stale.
      if (!event.incarnation_id.empty() &&
          it->second.meta.incarnation_id != event.incarnation_id) {
        stats_.stale_dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      model_.instances.erase(it);
      return true;
    }
    case Event::Kind::kHeartbeat: {
      auto it = model_.instances.find(event.instance_name);
      if (it == model_.instances.end()) {
        // Heartbeat before metainfo PUT: ignore; the etcd watch is the
        // authoritative source of instance existence.
        return false;
      }
      // Drop metrics from a stale incarnation.
      if (!event.incarnation_id.empty() &&
          it->second.meta.incarnation_id != event.incarnation_id) {
        stats_.stale_dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      bool changed = false;
      if (event.has_load) {
        it->second.load = event.load;
        changed = true;
      }
      if (event.has_latency) {
        it->second.latency = event.latency;
        changed = true;
      }
      return changed;
    }
    case Event::Kind::kCacheDelta:
    case Event::Kind::kCacheSnapshot:
      // Cache index construction lands in R3 (policy plug-in phase). For now
      // these events are accepted and ignored so the ingest wiring can be
      // built and tested end to end without the radix rewrite.
      return false;
    case Event::Kind::kServiceChanged:
      // Peer service topology is observability-only in peer mode.
      return false;
  }
  return false;
}

void ClusterStateActor::republish() {
  std::vector<InstanceView> views;
  views.reserve(model_.instances.size());
  for (const auto& [name, entry] : model_.instances) {
    InstanceView v;
    v.name = entry.meta.name;
    v.incarnation_id = entry.meta.incarnation_id;
    v.rpc_address = entry.meta.rpc_address;
    v.zmq_endpoint = entry.meta.zmq_endpoint;
    v.type = entry.meta.type;
    v.current_type = entry.meta.current_type;
    v.runtime_state = entry.meta.runtime_state;
    v.load = entry.load;
    v.latency = entry.latency;
    v.block_size = entry.meta.block_size;
    v.xxh3_128bits_seed = entry.meta.xxh3_128bits_seed;
    views.push_back(std::move(v));
  }
  std::sort(views.begin(), views.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.name < rhs.name;
  });
  uint64_t version = next_version_.fetch_add(1, std::memory_order_relaxed);
  auto snapshot = std::make_shared<const ClusterSnapshot>(
      version, std::move(views), /*cache_index=*/nullptr);
  published_.store(std::move(snapshot));
  stats_.snapshots_published.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace xllm_service
