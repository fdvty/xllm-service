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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace xllm_service {

// Immutable view of one instance as seen by the data plane. This is a plain
// value copied out of the control-plane model at publish time; it is never
// mutated after a ClusterSnapshot is constructed.
struct InstanceView {
  std::string name;
  std::string incarnation_id;
  std::string rpc_address;
  std::string zmq_endpoint;
  InstanceType type = InstanceType::DEFAULT;
  // Exact role for a MIX instance under SLO-aware scheduling.
  InstanceType current_type = InstanceType::PREFILL;
  InstanceRuntimeState runtime_state = InstanceRuntimeState::ACTIVE;

  LoadMetrics load;
  LatencyMetrics latency;

  int32_t block_size = 0;
  uint32_t xxh3_128bits_seed = 0;

  bool is_prefill_capable() const {
    return type == InstanceType::DEFAULT || type == InstanceType::PREFILL ||
           type == InstanceType::MIX;
  }
  bool is_decode_capable() const {
    return type == InstanceType::DEFAULT || type == InstanceType::DECODE ||
           type == InstanceType::MIX;
  }
};

// Opaque, immutable prefix-cache index. The concrete radix implementation is
// introduced in R3 (policy plug-in phase); until then the snapshot simply
// carries a null handle and cache-aware scorers fall back gracefully. Keeping
// it behind a forward declaration lets the snapshot type stabilize now without
// pulling the cache rewrite forward.
class CacheIndex;

// Immutable, versioned snapshot of the whole cluster. Constructed by the
// control-plane actor, published through SnapshotHolder, and read lock-free by
// the router. Never mutated after construction.
class ClusterSnapshot {
 public:
  ClusterSnapshot() = default;
  ClusterSnapshot(uint64_t version,
                  std::vector<InstanceView> instances,
                  std::shared_ptr<const CacheIndex> cache_index)
      : version_(version),
        instances_(std::move(instances)),
        cache_index_(std::move(cache_index)) {
    build_index();
  }

  uint64_t version() const { return version_; }

  const std::vector<InstanceView>& instances() const { return instances_; }
  size_t instance_count() const { return instances_.size(); }

  // Returns nullptr if the instance is not present in this snapshot.
  const InstanceView* find(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &instances_[it->second];
  }

  // May be null before the cache index is wired up (pre-R3) or when no
  // instance has published cache events yet. Callers must null-check.
  const CacheIndex* cache() const { return cache_index_.get(); }

 private:
  void build_index() {
    by_name_.reserve(instances_.size());
    for (size_t i = 0; i < instances_.size(); ++i) {
      by_name_.emplace(instances_[i].name, i);
    }
  }

  uint64_t version_ = 0;
  std::vector<InstanceView> instances_;
  std::unordered_map<std::string, size_t> by_name_;
  std::shared_ptr<const CacheIndex> cache_index_;
};

}  // namespace xllm_service
