/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "etcd_client.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>

#include "common/types.h"
#include "common/utils.h"

namespace xllm_service {
namespace {
constexpr auto kEtcdRequestTimeout = std::chrono::seconds(2);
}  // namespace

std::string get_event_key(const etcd::Event& event) {
  if (event.event_type() == etcd::Event::EventType::DELETE_ &&
      event.has_prev_kv()) {
    return event.prev_kv().key();
  }
  if (event.has_kv()) {
    return event.kv().key();
  }
  return "";
}

std::string get_event_value(const etcd::Event& event) {
  if (event.event_type() == etcd::Event::EventType::DELETE_ &&
      event.has_prev_kv()) {
    return event.prev_kv().as_string();
  }
  if (event.has_kv()) {
    return event.kv().as_string();
  }
  return "";
}

std::string get_event_key_suffix(const etcd::Event& event,
                                 uint64_t prefix_len) {
  const auto key = get_event_key(event);
  if (key.size() < prefix_len) {
    return "";
  }
  return key.substr(prefix_len);
}

EtcdClient::EtcdClient(const std::string& etcd_addr,
                       const std::string& etcd_namespace)
    : client_(etcd_addr),
      etcd_addr_(etcd_addr),
      etcd_namespace_prefix_(utils::normalize_etcd_namespace(etcd_namespace)) {
  client_.set_grpc_timeout(kEtcdRequestTimeout);
  LOG(INFO) << "EtcdClient init put start!";
  auto response = client_.put(namespaced_key("XLLM_PING"), "PING");
  LOG(INFO) << "EtcdClient init put end!";
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

EtcdClient::EtcdClient(const std::string& etcd_addr,
                       const std::string& username,
                       const std::string& password,
                       const std::string& etcd_namespace)
    : client_(etcd_addr, username, password),
      etcd_addr_(etcd_addr),
      etcd_namespace_prefix_(utils::normalize_etcd_namespace(etcd_namespace)) {
  client_.set_grpc_timeout(kEtcdRequestTimeout);
  LOG(INFO) << "EtcdClient init put start!";
  auto response = client_.put(namespaced_key("XLLM_PING"), "PING");
  LOG(INFO) << "EtcdClient init put end!";
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

std::string EtcdClient::namespaced_key(const std::string& logical_key) const {
  return utils::build_etcd_key_with_namespace(etcd_namespace_prefix_,
                                              logical_key);
}

EtcdClient::~EtcdClient() { stop_watch(); }

bool EtcdClient::set(const std::string& key, const std::string& value) {
  auto response = client_.put(namespaced_key(key), value);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd set " << key << " failed: " << response.error_message();
    return false;
  }

  return true;
}

bool EtcdClient::set(const std::string& key,
                     const std::string& value,
                     const int ttl) {
  return create_with_lease(key, value, ttl).created;
}

EtcdKeyLookupResult EtcdClient::lookup(const std::string& key) {
  try {
    const etcd::Response response = client_.get(namespaced_key(key));
    if (response.is_ok()) {
      return {EtcdKeyLookupStatus::FOUND,
              response.value().as_string(),
              response.value().lease(),
              {}};
    }
    if (response.error_code() == etcd::ERROR_KEY_NOT_FOUND) {
      return {EtcdKeyLookupStatus::NOT_FOUND, {}, 0, {}};
    }
    return {EtcdKeyLookupStatus::UNAVAILABLE,
            {},
            0,
            response.error_message()};
  } catch (const std::exception& error) {
    return {EtcdKeyLookupStatus::UNAVAILABLE, {}, 0, error.what()};
  }
}

EtcdLeaseCreateResult EtcdClient::create_with_lease(
    const std::string& key,
    const std::string& value,
    int ttl) {
  std::shared_ptr<etcd::KeepAlive> keep_alive;
  try {
    keep_alive = std::make_shared<etcd::KeepAlive>(client_, ttl);
    if (keep_alive->Lease() <= 0) {
      keep_alive->Cancel();
      return {false, 0, "etcd returned an invalid lease"};
    }
    etcdv3::Transaction transaction;
    transaction.add_compare_create(namespaced_key(key), 0);
    transaction.add_success_put(
        namespaced_key(key), value, keep_alive->Lease());
    const etcd::Response response = client_.txn(transaction);
    if (!response.is_ok()) {
      keep_alive->Cancel();
      return {false, 0, response.error_message()};
    }

    std::shared_ptr<etcd::KeepAlive> previous;
    {
      std::lock_guard<std::mutex> lock(keep_alives_mutex_);
      auto existing = keep_alives_.find(key);
      if (existing != keep_alives_.end()) {
        previous = std::move(existing->second);
      }
      keep_alives_[key] = keep_alive;
    }
    if (previous != nullptr) {
      previous->Cancel();
    }
    return {true, keep_alive->Lease(), {}};
  } catch (const std::exception& error) {
    if (keep_alive != nullptr) {
      keep_alive->Cancel();
    }
    return {false, 0, error.what()};
  }
}

bool EtcdClient::set(const std::string& key_prefix,
                     const XXH3KeyCacheMap& values) {
  bool rt = true;
  const std::string namespaced_prefix = namespaced_key(key_prefix);
  for (const auto& iter : values) {
    const std::string full_key = namespaced_prefix + iter.first.to_string();
    if (iter.second.empty()) {
      rt = rt && client_.rm(full_key).is_ok();
    } else {
      rt =
          rt &&
          client_.put(full_key, iter.second.serialize_to_json().dump()).is_ok();
    }
  }
  return rt;
}

bool EtcdClient::rm(const std::string& key) {
  auto response = client_.rm(namespaced_key(key));
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd rm " << key << " failed: " << response.error_message();
    return false;
  }

  return true;
}

bool EtcdClient::rm(const std::string& key_prefix,
                    const std::unordered_set<std::string>& keys) {
  etcdv3::Transaction transaction;
  transaction.add_compare_version(namespaced_key(ETCD_MASTER_SERVICE_KEY),
                                  etcdv3::CompareResult::GREATER,
                                  -1);
  const std::string namespaced_prefix = namespaced_key(key_prefix);
  for (const auto& iter : keys) {
    transaction.add_success_delete(namespaced_prefix + iter);
  }
  return client_.txn(transaction).is_ok();
}

bool EtcdClient::get(const std::string& key, std::string* value) {
  auto response = client_.get(namespaced_key(key));
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key << " failed: " << response.error_message();
    return false;
  }
  if (value) {
    *value = response.value().as_string();
  }
  return true;
}

bool EtcdClient::get_prefix(const std::string& key_prefix,
                            XXH3KeyCacheMap* values) {
  const std::string full_prefix = namespaced_key(key_prefix);
  auto response = client_.ls(full_prefix);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key_prefix
               << " failed: " << response.error_message();
    return false;
  }

  const size_t prefix_len = full_prefix.size();
  for (int i = 0; i < response.keys().size(); i++) {
    XXH3Key key(response.key(i).substr(prefix_len).c_str());
    auto json_str = response.value(i).as_string();

    CacheLocations value;
    if (!value.parse_from_json(json_str)) {
      LOG(ERROR) << "Parse json fail: " << json_str;
      continue;
    }

    values->insert_or_assign(std::move(key), std::move(value));
  }
  return true;
}

bool EtcdClient::get_prefix(
    const std::string& key_prefix,
    std::unordered_map<std::string, std::string>* values) {
  const std::string full_prefix = namespaced_key(key_prefix);
  auto response = client_.ls(full_prefix);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key_prefix
               << " failed: " << response.error_message();
    return false;
  }

  const size_t prefix_len = full_prefix.size();
  for (int i = 0; i < response.keys().size(); i++) {
    auto key_str = response.key(i).substr(prefix_len);
    auto str = response.value(i).as_string();

    values->insert_or_assign(std::move(key_str), std::move(str));
  }
  return true;
}

void EtcdClient::add_watch(const std::string& key_prefix,
                           Callback callback,
                           bool recursive) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  if (watchers_.find(key_prefix) != watchers_.end()) {
    watchers_[key_prefix].watcher->Cancel();
  }
  auto namespaced_prefix = namespaced_key(key_prefix);
  auto watcher = std::make_unique<etcd::Watcher>(
      client_,
      namespaced_prefix,
      [callback, namespaced_prefix](etcd::Response response) {
        callback(response, uint64_t(namespaced_prefix.size()));
      },
      recursive);

  watchers_[key_prefix] = {std::move(watcher), callback};
}

void EtcdClient::remove_watch(const std::string& key_prefix) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  auto it = watchers_.find(key_prefix);
  if (it != watchers_.end()) {
    it->second.watcher->Cancel();
    watchers_.erase(it);
  }
}

void EtcdClient::stop_watch() {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  for (auto& pair : watchers_) {
    pair.second.watcher->Cancel();
  }

  watchers_.clear();
}

}  // namespace xllm_service
