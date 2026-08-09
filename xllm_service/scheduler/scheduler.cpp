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

#include "scheduler/scheduler.h"

#include <brpc/http_status_code.h>

#include <utility>

#include "common/metrics.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "tokenizer/tokenizer_factory.h"

namespace {
constexpr int32_t kHeartbeatInterval = 3;  // in seconds

constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";
}  // namespace

namespace xllm_service {
namespace {

ScheduleResult make_schedule_error(ScheduleError error,
                                   const std::string& message) {
  return {error, message};
}

ScheduleError schedule_error_from_selection(RoutingSelectionError error) {
  switch (error) {
    case RoutingSelectionError::NONE:
      return ScheduleError::NONE;
    case RoutingSelectionError::LEGACY_SELECTOR_UNAVAILABLE:
    case RoutingSelectionError::LEGACY_SELECTION_FAILED:
      return ScheduleError::BACKEND_UNAVAILABLE;
    case RoutingSelectionError::EXTERNAL_DIRECTIVE_IN_LEGACY_MODE:
    case RoutingSelectionError::EXTERNAL_DIRECTIVE_IN_AGGREGATED_MODE:
    case RoutingSelectionError::MISSING_EXTERNAL_DIRECTIVE:
    case RoutingSelectionError::INVALID_EXTERNAL_DIRECTIVE:
    case RoutingSelectionError::UNSUPPORTED_EXTERNAL_DIRECTIVE_VERSION:
    case RoutingSelectionError::EXTERNAL_OWNER_MISMATCH:
      return ScheduleError::INVALID_ROUTING_DIRECTIVE;
    case RoutingSelectionError::UNKNOWN_MODE:
    case RoutingSelectionError::NULL_OUTPUT:
      return ScheduleError::INTERNAL_ERROR;
  }
  return ScheduleError::INTERNAL_ERROR;
}

}  // namespace

Scheduler::Scheduler(
    const Options& options,
    const RoutingConfiguration& routing_configuration,
    InstanceLifecycleEventDispatcher::Handler lifecycle_handler)
    : options_(options),
      routing_configuration_(routing_configuration),
      lifecycle_events_(std::vector<InstanceLifecycleEventDispatcher::Handler>{
          [this](const InstanceLifecycleEvent& event) {
            handle_instance_lifecycle_event(event);
          },
          std::move(lifecycle_handler)}) {
  GAUGE_SET(peer_service_enabled, options_.enable_peer_service() ? 1.0 : 0.0);

  tokenizer_ = TokenizerFactory::create_tokenizer(options_.tokenizer_path(),
                                                  &tokenizer_args_);
  chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);

  const std::string etcd_username =
      utils::get_optional_string_env(kEtcdUsernameEnvVar).value_or("");
  const std::string etcd_password =
      utils::get_optional_string_env(kEtcdPasswordEnvVar).value_or("");
  const bool has_etcd_auth_user = !etcd_username.empty();
  const bool has_etcd_auth_password = !etcd_password.empty();
  if (has_etcd_auth_user != has_etcd_auth_password) {
    LOG(FATAL) << "Both " << kEtcdUsernameEnvVar << " and "
               << kEtcdPasswordEnvVar << " must be set together.";
  }
  if (has_etcd_auth_user) {
    etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                                etcd_username,
                                                etcd_password,
                                                options_.etcd_namespace());
  } else {
    etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                                options_.etcd_namespace());
  }

  const std::string service_key =
      ETCD_XSERVICE_KEY_PREFIX + options_.service_name();
  service_registration_ = std::make_unique<ServiceRegistrationManager>(
      options_.service_name(),
      [this, service_key]() {
        const EtcdKeyLookupResult result = etcd_client_->lookup(service_key);
        switch (result.status) {
          case EtcdKeyLookupStatus::FOUND:
            return RegistrationLookupResult{RegistrationLookupStatus::FOUND,
                                            result.value,
                                            result.lease_id,
                                            result.error};
          case EtcdKeyLookupStatus::NOT_FOUND:
            return RegistrationLookupResult{RegistrationLookupStatus::MISSING,
                                            {},
                                            0,
                                            {}};
          case EtcdKeyLookupStatus::UNAVAILABLE:
            return RegistrationLookupResult{
                RegistrationLookupStatus::UNAVAILABLE,
                {},
                0,
                result.error};
        }
        return RegistrationLookupResult{RegistrationLookupStatus::UNAVAILABLE,
                                        {},
                                        0,
                                        "unknown etcd lookup status"};
      },
      [this, service_key]() {
        const EtcdLeaseCreateResult result = etcd_client_->create_with_lease(
            service_key, options_.service_name(), kHeartbeatInterval);
        return RegistrationCreateResult{
            result.created, result.lease_id, result.error};
      });
  if (!service_registration_->start()) {
    LOG(FATAL)
        << "Failed to register current xllm_service in etcd, service_name: "
        << options_.service_name();
  }

  auto handle_xservice = std::bind(&Scheduler::handle_xservice_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
  etcd_client_->add_watch(ETCD_XSERVICE_KEY_PREFIX, handle_xservice);

  if (!options_.enable_peer_service() &&
      !etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->set(
        ETCD_MASTER_SERVICE_KEY, options_.service_name(), kHeartbeatInterval);
    if (is_master_service_) {
      LOG(INFO) << "Set current service as master!";
    }
  }

  if (uses_legacy_routing()) {
    global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
        options, etcd_client_, is_master_service_);
  }

  if (uses_legacy_routing() && options_.enable_peer_service() &&
      options_.kv_event_zmq_enable()) {
    KvEventSubscriber::Options subscriber_options;
    subscriber_options.enabled(true)
        .poll_interval_ms(options_.kv_event_zmq_poll_interval_ms())
        .reconnect_interval_ms(options_.kv_event_zmq_reconnect_interval_ms())
        .reconnect_interval_max_ms(
            options_.kv_event_zmq_reconnect_interval_max_ms())
        .record_callback([this](const std::string& instance_name,
                                const proto::KvCacheEvent& cache_event) {
          record_instance_cache_event(instance_name, cache_event);
        })
        .snapshot_callback([this](const std::string& instance_name,
                                  const proto::KvCacheEvent& cache_event) {
          replace_instance_cache_snapshot(instance_name, cache_event);
        })
        .clear_callback([this](const std::string& instance_name) {
          clear_instance_cache(instance_name);
        });
    kv_event_subscriber_ =
        std::make_unique<KvEventSubscriber>(std::move(subscriber_options));
    kv_event_subscriber_->start();
  }

  instance_mgr_ = std::make_shared<InstanceMgr>(
      options, etcd_client_, is_master_service_, lifecycle_events_);

  request_registry_ = std::make_unique<RequestSessionRegistry>(
      [this](const std::shared_ptr<Request>& request,
             const llm::RequestOutput& output) {
        observe_session_generation(request, output);
      },
      [this](const std::shared_ptr<Request>& request,
             RequestTerminalReason reason) {
        handle_session_terminal(request, reason);
      });

  if (uses_legacy_routing()) {
    if (options.load_balance_policy() == "CAR") {
      lb_policy_ = std::make_unique<CacheAwareRouting>(instance_mgr_,
                                                       global_kvcache_mgr_);
    } else if (options.load_balance_policy() == "SLO_AWARE") {
      lb_policy_ = std::make_unique<SloAwarePolicy>(options, instance_mgr_);
    } else {
      lb_policy_ = std::make_unique<RoundRobin>(instance_mgr_);
    }
  }

  if (options_.enable_peer_service()) {
    LOG(INFO) << "Peer service mode enabled; skip master election and "
                 "metrics/cache etcd upload paths.";
  } else if (is_master_service_) {
    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);
  } else {
    auto handle_master = std::bind(&Scheduler::handle_master_service_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
    etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, handle_master);
  }
}

Scheduler::~Scheduler() {
  exited_.store(true, std::memory_order_release);
  if (etcd_client_ != nullptr) {
    etcd_client_->stop_watch();
  }
  if (service_registration_ != nullptr) {
    service_registration_->stop();
  }
  if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
    heartbeat_thread_->join();
  }
  lifecycle_events_.close();
  if (request_registry_ != nullptr) {
    request_registry_->close();
  }
  lb_policy_.reset();
  instance_mgr_.reset();
  if (kv_event_subscriber_ != nullptr) {
    kv_event_subscriber_->stop();
  }
}

void Scheduler::cancel_active_requests() {
  if (request_registry_ != nullptr) {
    request_registry_->close();
  }
}

size_t Scheduler::active_request_count() const {
  return request_registry_ == nullptr ? 0 : request_registry_->size();
}

ScheduleResult Scheduler::schedule(std::shared_ptr<Request> request) {
  // apply chat template
  if (request->messages.size() > 0) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return make_schedule_error(ScheduleError::INTERNAL_ERROR,
                                 "Chat template is not configured");
    }

    const std::vector<JsonTool> empty_tools;
    const std::vector<JsonTool>& tools_for_template =
        request->tool_choice == "none" ? empty_tools : request->tools;
    auto prompt = chat_template_->apply(
        request->messages, tools_for_template, request->chat_template_kwargs);
    if (!prompt.has_value()) {
      LOG(ERROR) << "Failed to construct prompt from messages";
      return make_schedule_error(ScheduleError::INVALID_REQUEST,
                                 "Failed to construct prompt from messages");
    }
    request->prompt = prompt.value();
  }

  // encode prompt
  if (request->prompt.size() != 0) {
    if (!get_tls_tokenizer()->encode(request->prompt, &request->token_ids)) {
      LOG(ERROR) << "Encode prompt failed: " << request->prompt;
      return make_schedule_error(ScheduleError::INVALID_REQUEST,
                                 "Failed to tokenize prompt");
    }
  }

  return prepare_routing_decision(request);
}

ScheduleResult Scheduler::prepare_routing_decision(
    const std::shared_ptr<Request>& request) {
  const RoutingSelectionResult selection = resolve_routing_decision(
      routing_configuration_,
      request->request_context.external_routing,
      [this, &request]() {
        return lb_policy_ != nullptr &&
               lb_policy_->select_instances_pair(request);
      },
      &request->routing);
  if (!routing_selection_result_ok(selection)) {
    LOG(ERROR) << "Failed to resolve routing authority: " << selection.message;
    return make_schedule_error(schedule_error_from_selection(selection.error),
                               selection.message);
  }

  const bool bound =
      uses_legacy_routing()
          ? instance_mgr_->bind_request_instance_incarnations(request)
          : instance_mgr_->bind_external_routing_decision(
                &request->routing, routing_configuration_.external_topology);
  if (!bound) {
    LOG(ERROR) << "Failed to bind request to instance incarnation ids. "
               << routing_decision_debug_string(request->routing);
    return make_schedule_error(
        uses_legacy_routing() ? ScheduleError::BACKEND_UNAVAILABLE
                              : ScheduleError::STALE_ROUTING_DECISION,
        uses_legacy_routing()
            ? "No schedulable backend instance is available"
            : "External routing decision no longer matches a schedulable "
              "backend incarnation");
  }
  const RoutingDecisionValidationResult validation =
      validate_routing_decision(request->routing);
  if (!routing_decision_validation_ok(validation)) {
    LOG(ERROR) << "Invalid routing decision: " << validation.message;
    return make_schedule_error(ScheduleError::INTERNAL_ERROR,
                               validation.message);
  }
  DLOG(INFO) << routing_decision_debug_string(request->routing);

  // update request metrics
  if (uses_legacy_routing() && !request->prompt.empty()) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
    instance_mgr_->record_dispatch(request);
  }

  return {};
}

bool Scheduler::uses_legacy_routing() const {
  return routing_configuration_.mode == RoutingMode::LEGACY;
}

void Scheduler::update_master_service_heartbeat() {
  while (!exited_.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));

    if (global_kvcache_mgr_ != nullptr) {
      global_kvcache_mgr_->upload_kvcache();
    }

    instance_mgr_->upload_load_metrics();
  }
}

bool Scheduler::handle_instance_heartbeat(const proto::HeartbeatRequest* req) {
  if (exited_.load(std::memory_order_acquire)) {
    return false;
  }
  COUNTER_INC(xservice_heartbeat_total);
  if (req->has_xtensor_info()) {
    COUNTER_INC(xservice_heartbeat_xtensor_total);
  }
  if (!instance_mgr_->record_instance_heartbeat(req->name(),
                                                req->incarnation_id())) {
    return false;
  }
  const auto& cache_event = req->cache_event();
  const bool has_heartbeat_cache_event = cache_event.stored_cache_size() > 0 ||
                                         cache_event.removed_cache_size() > 0 ||
                                         cache_event.offload_cache_size() > 0;
  if (global_kvcache_mgr_ != nullptr &&
      (!(options_.enable_peer_service() && options_.kv_event_zmq_enable()) ||
       has_heartbeat_cache_event)) {
    global_kvcache_mgr_->record_updated_kvcaches(req->name(), cache_event);
  }
  instance_mgr_->record_load_metrics_update(
      req->name(), req->incarnation_id(), req->load_metrics());
  instance_mgr_->update_latency_metrics(
      req->name(), req->incarnation_id(), req->latency_metrics());
  return true;
}

void Scheduler::handle_master_service_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (options_.enable_peer_service() ||
      exited_.load(std::memory_order_acquire) || response.events().empty()) {
    return;
  }

  if (etcd_client_->set(ETCD_MASTER_SERVICE_KEY,
                        options_.service_name(),
                        kHeartbeatInterval)) {
    is_master_service_ = true;

    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);

    if (global_kvcache_mgr_ != nullptr) {
      global_kvcache_mgr_->set_as_master();
    }
    instance_mgr_->set_as_master();
  }
}

void Scheduler::handle_xservice_watch(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (exited_.load(std::memory_order_acquire) || response.events().empty()) {
    return;
  }

  for (const auto& event : response.events()) {
    if (event.event_type() != etcd::Event::EventType::DELETE_) {
      continue;
    }

    std::string deleted_service;
    if (event.has_prev_kv()) {
      deleted_service = event.prev_kv().key().substr(prefix_len);
    } else if (event.has_kv()) {
      deleted_service = event.kv().key().substr(prefix_len);
    }

    if (deleted_service.empty()) {
      continue;
    }

    if (deleted_service == options_.service_name()) {
      LOG(INFO) << "Current xllm_service registration may have expired";
      if (service_registration_ != nullptr) {
        service_registration_->notify_registration_may_be_missing();
      }
      continue;
    }

    if (!options_.enable_peer_service()) {
      if (deleted_service == ETCD_MASTER_SERVICE_NAME) {
        continue;
      }

      if (!is_master_service_) {
        continue;
      }
    }

    LOG(INFO) << "Detected xllm_service offline: " << deleted_service;
  }
}

InstanceMetaInfo Scheduler::get_instance_info(
    const std::string& instance_name) {
  return instance_mgr_->get_instance_info(instance_name);
}

std::vector<std::string> Scheduler::get_static_decode_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_decode_list(instance_name);
}

std::vector<std::string> Scheduler::get_static_prefill_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_prefill_list(instance_name);
}

Tokenizer* Scheduler::get_tls_tokenizer() {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}

bool Scheduler::record_new_request(std::shared_ptr<ChatCallData> call_data,
                                   std::shared_ptr<Request> request) {
  request->latest_generate_time = absl::Now();
  auto tools_for_parse =
      (request->tool_choice == "none" ? std::vector<JsonTool>{}
                                      : request->tools);
  auto tool_call_parser_pref = options_.tool_call_parser();
  auto reasoning_parser_pref = options_.reasoning_parser();
  std::shared_ptr<ChatStreamParseState> stream_state;
  if (request->stream) {
    stream_state =
        response_handler_.create_chat_stream_parse_state(tools_for_parse,
                                                         request->model,
                                                         tool_call_parser_pref,
                                                         reasoning_parser_pref);
  }

  OutputCallback output_callback =
      [this,
       call_data,
       request,
       model = request->model,
       stream = request->stream,
       include_usage = request->include_usage,
       tools = std::move(tools_for_parse),
       tool_call_parser = std::move(tool_call_parser_pref),
       reasoning_parser = std::move(reasoning_parser_pref),
       stream_state = std::move(stream_state),
       created_time = absl::ToUnixSeconds(request->latest_generate_time)](
          const llm::RequestOutput& req_output) mutable -> bool {
    if (req_output.status.has_value()) {
      const auto& status = req_output.status.value();
      if (!status.ok()) {
        const bool retryable = transport_result_is_retryable(
            TransportResult{request->service_request_id,
                            request->last_transport_result_code,
                            TransportFailureStage::BEFORE_FIRST_TOKEN,
                            request->last_transport_retryability,
                            status.message()});
        return call_data->finish_with_error(
            status.message(),
            retryable ? brpc::HTTP_STATUS_SERVICE_UNAVAILABLE : 0,
            retryable ? transport_result_code_name(
                            request->last_transport_result_code)
                      : "",
            retryable);
      }
    }

    if (stream) {
      return response_handler_.send_delta_to_client(call_data,
                                                    include_usage,
                                                    created_time,
                                                    model,
                                                    req_output,
                                                    stream_state);
    } else if (!req_output.finished_on_prefill_instance) {
      // for non-stream request, only send final result from decode instance
      return response_handler_.send_result_to_client(call_data,
                                                     created_time,
                                                     model,
                                                     req_output,
                                                     tools,
                                                     tool_call_parser,
                                                     reasoning_parser);
    }
    return true;
  };

  if (!request_registry_->register_request(
          request, std::move(output_callback), [call_data]() {
            return call_data->is_disconnected();
          })) {
    LOG(ERROR) << "The request ID already exists or the registry is closed: "
               << request->service_request_id;
    if (uses_legacy_routing()) {
      instance_mgr_->record_request_finished(request);
      instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    }
    return false;
  }
  COUNTER_INC(server_request_in_total);

  return true;
}

bool Scheduler::record_new_request(
    std::shared_ptr<CompletionCallData> call_data,
    std::shared_ptr<Request> request) {
  request->latest_generate_time = absl::Now();
  OutputCallback output_callback =
      [this,
       call_data,
       request,
       model = request->model,
       stream = request->stream,
       include_usage = request->include_usage,
       created_time = absl::ToUnixSeconds(request->latest_generate_time)](
          const llm::RequestOutput& req_output) mutable -> bool {
    if (req_output.status.has_value()) {
      const auto& status = req_output.status.value();
      if (!status.ok()) {
        const bool retryable = transport_result_is_retryable(
            TransportResult{request->service_request_id,
                            request->last_transport_result_code,
                            TransportFailureStage::BEFORE_FIRST_TOKEN,
                            request->last_transport_retryability,
                            status.message()});
        return call_data->finish_with_error(
            status.message(),
            retryable ? brpc::HTTP_STATUS_SERVICE_UNAVAILABLE : 0,
            retryable ? transport_result_code_name(
                            request->last_transport_result_code)
                      : "",
            retryable);
      }
    }

    if (stream) {
      return response_handler_.send_delta_to_client(
          call_data, include_usage, created_time, model, req_output);
    } else if (!req_output.finished_on_prefill_instance) {
      // for non-stream request, only send final result from decode instance
      return response_handler_.send_result_to_client(
          call_data, created_time, model, req_output);
    }
    return true;
  };

  if (!request_registry_->register_request(
          request, std::move(output_callback), [call_data]() {
            return call_data->is_disconnected();
          })) {
    LOG(ERROR) << "The request ID already exists or the registry is closed: "
               << request->service_request_id;
    if (uses_legacy_routing()) {
      instance_mgr_->record_request_finished(request);
      instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    }
    return false;
  }
  COUNTER_INC(server_request_in_total);

  return true;
}

bool Scheduler::handle_transport_failure(const TransportResult& result) {
  return request_registry_->on_transport_failure(result);
}

bool Scheduler::handle_transport_failure(const std::string& service_request_id,
                                         TransportFailureStage stage,
                                         const std::string& message) {
  return request_registry_->on_transport_failure(
      service_request_id, stage, message);
}

void Scheduler::handle_instance_lifecycle_event(
    const InstanceLifecycleEvent& event) {
  const InstanceMetaInfo& instance = event.instance;
  switch (event.type) {
    case InstanceLifecycleEventType::REGISTERED:
      add_kv_event_source(instance);
      return;
    case InstanceLifecycleEventType::REGISTRATION_UPDATED:
      clear_instance_cache(instance.name);
      remove_kv_event_source(instance.name, instance.incarnation_id);
      add_kv_event_source(instance);
      return;
    case InstanceLifecycleEventType::DEREGISTERING:
      remove_kv_event_source(instance.name, instance.incarnation_id);
      return;
    case InstanceLifecycleEventType::DEREGISTERED:
      clear_requests_on_failed_instance(instance.name,
                                        instance.incarnation_id,
                                        instance.type == InstanceType::MIX
                                            ? instance.current_type
                                            : instance.type);
      clear_instance_cache(instance.name);
      return;
  }
}

void Scheduler::clear_requests_on_failed_instance(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  request_registry_->on_instance_failure({instance_name, incarnation_id, type});
}

void Scheduler::clear_instance_cache(const std::string& instance_name) {
  if (global_kvcache_mgr_ != nullptr) {
    global_kvcache_mgr_->clear_instance_cache(instance_name);
  }
}

void Scheduler::add_kv_event_source(const InstanceMetaInfo& info) {
  if (kv_event_subscriber_ != nullptr) {
    kv_event_subscriber_->add_or_update_source(info);
  }
}

void Scheduler::remove_kv_event_source(const std::string& instance_name,
                                       const std::string& incarnation_id) {
  if (kv_event_subscriber_ != nullptr) {
    kv_event_subscriber_->remove_source(instance_name, incarnation_id);
  }
}

void Scheduler::record_instance_cache_event(
    const std::string& instance_name,
    const proto::KvCacheEvent& cache_event) {
  if (global_kvcache_mgr_ != nullptr) {
    global_kvcache_mgr_->record_updated_kvcaches(instance_name, cache_event);
  }
}

void Scheduler::replace_instance_cache_snapshot(
    const std::string& instance_name,
    const proto::KvCacheEvent& cache_event) {
  if (global_kvcache_mgr_ != nullptr) {
    global_kvcache_mgr_->replace_instance_kvcaches(instance_name, cache_event);
  }
}

bool Scheduler::handle_generation(const llm::RequestOutput& request_output) {
  const GenerationDispatchResult result =
      request_registry_->on_generation(request_output);
  if (result == GenerationDispatchResult::ACCEPTED) {
    return true;
  }
  if (result == GenerationDispatchResult::CLIENT_DISCONNECTED) {
    LOG(INFO) << "Client disconnected; request session was cancelled, "
              << "request id: " << request_output.service_request_id;
    return false;
  }
  if (result == GenerationDispatchResult::REGISTRY_CLOSED) {
    LOG(WARNING) << "Generation ignored because request registry is closed, "
                 << "request id: " << request_output.service_request_id;
    return false;
  }
  LOG(ERROR) << "Cannot find an active session for generation, request id: "
             << request_output.service_request_id;
  return false;
}

void Scheduler::observe_session_generation(
    const std::shared_ptr<Request>& request,
    const llm::RequestOutput& output) {
  update_request_metrics(request, output.finished_on_prefill_instance);
  update_token_latency_metrics(request, output.finished_on_prefill_instance);
}

void Scheduler::handle_session_terminal(const std::shared_ptr<Request>& request,
                                        RequestTerminalReason reason) {
  if (uses_legacy_routing()) {
    instance_mgr_->record_request_finished(request);
    instance_mgr_->update_request_metrics(
        request,
        reason == RequestTerminalReason::COMPLETED
            ? RequestAction::FINISH_DECODE
            : RequestAction::CANCEL);
  }
  if (reason != RequestTerminalReason::COMPLETED) {
    LOG(INFO) << "Request session terminated, request id: "
              << request->service_request_id
              << ", reason: " << request_terminal_reason_name(reason);
  }
}

void Scheduler::update_request_metrics(std::shared_ptr<Request> request,
                                       bool finished_on_prefill_instance) {
  request->num_generated_tokens += 1;
  if (finished_on_prefill_instance) {
    if (uses_legacy_routing()) {
      instance_mgr_->record_prefill_finished(request);
      instance_mgr_->update_request_metrics(request,
                                            RequestAction::FINISH_PREFILL);
    }
    request->prefill_stage_finished = true;
  } else if (uses_legacy_routing()) {
    instance_mgr_->update_request_metrics(request, RequestAction::GENERATE);
  }
}

void Scheduler::update_token_latency_metrics(
    std::shared_ptr<Request> request,
    bool finished_on_prefill_instance) {
  int64_t tbt_milliseconds =
      absl::ToInt64Milliseconds(absl::Now() - request->latest_generate_time);
  request->latest_generate_time = absl::Now();
  if (finished_on_prefill_instance) {
    HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                      tbt_milliseconds);
  } else {
    HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
  }
}

bool Scheduler::has_available_instances() const {
  if (!uses_legacy_routing()) {
    return instance_mgr_->has_available_external_endpoint(
        routing_configuration_.external_backend_endpoint,
        routing_configuration_.external_topology);
  }
  return instance_mgr_->has_available_instances();
}

std::string Scheduler::backend_unavailable_reason() const {
  if (!uses_legacy_routing()) {
    return "configured external xLLM endpoint is unavailable: " +
           routing_configuration_.external_backend_endpoint;
  }
  return "no schedulable xLLM endpoint is available";
}

nlohmann::json Scheduler::debug_summary() const {
  nlohmann::json summary;
  summary["service_name"] = options_.service_name();
  summary["routing_mode"] = routing_mode_name(routing_configuration_.mode);
  if (!uses_legacy_routing()) {
    summary["external_backend_endpoint"] =
        routing_configuration_.external_backend_endpoint;
    summary["external_routing_topology"] = external_routing_topology_name(
        routing_configuration_.external_topology);
  }
  summary["enable_peer_service"] = options_.enable_peer_service();
  summary["is_master_service"] = is_master_service_;
  summary["service_registration"] = {
      {"healthy",
       service_registration_ != nullptr && service_registration_->healthy()},
      {"lease_id",
       service_registration_ != nullptr
           ? service_registration_->owned_lease_id()
           : 0}};
  summary["instance_view"] =
      instance_mgr_ ? instance_mgr_->debug_summary() : nlohmann::json::object();
  summary["cache_index"] = global_kvcache_mgr_
                               ? global_kvcache_mgr_->debug_summary()
                               : nlohmann::json::object();
  summary["kv_event_subscriber"] = kv_event_subscriber_
                                       ? kv_event_subscriber_->debug_summary()
                                       : nlohmann::json::object();
  summary["active_request_sessions"] =
      request_registry_ ? request_registry_->size() : 0;
  return summary;
}

}  // namespace xllm_service
