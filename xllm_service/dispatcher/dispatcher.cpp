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

#include "dispatcher/dispatcher.h"

#include <brpc/controller.h>
#include <glog/logging.h>
#include <google/protobuf/message.h>
#include <google/protobuf/stubs/callback.h>

#include <cerrno>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xllm_service.pb.h"

namespace xllm_service {

class DispatcherState final {
 public:
  explicit DispatcherState(Dispatcher::TransportObserver transport_observer)
      : transport_observer_(std::move(transport_observer)) {}

  bool begin_dispatch() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return false;
    }
    ++inflight_;
    return true;
  }

  bool register_call(brpc::CallId call_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return false;
    }
    active_calls_.emplace(call_id.value, call_id);
    return true;
  }

  bool is_closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  void complete_generation(const TransportResult& result,
                           brpc::CallId call_id = {0}) {
    complete(result, call_id, [this, &result]() {
      if (transport_observer_) {
        transport_observer_(result);
      }
    });
  }

  void complete_models(const TransportResult& result,
                       const xllm::proto::ModelListResponse& response,
                       const Dispatcher::ModelResultCallback& callback,
                       brpc::CallId call_id = {0}) {
    complete(result, call_id, [&result, &response, &callback]() {
      if (callback) {
        callback(result, response);
      }
    });
  }

  void close() {
    std::vector<brpc::CallId> active_calls;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!closed_) {
        closed_ = true;
        active_calls.reserve(active_calls_.size());
        for (const auto& item : active_calls_) {
          active_calls.push_back(item.second);
        }
      }
    }

    for (brpc::CallId call_id : active_calls) {
      brpc::StartCancel(call_id);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]() { return inflight_ == 0; });
  }

  DispatcherStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    DispatcherStats result;
    result.closed = closed_;
    result.inflight = inflight_;
    result.transport_failure_total = transport_failure_total_;
    result.stale_routing_decision_total = stale_routing_decision_total_;
    return result;
  }

 private:
  template <typename Callback>
  void complete(const TransportResult& result,
                brpc::CallId call_id,
                Callback callback) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (result.code != TransportResultCode::SUCCESS) {
        ++transport_failure_total_;
      }
      if (result.code == TransportResultCode::STALE_ROUTING_DECISION) {
        ++stale_routing_decision_total_;
      }
      if (call_id.value != 0) {
        active_calls_.erase(call_id.value);
      }
    }

    try {
      callback();
    } catch (const std::exception& error) {
      LOG(ERROR) << "Transport completion callback failed: " << error.what();
    } catch (...) {
      LOG(ERROR) << "Transport completion callback failed with unknown error";
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (inflight_ > 0) {
      --inflight_;
    }
    if (inflight_ == 0) {
      condition_.notify_all();
    }
  }

  const Dispatcher::TransportObserver transport_observer_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool closed_ = false;
  uint64_t inflight_ = 0;
  uint64_t transport_failure_total_ = 0;
  uint64_t stale_routing_decision_total_ = 0;
  std::unordered_map<uint64_t, brpc::CallId> active_calls_;
};

namespace {

TransportResult make_failure(
    const std::string& request_id,
    TransportResultCode code,
    const std::string& message,
    TransportRetryability retryability = TransportRetryability::NOT_RETRYABLE) {
  TransportResult result;
  result.request_id = request_id;
  result.code = code;
  result.failure_stage = TransportFailureStage::BEFORE_FIRST_TOKEN;
  result.retryability = retryability;
  result.message = message;
  return result;
}

class GenerationRpcClosure final : public google::protobuf::Closure {
 public:
  GenerationRpcClosure(std::shared_ptr<DispatcherState> state,
                       std::shared_ptr<brpc::Channel> channel,
                       std::shared_ptr<const google::protobuf::Message> request,
                       std::string request_id)
      : state_(std::move(state)),
        channel_(std::move(channel)),
        request_(std::move(request)),
        request_id_(std::move(request_id)) {}

  brpc::Controller* controller() { return &controller_; }
  brpc::CallId call_id() { return controller_.call_id(); }

  void reject(const TransportResult& result) {
    state_->complete_generation(result);
    delete this;
  }

  void Run() override {
    TransportResult result;
    result.request_id = request_id_;
    if (controller_.Failed()) {
      const bool dispatcher_closed =
          controller_.ErrorCode() == ECANCELED && state_->is_closed();
      const auto retryability =
          dispatcher_closed
              ? TransportRetryability::NOT_RETRYABLE
              : TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN;
      result = make_failure(request_id_,
                            dispatcher_closed
                                ? TransportResultCode::DISPATCHER_CLOSED
                                : TransportResultCode::RPC_FAILURE,
                            controller_.ErrorText(),
                            retryability);
    }
    state_->complete_generation(result, controller_.call_id());
    delete this;
  }

 private:
  const std::shared_ptr<DispatcherState> state_;
  const std::shared_ptr<brpc::Channel> channel_;
  const std::shared_ptr<const google::protobuf::Message> request_;
  const std::string request_id_;
  brpc::Controller controller_;
};

class ModelsRpcClosure final : public google::protobuf::Closure {
 public:
  ModelsRpcClosure(std::shared_ptr<DispatcherState> state,
                   std::shared_ptr<brpc::Channel> channel,
                   std::shared_ptr<const xllm::proto::ModelListRequest> request,
                   Dispatcher::ModelResultCallback callback)
      : state_(std::move(state)),
        channel_(std::move(channel)),
        request_(std::move(request)),
        callback_(std::move(callback)) {}

  brpc::Controller* controller() { return &controller_; }
  brpc::CallId call_id() { return controller_.call_id(); }
  xllm::proto::ModelListResponse* response() { return &response_; }

  void reject(const TransportResult& result) {
    state_->complete_models(result, response_, callback_);
    delete this;
  }

  void Run() override {
    TransportResult result;
    if (controller_.Failed()) {
      const bool dispatcher_closed =
          controller_.ErrorCode() == ECANCELED && state_->is_closed();
      const auto retryability =
          dispatcher_closed
              ? TransportRetryability::NOT_RETRYABLE
              : TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN;
      result = make_failure("",
                            dispatcher_closed
                                ? TransportResultCode::DISPATCHER_CLOSED
                                : TransportResultCode::RPC_FAILURE,
                            controller_.ErrorText(),
                            retryability);
    }
    state_->complete_models(
        result, response_, callback_, controller_.call_id());
    delete this;
  }

 private:
  const std::shared_ptr<DispatcherState> state_;
  const std::shared_ptr<brpc::Channel> channel_;
  const std::shared_ptr<const xllm::proto::ModelListRequest> request_;
  const Dispatcher::ModelResultCallback callback_;
  brpc::Controller controller_;
  xllm::proto::ModelListResponse response_;
};

template <typename Request, typename RpcInvoker>
bool dispatch_generation(const std::shared_ptr<ChannelPool>& channel_pool,
                         const std::shared_ptr<DispatcherState>& state,
                         const RoutingDecision& decision,
                         const std::string& request_id,
                         const Request& request,
                         RpcInvoker invoke) {
  if (!state->begin_dispatch()) {
    return false;
  }

  const RoutingDecisionValidationResult validation =
      validate_routing_decision(decision);
  if (!routing_decision_validation_ok(validation)) {
    state->complete_generation(
        make_failure(request_id,
                     TransportResultCode::INVALID_ROUTING_DECISION,
                     validation.message));
    return true;
  }

  const ChannelLookupResult channel_result =
      channel_pool->get_or_create_with_status(decision.prefill_endpoint,
                                              decision.prefill_incarnation);
  if (!channel_result.available()) {
    const bool stale = channel_result.stale();
    state->complete_generation(
        make_failure(request_id,
                     stale ? TransportResultCode::STALE_ROUTING_DECISION
                           : TransportResultCode::CHANNEL_UNAVAILABLE,
                     channel_result.message,
                     stale ? TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN
                           : TransportRetryability::NOT_RETRYABLE));
    return true;
  }

  auto owned_request = std::make_shared<Request>(request);
  auto* closure = new GenerationRpcClosure(
      state, channel_result.channel, owned_request, request_id);
  if (!state->register_call(closure->call_id())) {
    closure->reject(make_failure(request_id,
                                 TransportResultCode::DISPATCHER_CLOSED,
                                 "Backend dispatcher is closed"));
    return true;
  }
  xllm::proto::XllmAPIService_Stub stub(channel_result.channel.get());
  invoke(stub, closure->controller(), owned_request.get(), closure);
  return true;
}

}  // namespace

Dispatcher::Dispatcher(std::shared_ptr<ChannelPool> channel_pool,
                       TransportObserver transport_observer)
    : channel_pool_(std::move(channel_pool)),
      state_(std::make_shared<DispatcherState>(std::move(transport_observer))) {
}

Dispatcher::~Dispatcher() { close(); }

bool Dispatcher::dispatch_completion(
    const RoutingDecision& decision,
    const std::string& request_id,
    const xllm::proto::CompletionRequest& request) {
  return dispatch_generation(
      channel_pool_,
      state_,
      decision,
      request_id,
      request,
      [](xllm::proto::XllmAPIService_Stub& stub,
         brpc::Controller* controller,
         const xllm::proto::CompletionRequest* owned_request,
         google::protobuf::Closure* closure) {
        stub.Completions(controller, owned_request, nullptr, closure);
      });
}

bool Dispatcher::dispatch_chat(const RoutingDecision& decision,
                               const std::string& request_id,
                               const xllm::proto::ChatRequest& request) {
  return dispatch_generation(channel_pool_,
                             state_,
                             decision,
                             request_id,
                             request,
                             [](xllm::proto::XllmAPIService_Stub& stub,
                                brpc::Controller* controller,
                                const xllm::proto::ChatRequest* owned_request,
                                google::protobuf::Closure* closure) {
                               stub.ChatCompletions(
                                   controller, owned_request, nullptr, closure);
                             });
}

bool Dispatcher::dispatch_models(const std::string& endpoint,
                                 const std::string& incarnation_id,
                                 const xllm::proto::ModelListRequest& request,
                                 ModelResultCallback callback) {
  if (!state_->begin_dispatch()) {
    return false;
  }

  const ChannelLookupResult channel_result =
      channel_pool_->get_or_create_with_status(endpoint, incarnation_id);
  if (!channel_result.available()) {
    const bool stale = channel_result.stale();
    const xllm::proto::ModelListResponse response;
    state_->complete_models(
        make_failure("",
                     stale ? TransportResultCode::STALE_ROUTING_DECISION
                           : TransportResultCode::CHANNEL_UNAVAILABLE,
                     channel_result.message,
                     stale ? TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN
                           : TransportRetryability::NOT_RETRYABLE),
        response,
        callback);
    return true;
  }

  auto owned_request = std::make_shared<xllm::proto::ModelListRequest>(request);
  auto* closure = new ModelsRpcClosure(
      state_, channel_result.channel, owned_request, std::move(callback));
  if (!state_->register_call(closure->call_id())) {
    closure->reject(make_failure("",
                                 TransportResultCode::DISPATCHER_CLOSED,
                                 "Backend dispatcher is closed"));
    return true;
  }
  xllm::proto::XllmAPIService_Stub stub(channel_result.channel.get());
  stub.Models(
      closure->controller(), owned_request.get(), closure->response(), closure);
  return true;
}

void Dispatcher::close() { state_->close(); }

DispatcherStats Dispatcher::stats() const {
  DispatcherStats result = state_->stats();
  result.channel_count = channel_pool_->channel_count();
  result.endpoint_count = channel_pool_->endpoint_count();
  return result;
}

}  // namespace xllm_service
