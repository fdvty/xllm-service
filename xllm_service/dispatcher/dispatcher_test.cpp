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

#include <brpc/server.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/options.h"
#include "xllm_service.pb.h"

namespace xllm_service {
namespace {

RoutingDecision make_routing_decision(const std::string& endpoint,
                                      const std::string& incarnation) {
  RoutingDecision decision;
  decision.prefill_endpoint = endpoint;
  decision.prefill_incarnation = incarnation;
  return decision;
}

class DelayedCompletionService final : public xllm::proto::XllmAPIService {
 public:
  void Completions(google::protobuf::RpcController*,
                   const xllm::proto::CompletionRequest* request,
                   xllm::proto::CompletionResponse*,
                   google::protobuf::Closure* done) override {
    std::lock_guard<std::mutex> lock(mutex_);
    request_id_ = request->service_request_id();
    done_ = done;
    condition_.notify_all();
  }

  bool wait_until_received() {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(
        lock, std::chrono::seconds(5), [this]() { return done_ != nullptr; });
  }

  std::string request_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return request_id_;
  }

  void finish() {
    google::protobuf::Closure* done = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done = done_;
      done_ = nullptr;
    }
    if (done != nullptr) {
      done->Run();
    }
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::string request_id_;
  google::protobuf::Closure* done_ = nullptr;
};

TEST(DispatcherTest, ReportsStaleGenerationDecisionAsRetryable) {
  auto pool = std::make_shared<ChannelPool>(
      [](const std::string&) { return std::shared_ptr<brpc::Channel>(); });
  std::vector<TransportResult> results;
  Dispatcher dispatcher(pool, [&results](const TransportResult& result) {
    results.push_back(result);
  });

  xllm::proto::CompletionRequest request;
  EXPECT_TRUE(dispatcher.dispatch_completion(
      make_routing_decision("127.0.0.1:8000", "incarnation-1"),
      "request-1",
      request));

  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].request_id, "request-1");
  EXPECT_EQ(results[0].code, TransportResultCode::STALE_ROUTING_DECISION);
  EXPECT_EQ(results[0].failure_stage,
            TransportFailureStage::BEFORE_FIRST_TOKEN);
  EXPECT_EQ(results[0].retryability,
            TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN);
  EXPECT_TRUE(transport_result_is_retryable(results[0]));
  EXPECT_EQ(dispatcher.stats().inflight, 0);
  EXPECT_EQ(dispatcher.stats().transport_failure_total, 1);
  EXPECT_EQ(dispatcher.stats().stale_routing_decision_total, 1);
}

TEST(DispatcherTest, ReportsChannelInitializationFailureAsNonRetryable) {
  auto pool = std::make_shared<ChannelPool>(
      [](const std::string&) { return std::shared_ptr<brpc::Channel>(); });
  ASSERT_TRUE(pool->activate("127.0.0.1:8000", "incarnation-1"));
  TransportResult observed;
  Dispatcher dispatcher(
      pool, [&observed](const TransportResult& result) { observed = result; });

  xllm::proto::CompletionRequest request;
  EXPECT_TRUE(dispatcher.dispatch_completion(
      make_routing_decision("127.0.0.1:8000", "incarnation-1"),
      "request-1",
      request));

  EXPECT_EQ(observed.code, TransportResultCode::CHANNEL_UNAVAILABLE);
  EXPECT_EQ(observed.retryability, TransportRetryability::NOT_RETRYABLE);
  EXPECT_FALSE(transport_result_is_retryable(observed));
  EXPECT_EQ(dispatcher.stats().stale_routing_decision_total, 0);
}

TEST(DispatcherTest, ReportsInvalidRoutingDecision) {
  auto pool = std::make_shared<ChannelPool>(
      [](const std::string&) { return std::shared_ptr<brpc::Channel>(); });
  TransportResult observed;
  int observer_count = 0;
  Dispatcher dispatcher(pool, [&](const TransportResult& result) {
    observed = result;
    ++observer_count;
  });

  RoutingDecision decision;
  decision.prefill_endpoint = "127.0.0.1:8000";
  xllm::proto::CompletionRequest request;
  EXPECT_TRUE(dispatcher.dispatch_completion(decision, "request-1", request));

  EXPECT_EQ(observer_count, 1);
  EXPECT_EQ(observed.request_id, "request-1");
  EXPECT_EQ(observed.code, TransportResultCode::INVALID_ROUTING_DECISION);
  EXPECT_EQ(dispatcher.stats().inflight, 0);
  EXPECT_EQ(dispatcher.stats().transport_failure_total, 1);
}

TEST(DispatcherTest, ReportsBackendRpcFailureAsRetryableBeforeFirstToken) {
  DelayedCompletionService service;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1", brpc::PortRange(20000, 40000), nullptr),
            0);

  const std::string endpoint =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  Options options;
  options.timeout_ms(1000).connect_timeout_ms(100);
  auto pool = std::make_shared<ChannelPool>(options);
  ASSERT_TRUE(pool->activate(endpoint, "incarnation-1"));
  server.Stop(0);
  server.Join();

  std::mutex mutex;
  std::condition_variable condition;
  bool observed = false;
  TransportResult result;
  Dispatcher dispatcher(pool, [&](const TransportResult& failure) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      result = failure;
      observed = true;
    }
    condition.notify_all();
  });

  xllm::proto::CompletionRequest request;
  ASSERT_TRUE(dispatcher.dispatch_completion(
      make_routing_decision(endpoint, "incarnation-1"), "request-1", request));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(
        lock, std::chrono::seconds(5), [&observed]() { return observed; }));
  }

  EXPECT_EQ(result.code, TransportResultCode::RPC_FAILURE);
  EXPECT_EQ(result.failure_stage, TransportFailureStage::BEFORE_FIRST_TOKEN);
  EXPECT_EQ(result.retryability,
            TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN);
  EXPECT_TRUE(transport_result_is_retryable(result));
  EXPECT_EQ(dispatcher.stats().inflight, 0);
}

TEST(DispatcherTest, ReportsUnavailableModelsChannelToCaller) {
  auto pool = std::make_shared<ChannelPool>(
      [](const std::string&) { return std::shared_ptr<brpc::Channel>(); });
  Dispatcher dispatcher(pool, {});
  TransportResult observed;
  int callback_count = 0;

  xllm::proto::ModelListRequest request;
  EXPECT_TRUE(dispatcher.dispatch_models(
      "127.0.0.1:8000",
      "incarnation-1",
      request,
      [&observed, &callback_count](const TransportResult& result,
                                   const xllm::proto::ModelListResponse&) {
        observed = result;
        ++callback_count;
      }));

  EXPECT_EQ(callback_count, 1);
  EXPECT_EQ(observed.code, TransportResultCode::STALE_ROUTING_DECISION);
  EXPECT_EQ(observed.retryability,
            TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN);
  EXPECT_EQ(dispatcher.stats().inflight, 0);
}

TEST(DispatcherTest, RejectsDispatchAfterClose) {
  auto pool = std::make_shared<ChannelPool>(
      [](const std::string&) { return std::shared_ptr<brpc::Channel>(); });
  int observer_count = 0;
  Dispatcher dispatcher(
      pool, [&observer_count](const TransportResult&) { ++observer_count; });
  dispatcher.close();

  xllm::proto::ChatRequest request;
  EXPECT_FALSE(dispatcher.dispatch_chat(
      make_routing_decision("127.0.0.1:8000", "incarnation-1"),
      "request-1",
      request));
  EXPECT_EQ(observer_count, 0);
}

TEST(DispatcherTest, CloseCancelsInflightRpc) {
  DelayedCompletionService service;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1", brpc::PortRange(20000, 40000), nullptr),
            0);

  const std::string endpoint =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  Options options;
  options.timeout_ms(5000).connect_timeout_ms(1000);
  auto pool = std::make_shared<ChannelPool>(options);
  ASSERT_TRUE(pool->activate(endpoint, "incarnation-1"));

  std::atomic<int32_t> observer_count{0};
  std::atomic<TransportResultCode> observed_code{TransportResultCode::SUCCESS};
  Dispatcher dispatcher(
      pool, [&observer_count, &observed_code](const TransportResult& result) {
        observed_code.store(result.code);
        ++observer_count;
      });
  xllm::proto::CompletionRequest request;
  request.set_service_request_id("request-owned-by-dispatcher");
  ASSERT_TRUE(dispatcher.dispatch_completion(
      make_routing_decision(endpoint, "incarnation-1"), "request-1", request));
  const bool request_received = service.wait_until_received();
  EXPECT_TRUE(request_received);
  if (!request_received) {
    server.Stop(0);
    server.Join();
    dispatcher.close();
    return;
  }
  EXPECT_EQ(service.request_id(), "request-owned-by-dispatcher");

  std::atomic<bool> close_returned{false};
  std::mutex close_mutex;
  std::condition_variable close_condition;
  std::thread close_thread([&]() {
    dispatcher.close();
    {
      std::lock_guard<std::mutex> lock(close_mutex);
      close_returned.store(true);
    }
    close_condition.notify_all();
  });

  bool cancelled = false;
  {
    std::unique_lock<std::mutex> lock(close_mutex);
    cancelled = close_condition.wait_for(
        lock, std::chrono::seconds(2), [&close_returned]() {
          return close_returned.load();
        });
  }

  service.finish();
  close_thread.join();

  EXPECT_TRUE(cancelled);
  EXPECT_EQ(observer_count.load(), 1);
  EXPECT_EQ(observed_code.load(), TransportResultCode::DISPATCHER_CLOSED);
  EXPECT_EQ(dispatcher.stats().inflight, 0);
  server.Stop(0);
  server.Join();
}

TEST(DispatcherTest, CloseRejectsDispatchWaitingForChannelCreation) {
  std::mutex factory_mutex;
  std::condition_variable factory_condition;
  bool factory_entered = false;
  bool release_factory = false;
  auto pool = std::make_shared<ChannelPool>(
      [&](const std::string&) -> std::shared_ptr<brpc::Channel> {
        std::unique_lock<std::mutex> lock(factory_mutex);
        factory_entered = true;
        factory_condition.notify_all();
        factory_condition.wait(
            lock, [&release_factory]() { return release_factory; });
        return std::make_shared<brpc::Channel>();
      });
  ASSERT_TRUE(pool->activate("127.0.0.1:8000", "incarnation-1"));

  TransportResult observed;
  std::atomic<int32_t> observer_count{0};
  Dispatcher dispatcher(pool, [&](const TransportResult& result) {
    observed = result;
    ++observer_count;
  });
  xllm::proto::CompletionRequest request;
  std::atomic<bool> dispatched{false};
  std::thread dispatch_thread([&]() {
    dispatched.store(dispatcher.dispatch_completion(
        make_routing_decision("127.0.0.1:8000", "incarnation-1"),
        "request-1",
        request));
  });

  bool factory_was_entered = false;
  {
    std::unique_lock<std::mutex> lock(factory_mutex);
    factory_was_entered = factory_condition.wait_for(
        lock, std::chrono::seconds(2), [&factory_entered]() {
          return factory_entered;
        });
  }
  EXPECT_TRUE(factory_was_entered);

  std::thread close_thread([&dispatcher]() { dispatcher.close(); });
  const auto close_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!dispatcher.stats().closed &&
         std::chrono::steady_clock::now() < close_deadline) {
    std::this_thread::yield();
  }
  EXPECT_TRUE(dispatcher.stats().closed);
  {
    std::lock_guard<std::mutex> lock(factory_mutex);
    release_factory = true;
  }
  factory_condition.notify_all();

  dispatch_thread.join();
  close_thread.join();

  EXPECT_TRUE(dispatched.load());
  EXPECT_EQ(observer_count.load(), 1);
  EXPECT_EQ(observed.code, TransportResultCode::DISPATCHER_CLOSED);
  EXPECT_EQ(dispatcher.stats().inflight, 0);
}

}  // namespace
}  // namespace xllm_service
