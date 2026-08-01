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

#include "execution/request_session.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace xllm_service {
namespace {

std::shared_ptr<Request> make_request() {
  auto request = std::make_shared<Request>();
  request->service_request_id = "request-1";
  request->routing.prefill_endpoint = "prefill-0";
  request->routing.decode_endpoint = "decode-0";
  request->routing.prefill_incarnation = "prefill-incarnation";
  request->routing.decode_incarnation = "decode-incarnation";
  return request;
}

llm::RequestOutput make_output(bool finished_on_prefill, bool finished) {
  llm::RequestOutput output;
  output.service_request_id = "request-1";
  output.finished_on_prefill_instance = finished_on_prefill;
  output.finished = finished;
  return output;
}

TEST(RequestSessionTest, ProcessesGenerationAndCompletesOnce) {
  auto request = make_request();
  int output_count = 0;
  int generation_count = 0;
  std::vector<RequestTerminalReason> terminal_reasons;
  RequestSession session(
      request,
      [&output_count](llm::RequestOutput) {
        ++output_count;
        return true;
      },
      []() { return false; },
      [&generation_count](const std::shared_ptr<Request>& observed_request,
                          const llm::RequestOutput& output) {
        ++generation_count;
        if (output.finished_on_prefill_instance) {
          observed_request->prefill_stage_finished = true;
        }
      },
      [&terminal_reasons](const std::shared_ptr<Request>&,
                          RequestTerminalReason reason) {
        terminal_reasons.push_back(reason);
      });

  EXPECT_TRUE(session.on_dispatched());
  EXPECT_TRUE(session.on_generation(make_output(true, false)));
  EXPECT_EQ(session.state(), RequestSessionState::DECODE_RUNNING);
  EXPECT_TRUE(session.on_generation(make_output(false, true)));
  EXPECT_EQ(session.state(), RequestSessionState::COMPLETED);
  EXPECT_FALSE(session.on_generation(make_output(false, true)));

  EXPECT_EQ(output_count, 2);
  EXPECT_EQ(generation_count, 2);
  EXPECT_EQ(
      terminal_reasons,
      (std::vector<RequestTerminalReason>{RequestTerminalReason::COMPLETED}));
}

TEST(RequestSessionTest, TransportFailureWritesErrorAndFailsOnce) {
  auto request = make_request();
  std::vector<llm::StatusCode> status_codes;
  std::vector<RequestTerminalReason> terminal_reasons;
  RequestSession session(
      request,
      [&status_codes](llm::RequestOutput output) {
        status_codes.push_back(output.status->code());
        return true;
      },
      []() { return false; },
      {},
      [&terminal_reasons](const std::shared_ptr<Request>&,
                          RequestTerminalReason reason) {
        terminal_reasons.push_back(reason);
      });

  ASSERT_TRUE(session.on_dispatched());
  EXPECT_TRUE(session.on_transport_failure(
      TransportFailureStage::BEFORE_FIRST_TOKEN, "connection refused"));
  EXPECT_FALSE(session.on_transport_failure(
      TransportFailureStage::BEFORE_FIRST_TOKEN, "duplicate failure"));

  EXPECT_EQ(session.state(), RequestSessionState::FAILED);
  EXPECT_EQ(status_codes,
            (std::vector<llm::StatusCode>{llm::StatusCode::UNAVAILABLE}));
  EXPECT_EQ(terminal_reasons,
            (std::vector<RequestTerminalReason>{
                RequestTerminalReason::TRANSPORT_FAILURE_BEFORE_FIRST_TOKEN}));
}

TEST(RequestSessionTest, PreservesRetryableStaleFailureBeforeFirstToken) {
  auto request = make_request();
  RequestTerminalReason terminal_reason = RequestTerminalReason::COMPLETED;
  RequestSession session(
      request,
      [](llm::RequestOutput) { return true; },
      []() { return false; },
      {},
      [&terminal_reason](const std::shared_ptr<Request>&,
                         RequestTerminalReason reason) {
        terminal_reason = reason;
      });

  TransportResult result;
  result.request_id = request->service_request_id;
  result.code = TransportResultCode::STALE_ROUTING_DECISION;
  result.retryability = TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN;
  result.message = "Backend endpoint incarnation is stale";

  ASSERT_TRUE(session.on_dispatched());
  EXPECT_TRUE(session.on_transport_failure(result));
  EXPECT_EQ(request->last_transport_result_code,
            TransportResultCode::STALE_ROUTING_DECISION);
  EXPECT_EQ(request->last_transport_retryability,
            TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN);
  EXPECT_EQ(terminal_reason,
            RequestTerminalReason::TRANSPORT_FAILURE_BEFORE_FIRST_TOKEN);
}

TEST(RequestSessionTest, RemovesRetryabilityAfterGenerationStarts) {
  auto request = make_request();
  RequestSession session(
      request,
      [](llm::RequestOutput) { return true; },
      []() { return false; },
      {},
      {});

  TransportResult result;
  result.request_id = request->service_request_id;
  result.code = TransportResultCode::STALE_ROUTING_DECISION;
  result.retryability = TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN;
  result.message = "Backend endpoint incarnation changed";

  ASSERT_TRUE(session.on_dispatched());
  ASSERT_TRUE(session.on_generation(make_output(true, false)));
  EXPECT_TRUE(session.on_transport_failure(result));
  EXPECT_EQ(request->last_transport_retryability,
            TransportRetryability::NOT_RETRYABLE);
}

TEST(RequestSessionTest, ClassifiesTransportFailureAfterGenerationStarts) {
  auto request = make_request();
  RequestTerminalReason terminal_reason = RequestTerminalReason::COMPLETED;
  RequestSession session(
      request,
      [](llm::RequestOutput) { return true; },
      []() { return false; },
      {},
      [&terminal_reason](const std::shared_ptr<Request>&,
                         RequestTerminalReason reason) {
        terminal_reason = reason;
      });

  ASSERT_TRUE(session.on_dispatched());
  ASSERT_TRUE(session.on_generation(make_output(true, false)));
  EXPECT_TRUE(session.on_transport_failure(
      TransportFailureStage::BEFORE_FIRST_TOKEN, "connection reset"));

  EXPECT_EQ(terminal_reason,
            RequestTerminalReason::TRANSPORT_FAILURE_AFTER_FIRST_TOKEN);
}

TEST(RequestSessionTest, RuntimeCancellationWritesErrorAndCompletesOnce) {
  auto request = make_request();
  std::vector<llm::StatusCode> status_codes;
  std::vector<RequestTerminalReason> terminal_reasons;
  RequestSession session(
      request,
      [&status_codes](llm::RequestOutput output) {
        status_codes.push_back(output.status->code());
        return true;
      },
      []() { return false; },
      {},
      [&terminal_reasons](const std::shared_ptr<Request>&,
                          RequestTerminalReason reason) {
        terminal_reasons.push_back(reason);
      });

  ASSERT_TRUE(session.on_dispatched());
  EXPECT_TRUE(session.on_runtime_cancel("Runtime is shutting down"));
  EXPECT_FALSE(session.on_runtime_cancel("Duplicate cancellation"));

  EXPECT_EQ(session.state(), RequestSessionState::CANCELLED);
  EXPECT_EQ(status_codes,
            (std::vector<llm::StatusCode>{llm::StatusCode::CANCELLED}));
  EXPECT_EQ(terminal_reasons,
            (std::vector<RequestTerminalReason>{
                RequestTerminalReason::RUNTIME_CANCELLED}));
}

TEST(RequestSessionTest, CancelsDisconnectedClientWithoutOutput) {
  auto request = make_request();
  int output_count = 0;
  RequestTerminalReason terminal_reason = RequestTerminalReason::COMPLETED;
  RequestSession session(
      request,
      [&output_count](llm::RequestOutput) {
        ++output_count;
        return true;
      },
      []() { return true; },
      {},
      [&terminal_reason](const std::shared_ptr<Request>&,
                         RequestTerminalReason reason) {
        terminal_reason = reason;
      });

  ASSERT_TRUE(session.on_dispatched());
  EXPECT_TRUE(session.on_generation(make_output(false, false)));

  EXPECT_EQ(session.state(), RequestSessionState::CANCELLED);
  EXPECT_EQ(output_count, 0);
  EXPECT_EQ(terminal_reason, RequestTerminalReason::CLIENT_DISCONNECTED);
}

TEST(RequestSessionTest, PrefillFailureOnlyMatchesBeforePrefillCompletes) {
  auto request = make_request();
  int output_count = 0;
  int terminal_count = 0;
  RequestSession session(
      request,
      [&output_count](llm::RequestOutput output) {
        EXPECT_EQ(output.status->code(), llm::StatusCode::CANCELLED);
        ++output_count;
        return true;
      },
      []() { return false; },
      {},
      [&terminal_count](const std::shared_ptr<Request>&,
                        RequestTerminalReason reason) {
        EXPECT_EQ(reason, RequestTerminalReason::INSTANCE_FAILURE);
        ++terminal_count;
      });

  ASSERT_TRUE(session.on_dispatched());
  EXPECT_FALSE(session.on_instance_failure(
      {"prefill-0", "stale-incarnation", InstanceType::PREFILL}));
  EXPECT_TRUE(session.on_instance_failure(
      {"prefill-0", "prefill-incarnation", InstanceType::PREFILL}));
  EXPECT_FALSE(session.on_instance_failure(
      {"prefill-0", "prefill-incarnation", InstanceType::PREFILL}));

  EXPECT_EQ(output_count, 1);
  EXPECT_EQ(terminal_count, 1);
  EXPECT_EQ(session.state(), RequestSessionState::CANCELLED);
  EXPECT_EQ(request->last_transport_result_code,
            TransportResultCode::STALE_ROUTING_DECISION);
  EXPECT_EQ(request->last_transport_retryability,
            TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN);
}

TEST(RequestSessionTest, InstanceFailureAfterGenerationIsNotRetryable) {
  auto request = make_request();
  RequestSession session(
      request,
      [](llm::RequestOutput) { return true; },
      []() { return false; },
      {},
      {});

  ASSERT_TRUE(session.on_dispatched());
  ASSERT_TRUE(session.on_generation(make_output(true, false)));
  EXPECT_TRUE(session.on_instance_failure(
      {"prefill-0", "prefill-incarnation", InstanceType::DEFAULT}));
  EXPECT_EQ(request->last_transport_result_code,
            TransportResultCode::STALE_ROUTING_DECISION);
  EXPECT_EQ(request->last_transport_retryability,
            TransportRetryability::NOT_RETRYABLE);
}

TEST(RequestSessionTest, OutputFailureTerminatesSuccessfulGeneration) {
  auto request = make_request();
  RequestTerminalReason terminal_reason = RequestTerminalReason::COMPLETED;
  RequestSession session(
      request,
      [](llm::RequestOutput) { return false; },
      []() { return false; },
      {},
      [&terminal_reason](const std::shared_ptr<Request>&,
                         RequestTerminalReason reason) {
        terminal_reason = reason;
      });

  ASSERT_TRUE(session.on_dispatched());
  EXPECT_TRUE(session.on_generation(make_output(false, false)));
  EXPECT_EQ(session.state(), RequestSessionState::FAILED);
  EXPECT_EQ(terminal_reason, RequestTerminalReason::OUTPUT_FAILURE);
}

}  // namespace
}  // namespace xllm_service
