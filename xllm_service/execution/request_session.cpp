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

#include <utility>

#include "common/xllm/status.h"

namespace xllm_service {

const char* request_terminal_reason_name(RequestTerminalReason reason) {
  switch (reason) {
    case RequestTerminalReason::COMPLETED:
      return "completed";
    case RequestTerminalReason::GENERATION_FAILURE:
      return "generation_failure";
    case RequestTerminalReason::OUTPUT_FAILURE:
      return "output_failure";
    case RequestTerminalReason::TRANSPORT_FAILURE_BEFORE_FIRST_TOKEN:
      return "transport_failure_before_first_token";
    case RequestTerminalReason::TRANSPORT_FAILURE_AFTER_FIRST_TOKEN:
      return "transport_failure_after_first_token";
    case RequestTerminalReason::CLIENT_DISCONNECTED:
      return "client_disconnected";
    case RequestTerminalReason::INSTANCE_FAILURE:
      return "instance_failure";
    case RequestTerminalReason::RUNTIME_CANCELLED:
      return "runtime_cancelled";
  }
  return "unknown";
}

RequestSession::RequestSession(std::shared_ptr<Request> request,
                               OutputCallback output_callback,
                               DisconnectCheck disconnect_check,
                               GenerationObserver generation_observer,
                               TerminalObserver terminal_observer)
    : request_(std::move(request)),
      output_callback_(std::move(output_callback)),
      disconnect_check_(std::move(disconnect_check)),
      generation_observer_(std::move(generation_observer)),
      terminal_observer_(std::move(terminal_observer)) {}

bool RequestSession::on_dispatched() {
  RequestSessionState expected = RequestSessionState::CREATED;
  return state_.compare_exchange_strong(expected,
                                        RequestSessionState::DISPATCHED);
}

bool RequestSession::on_generation(llm::RequestOutput output) {
  if (is_terminal()) {
    return false;
  }

  if (client_disconnected()) {
    return on_client_disconnect();
  }

  const bool status_error =
      output.status.has_value() && !output.status.value().ok();
  const bool runtime_cancelled = status_error && output.status.value().code() ==
                                                     llm::StatusCode::CANCELLED;
  const bool finished = output.finished;

  if (!status_error) {
    state_.store(output.finished_on_prefill_instance
                     ? RequestSessionState::PREFILL_RUNNING
                     : RequestSessionState::DECODE_RUNNING);
    if (generation_observer_) {
      generation_observer_(request_, output);
    }
  }

  if (!deliver_output(std::move(output))) {
    return finish(RequestSessionState::FAILED,
                  RequestTerminalReason::OUTPUT_FAILURE);
  }
  if (status_error) {
    return finish(runtime_cancelled ? RequestSessionState::CANCELLED
                                    : RequestSessionState::FAILED,
                  runtime_cancelled
                      ? RequestTerminalReason::RUNTIME_CANCELLED
                      : RequestTerminalReason::GENERATION_FAILURE);
  }
  if (finished) {
    return finish(RequestSessionState::COMPLETED,
                  RequestTerminalReason::COMPLETED);
  }
  if (state_.load() == RequestSessionState::PREFILL_RUNNING) {
    state_.store(RequestSessionState::DECODE_RUNNING);
  }
  return true;
}

bool RequestSession::on_transport_failure(const TransportResult& result) {
  if (is_terminal()) {
    return false;
  }

  const RequestSessionState current_state = state_.load();
  const bool generation_started =
      current_state == RequestSessionState::PREFILL_RUNNING ||
      current_state == RequestSessionState::DECODE_RUNNING;
  const bool after_first_token =
      result.failure_stage == TransportFailureStage::AFTER_FIRST_TOKEN ||
      generation_started;
  request_->last_transport_result_code = result.code;
  request_->last_transport_retryability =
      !after_first_token && transport_result_is_retryable(result)
          ? result.retryability
          : TransportRetryability::NOT_RETRYABLE;

  llm::RequestOutput output;
  output.service_request_id = request_->service_request_id;
  output.status = llm::Status(llm::StatusCode::UNAVAILABLE, result.message);
  deliver_output(std::move(output));
  return finish(
      RequestSessionState::FAILED,
      after_first_token
          ? RequestTerminalReason::TRANSPORT_FAILURE_AFTER_FIRST_TOKEN
          : RequestTerminalReason::TRANSPORT_FAILURE_BEFORE_FIRST_TOKEN);
}

bool RequestSession::on_transport_failure(TransportFailureStage stage,
                                          const std::string& message) {
  TransportResult result;
  result.code = TransportResultCode::RPC_FAILURE;
  result.failure_stage = stage;
  result.message = message;
  return on_transport_failure(result);
}

bool RequestSession::on_instance_failure(const InstanceFailure& failure) {
  if (is_terminal() || !matches(failure)) {
    return false;
  }

  const RequestSessionState current_state = state_.load();
  const bool generation_started =
      current_state == RequestSessionState::PREFILL_RUNNING ||
      current_state == RequestSessionState::DECODE_RUNNING;
  request_->last_transport_result_code =
      TransportResultCode::STALE_ROUTING_DECISION;
  request_->last_transport_retryability =
      generation_started
          ? TransportRetryability::NOT_RETRYABLE
          : TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN;

  llm::RequestOutput output;
  output.service_request_id = request_->service_request_id;
  output.status =
      llm::Status(llm::StatusCode::CANCELLED, "Instance is failed and deleted");
  deliver_output(std::move(output));
  return finish(RequestSessionState::CANCELLED,
                RequestTerminalReason::INSTANCE_FAILURE);
}

bool RequestSession::on_runtime_cancel(const std::string& message) {
  if (is_terminal()) {
    return false;
  }

  llm::RequestOutput output;
  output.service_request_id = request_->service_request_id;
  output.status = llm::Status(llm::StatusCode::CANCELLED, message);
  deliver_output(std::move(output));
  return finish(RequestSessionState::CANCELLED,
                RequestTerminalReason::RUNTIME_CANCELLED);
}

bool RequestSession::on_client_disconnect() {
  return finish(RequestSessionState::CANCELLED,
                RequestTerminalReason::CLIENT_DISCONNECTED);
}

bool RequestSession::client_disconnected() const {
  return disconnect_check_ && disconnect_check_();
}

bool RequestSession::matches(const InstanceFailure& failure) const {
  if (failure.type == InstanceType::DEFAULT) {
    return request_->routing.prefill_endpoint == failure.instance_name &&
           request_->routing.prefill_incarnation == failure.incarnation_id;
  }
  if (failure.type == InstanceType::PREFILL) {
    return request_->routing.prefill_endpoint == failure.instance_name &&
           request_->routing.prefill_incarnation == failure.incarnation_id &&
           !request_->prefill_stage_finished;
  }
  if (failure.type == InstanceType::DECODE) {
    return request_->routing.decode_endpoint == failure.instance_name &&
           request_->routing.decode_incarnation == failure.incarnation_id;
  }
  return false;
}

bool RequestSession::deliver_output(llm::RequestOutput output) {
  return !output_callback_ || output_callback_(std::move(output));
}

bool RequestSession::finish(RequestSessionState state,
                            RequestTerminalReason reason) {
  if (is_terminal()) {
    return false;
  }
  state_.store(state);
  if (terminal_observer_) {
    terminal_observer_(request_, reason);
  }
  return true;
}

bool RequestSession::is_terminal() const {
  const RequestSessionState state = state_.load();
  return state == RequestSessionState::COMPLETED ||
         state == RequestSessionState::FAILED ||
         state == RequestSessionState::CANCELLED;
}

}  // namespace xllm_service
