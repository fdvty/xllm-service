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

#include "http_service/service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <brpc/http_status_code.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <cctype>
#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include "backend_http/request_context.h"
#include "chat.pb.h"
#include "common/call_data.h"
#include "common/closure_guard.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "common/xllm/uuid.h"
#include "completion.pb.h"
#include "dispatcher/dispatcher.h"
#include "runtime/runtime_state.h"
#include "scheduler/scheduler.h"
#include "telemetry/prometheus_metrics.h"

namespace xllm_service {

void apply_schedule_failure(brpc::Controller* controller,
                            const ScheduleResult& result) {
  if (controller == nullptr || schedule_result_ok(result)) {
    return;
  }

  int status_code = brpc::HTTP_STATUS_INTERNAL_SERVER_ERROR;
  switch (result.error) {
    case ScheduleError::INVALID_REQUEST:
    case ScheduleError::INVALID_ROUTING_DIRECTIVE:
      status_code = brpc::HTTP_STATUS_BAD_REQUEST;
      break;
    case ScheduleError::BACKEND_UNAVAILABLE:
    case ScheduleError::STALE_ROUTING_DECISION:
      status_code = brpc::HTTP_STATUS_SERVICE_UNAVAILABLE;
      break;
    case ScheduleError::INTERNAL_ERROR:
    case ScheduleError::NONE:
      break;
  }

  const char* error_code = schedule_error_code(result.error);
  const bool retryable = schedule_result_is_retryable(result);
  nlohmann::json error = {
      {"error", {{"message", result.message}, {"type", error_code}}}};
  controller->http_response().set_content_type("application/json");
  controller->http_response().set_status_code(status_code);
  controller->http_response().SetHeader("x-llm-d-error-code", error_code);
  if (retryable) {
    controller->http_response().SetHeader("x-llm-d-retryable", "true");
    controller->http_response().SetHeader("Retry-After", "0");
  }
  controller->response_attachment().append(error.dump());
}

namespace {
thread_local llm::ShortUUID short_uuid;
std::string generate_service_request_id(const std::string& method) {
  std::stringstream ss;
  ss << method << "-";
  ss << std::this_thread::get_id();
  ss << "-";
  ss << short_uuid.random();
  return ss.str();
}

nlohmann::json proto_value_to_json(const google::protobuf::Value& pb_value);

nlohmann::json proto_struct_to_json(const google::protobuf::Struct& pb_struct) {
  nlohmann::json result = nlohmann::json::object();
  for (const auto& field : pb_struct.fields()) {
    result[field.first] = proto_value_to_json(field.second);
  }
  return result;
}

nlohmann::json proto_value_to_json(const google::protobuf::Value& pb_value) {
  switch (pb_value.kind_case()) {
    case google::protobuf::Value::kNullValue:
      return nlohmann::json(nullptr);
    case google::protobuf::Value::kNumberValue:
      return nlohmann::json(pb_value.number_value());
    case google::protobuf::Value::kStringValue:
      return nlohmann::json(pb_value.string_value());
    case google::protobuf::Value::kBoolValue:
      return nlohmann::json(pb_value.bool_value());
    case google::protobuf::Value::kStructValue:
      return proto_struct_to_json(pb_value.struct_value());
    case google::protobuf::Value::kListValue: {
      nlohmann::json result = nlohmann::json::array();
      for (const auto& item : pb_value.list_value().values()) {
        result.push_back(proto_value_to_json(item));
      }
      return result;
    }
    case google::protobuf::Value::KIND_NOT_SET:
    default:
      return nlohmann::json(nullptr);
  }
}
std::vector<JsonTool> parse_tools_from_proto(
    const google::protobuf::RepeatedPtrField<::xllm::proto::Tool>&
        proto_tools) {
  std::vector<JsonTool> tools;
  tools.reserve(proto_tools.size());

  for (const auto& proto_tool : proto_tools) {
    JsonTool json_tool;
    json_tool.type = proto_tool.type();
    json_tool.function.name = proto_tool.function().name();
    json_tool.function.description = proto_tool.function().description();
    if (proto_tool.function().has_parameters()) {
      json_tool.function.parameters =
          proto_struct_to_json(proto_tool.function().parameters());
    } else {
      json_tool.function.parameters = nlohmann::json::object();
    }
    tools.emplace_back(std::move(json_tool));
  }
  return tools;
}
}  // namespace

XllmHttpServiceImpl::XllmHttpServiceImpl(const Options& options,
                                         Scheduler* scheduler,
                                         RuntimeState& runtime_state,
                                         Dispatcher* dispatcher)
    : options_(options),
      scheduler_(scheduler),
      dispatcher_(dispatcher),
      runtime_state_(runtime_state) {
  initialized_ = true;
  request_tracer_ =
      std::make_unique<RequestTracer>(options_.enable_request_trace());
}

XllmHttpServiceImpl::~XllmHttpServiceImpl() {}

void XllmHttpServiceImpl::Hello(::google::protobuf::RpcController* controller,
                                const proto::HttpHelloRequest* request,
                                proto::HttpHelloResponse* response,
                                ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  LOG(INFO) << "Get request: " << request->ping();

  response->set_pong(request->ping());
}

void XllmHttpServiceImpl::Health(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)
          ->SetFailed("brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  const RuntimeHealthSnapshot snapshot = runtime_state_.health_snapshot();
  cntl->http_response().set_content_type("text/plain");
  cntl->http_response().set_status_code(
      snapshot.ready ? brpc::HTTP_STATUS_OK
                     : brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  cntl->response_attachment().append(snapshot.ready ? "ok\n" : "unavailable\n");
}

void XllmHttpServiceImpl::Liveness(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)
          ->SetFailed("brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  const RuntimeHealthSnapshot snapshot = runtime_state_.health_snapshot();
  cntl->http_response().set_content_type("text/plain");
  cntl->http_response().set_status_code(
      snapshot.live ? brpc::HTTP_STATUS_OK
                    : brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  cntl->response_attachment().append(snapshot.live ? "ok\n" : "stopped\n");
}

void XllmHttpServiceImpl::Readiness(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)
          ->SetFailed("brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  const RuntimeHealthSnapshot snapshot = runtime_state_.health_snapshot();
  cntl->http_response().set_content_type("text/plain");
  cntl->http_response().set_status_code(
      snapshot.ready ? brpc::HTTP_STATUS_OK
                     : brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  cntl->response_attachment().append(snapshot.ready ? "ready\n"
                                                    : snapshot.reason + "\n");
}

bool XllmHttpServiceImpl::ensure_backend_ready(
    brpc::Controller* controller) const {
  const RuntimeHealthSnapshot snapshot = runtime_state_.health_snapshot();
  if (snapshot.ready) {
    return true;
  }

  nlohmann::json error = {
      {"error",
       {{"message",
         snapshot.reason.empty() ? "backend is not ready" : snapshot.reason},
        {"type", "service_unavailable"}}}};
  controller->http_response().set_content_type("application/json");
  controller->http_response().set_status_code(
      brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  controller->response_attachment().append(error.dump());
  return false;
}

namespace {

constexpr char kInferContentLength[] = "Infer-Content-Length";
constexpr char kContentLength[] = "Content-Length";

bool TryParseContentLength(const char* header_name,
                           const std::string& header_value,
                           size_t attachment_size,
                           size_t* content_len) {
  try {
    size_t parsed_size = 0;
    const auto parsed_value = std::stoull(header_value, &parsed_size, 10);
    while (
        parsed_size < header_value.size() &&
        std::isspace(static_cast<unsigned char>(header_value[parsed_size]))) {
      ++parsed_size;
    }
    if (parsed_size != header_value.size() ||
        parsed_value > std::numeric_limits<size_t>::max()) {
      LOG(WARNING) << "Invalid " << header_name
                   << " header value: " << header_value;
      return false;
    }

    *content_len = static_cast<size_t>(parsed_value);
    if (*content_len > attachment_size) {
      LOG(WARNING) << header_name << " header value " << *content_len
                   << " exceeds request attachment size " << attachment_size
                   << ", use attachment size instead.";
      *content_len = attachment_size;
    }
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Invalid " << header_name
                 << " header value: " << header_value
                 << ", error: " << e.what();
    return false;
  }
}

size_t GetJsonContentLength(const brpc::Controller* ctrl,
                            size_t attachment_size) {
  const auto infer_content_len =
      ctrl->http_request().GetHeader(kInferContentLength);
  size_t content_len = 0;
  if (infer_content_len != nullptr && TryParseContentLength(kInferContentLength,
                                                            *infer_content_len,
                                                            attachment_size,
                                                            &content_len)) {
    return content_len;
  }

  const auto content_len_header =
      ctrl->http_request().GetHeader(kContentLength);
  if (content_len_header != nullptr &&
      TryParseContentLength(
          kContentLength, *content_len_header, attachment_size, &content_len)) {
    return content_len;
  }

  if (attachment_size > 0) {
    LOG(WARNING) << "Content-Length header is missing, use request attachment "
                    "size instead: "
                 << attachment_size;
  }
  return attachment_size;
}

}  // namespace

template <typename T>
void XllmHttpServiceImpl::handle(std::shared_ptr<T> call_data,
                                 std::shared_ptr<Request> request) {
  // record request
  auto& req_pb = call_data->request();
  bool success = scheduler_->record_new_request(call_data, request);
  if (!success) {
    LOG(ERROR) << "rpc service add new request error: "
               << request->service_request_id;
    call_data->finish_with_error("Internal runtime error.");
    return;
  }

  bool dispatched = false;
  if constexpr (std::is_same_v<T, CompletionCallData>) {
    dispatched = dispatcher_ != nullptr &&
                 dispatcher_->dispatch_completion(
                     request->routing, request->service_request_id, req_pb);
  } else if constexpr (std::is_same_v<T, ChatCallData>) {
    dispatched = dispatcher_ != nullptr &&
                 dispatcher_->dispatch_chat(
                     request->routing, request->service_request_id, req_pb);
  } else {
    LOG(ERROR) << "Unknown call_data type";
  }

  if (!dispatched) {
    scheduler_->handle_transport_failure(
        request->service_request_id,
        TransportFailureStage::BEFORE_FIRST_TOKEN,
        "Backend dispatcher is unavailable");
  }
}

template <typename T>
std::shared_ptr<Request> XllmHttpServiceImpl::generate_request(
    T* req_pb,
    const brpc::Controller& controller,
    const std::string& method) {
  std::shared_ptr<Request> request = std::make_shared<Request>();
  request->request_context = parse_request_context(controller);

  const std::string request_body_model = req_pb->model();
  request->model = resolve_effective_model_name(request_body_model,
                                                request->request_context);
  if (request->model != request_body_model) {
    req_pb->set_model(request->model);
  }

  // TODO: add `created_time` fileds etc.
  // create xllm_service request_id: service_request_id
  request->service_request_id = generate_service_request_id(method);

  const auto& context = request->request_context;
  nlohmann::json identity_event = {
      {"event", "request_context_accepted"},
      {"request_id", context.request_id},
      {"service_request_id", request->service_request_id},
      {"traceparent", context.traceparent},
      {"tenant_id", context.tenant_id},
      {"service_tier", context.service_tier},
      {"slo_class", context.slo_class},
      {"method", method},
  };
  LOG(INFO) << identity_event.dump();

  if (req_pb->has_stream()) {
    request->stream = req_pb->stream();
  }

  if (req_pb->has_stream_options()) {
    request->include_usage = req_pb->stream_options().include_usage();
  }

  if (options_.enable_request_trace()) {
    request->trace_callback =
        [this, service_request_id = request->service_request_id](
            const std::string& message) {
          request_tracer_->log(service_request_id, message);
        };
  }

  return request;
}

namespace {
void handle_get_model_response(std::shared_ptr<CompletionCallData> call_data,
                               const TransportResult& result,
                               const xllm::proto::ModelListResponse& response) {
  if (result.code != TransportResultCode::SUCCESS) {
    LOG(ERROR) << "Failed to get serving models: " << result.message;
    const bool retryable = transport_result_is_retryable(result);
    call_data->finish_with_error(
        result.message,
        retryable ? brpc::HTTP_STATUS_SERVICE_UNAVAILABLE : 0,
        retryable ? transport_result_code_name(result.code) : "",
        retryable);
    return;
  }
  std::string err_msg;
  std::string json_output;
  if (!json2pb::ProtoMessageToJson(response, &json_output, &err_msg)) {
    call_data->finish_with_error(err_msg);
    LOG(ERROR) << "ProtoMessageToJson failed: " << err_msg;
    return;
  }
  LOG(INFO) << "ProtoMessageToJson: " << json_output;
  call_data->write_and_finish(json_output);
}
}  // namespace

void XllmHttpServiceImpl::get_serving_models(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (cntl) {
      cntl->SetFailed("brpc request | response | controller is null");
    }
    return;
  }
  if (!ensure_backend_ready(cntl)) {
    return;
  }
  auto call_data = std::make_shared<CompletionCallData>(
      cntl, false, done_guard.release(), nullptr, nullptr);

  auto service_request = std::make_shared<Request>();
  service_request->request_context = parse_request_context(*cntl);
  const ScheduleResult schedule_result = scheduler_->schedule(service_request);
  if (!schedule_result_ok(schedule_result)) {
    apply_schedule_failure(cntl, schedule_result);
    LOG(ERROR) << "Schedule request failed: " << schedule_result.message;
    return;
  }

  xllm::proto::ModelListRequest model_request;
  const bool dispatched =
      dispatcher_ != nullptr &&
      dispatcher_->dispatch_models(
          service_request->routing.prefill_endpoint,
          service_request->routing.prefill_incarnation,
          model_request,
          [call_data](const TransportResult& result,
                      const xllm::proto::ModelListResponse& model_response) {
            handle_get_model_response(call_data, result, model_response);
          });
  if (!dispatched) {
    call_data->finish_with_error("Backend dispatcher is unavailable");
  }
}

void XllmHttpServiceImpl::Completions(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (cntl) {
      cntl->SetFailed("brpc request | response | controller is null");
    }
    return;
  }
  if (!ensure_backend_ready(cntl)) {
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::CompletionRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::CompletionResponse>(
          arena);

  std::string attachment = std::move(cntl->request_attachment().to_string());
  std::string error;
  auto st = json2pb::JsonToProtoMessage(attachment, req_pb, &error);
  if (!st) {
    cntl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  auto service_request = generate_request(req_pb, *cntl, "/v1/completions");

  if (!req_pb->prompt().empty()) {
    service_request->prompt = req_pb->prompt();
    // select instance for request
    const ScheduleResult schedule_result =
        scheduler_->schedule(service_request);
    if (!schedule_result_ok(schedule_result)) {
      apply_schedule_failure(cntl, schedule_result);
      LOG(ERROR) << "Schedule request failed: " << schedule_result.message;
      return;
    }
  } else {
    cntl->SetFailed("Prompt is empty!");
    LOG(ERROR) << "Prompt is empty!";
    return;
  }

  // update request protobuf
  req_pb->set_service_request_id(service_request->service_request_id);
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_endpoint);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_endpoint);
  req_pb->mutable_routing()->set_prefill_incarnation(
      service_request->routing.prefill_incarnation);
  req_pb->mutable_routing()->set_decode_incarnation(
      service_request->routing.decode_incarnation);

  auto call_data = std::make_shared<CompletionCallData>(
      cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
  handle(call_data, service_request);
}

void XllmHttpServiceImpl::ChatCompletions(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (cntl) {
      cntl->SetFailed("brpc request | response | controller is null");
    }
    return;
  }
  if (!ensure_backend_ready(cntl)) {
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatResponse>(
          arena);

  const auto attachment_size = cntl->request_attachment().size();
  auto content_len = GetJsonContentLength(cntl, attachment_size);
  std::string attachment;
  cntl->request_attachment().copy_to(&attachment, content_len, 0);

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status =
      google::protobuf::util::JsonStringToMessage(attachment, req_pb, options);
  if (!status.ok()) {
    cntl->SetFailed(status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << status.ToString();
    return;
  }

  auto service_request =
      generate_request(req_pb, *cntl, "/v1/chat/completions");

  if (req_pb->messages_size() > 0) {
    service_request->messages.reserve(req_pb->messages_size());
    for (const auto& message : req_pb->messages()) {
      service_request->messages.emplace_back(message.role(), message.content());
    }
    if (req_pb->has_chat_template_kwargs()) {
      service_request->chat_template_kwargs =
          proto_struct_to_json(req_pb->chat_template_kwargs());
    }
    service_request->tools = parse_tools_from_proto(req_pb->tools());
    if (req_pb->has_tool_choice()) {
      service_request->tool_choice = req_pb->tool_choice();
    }

    const ScheduleResult schedule_result =
        scheduler_->schedule(service_request);
    if (!schedule_result_ok(schedule_result)) {
      apply_schedule_failure(cntl, schedule_result);
      LOG(ERROR) << "Schedule request failed: " << schedule_result.message;
      return;
    }
  } else {
    cntl->SetFailed("Messages is empty!");
    LOG(ERROR) << "Messages is empty!";
    return;
  }

  // update request protobuf
  req_pb->set_service_request_id(service_request->service_request_id);
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_endpoint);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_endpoint);
  req_pb->mutable_routing()->set_prefill_incarnation(
      service_request->routing.prefill_incarnation);
  req_pb->mutable_routing()->set_decode_incarnation(
      service_request->routing.decode_incarnation);

  auto call_data = std::make_shared<ChatCallData>(
      cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
  handle(call_data, service_request);
}

void XllmHttpServiceImpl::Embeddings(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (cntl) {
      cntl->SetFailed("brpc request | response | controller is null");
    }
    return;
  }

  cntl->SetFailed("not support Embeddings");
  return;
}

void XllmHttpServiceImpl::Models(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  get_serving_models(controller, request, response, done);
}

void XllmHttpServiceImpl::Metrics(::google::protobuf::RpcController* controller,
                                  const proto::HttpRequest* request,
                                  proto::HttpResponse* response,
                                  ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)
          ->SetFailed("brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  nlohmann::json summary = nlohmann::json::object();
  if (scheduler_ != nullptr) {
    summary = scheduler_->debug_summary();
  }
  if (dispatcher_ != nullptr) {
    const DispatcherStats stats = dispatcher_->stats();
    summary["dispatcher"] = {
        {"closed", stats.closed},
        {"inflight", stats.inflight},
        {"transport_failure_total", stats.transport_failure_total},
        {"stale_routing_decision_total", stats.stale_routing_decision_total},
        {"channel_count", stats.channel_count},
        {"endpoint_count", stats.endpoint_count}};
  }

  const RuntimeHealthSnapshot health = runtime_state_.health_snapshot();
  PrometheusMetricsSnapshot snapshot =
      build_prometheus_metrics_snapshot(summary,
                                        options_.service_name(),
                                        options_.block_size(),
                                        health.ready,
                                        runtime_phase_name(health.phase));
  cntl->http_response().set_content_type("text/plain; version=0.0.4");
  cntl->response_attachment().append(render_prometheus_metrics(snapshot));
}

void XllmHttpServiceImpl::DebugSummary(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)
          ->SetFailed("brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  nlohmann::json summary = nlohmann::json::object();
  if (scheduler_ != nullptr) {
    summary = scheduler_->debug_summary();
  } else {
    summary["service_name"] = options_.service_name();
  }
  if (dispatcher_ != nullptr) {
    const DispatcherStats stats = dispatcher_->stats();
    summary["dispatcher"] = {
        {"closed", stats.closed},
        {"inflight", stats.inflight},
        {"transport_failure_total", stats.transport_failure_total},
        {"stale_routing_decision_total", stats.stale_routing_decision_total},
        {"channel_count", stats.channel_count},
        {"endpoint_count", stats.endpoint_count}};
  }
  const RuntimeHealthSnapshot health = runtime_state_.health_snapshot();
  summary["live"] = health.live;
  summary["ready"] = health.ready;
  summary["readiness_reason"] = health.reason;
  summary["runtime_phase"] = runtime_phase_name(health.phase);
  cntl->http_response().set_content_type("application/json");
  cntl->response_attachment().append(summary.dump());
}

}  // namespace xllm_service
