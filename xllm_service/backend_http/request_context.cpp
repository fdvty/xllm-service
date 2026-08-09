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

#include "backend_http/request_context.h"

#include <brpc/controller.h>

#include <charconv>
#include <cstddef>
#include <exception>
#include <system_error>
#include <utility>

namespace xllm_service {
namespace {

constexpr char kFairnessIdHeader[] = "x-llm-d-inference-fairness-id";
constexpr char kFairnessIdAlias[] = "x-gateway-inference-fairness-id";
constexpr char kInferenceObjectiveHeader[] = "x-llm-d-inference-objective";
constexpr char kInferenceObjectiveAlias[] = "x-gateway-inference-objective";
constexpr char kModelNameRewriteHeader[] = "x-llm-d-model-name-rewrite";
constexpr char kModelNameRewriteAlias[] = "x-gateway-model-name-rewrite";
constexpr char kSloTtftMsHeader[] = "x-llm-d-slo-ttft-ms";
constexpr char kSloTtftMsAlias[] = "x-slo-ttft-ms";
constexpr char kSloTpotMsHeader[] = "x-llm-d-slo-tpot-ms";
constexpr char kSloTpotMsAlias[] = "x-slo-tpot-ms";
constexpr char kRoutingVersionHeader[] = "x-llm-d-routing-decision-version";
constexpr char kPrefillEndpointHeader[] = "x-llm-d-prefill-endpoint";
constexpr char kDecodeEndpointHeader[] = "x-llm-d-decode-endpoint";
constexpr char kRoutingAttemptHeader[] = "x-llm-d-routing-attempt";
constexpr char kRequestIdHeader[] = "x-request-id";
constexpr char kTraceparentHeader[] = "traceparent";
constexpr char kTenantIdHeader[] = "x-maas-tenant-id";
constexpr char kServiceTierHeader[] = "x-maas-service-tier";
constexpr char kSloClassHeader[] = "x-maas-slo-class";

const std::string* get_header(const brpc::Controller& controller,
                              const char* header,
                              const char* alias,
                              bool* found) {
  const std::string* value = controller.http_request().GetHeader(header);
  if (value != nullptr) {
    *found = true;
    return value;
  }

  value = controller.http_request().GetHeader(alias);
  if (value != nullptr) {
    *found = true;
    return value;
  }
  return nullptr;
}

std::string get_string_header(const brpc::Controller& controller,
                              const char* header,
                              const char* alias,
                              bool* found) {
  const std::string* value = get_header(controller, header, alias, found);
  if (value == nullptr) {
    return "";
  }
  return *value;
}

int64_t get_non_negative_int_header(const brpc::Controller& controller,
                                    const char* header,
                                    const char* alias,
                                    bool* found) {
  const std::string* value = get_header(controller, header, alias, found);
  if (value == nullptr || value->empty()) {
    return -1;
  }

  try {
    size_t parsed_size = 0;
    int64_t parsed_value = std::stoll(*value, &parsed_size, 10);
    if (parsed_size != value->size() || parsed_value < 0) {
      return -1;
    }
    return parsed_value;
  } catch (const std::exception&) {
    return -1;
  }
}

const std::string* get_exact_header(const brpc::Controller& controller,
                                    const char* header,
                                    bool* found) {
  const std::string* value = controller.http_request().GetHeader(header);
  if (value != nullptr) {
    *found = true;
  }
  return value;
}

std::string get_exact_string_header(const brpc::Controller& controller,
                                    const char* header,
                                    bool* found) {
  const std::string* value = get_exact_header(controller, header, found);
  return value == nullptr ? "" : *value;
}

bool parse_uint32_header(const std::string* value, uint32_t* output) {
  if (value == nullptr || value->empty() || output == nullptr) {
    return false;
  }
  const char* begin = value->data();
  const char* end = begin + value->size();
  const std::from_chars_result result = std::from_chars(begin, end, *output);
  return result.ec == std::errc() && result.ptr == end;
}

void invalidate_directive(ExternalRoutingDirective* directive,
                          const std::string& error) {
  if (directive->valid) {
    directive->valid = false;
    directive->error = error;
  }
}

}  // namespace

RequestContext parse_request_context(const brpc::Controller& controller) {
  RequestContext context;
  bool found = false;

  context.request_id =
      get_exact_string_header(controller, kRequestIdHeader, &found);
  context.traceparent =
      get_exact_string_header(controller, kTraceparentHeader, &found);
  context.tenant_id =
      get_exact_string_header(controller, kTenantIdHeader, &found);
  context.service_tier =
      get_exact_string_header(controller, kServiceTierHeader, &found);
  context.slo_class =
      get_exact_string_header(controller, kSloClassHeader, &found);

  std::string fairness_id = get_string_header(
      controller, kFairnessIdHeader, kFairnessIdAlias, &found);
  if (!fairness_id.empty()) {
    context.inference_fairness_id = std::move(fairness_id);
  }

  context.inference_objective = get_string_header(
      controller, kInferenceObjectiveHeader, kInferenceObjectiveAlias, &found);
  context.model_name_rewrite = get_string_header(
      controller, kModelNameRewriteHeader, kModelNameRewriteAlias, &found);
  context.slo_ttft_ms = get_non_negative_int_header(
      controller, kSloTtftMsHeader, kSloTtftMsAlias, &found);
  context.slo_tpot_ms = get_non_negative_int_header(
      controller, kSloTpotMsHeader, kSloTpotMsAlias, &found);

  bool has_routing_version = false;
  bool has_prefill_endpoint = false;
  bool has_decode_endpoint = false;
  bool has_routing_attempt = false;
  const std::string* routing_version =
      get_exact_header(controller, kRoutingVersionHeader, &has_routing_version);
  const std::string* prefill_endpoint = get_exact_header(
      controller, kPrefillEndpointHeader, &has_prefill_endpoint);
  const std::string* decode_endpoint =
      get_exact_header(controller, kDecodeEndpointHeader, &has_decode_endpoint);
  const std::string* routing_attempt =
      get_exact_header(controller, kRoutingAttemptHeader, &has_routing_attempt);

  context.external_routing.present = has_routing_version ||
                                     has_prefill_endpoint ||
                                     has_decode_endpoint || has_routing_attempt;
  found = found || context.external_routing.present;
  if (context.external_routing.present) {
    if (!parse_uint32_header(routing_version,
                             &context.external_routing.version)) {
      invalidate_directive(&context.external_routing,
                           "Routing decision version header is missing or "
                           "invalid");
    }
    if (prefill_endpoint != nullptr) {
      context.external_routing.prefill_endpoint = *prefill_endpoint;
    }
    if (decode_endpoint != nullptr) {
      context.external_routing.decode_endpoint = *decode_endpoint;
    }
    if (has_routing_attempt &&
        !parse_uint32_header(routing_attempt,
                             &context.external_routing.attempt)) {
      invalidate_directive(&context.external_routing,
                           "Routing attempt header is invalid");
    }
  }
  context.has_llm_d_context = found;
  return context;
}

std::string resolve_effective_model_name(const std::string& request_model,
                                         const RequestContext& context) {
  if (!context.model_name_rewrite.empty()) {
    return context.model_name_rewrite;
  }
  return request_model;
}

}  // namespace xllm_service
