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

#include <cstdint>
#include <string>

#include "routing/routing_configuration.h"

namespace brpc {
class Controller;
}  // namespace brpc

namespace xllm_service {

struct RequestContext {
  bool has_llm_d_context = false;
  std::string request_id;
  std::string traceparent;
  std::string tenant_id;
  std::string service_tier;
  std::string slo_class;
  std::string inference_fairness_id = "default-flow";
  std::string inference_objective;
  std::string model_name_rewrite;
  int64_t slo_ttft_ms = -1;
  int64_t slo_tpot_ms = -1;
  ExternalRoutingDirective external_routing;
};

RequestContext parse_request_context(const brpc::Controller& controller);
std::string resolve_effective_model_name(const std::string& request_model,
                                         const RequestContext& context);

}  // namespace xllm_service
