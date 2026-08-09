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
#include <gtest/gtest.h>

namespace xllm_service {
namespace {

TEST(RequestContextTest, ParsesCanonicalLlmDHeaders) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-request-id", "request-observe-1");
  controller.http_request().SetHeader(
      "traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
  controller.http_request().SetHeader("x-maas-tenant-id", "retail-search");
  controller.http_request().SetHeader("x-maas-service-tier", "high");
  controller.http_request().SetHeader("x-maas-slo-class", "latency");
  controller.http_request().SetHeader("x-llm-d-inference-fairness-id",
                                      "tenant-a");
  controller.http_request().SetHeader("x-llm-d-inference-objective",
                                      "premium-traffic");
  controller.http_request().SetHeader("x-llm-d-model-name-rewrite", "qwen3");
  controller.http_request().SetHeader("x-llm-d-slo-ttft-ms", "275");
  controller.http_request().SetHeader("x-llm-d-slo-tpot-ms", "40");

  RequestContext context = parse_request_context(controller);

  EXPECT_TRUE(context.has_llm_d_context);
  EXPECT_EQ(context.request_id, "request-observe-1");
  EXPECT_EQ(context.traceparent,
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
  EXPECT_EQ(context.tenant_id, "retail-search");
  EXPECT_EQ(context.service_tier, "high");
  EXPECT_EQ(context.slo_class, "latency");
  EXPECT_EQ(context.inference_fairness_id, "tenant-a");
  EXPECT_EQ(context.inference_objective, "premium-traffic");
  EXPECT_EQ(context.model_name_rewrite, "qwen3");
  EXPECT_EQ(context.slo_ttft_ms, 275);
  EXPECT_EQ(context.slo_tpot_ms, 40);
}

TEST(RequestContextTest, CanonicalHeaderWinsOverAlias) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-gateway-inference-fairness-id",
                                      "legacy-tenant");
  controller.http_request().SetHeader("x-llm-d-inference-fairness-id",
                                      "tenant-a");
  controller.http_request().SetHeader("x-slo-ttft-ms", "1000");
  controller.http_request().SetHeader("x-llm-d-slo-ttft-ms", "300");

  RequestContext context = parse_request_context(controller);

  EXPECT_TRUE(context.has_llm_d_context);
  EXPECT_EQ(context.inference_fairness_id, "tenant-a");
  EXPECT_EQ(context.slo_ttft_ms, 300);
}

TEST(RequestContextTest, MissingHeadersUseCompatibilityDefaults) {
  brpc::Controller controller;

  RequestContext context = parse_request_context(controller);

  EXPECT_FALSE(context.has_llm_d_context);
  EXPECT_EQ(context.inference_fairness_id, "default-flow");
  EXPECT_EQ(context.slo_ttft_ms, -1);
  EXPECT_EQ(context.slo_tpot_ms, -1);
}

TEST(RequestContextTest, InvalidSloHeadersAreUnset) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-llm-d-slo-ttft-ms", "abc");
  controller.http_request().SetHeader("x-llm-d-slo-tpot-ms", "-1");

  RequestContext context = parse_request_context(controller);

  EXPECT_TRUE(context.has_llm_d_context);
  EXPECT_EQ(context.slo_ttft_ms, -1);
  EXPECT_EQ(context.slo_tpot_ms, -1);
}

TEST(RequestContextTest, ModelNameRewriteOverridesRequestBodyModel) {
  RequestContext context;
  context.model_name_rewrite = "backend-model";

  EXPECT_EQ(resolve_effective_model_name("public-model", context),
            "backend-model");

  context.model_name_rewrite.clear();
  EXPECT_EQ(resolve_effective_model_name("public-model", context),
            "public-model");
}

TEST(RequestContextTest, ParsesVersionedExternalPdDirective) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-llm-d-routing-decision-version", "1");
  controller.http_request().SetHeader("x-llm-d-prefill-endpoint",
                                      "prefill:8000");
  controller.http_request().SetHeader("x-llm-d-decode-endpoint", "decode:8000");
  controller.http_request().SetHeader("x-llm-d-routing-attempt", "2");

  const RequestContext context = parse_request_context(controller);

  EXPECT_TRUE(context.has_llm_d_context);
  EXPECT_TRUE(context.external_routing.present);
  EXPECT_TRUE(context.external_routing.valid);
  EXPECT_EQ(context.external_routing.version, 1);
  EXPECT_EQ(context.external_routing.prefill_endpoint, "prefill:8000");
  EXPECT_EQ(context.external_routing.decode_endpoint, "decode:8000");
  EXPECT_EQ(context.external_routing.attempt, 2);
}

TEST(RequestContextTest, RejectsMalformedExternalPdDirectiveNumbers) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-llm-d-routing-decision-version",
                                      "latest");
  controller.http_request().SetHeader("x-llm-d-prefill-endpoint",
                                      "prefill:8000");
  controller.http_request().SetHeader("x-llm-d-decode-endpoint", "decode:8000");
  controller.http_request().SetHeader("x-llm-d-routing-attempt", "-1");

  const RequestContext context = parse_request_context(controller);

  EXPECT_TRUE(context.external_routing.present);
  EXPECT_FALSE(context.external_routing.valid);
  EXPECT_FALSE(context.external_routing.error.empty());
}

}  // namespace
}  // namespace xllm_service
