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

#include "metrics.h"

DEFINE_COUNTER(server_request_in_total,
               "Total number of request that server received");

DEFINE_GAUGE(peer_service_enabled,
             "Whether peer-service rollout mode is enabled");
DEFINE_GAUGE(xservice_instance_view_size,
             "Number of instances in the local xllm-service view");
DEFINE_GAUGE(xservice_load_metrics_size,
             "Number of load metrics entries in the local xllm-service view");
DEFINE_GAUGE(xservice_kvcache_index_size,
             "Number of KV cache index entries in the local xllm-service view");
DEFINE_GAUGE(kv_event_zmq_source_count,
             "Number of connected KV event ZMQ sources");
DEFINE_COUNTER(xservice_heartbeat_total,
               "Total number of xllm instance heartbeats received");
DEFINE_COUNTER(xservice_heartbeat_xtensor_total,
               "Total number of xllm instance heartbeats with XTensor info");
DEFINE_GAUGE(xservice_registration_healthy,
             "Whether the current xllm-service registration lease is owned");
DEFINE_COUNTER(xservice_registration_reconcile_total,
               "Total number of xllm-service registration reconcile cycles");
DEFINE_COUNTER(xservice_registration_attempt_total,
               "Total number of xllm-service registration create attempts");
DEFINE_COUNTER(xservice_registration_success_total,
               "Total number of successful xllm-service registrations");
DEFINE_COUNTER(xservice_registration_recovery_total,
               "Total number of recovered xllm-service registrations");
DEFINE_COUNTER(xservice_registration_failure_total,
               "Total number of failed xllm-service registration checks");
DEFINE_COUNTER(kv_event_zmq_received_total,
               "Total number of KV cache events received through ZMQ");
DEFINE_COUNTER(kv_event_zmq_snapshot_received_total,
               "Total number of KV cache snapshots received through ZMQ");
DEFINE_COUNTER(kv_event_zmq_parse_failure_total,
               "Total number of failed KV cache event ZMQ parses");
DEFINE_COUNTER(kv_event_zmq_stale_total,
               "Total number of stale KV cache event ZMQ messages");
DEFINE_COUNTER(kv_event_zmq_gap_total,
               "Total number of KV cache event ZMQ sequence gaps");

// ttft latency histogram
DEFINE_HISTOGRAM(time_to_first_token_latency_milliseconds,
                 "Histogram of time to first token latency in milliseconds");
// inter token latency histogram
DEFINE_HISTOGRAM(inter_token_latency_milliseconds,
                 "Histogram of inter token latency in milliseconds");
