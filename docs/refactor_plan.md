# xllm-service 收缩重构方案

> 总体方案见 [maas_llmd_development_plan.md](./maas_llmd_development_plan.md)。
>
> 本文只描述 xllm-service 仓库内的代码重构和迁移步骤。

## 一、重构定位

xllm-service 不再以“自研完整 MaaS 调度平台”为最终目标，而是逐步收缩为
**xLLM Backend Adapter + Runtime Control**：

- llm-d 负责通用的流量入口、队列、租户公平性、Admission、SLO/KV/负载感知
  调度和 Endpoint 选择。
- Kubernetes + InferencePool 负责实例发现、健康成员关系和通用副本管理。
- xllm-service 保留 xLLM 特有的协议适配、请求执行、P/D Link/Unlink、KV 传输、
  结果回传、取消、迁移和运行时控制。
- 现有 Scheduler/InstanceMgr/GlobalKVCacheMgr 在兼容期继续工作，完成切流后删除其
  通用平台职责。

这次重构使用渐进迁移，不进行一次性 ground-up rewrite，也不在 C++ 中重新实现
llm-d Router。

## 二、当前问题

### 2.1 状态所有权不清晰

同一个实例的状态分散在：

- `instances_`
- `prefill_index_` / `decode_index_`
- `cached_channels_`
- `load_metrics_`
- `latency_metrics_`
- `request_metrics_`
- `inflight_request_counts_`
- `time_predictors_`
- `GlobalKVCacheMgr::kvcache_infos_`

一次实例注册、删除或 incarnation 变化必须跨多张表更新，导致 `cluster_mutex_`、
`metrics_mutex_`、`kvcache_mutex_` 和 `update_mutex_` 形成手工维护的锁顺序。

### 2.2 模块双向依赖

`Scheduler` 拥有 `InstanceMgr`，`InstanceMgr` 又通过裸指针回调 `Scheduler` 清理请求、
KV cache 和订阅源：

```text
Scheduler -> InstanceMgr -> Scheduler
```

模块接口无法表达完整副作用，也无法局部判断一次调用会获取哪些锁。

### 2.3 Scheduler 职责过载

当前 Scheduler 同时负责：

- tokenizer 和 chat template
- etcd 注册、watch 和 master/peer 行为
- 实例发现和 channel
- KV cache 订阅和索引
- RR/CAR/SLO 调度
- 请求注册、输出排序和完成清理
- 实例故障时的请求取消
- 指标更新和 debug summary

这些职责必须拆开，且大部分通用平台职责最终交给 llm-d/Kubernetes。

### 2.4 锁覆盖慢操作和外部回调

当前存在以下高风险模式：

- 持有 `request_mutex_` 调用客户端 `output_callback`。
- 持有 `update_mutex_` 调用 etcd 网络操作。
- KV match 在持有共享锁期间处理完整 prompt block。
- 一个请求的状态由 `requests_` 和 `remote_requests_output_thread_map_` 两把锁分别保护。
- 公共/私有函数通过注释要求调用方提前持有某把锁。

重构目标不是“没有锁”，而是锁只存在于单一 owner 的私有实现中，不跨模块和 I/O。

## 三、目标边界

### 3.1 llm-d/Kubernetes 接管的职责

- 外部请求路由和 Endpoint 选择。
- 租户 flow、优先级、公平性和 Admission。
- RR、负载、SLO、KV affinity 等调度策略。
- Kubernetes Endpoint 发现和 readiness。
- 全局 prefix-to-endpoint KV 目录。
- HPA/KEDA 或自定义 autoscaler 的副本决策。

这些能力不在新 xllm-service 中维护第二份权威状态。

### 3.2 xllm-service 最终保留的职责

- OpenAI/backend 请求到 xLLM brpc/protobuf 的适配。
- 单请求执行状态、输出顺序、取消和失败分类。
- brpc ChannelPool 和 Dispatcher。
- xLLM P/D Link/Unlink、KV 传输和回调协议。
- xLLM KV event 到 llm-d event contract 的转换。
- xLLM engine health、metrics、drain、warmup 和 role control。
- 首 token 前重试和受支持请求的中途迁移。

### 3.3 兼容期保留的职责

在 llm-d 原生调度完成前，legacy mode 继续支持：

- etcd 实例发现。
- RR/CAR/SLO 调度。
- 现有 aggregated 和 P/D 路由。
- 本地全局 KV cache index。
- peer-service 行为。

兼容逻辑必须放在明确的 `legacy/` 边界后，不允许新代码继续依赖 legacy 内部容器。

### 3.4 最终部署单元

收缩后的仓库产生两个二进制，而不是继续运行一个隐藏全部 worker 的中心服务：

1. `xllm-backend-adapter`：与一个可路由 xLLM engine group 共置或直接部署在其前面，
   作为 llm-d InferencePool Endpoint。它拥有该 Endpoint 接收请求的 HTTP stream、
   RequestSession 和本地 Dispatcher。
2. `xllm-runtime-controller`：不进入 token 热路径，负责 P/D Link/Unlink reconcile、
   role change、warmup、model-ready 和 runtime lifecycle。

P/D 请求由 llm-d 选择 prefill Endpoint 和兼容 decode partner。prefill 侧 adapter 保持
HTTP stream owner，并通过版本化 RoutingDecision 调用现有 xLLM P/D 协议；decode 输出
沿现有 callback 语义返回该 adapter。

中心化 xllm-service 只存在于 compatibility mode。否则它会隐藏各个推理 Endpoint，
使 llm-d 无法完成 per-instance 负载/KV/SLO 调度。

## 四、状态所有权

| 状态 | 迁移期 Owner | 最终 Owner |
| --- | --- | --- |
| 实例拓扑和 incarnation | LegacyClusterState | Kubernetes + InferencePool |
| 调度候选和评分 | LegacyRoutingEngine | llm-d EPP |
| 全局 KV prefix 目录 | LegacyCacheIndex | llm-d KV data layer/indexer |
| 单请求执行状态 | RequestSession | RequestSession |
| 请求到输出执行器的映射 | RequestSessionRegistry | RequestSessionRegistry |
| brpc channels | ChannelPool | ChannelPool |
| transport retry | Dispatcher | Dispatcher |
| xLLM P/D 链接状态 | Legacy InstanceMgr | xLLM Runtime Controller / PdCoordinator |
| xLLM 物理 KV 状态 | xLLM engine | xLLM engine |
| adapter 本地指标 | Telemetry | Telemetry |

约束：

1. 每份可变状态只有一个 writer。
2. 模块之间只传命令、事件、值对象或不可变 snapshot。
3. 观察副本可以重复，权威副本不能重复。
4. incarnation ID 和 event sequence 必须参与异步事件判定。

## 五、目标模块

```text
xllm_service/
├── backend_http/
│   ├── openai_service.{h,cpp}       # 内部 backend API，不做全局调度
│   └── request_context.{h,cpp}      # 可信请求上下文读取
├── execution/
│   ├── request_session.{h,cpp}      # 单请求状态机和输出串行化
│   ├── request_registry.{h,cpp}     # 分片索引，不保存调度状态
│   └── migration.{h,cpp}            # retry/token replay 策略
├── dispatcher/
│   ├── dispatcher.{h,cpp}           # 请求发送、取消和 transport 结果
│   └── channel_pool.{h,cpp}         # brpc channel 生命周期
├── runtime/
│   ├── runtime_client.h             # xLLM runtime 接口
│   ├── runtime_state.{h,cpp}        # adapter 本地状态
│   └── lifecycle.{h,cpp}            # drain、warmup、role control
├── pd/
│   ├── pd_coordinator.{h,cpp}       # controller 侧 Link/Unlink reconcile
│   ├── pd_executor.{h,cpp}          # endpoint 侧 P/D 执行协议
│   └── pd_contract.h                # llm-d 选择结果到 xLLM 的契约
├── kv/
│   ├── kv_event_adapter.{h,cpp}     # xLLM event -> llm-d contract
│   └── kv_event_publisher.{h,cpp}
├── telemetry/
│   ├── health.{h,cpp}
│   └── metrics.{h,cpp}
├── controller/
│   ├── runtime_controller.{h,cpp}   # out-of-path runtime reconcile
│   └── warm_pool.{h,cpp}            # warmup、model preload 和 ready state
├── legacy/
│   ├── legacy_service.{h,cpp}
│   ├── legacy_routing_engine.{h,cpp}
│   ├── legacy_cluster_state.{h,cpp}
│   └── legacy_cache_index.{h,cpp}
└── apps/
    ├── backend_adapter_main.cpp
    └── runtime_controller_main.cpp
```

迁移期可以通过 facade 包装现有文件，不要求第一步就移动所有源文件。

目标依赖方向：

```text
backend_http -> execution -> runtime interfaces
execution    -> dispatcher -> transport
pd executor  -> dispatcher + runtime interfaces
controller   -> pd coordinator + runtime interfaces
kv           -> runtime interfaces
telemetry    -> runtime interfaces
legacy       -> 上述公开接口
```

底层模块不得反向持有 `Scheduler*`、HTTP service 或 orchestration parent。

## 六、并发模型

### 6.1 请求状态

每个请求由一个 `RequestSession` 独占状态：

```text
Created -> Dispatched -> PrefillRunning -> DecodeRunning
        -> Completed | Failed | Cancelled | Migrating
```

同一请求的 generation、disconnect、timeout、instance failure、cancel 和 finish 事件进入
同一串行执行上下文。这样可以删除当前独立的 request map/thread map 协调逻辑。

`RequestSessionRegistry` 只负责 `request_id -> session` 查找，不参与请求状态迁移。

### 6.2 Transport

`Dispatcher` 负责：

- 从不可变 RoutingDecision 构建请求。
- 获取 ChannelPool 中的连接。
- 发送 brpc 请求。
- 将 transport success/failure 转换成 RequestSession event。
- 维护 adapter 执行层需要的 inflight 指标。

它不选择实例，也不修改集群拓扑。

### 6.3 控制状态

现有 `ClusterStateActor` 和 `ClusterSnapshot` 可以作为 legacy 兼容期的单写者状态容器，
但约束如下：

- 不扩展成新的 MaaS 全局控制面。
- 不重新实现 llm-d 的 Flow Control、EPP 或 autoscaler。
- migrated traffic 不读取 legacy snapshot 做第二次调度。
- Kubernetes/InferencePool 接管发现后删除对应 legacy state。

现有 `InflightTable` 在兼容期用于修正心跳延迟；最终调度 inflight 由 llm-d 拥有，
adapter 只保留请求执行和 transport 指标。

### 6.4 锁规则

- 锁只能保护当前模块的私有状态。
- 公共接口不得要求调用方持锁。
- 不得持锁调用 RPC、etcd、ZMQ、Kubernetes、磁盘、callback 或 plugin。
- 多份状态的一致变更由一个 owner 内的事件状态机完成，不通过多锁事务完成。
- 读多写少的控制状态使用不可变 snapshot。
- Channel 初始化、Link/Unlink 和客户端输出均在锁外执行。

## 七、核心接口

```cpp
struct RoutingDecision {
  std::string prefill_endpoint;
  std::string decode_endpoint;
  std::string prefill_incarnation;
  std::string decode_incarnation;
  uint32_t attempt = 0;
};

class RequestSession {
 public:
  void on_dispatched(const RoutingDecision& decision);
  void on_generation(RequestOutput output);
  void on_transport_failure(TransportError error);
  void on_instance_failure(InstanceFailure failure);
  void on_client_disconnect();
  void cancel(CancelReason reason);
};

class Dispatcher {
 public:
  void dispatch(std::shared_ptr<RequestSession> session,
                const RoutingDecision& decision,
                BackendRequest request);
  void cancel(const RequestId& request_id, const RoutingDecision& decision);
};

class PdCoordinator {
 public:
  Status link(const PdEndpoint& prefill, const PdEndpoint& decode);
  Status unlink(const PdEndpoint& prefill, const PdEndpoint& decode);
};

class PdExecutor {
 public:
  Status validate(const RoutingDecision& decision) const;
};
```

legacy routing 和 llm-d routing 只负责产生相同的 `RoutingDecision`，执行层不关心来源。

## 八、分阶段重构

### R0：冻结行为和建立护栏

目标：固定当前 aggregated/P/D 行为，停止继续扩大 Scheduler。

- 补齐现有 RR、CAR、SLO、P/D、peer service 和故障清理基线测试。
- 增加锁等待、请求表大小、channel 数、callback 时延和 KV index 更新时延指标。
- 建立规则：新平台能力不得直接加入 Scheduler/InstanceMgr。
- 保留当前 `core/` 原型和测试，不直接切换生产读写路径。

退出条件：现有 `phase1_tests/` 可重复运行，并能对比重构前后行为。

### R1：建立 facade 和单向依赖

目标：在不改变行为的前提下切断双向依赖。

- 用接口封装现有实例发现、routing、cache index 和 transport。
- `InstanceMgr` 不再保存或调用 `Scheduler*`。
- 实例删除产生 `InstanceRemoved` 事件，由订阅者分别处理：
  - RequestSessionRegistry 处理关联请求。
  - KvEventAdapter/legacy cache 处理 cache。
  - ChannelPool 处理连接。
  - PdCoordinator 处理 Link/Unlink。
- 所有外部调用先在锁内构造 action list，再在锁外执行。

退出条件：源码中不存在 `InstanceMgr -> Scheduler` 回调。

### R2：RequestSession 和 Dispatcher

目标：解决请求生命周期由多张表和多把锁共同维护的问题。

- 引入 `RequestSession` 状态机。
- 合并 `requests_` 与 `remote_requests_output_thread_map_` 的生命周期所有权。
- 以 per-request strand/serial executor 保证 token 顺序。
- 将 brpc channel 和发送逻辑从 HTTP service/Scheduler 移到 Dispatcher。
- 客户端 callback、finish 和 cancel 全部在 registry 锁外执行。
- 明确 first-token-before/after failure 分类。

退出条件：Scheduler 不再拥有请求表和 128 个固定 output thread pools。

### R3：LegacyClusterState 单写者

目标：在兼容期集中实例、指标和 incarnation 变化，降低旧路径的锁复杂度。

- 将 etcd watch、heartbeat 和 reconcile 转成 typed events。
- 使用现有 ClusterStateActor 串行应用兼容期状态变更。
- 发布不可变 snapshot 给 LegacyRoutingEngine。
- channel、P/D side effects 和 callback 不进入 actor 状态修改逻辑。
- KV 高频事件优先转发给外部 index；只有 legacy CAR 必需的状态留在本地。

退出条件：InstanceMgr 中的 topology/metrics 多锁状态被 facade/actor 替代，旧类可删除或
仅作为协议兼容包装。

### R4：llm-d Compatibility Mode

目标：llm-d 接管外部流控，但不改变现有 P/D 执行。

- xllm-service 作为 llm-d InferencePool endpoint。
- 接收可信 tenant/SLO/request context。
- 旧 Scheduler 继续产生 P/D RoutingDecision。
- 输出 goodput、SLO、queue 和失败分类指标。
- 提供 `routing_mode=legacy` 明确标识临时权威来源。

退出条件：llm-d 可安全前置和回滚，现有 P/D 无功能回归。

### R5：外部 RoutingDecision

目标：执行层接受 llm-d 的选择结果，不再读取本地全局状态重新选择实例。

- 增加 `routing_mode=external`。
- 定义并版本化 aggregated/P/D RoutingDecision contract。
- 校验 endpoint incarnation、模型/config digest、block size 和 P/D compatibility。
- 对 stale decision 返回明确的 retryable error。
- aggregated 流量首先切到 external routing。
- P/D 保持 legacy 与 external shadow compare，随后切换。

退出条件：migrated traffic 对 LegacyRoutingEngine、InstanceMgr 候选集和
GlobalKVCacheMgr match 零依赖。

### R6：KV Event Adapter 和 llm-d P/D

目标：迁移 CAR 和 P/D 候选评分，同时保留 xLLM 执行协议。

- xLLM KV delta/snapshot 转换成 llm-d contract。
- 支持 seq gap、snapshot rebuild 和 incarnation reset。
- llm-d 负责 prefix affinity、负载、SLO 和拓扑评分。
- PdExecutor 验证并执行 llm-d 选择的 pair。
- Runtime Controller/PdCoordinator reconcile Link/Unlink；KV transfer 和 decode
  response callback 保留在 xLLM 层。

退出条件：关闭 legacy cache index 和 P/D selector 后，完整 P/D 测试通过。

### R7：请求迁移和运行时控制

目标：补齐 adapter 应拥有的差异化能力。

- 首 token 前 transport failure 自动重试。
- RequestSession 保存已确认输出 token，支持受限的 token replay migration。
- 增加 drain、cancel、warmup、model-ready 和 role-control 接口。
- 对不支持迁移的 sampling/structured-output 模式显式 fail fast。
- 故障和迁移全部受 retry budget、deadline 和 SLO 约束。

退出条件：支持范围内的迁移不丢失或重复客户端已接收 token。

### R8：删除 Legacy

目标：完成收缩，不在当前二进制中长期维护两套平台。

- 删除 Scheduler 通用编排职责。
- 删除 LoadBalancePolicy、InstanceMgr 全局 routing state 和本地 CAR match。
- 删除 etcd master/peer service request-path dependency。
- 删除重复的外部 OpenAI gateway 职责。
- 将回滚方式改成部署旧版本，而不是保留 dead path。

最终保留：RequestSession、Dispatcher、ChannelPool、PdExecutor、PdCoordinator、
Runtime Controller、KV Event Adapter、Telemetry、Runtime Control。

## 九、迁移开关

迁移期开关必须表达单一权威来源：

```text
routing_mode: legacy | external
discovery_mode: etcd | kubernetes
kv_index_mode: local | external
request_execution_mode: legacy | session
```

不允许同一个请求同时由 legacy 和 external routing 执行。Shadow mode 只能记录第二份
decision，不能产生第二次副作用。

开关删除顺序：

1. 删除 `request_execution_mode=legacy`。
2. 删除 `routing_mode=legacy`。
3. 删除 `kv_index_mode=local`。
4. 删除 `discovery_mode=etcd`。

## 十、测试与验收

### 10.1 单元测试

- RequestSession 的所有状态迁移和重复事件。
- finish/cancel/failure/client disconnect 并发竞态。
- ChannelPool incarnation replacement。
- InstanceRemoved 事件幂等性。
- KV seq gap、snapshot、重复和乱序。
- P/D compatibility 和 Link/Unlink action 生成。

### 10.2 对拍测试

- legacy 与 external RR 决策。
- legacy CAR 与 llm-d KV scorer。
- legacy SLO 与 llm-d SLO policy。
- legacy 与 llm-d P/D pair decision。

Shadow decision 不发送请求，只记录差异及原因。

### 10.3 并发和故障测试

- 多路 generation 与实例删除并发。
- callback 阻塞不阻塞 registry 和其他请求。
- etcd/Kubernetes watch 重连。
- ZMQ 丢包和 publisher 重启。
- prefill/decode crash、重启和 incarnation 变化。
- EPP/adapter 滚动升级和 active request drain。

### 10.4 性能门槛

- 请求热路径不读取全局可变 map。
- callback、RPC、watch 和 KV update 不形成长临界区。
- adapter CPU、内存和排队延迟有独立指标。
- weighted goodput 不低于迁移前基线，并满足约定的尾延迟回归预算。

## 十一、代码审查清单

- 这份可变状态的唯一 owner 是谁？
- 是否在另一个模块维护了同一份权威状态？
- 公共 API 是否泄漏锁或内部容器？
- 是否持锁执行了 I/O、callback 或 plugin？
- lower-level 模块是否反向调用 parent/orchestrator？
- 异步事件是否包含 incarnation/sequence 并且可幂等处理？
- failure path 是否只完成一次 RequestSession？
- legacy 和 external mode 是否可能同时产生副作用？
- 新功能应落在 llm-d、adapter、xLLM engine 还是 runtime controller？

## 十二、近期执行顺序

1. 完成 R0 基线和 contracts。
2. 完成 R1，消除 `InstanceMgr -> Scheduler`。
3. 完成 R2，将请求生命周期迁入 RequestSession。
4. 将现有 `core/` actor 接入 R3 的 legacy facade，而不是新 Router。
5. 并行部署 llm-d compatibility mode。
6. 先完成 aggregated external routing，再迁移 KV 和 P/D decision。
7. 完成 legacy 删除后，再投入 request migration 和预测 autoscaling 的复杂算法。
