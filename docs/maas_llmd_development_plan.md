# 京东零售大模型 MaaS 推理平台开发计划

> 文档定位：技术路线汇报和阶段开发计划。
>
> 场景：一种基础模型、大量相同实例、多个业务和租户、不同 SLO 优先级。

## 一、汇报结论

建议采用：

```text
llm-d 做通用 MaaS 平台
xllm-service 收缩为 xLLM 适配和运行时控制
Dynamo 作为容错、Planner 和快速启动的参考
```

具体分工：

- **llm-d**：负责多租户队列、优先级、公平性、SLO/KV/负载调度、实例发现和
  自动扩缩容框架。
- **xllm-service**：保留 xLLM 特有的 P/D、KV 传输、brpc、NPU 适配、请求执行和
  运行时控制能力。
- **Dynamo**：重点借鉴请求迁移、故障恢复、Planner 和模型快速启动设计，不直接
  替换当前 xLLM 运行时。

这不是推倒重写。现有 P/D、KV 和 peer-service 能力继续保留，先兼容接入 llm-d，
再逐步把通用平台能力迁移出去。

## 二、业务目标

平台服务同一个基础模型，但要同时承接不同业务、租户、用户和 SLO 等级。核心目标
不只是提高总吞吐，而是提高 **Goodput**：

> 在满足 TTFT、TPOT 等 SLO 的前提下，每秒实际完成的有效 token 数量。

对于不同业务，可进一步计算加权 Goodput：

```text
加权 Goodput = 业务优先级权重 x 满足 SLO 的有效 token 数 / 时间
```

平台需要逐步具备：

- 多租户排队、优先级和公平调度。
- SLO-aware 和 KV-aware 请求调度。
- P/D 分离及独立扩缩容。
- 请求重试、实例故障恢复和部分请求中途迁移。
- 模型预加载、实例预热和快速扩容。

## 三、xllm-service 现状判断

### 3.1 已有基础

xllm-service 已经具备较好的 xLLM 集群服务能力：

- 支持 aggregated 和 P/D 分离部署，以及 Prefill、Decode、Mix 角色。
- 支持 P/D 选择、Link/Unlink 和 Decode 结果回传。
- 支持 RR、Cache-Aware Routing 和基础 SLO-Aware Routing。
- 支持 etcd 发现、心跳、incarnation、故障实例摘除。
- 支持 KV cache 增量事件、snapshot 和 peer-service。

这些能力，特别是 xLLM P/D、KV 传输、NPU 运行时和底层引擎适配，需要继续保留。

### 3.2 当前局限

| 当前问题 | 对平台建设的影响 |
| --- | --- |
| 缺少租户、业务、优先级和每请求 SLO | 无法直接支撑零售多业务统一调度 |
| TTFT/TPOT 主要是全局配置 | 无法同时保障多种 SLO 等级 |
| Scheduler 同时负责入口、发现、调度、KV、请求和响应 | 模块耦合高，继续加功能风险大 |
| 实例、指标、channel、KV 和请求状态分散 | 需要大量锁维护一致性，容易出现竞态和尾延迟 |
| InstanceMgr 反向调用 Scheduler | 模块边界和状态 Owner 不清晰 |
| 实例失败后主要取消关联请求 | 缺少真正的请求 failover 和中途恢复 |
| 缺少成熟的 Kubernetes Operator 和扩缩容闭环 | 需要自行建设完整平台控制面 |

结论是：xllm-service 适合继续承担 xLLM 特有能力，但不适合在当前 Scheduler 上继续
堆叠完整 MaaS 平台功能。否则短期看改动直接，长期会持续增加锁、共享状态和维护成本。

## 四、为什么选择 llm-d

### 4.1 三种路线对比

| 路线 | 优点 | 主要问题 | 建议 |
| --- | --- | --- | --- |
| 继续独立完善 xllm-service | 最熟悉 xLLM，短期改动直接 | 需要自研多租户流控、Kubernetes 控制面、HA、扩缩容和平台生态 | 不作为总体路线 |
| 基于 llm-d 二次开发 | Kubernetes-native，已有多租户、KV/SLO 调度和扩缩容框架，推理引擎相对解耦 | 需要补 xLLM、P/D、KV 和容错适配 | **推荐** |
| 基于 Dynamo 二次开发 | 请求迁移、Planner、分布式运行时和快速启动较完整 | 与 NVIDIA/CUDA 和自身 Python/Rust Runtime 结合较深，xLLM/NPU 接入成本高 | 作为参考 |

### 4.2 选择 llm-d 的主要原因

1. **适合单模型共享大池**

   一个 InferencePool 管理大量相同模型实例，业务和用户作为请求级 Flow 处理，避免
   为每个业务单独部署一套模型，资源利用率更高。

2. **与多租户 SLO 目标匹配**

   llm-d Flow Control 已提供 priority、tenant fairness、集中排队和 Admission；EPP
   可以扩展京东自己的 Goodput、SLO、KV 和业务优先级算法。

3. **可以保留 xLLM 引擎**

   llm-d 面向 Model Server Endpoint 工作，不要求替换 xLLM 内部运行时。补齐 HTTP、
   health、metrics 和 KV event contract 后即可接入。

4. **减少通用平台重复建设**

   Kubernetes/llm-d 可以承担实例发现、readiness、滚动升级、HPA/KEDA 和路由框架，
   xLLM 团队可以集中投入 P/D、KV、NPU 和快速启动。

### 4.3 为什么不直接选择 Dynamo

Dynamo 的请求迁移、Planner、Snapshot 和快速启动值得借鉴，但当前不适合作为主底座：

- 与 NVIDIA GPU、CUDA、NIXL、CRIU 结合更深。
- xLLM 的 C++、brpc、P/D 和 NPU 协议需要较大改造才能成为 Dynamo Backend。
- 当前首要问题是多业务、租户公平性和 SLO Goodput，llm-d 的 Flow Control 更贴近
  目标。

如果未来生产环境转为 NVIDIA GPU + vLLM/SGLang 为主，可以重新评估 Dynamo。

### 4.4 llm-d 也需要二次开发

llm-d 不是开箱即用，仍需补齐：

- xLLM metrics 和 KV event adapter。
- xLLM P/D pair 选择及内部协议适配。
- 京东租户、配额、SLOClass 和 Goodput 算法。
- 多 EPP 副本下的公平性和队列恢复。
- 请求中途迁移及 NPU 快速启动。

原则是优先使用扩展接口，减少对 llm-d 核心代码的长期 fork。

## 五、目标架构

```text
业务调用方
    |
    v
京东 API Gateway / IAM
  鉴权、租户、业务、配额、SLOClass
    |
    v
llm-d Gateway + EPP
  多租户队列、优先级、公平性、Admission
  SLO、负载和 KV-aware 调度
    |
    v
InferencePool：同一个基础模型
  Aggregated / Prefill / Decode Endpoint
    |
    v
xLLM Endpoint Pod
  xllm-backend-adapter + xLLM engine group

旁路控制面：
Prometheus -> HPA/KEDA -> Kubernetes Deployment
                         -> xllm-runtime-controller
                            P/D Link/Unlink、预热和运行时控制
```

xllm-service 最终收缩为两个部署单元：

- `xllm-backend-adapter`：部署在每个可路由 Endpoint 旁边，负责 RequestSession、
  HTTP stream 和 xLLM 协议适配，不再维护全局调度状态。
- `xllm-runtime-controller`：不进入 token 热路径，负责 P/D Link/Unlink、角色变化、
  模型预热和运行时生命周期。

中心化 xllm-service 仅在迁移期保留。最终 llm-d 需要看到具体 xLLM Endpoint，才能
完成 per-instance 的负载、KV 和 SLO 调度。

## 六、状态所有权原则

| 状态或能力 | 最终 Owner |
| --- | --- |
| 用户、租户、业务身份和配额 | 京东 API Gateway / 策略服务 |
| 请求队列、公平性和 Admission | llm-d Flow Control |
| Endpoint 发现和 readiness | Kubernetes + InferencePool |
| SLO、负载、KV 调度决策 | llm-d EPP |
| 全局 prefix-to-endpoint KV 目录 | llm-d KV data layer/indexer |
| 单请求执行、输出顺序和取消 | xLLM RequestSession |
| P/D Link/Unlink 拓扑 | xLLM Runtime Controller |
| 实例内部物理 KV | xLLM Engine |
| 副本数 | HPA/KEDA 或唯一的自定义 Autoscaler |

核心原则：**一份可变状态只有一个权威 Owner，其他模块只订阅事件或读取快照。**
这样可以从根本上减少跨模块共享状态和锁。

## 七、开发阶段

| 阶段 | 主要工作 | 阶段成果 |
| --- | --- | --- |
| 0. 基线和契约 | 固化 aggregated/P/D/CAR/SLO 行为；统一 RequestContext、metrics、KV event；建立多租户 Goodput 基线 | 后续改造可测试、可量化对比 |
| 1. llm-d 前置兼容 | `Gateway -> llm-d -> 现有 xllm-service`；llm-d 先接管多租户流控，旧 Scheduler 继续 P/D 和 KV 调度 | 不改变现有 P/D 链路，快速验证 llm-d 并支持回滚 |
| 2. xllm-service 解耦 | 引入 RequestSession、Dispatcher、ChannelPool；删除反向调用；禁止持锁 I/O；legacy 调度隔离 | 明确模块边界和状态 Owner，降低锁和竞态风险 |
| 3. Endpoint 直连 | 每个 xLLM Endpoint 补齐 HTTP、health、metrics；aggregated 流量先由 llm-d 直接调度；接入 HPA/KEDA | 跑通 llm-d + xLLM adapter 最小闭环 |
| 4. KV 和 P/D 迁移 | KV event 接入 llm-d；llm-d 负责 P/D 候选和评分；xLLM 保留 Link/Unlink、KV transfer 和 callback | 关闭 legacy CAR/P/D selector 后仍保持完整 P/D 能力 |
| 5. Goodput 和可靠性 | Goodput 策略、首 token 前重试、部分请求迁移、warm pool、权重预加载和预测扩容 | 从统一调度升级为围绕 SLO Goodput 自动优化 |
| 6. Legacy 删除 | 删除旧 Scheduler 的通用调度、etcd master/peer 请求路径和重复全局状态 | xllm-service 只保留 xLLM 差异化能力 |

代码级重构步骤见 [refactor_plan.md](./refactor_plan.md)。

## 八、第一阶段 MVP

第一版优先完成：

- 一个共享模型池。
- 多租户和三类 SLO 优先级。
- Aggregated 和现有 P/D 兼容路径。
- 基础负载调度、健康摘除和 HPA/KEDA 扩缩容。
- 按租户和 SLOClass 统计 Goodput。
- 可重复的压测、灰度和故障注入环境。

第一版暂不承诺所有场景的流式无缝迁移、跨租户 KV 共享、复杂预测扩容和多 EPP
副本下完全精确的全局公平性。先形成稳定、可观测、可压测的服务骨架，再逐步增加
复杂算法。

## 九、主要风险

| 风险 | 应对措施 |
| --- | --- |
| llm-d 部分能力仍在演进 | 固定版本，优先扩展接口，控制长期 patch 数量 |
| Flow Control 队列为内存状态 | 初期依赖幂等重试，后续评估 tenant 分片或外部队列 |
| xLLM KV hash 与 llm-d 不一致 | 版本化 tokenizer、block size、hash algorithm 和 seed |
| P/D 迁移影响稳定链路 | legacy/external 先做 shadow decision，对拍后分批切流 |
| 中途请求迁移有语义限制 | 先支持普通单路生成，结构化输出等场景显式降级 |
| 扩容后实例启动慢 | 分别优化节点、镜像、权重和引擎初始化时间 |

## 十、预期收益

- **研发效率**：复用 llm-d/Kubernetes 通用能力，减少重复建设。
- **技术聚焦**：xLLM 团队集中投入 P/D、KV、NPU 和快速启动。
- **系统稳定性**：明确状态 Owner，减少锁、竞态和跨模块副作用。
- **资源效率**：多个业务共享同一模型池，提高总体利用率。
- **业务保障**：按优先级保障核心流量，以 SLO Goodput 衡量真实服务效果。
- **迁移安全**：每个阶段可灰度、可对拍、可回滚，现有 P/D 始终作为稳定路径。

## 十一、建议确认事项

1. 确认“llm-d 平台骨架 + xLLM Adapter/Runtime Controller”的总体方向。
2. 冻结向旧 Scheduler 继续增加通用平台能力。
3. 优先建设基线、接口契约和 llm-d compatibility mode。
4. 保留现有 P/D 作为迁移期稳定生产路径。
5. 将 Dynamo 定位为请求迁移、Planner 和快速启动的参考实现。
6. 以 Goodput、SLO 命中率、稳定性和回滚能力作为阶段验收标准。
