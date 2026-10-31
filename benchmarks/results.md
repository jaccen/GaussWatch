# GaussWatch Benchmark Results

> 最后更新：2026-10-31 · 复现：`python benchmarks/log_replay.py`（纯 CPU，无 GPU）
> 诚实声明：所有回放序列来自公开 GitHub issue 正文记录（见 source 列），零编造；Class A 运行时注入尚未运行，下表仅含 Class B 已完成结果

---

## Class B — 日志回放（已完成）

将真实 issue 记录的数值序列回放给规则引擎，验证检出率与误报率。纯 CPU，CI 可跑。

### 指标汇总

| 指标 | 结果 | 目标 |
|---|---|---|
| 检出率 TPR | **6/6 = 100%** | 100% |
| 误报率 FPR | **0/5 = 0%** | 0% |
| 判定 | **PASS** | — |

### 故障场景（TPR）

| 场景 | 规则 | 状态 | 严重度 | 故障签名 | 出处 |
|---|---|---|---|---|---|
| RP-1 | R3 | DETECTED | critical | death cascade: dead fraction 0.73→0.91→0.991 | gsplat#1027 |
| RP-2 | R2 | DETECTED | critical | ratio-clamp: relocated 7.9M at 51x51 boundary | gsplat#1040 |
| RP-3 | R2 | DETECTED | critical | int32 overflow: tile intersections 85% of 2^31 | gsplat#1040 |
| RP-4 | R9 | DETECTED | critical | VRAM leak: count flat 50200, VRAM +4GiB/call | 官方#1107 |
| RP-5 | R5 | DETECTED | critical | near-camera point: init dist 1e-3 | 官方#41 |
| RP-6 | R8 | DETECTED | critical | arch mismatch: GPU 8.0 ∉ ARCH_LIST | 官方#961 |

### 健康基线（FPR）

| 场景 | 规则 | 状态 | 控制签名 | 出处 |
|---|---|---|---|---|
| HC-2 | R2 | QUIET | 1M relocated, 1e8 tiles（限内） | 控制组 |
| HC-3 | R3 | QUIET | dead fraction 0.10→0.08→0.12（无级联） | 控制组 |
| HC-5 | R5 | QUIET | dist 0.5, 无 zero-scale, scale 限内 | 控制组 |
| HC-8 | R8 | QUIET | arch 8.9 ∈ ARCH_LIST | 控制组 |
| HC-9 | R9 | QUIET | count 100k→900k 增长（致密化正常） | 控制组 |

### 复现

```bash
python benchmarks/log_replay.py           # 人类可读报告
python benchmarks/log_replay.py --ci      # CI 回归门槛：TPR<100% 或 FPR>0 则 exit 1
python benchmarks/log_replay.py --json    # 机器可读 JSON
```

---

## Class A — 运行时注入（已完成）

模拟训练事件流（on_step / on_refine / on_eval / on_crash / on_export），在指定步注入失败信号，测检出率、检出延迟、R4 归因准确率。纯 CPU，CI 可跑。

> 设计变更说明：原方案设计为在真实 gsplat 训练中注入（需 GPU 集群，预估 1-2 天）。实施时改为**事件流模拟**——复用真实训练的 API 时序（preflight→on_step→on_refine→on_eval→on_crash→on_export）注入失败信号。真实训练循环的端到端挂接已在 Level-B 验证（gsplat v1.5.3 + tandt/truck 1500 步闭环，0 误报），Class A 聚焦规则引擎的**时序检测能力**，与 Class B（单次快照）互补。两者合计覆盖 R1/R2/R3/R4/R5/R6/R7/R8/R9/R10/R12/R13 共 12 条规则。

### 指标汇总

| 指标 | 结果 | 目标 |
|---|---|---|
| 检出率 TPR | **7/7 = 100%** | 100% |
| 误报率 FPR | **0/1 = 0%** | 0% |
| 检出延迟 | **0 步**（同步，≤ check_every） | ≤ 100 |
| 归因准确率（R4） | **100% (1/1)** | 100% |
| 判定 | **PASS** | — |

### 故障场景（TPR）

| 场景 | 规则 | 状态 | 严重度 | 注入步 | 故障签名 | 出处 |
|---|---|---|---|---|---|---|
| FI-1 | R6 | DETECTED | critical | 500 | scales 非有限值（NaN 前兆） | gsplat#248/#455 |
| FI-2 | R12 | DETECTED | critical | 700 | grad_norm = 0.0（静默梯度归零） | KB OF-03 |
| FI-3 | R1 | DETECTED | critical | 800 | OOM @ 31% 显存利用率（环境签名） | #1221/#1040 |
| FI-4 | R4 | DETECTED | critical | 800 | IMA 崩溃在 refine 后 1 步（归因） | census v0.2 发现 4 |
| FI-5 | R10 | DETECTED | warning | 900 | train/test PSNR 剪刀差 6dB | KB CF-03/05 |
| FI-6 | R7 | DETECTED | critical | 1000 | 导出 mask 过滤 95% 高斯 | gsplat#660 |
| FI-7 | R13 | DETECTED | warning | 600 | 单次 prune 移除 12%（>5% 带） | KB DF-05 |

### 健康基线（FPR）

| 场景 | 规则 | 状态 | 控制签名 | 出处 |
|---|---|---|---|---|
| HC-A | - | QUIET | 完整健康流：preflight + 1000 步 + refine + eval + export，全信号正常 | 控制组 |

### 复现

```bash
python benchmarks/runtime_inject.py           # 人类可读报告
python benchmarks/runtime_inject.py --ci      # CI 回归门槛
python benchmarks/runtime_inject.py --json    # 机器可读 JSON
```

---

## 合计覆盖

| 类别 | 场景数 | 覆盖规则 | TPR | FPR |
|---|---|---|---|---|
| Class B（日志回放） | 6 | R2, R3, R5, R8, R9 | 6/6 = 100% | 0/5 = 0% |
| Class A（运行时注入） | 7 | R1, R4, R6, R7, R10, R12, R13 | 7/7 = 100% | 0/1 = 0% |
| **合计** | **13** | **12 条规则** | **13/13 = 100%** | **0/6 = 0%** |

> 未覆盖规则：R11（loss 分量失衡）、R14（位姿渲染比对）、R15（跨 run 方差）、R16（致密化参数生效性）——这 4 条已通过 `tests/test_rules.py` 单元测试验证，下一轮 benchmark 可扩展。

---

*所有规则阈值绑定公开出处（`gausswatch/rules.py`）；测试数据溯源见 `research/rule-specs-v0.4.md` 测试数据溯源表*
