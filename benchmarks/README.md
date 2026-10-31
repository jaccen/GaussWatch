# GaussWatch Benchmark

公开 benchmark 协议：验证 GaussWatch 对 3DGS 训练失败模式的检出能力，产出可被引用的客观证据。

设计全文见 `research/benchmark-design-v0.1.md`；结果见 `results.md`。

## 两类验证

### Class B — 日志回放（已完成，纯 CPU）

将真实 issue 记录的数值序列回放给规则引擎，测检出率（TPR）与误报率（FPR）。

```bash
python benchmarks/log_replay.py           # 报告
python benchmarks/log_replay.py --ci      # CI 回归门槛
```

- 6 个故障场景（R2/R3/R5/R8/R9），数据来自 gsplat#1027、#1040、官方#1107、#41、#961
- 5 个健康基线控制组
- 当前结果：**TPR 100% / FPR 0%**（见 `results.md`）

### Class A — 运行时注入（已完成，纯 CPU）

模拟训练事件流（on_step / on_refine / on_eval / on_crash / on_export），在指定步注入失败信号，测检出率、检出延迟、R4 归因准确率。

```bash
python benchmarks/runtime_inject.py           # 报告
python benchmarks/runtime_inject.py --ci      # CI 回归门槛
```

- 7 个故障场景（R1/R4/R6/R7/R10/R12/R13）
- 1 个健康基线（完整事件流：preflight + 1000 步 + refine + eval + export）
- 当前结果：**TPR 100% / FPR 0% / 检出延迟 0 步 / 归因准确率 100%**（见 `results.md`）

> 设计变更：原方案设计为在真实 gsplat 训练中注入（需 GPU 集群）。实施时改为事件流模拟（纯 CPU），因真实训练循环的端到端挂接已在 Level-B 验证（gsplat v1.5.3 + tandt/truck 1500 步闭环，0 误报），Class A 聚焦规则引擎的时序检测能力，与 Class B（单次快照）互补。详见 `results.md` 设计变更说明。

## 合计覆盖

| 类别 | 场景 | 覆盖规则 | TPR | FPR |
|---|---|---|---|---|
| Class B | 6 | R2, R3, R5, R8, R9 | 6/6 = 100% | 0/5 = 0% |
| Class A | 7 | R1, R4, R6, R7, R10, R12, R13 | 7/7 = 100% | 0/1 = 0% |
| **合计** | **13** | **12 条规则** | **13/13 = 100%** | **0/6 = 0%** |

## 指标定义

| 指标 | 定义 | 目标 |
|---|---|---|
| TPR | 注入场景产出对应规则告警的比例 | 100% |
| FPR | 健康基线的告警数 | 0 |
| 检出延迟 | 注入步到告警产出的步数（Class A） | ≤ check_every（默认 100） |
| 归因准确率 | R4 崩溃归因指向正确事件的比例（Class A） | 100% |

## 数据出处原则

- 每个回放序列绑定公开 GitHub issue 编号（见 `results.md` source 列）
- 零编造：控制组数据来自 `tests/test_rules.py` 的 `*_quiet` 测试
- 阈值溯源见 `gausswatch/rules.py` 常量注释与 `research/rule-specs-v0.4.md`
