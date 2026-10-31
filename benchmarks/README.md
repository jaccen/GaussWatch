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

### Class A — 运行时注入（待跑，需 GPU）

在真实 gsplat v1.5.3 训练中通过注入器锚点注入失败信号，测端到端检出 + 检出延迟 + 崩溃归因准确率。

- 7 场景（R1/R4/R6/R7/R10/R12/R13），设计见 `research/benchmark-design-v0.1.md` 第 3.1 节
- 数据集：nerf_synthetic/lego + tandt/truck + 360_v2/bonsai
- 注入器复用 `scripts/inject_gausswatch.py` 的 5 锚点机制

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
