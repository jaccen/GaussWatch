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

## Class A — 运行时注入（待跑）

在真实 gsplat 训练中注入失败信号，验证端到端检出 + 检出延迟 + 归因准确率。需 GPU（tiger2 集群）。

> 状态：**待实施**。设计见 `research/benchmark-design-v0.1.md` 第 3.1 节（7 场景）。数据集：nerf_synthetic/lego + tandt/truck + 360_v2/bonsai。跑完后在本文件续表。

---

*所有规则阈值绑定公开出处（`gausswatch/rules.py`）；测试数据溯源见 `research/rule-specs-v0.4.md` 测试数据溯源表*
