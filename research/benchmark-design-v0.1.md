# GaussWatch 公开 Benchmark 设计方案 v0.1

> 日期：2026-10-31 · 性质：设计稿，待老板拍板后实施 · 依据：rules.py（R1-R16）、rule-specs-v0.4、failure-mode-census-v0.1~v0.3、决策稿 90 天里程碑第 7-10 周
> 状态：**方案稿**，所有数字均为"目标值/预期"，非实测——benchmark 尚未运行

---

## 1. 目标

产出**可被引用的客观证据**，回答一个问题：

> GaussWatch 接入后，能否在失败发生时实时检出（且不误报健康训练），相比"裸跑 + 肉眼回看"快多少、准多少？

该证据服务三个出口：
- README 首屏的 benchmark 引用块（star 运营前置）
- 论文实验章节素材（一鱼两吃：诊断检出率 = tool 章节实验）
- 社区发帖的硬数据（HN/X/知乎帖子的 credibility）

## 2. 对比对象

| 组别 | 配置 | 代表什么 |
|---|---|---|
| **Baseline A** | gsplat v1.5.3 裸跑，无监控 | 现状："重跑 + 肉眼回看 loss/图" |
| **Baseline B** | 官方 graphdeco-inria/gaussian-splatting 裸跑 | 第二主流实现的现状 |
| **Treatment** | gsplat + GaussWatch 接入（wrap_strategy + wrap_eval） | 本工具 |

> Baseline 的"检出"定义：训练崩溃（OOM/NaN 异常退出）或人眼发现 loss/PSNR 曲线异常——延迟通常是"崩溃后回看"或"训练结束看图"，量级 = 全程。Treatment 的"检出"= 规则在 check_every 窗口内产出 RuleAlert。

## 3. 失败注入场景矩阵

分两类注入，覆盖 R1-R16 中可受控复现的规则：

### 3.1 A 类·运行时注入（runtime injection，需 GPU）

在真实 gsplat 训练中，通过注入器锚点在指定步注入失败信号，验证端到端检出。

| 场景 | 规则 | 注入手法 | 预期 alert | 真实出处 |
|---|---|---|---|---|
| FI-1 NaN 前兆 | R6 | step=500 向 `scales` 注入 NaN | R6 critical | gsplat#248/#455 |
| FI-2 梯度归零 | R12 | step=500 强制 `grad_norm=0.0` | R12 critical | KB OF-03 |
| FI-3 env-OOM 签名 | R1 | 限制可用显存至 30%，触发 OOM | R1 critical（env 型） | #1221/#1040 |
| FI-4 崩溃归因 | R4 | FI-3 后记录 crash_step 与 last_event_step 步距 ≤2 | R4 critical | census v0.2 发现 4 |
| FI-5 train/test 剪刀差 | R10 | step≥800 上报 train_psnr 虚高 / test_psnr 压低（gap>3dB） | R10 warning | KB CF-03/05 |
| FI-6 导出过滤 | R7 | export 时构造 mask_filtered_ratio=0.95 | R7 critical | gsplat#660 |
| FI-7 over-prune | R13 | 单次 refine 构造 pruned/total=12% | R13 warning | KB DF-05 |

### 3.2 B 类·日志回放（log replay，无需 GPU，CI 可跑）

用真实 issue 记录的数值序列回放给规则引擎，验证判定准确率。复用 `tests/` 已有测试数据（零新增数据，零编造）。

| 场景 | 规则 | 回放序列 | 预期 | 数据出处 |
|---|---|---|---|---|
| RP-1 death cascade | R3 | dead_fraction [0.73,0.91,0.991,1.00] | R3 critical | gsplat#1027 正文 |
| RP-2 规模钳制 | R2 | relocated=7.9M | R2 critical | gsplat#1040 |
| RP-3 int32 溢出 | R2 | tile_intersections>0.8×2^31 | R2 critical | gsplat#1040 |
| RP-4 VRAM 泄漏 | R9 | 计数恒 50200 + VRAM 每次 +4GiB | R9 critical | 官方#1107 |
| RP-5 近相机点 | R5 | init_min_camera_dist=1e-3 | R5 critical | 官方#41 |
| RP-6 arch mismatch | R8 | gpu_arch="8.9" ∉ ARCH_LIST | R8 critical | 官方#961 |

### 3.3 健康基线（负控）

每个数据集跑一次**无注入**的完整训练，记录 alert 数——**预期 = 0**（Level-B 已在 tandt/truck 1500 步实测 0 alerts，本 benchmark 扩到更多步/场景验证 FPR）。

## 4. 指标体系

| 指标 | 定义 | 目标 | 计算 |
|---|---|---|---|
| **检出率 TPR** | 注入场景中产出对应 rule alert 的比例 | 100%（A 类 7 + B 类 6 = 13 场景全中） | 命中场景数 / 总注入场景数 |
| **误报率 FPR** | 健康基线的 alert 数 | 0 | 健康训练 alert 数 / 健康训练总步数 |
| **检出延迟** | 从注入步到 alert 产出的步数 | ≤ check_every（默认 100 步） | alert_step − inject_step |
| **归因准确率** | R4 场景中 crash 归因指向正确事件的比例 | 100% | 正确归因 / R4 场景数 |
| **Baseline 延迟** | 裸跑从失败到人眼发现 | 量级 = 全程（崩溃后或训练结束） | 定性对比，不做精确计时 |

> 诚实声明：Baseline 延迟无法精确计时（"人眼发现"无客观 timestamp），benchmark 只做定性对比——"GaussWatch 实时（≤100 步）vs 裸跑（崩溃后/全程）"。不编造 baseline 的精确秒数。

## 5. 数据集

全部用集群 `/home/jiaxb/3dgs/data/` 已有数据集（零下载）：

| 数据集 | 场景 | 用途 | 理由 |
|---|---|---|---|
| nerf_synthetic | lego | 快速冒烟（小场景） | 训练快，适合高频注入验证 |
| tandt | truck | 中等场景（Level-B 已用） | 已验证闭环，延续 |
| 360_v2 | bonsai 或 garden | 大场景 FPR 验证 | 步数多，验证健康基线 0 误报的稳健性 |

## 6. 跑法与可复现性

```
benchmarks/
  README.md          # benchmark 协议（场景定义、指标、复现命令）
  runtime_inject.py  # A 类运行时注入器（复用 inject_gausswatch.py 锚点）
  log_replay.py      # B 类日志回放（复用 tests/ 数据，纯 CPU）
  run_all.sh         # 一键脚本：健康基线 + A类注入 + B类回放
  results.md         # 结果表（跑完后填写，README 引用）
```

**一键复现**：`bash benchmarks/run_all.sh`（集群）+ `python benchmarks/log_replay.py`（CI/本机，无 GPU）

**注入器复用**：runtime_inject.py 基于 `scripts/inject_gausswatch.py` 的 5 锚点机制，加 `--fault FI-1 --at-step 500` 参数化注入。

**B 类零 GPU**：log_replay.py 直接喂 `TrainingSnapshot` 给 `rules.py`，CI 矩阵可跑，复现门槛最低。

## 7. 产出物

1. `benchmarks/results.md` — 场景 × 检出 × 延迟 × 误报 结果表
2. README 首屏引用块：`> Benchmark: 13/13 fault scenarios detected, 0 false positives across N k steps (see benchmarks/results.md)`
3. 论文实验章节素材：检出率/误报率表 + 归因准确率（SignedGS / tool 章节复用）
4. 社区发帖硬数据：HN/X 帖子引用"13 fault scenarios, 0 false positives"

## 8. 工作量估计

| 子任务 | 预估 | 依赖 |
|---|---|---|
| B 类日志回放脚本 + 跑 | 0.5 天 | 无 GPU，可立即做 |
| A 类运行时注入器扩展 | 1 天 | 复用注入器锚点 |
| 集群跑 A 类（3 数据集 × 7 场景 + 3 健康基线） | 1-2 天（含排队） | tiger2 可用 + /var/tmp 旁路环境 |
| results.md 汇总 + README 引用 | 0.5 天 | 跑完 |
| **合计** | **3-4 天** | — |

> 风险：tiger2 `/home` NFS 配额已满，A 类需继续走 `/var/tmp` 旁路（已验证可行）；tiger 网络不稳，长任务 nohup 后台。

## 9. 待老板拍板的决策点

1. **场景范围**：A 类 7 + B 类 6 = 13 场景是否够？还是先做 B 类（0.5 天出结果）再决定 A 类范围？
2. **数据集选择**：nerf_synthetic/lego + tandt/truck + 360_v2/bonsai 三场景是否 OK？还是聚焦 truck 一个？
3. **论文复用**：benchmark 结果是否同步设计成 SignedGS/tool 章节实验（一鱼两吃）？若是，指标需加"多种子统计 + 配对 t 检验"。
4. **CI 集成**：B 类日志回放是否纳入 CI（无 GPU 矩阵每次跑），作为回归门槛？

---

*设计：星辰超级智能体 · 2026-10-31 · 全部规则阈值绑定公开出处（rules.py）；benchmark 未运行，所有数字为目标值*
