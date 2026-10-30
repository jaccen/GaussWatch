# GaussWatch — 规则规格 v0.4（R1-R16 伪代码 + 测试映射）

> 日期：2026-10-30 · 上游：census v0.2（issue 实证）/ v0.3（技能库对齐）· 本版完成规则规格定稿并落地可运行参考实现
> 代码：`gausswatch/rules.py`（P0 六条可运行）· 测试：`tests/test_rules.py`（35 passed + 3 xfailed）

## 设计契约

1. **纯数值解耦**：规则只消费 `TrainingSnapshot`（训练状态快照，plain Python 数值），不 import torch——B 阶段采集层（torch 侧）与规则层（本文件）完全解耦，可先用日志回放验证
2. **缺字段跳过**：快照字段 None = 本步未采集，规则静默跳过，绝不在退化输入上抛异常（采集层可喂部分快照）
3. **阈值溯源**：每个常量注释出处（issue 编号或 3dgs-training-debugger KB 模式号），延续不编造数值红线
4. **测试即证据**：测试数据全部来自真实 issue 证据（#1027 死亡级联序列、#1040 规模钳制、#1107 泄漏曲线等），B 阶段实现规则时测试即生效

## 规则规格表

### R1 环境类 OOM 签名（P0，已实现）
- 输入：error_type ∈ {OOM, IMA}、vram_allocated/total
- 伪代码：`if error_type in {OOM,IMA}: util = alloc/total; if util < 0.70 → env-signature(critical) else → capacity-type(warning)`
- 出处：census v0.2 横向发现 1；#1221（1281 点 OOM）/#961/#1040（96G 用 30-60G）
- 测试：`test_r1_env_oom_signature`（45G/96G=47% → env）、`test_r1_capacity_oom`（95% → capacity）、`test_r1_no_error_no_alert`

### R2 规模外推三指标（P0，已实现）
- 输入：relocated_count、tile_intersections、relocated_runaway
- 伪代码：`if relocated >= 7.9M → ratio-clamp 告警；if tile_intersections > 0.8×2^31 → int32 溢出预警；if runaway → 告警`
- 出处：gsplat#1040（51×51 二项 ratio 表钳制 ~7.9M、int32 交点溢出 PR #1054、relocated runaway）
- 测试：`test_r2_ratio_clamp_breach`（7.9M）、`test_r2_tile_int32_overflow_warning`、`test_r2_runaway`、`test_r2_healthy_quiet`

### R3 refine 事件曲线族（P0，已实现）
- 输入：dead_fraction_history（每轮 refine 后死亡率）
- 伪代码：`if 最近 3 轮连续上升且末值 > 0.8 → death-cascade 告警`（#1027 真实序列 73→91→99.1→100%）
- 出处：gsplat#1027；KB DF-07（opacity reset 周期回归为第三类周期事件，B 阶段并入）
- 测试：`test_r3_death_cascade_real_sequence`、`test_r3_healthy_fluctuation_quiet`、`test_r3_insufficient_history_quiet`

### R4 崩溃归因锚点（P0，已实现）
- 输入：crash_step、last_event_step（最近 densify/relocate 步）、crash_error_type
- 伪代码：`if error ∈ {OOM,IMA} and 0 ≤ crash_step - event_step ≤ 2 → 归因该事件`
- 出处：census v0.2 横向发现 4（异步上报栈不可信；#41/#1040 步距均 ≤2）
- 测试：`test_r4_crash_one_step_after_relocate`、`test_r4_far_from_event_quiet`、`test_r4_non_ima_quiet`

### R5 初始化自检·参数（P0，已实现）
- 输入：init_min_camera_dist、init_zero_scale_fraction、init_max_scale/bound
- 伪代码：`近相机点（<1e-2）→ 建议过滤；zero-scale 占比 >0 → 告警；max scale 越界 → 告警`
- 出处：官方 #41（2026-09 评论：COLMAP 点与相机中心重合 → 退化 scale → NaN→IMA 链）；KB IF-02/IF-03
- 测试：`test_r5_near_camera_point`（1e-3）、`test_r5_zero_scale_fraction`、`test_r5_max_scale_over_bound`、`test_r5_clean_init_quiet`

### R6 NaN 前兆（P0，已实现）
- 输入：scales_finite/opacities_finite、grad_norm_history
- 伪代码：`isfinite 周期检查失败 → 告警；grad norm 序列"≥2 步正常后突现非有限"→ stable-then-collapse 签名`（#248：几百步正常 → 2 步内全崩）
- 出处：gsplat#248/#455
- 测试：`test_r6_scales_nonfinite`、`test_r6_stable_then_collapse`、`test_r6_grad_normal_quiet`
- 实现注记：NaN 判定用自不等式（`x != x`），`in (nan, inf)` 对 NaN 恒 False 是已知陷阱（v0.4 开发中实际踩到并修正）

### R7 导出保护（P0，已实现）
- 输入：export_mask_filtered_ratio、export_opacities_ndim
- 伪代码：`过滤比例 ≥90% → 告警；opacities 一维 → 已知 any(axis=0) 过滤 bug 形状告警`
- 出处：gsplat#660（维护者确认 + PR #663 合并）
- 测试：`test_r7_mask_over_filtering`（95%）、`test_r7_1d_opacities_bug_shape`、`test_r7_normal_export_quiet`

### R8 环境预检（P0，已实现）
- 输入：gpu_arch、torch_cuda_arch_list
- 伪代码：`启动时 gpu_arch ∉ ARCH_LIST → build/runtime mismatch 告警`
- 出处：官方 #961（Docker 中 ARCH_LIST 未含本卡架构）
- 测试：`test_r8_arch_missing`、`test_r8_arch_present_quiet`

### R9 VRAM-计数失联（P0，已实现）
- 输入：vram_history、num_gaussians_history（≥3 采样点）
- 伪代码：`计数稳定（变化 <1%）且 VRAM 单调增 → 泄漏/优化器膨胀告警`
- 出处：官方 #1107（计数恒 50200，每次调用 +4GiB）；KB MF-05+MF-03
- 测试：`test_r9_leak_signature`、`test_r9_count_growing_quiet`（致密化期计数增长属正常，不告警）

### R12 梯度归零（P0，已实现）
- 输入：grad_norm
- 伪代码：`grad_norm == 0.0（严格相等）→ 静默失败告警（detach/计算图断裂）`
- 出处：KB OF-03 + 库监控表"exactly 0.0"条目；v0.3 top-10 #6
- 测试：`test_r12_exact_zero`、`test_r12_normal_grad_quiet`

### R13 剪枝比例带（P1，已实现）
- 输入：pruned_count、total_gaussians
- 伪代码：`单次 prune 移除 >5% → over-pruning 告警`
- 出处：KB DF-05（正常只删透明高斯，计数骤降出洞）
- 测试：`test_r13_over_prune`（12%）、`test_r13_normal_prune_quiet`（3%）

### R14 初始化自检·位姿渲染比对（P0，已实现）
- 输入：render_metric（早期位姿渲染指标）、input_metric（输入参照指标，同尺度）
- 伪代码：`|render - input|/|input| > 50% → 位姿/初始化告警`
- 出处：KB IF-04（相机位姿错误：渲染与输入不符）
- 测试：`test_r14_parity_divergence`（70%）、`test_r14_parity_ok_quiet`（5%）

### R16 致密化参数生效性（P1，已实现）
- 输入：render_metric、render_metric_prev（参数变更前后行为代理）
- 伪代码：`变化 <1% → 参数无行为效果 → config-code 一致性检查`
- 出处：官方 #1108/#741/#544（max_radii2D 等静默失效三连）
- 测试：`test_r16_param_ineffective`、`test_r16_param_effective_quiet`

### R10 eval 剪刀差（P2 stub，B 阶段实现）
`train/test PSNR gap > 3dB 或峰值后连降`（KB CF-03/CF-05）。测试：`test_r10_spec_stub`（xfail）

### R11 loss 分量失衡（P2 stub，B 阶段实现）
`reg_loss > 10× render_loss；mask 覆盖 >90% 或反转`（KB OF-04/OF-06）。测试：`test_r11_spec_stub`（xfail）

### R15 跨 run 方差（P2 stub，B 阶段实现）
`同配置多种子 PSNR std > 0.5dB → 非确定性算子`（KB CF-07）。测试：`test_r15_spec_stub`（xfail）

## 测试数据溯源表

| 测试数据 | 真实来源 |
|---|---|
| dead_fraction [0.73, 0.91, 0.991, 1.00] | gsplat#1027 正文记录的真实级联序列 |
| relocated ~7.9M（ratio-clamp 边界） | gsplat#1040（7.9M/8.0M 时 ratio 均值 ~79 超 51） |
| 45G/96G 利用率下 IMA | gsplat#1040（96GB 仅用 30-60GB） |
| 计数恒 50200 + VRAM 每调用 +4GiB | 官方 #1107 |
| grad_norm 稳定序列后 NaN | gsplat#248（数百步正常 → 1-2 步全崩） |
| 1281 点 OOM → env 签名 | 官方 #1221 |
| 1e-3 相机距离点 | 官方 #41（2026-09 补充路径） |
| 一维 opacities → any(axis=0) 过滤 | gsplat#660（维护者确认） |

## B 阶段集成说明

1. **采集层接口**：collector（torch 侧）把每步/每 refine 事件填充 `TrainingSnapshot`，规则层零修改
2. **R8/R14 的特殊调用时机**：R8 在启动时调用一次；R14 在 iter<100 调用一次；其余规则每步/每事件流式调用
3. **P1/P2 stub 实现**：R10/R11/R15 数值逻辑简单，实现后删 xfail 标记即转绿
4. **告警路由**：RuleAlert.severity 用于面板分级展示；source 字段保证每个告警可回溯证据

---
*规格与实现：星辰超级智能体 · 2026-10-30 · 全部阈值绑定公开出处；测试 35 passed + 3 xfailed（pytest）*
