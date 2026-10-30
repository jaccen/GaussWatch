# GaussWatch — 失败模式普查 v0.2（issue 正文级）

> 普查日期：2026-10-30 · 方法：逐条阅读 14 个代表性 issue 正文与评论（子代理执行，主会话审核）
> 上游：v0.1 标题级普查（`failure-mode-census-v0.1.md`）· 用途：P0 检测规则引擎的知识规格
> 置信分级纪律：gsplat 仓库维护者/协作者的结论 = 高置信；官方仓库评论者共识 = 中置信；单评论者建议 = 低置信。下文逐条标注。

---

## 一、四类失败模式规格（触发条件 → 错误信号 → 根因/解法 → 可观测信号）

### F1. 显存类（OOM）

| Issue | 触发条件 | 错误信号 | 根因/解法（置信） | 可观测信号 |
|---|---|---|---|---|
| 官方 #1221 | A100 80GB，`create_from_pcd`→`distCUDA2`；初始化仅 1281 点 | `MemoryError: bad allocation: cudaErrorMemoryAllocation` | 页面未证实（低置信：点数极少仍 OOM 指向环境类而非容量类） | 初始化点数 <10k 即 OOM → 判环境类失败 |
| 官方 #961 | RTX 8000 48GB，distCUDA2 | `std::bad_alloc: cudaErrorMemoryAllocation` | 两解并陈：换 4090 自答（低）；`TORCH_CUDA_ARCH_LIST` 未含本架构（评论建议，中） | 启动时校验 GPU arch ∈ TORCH_CUDA_ARCH_LIST |
| 官方 #1107 | 循环调用 rasterizer（50200 高斯，1024²，antialiasing=True） | 显存每次调用恒增 4GiB 直至 OOM | 未明（无维护者回复，低） | `memory_allocated` 单调上升且增量与输入规模无关 → 泄漏签名 |
| 官方 #1146 | 无人机图 1024→2048 升分辨率；分批加载避免 OOM 后质量退化 | 非崩溃，PSNR 退化 | 线索：分辨率缩放时相机内参 cx/cy 未随之调整（评论推测，低）；无共识解法 | 分辨率变更后 PSNR 对基线差值；内参缩放一致性检查 |

### F2. 数值类（NaN/发散）

| Issue | 触发条件 | 错误信号 | 根因/解法（置信） | 可观测信号 |
|---|---|---|---|---|
| gsplat #248 | 输入无 NaN，几百步后 1-2 步内全部高斯 NaN | `_FullyFusedProjectionPackedBackward` returned nan；grad norm NaN | fp16 坐标变换精度误差致 scales NaN（多人佐证，**高**）；解法：场景整体放大 / 参数转 fp32 / clamp_min 诊断法 | scales 最小值分布 + fp16 标记监控；"数百步正常→2 步内全崩"突变特征 |
| gsplat #455 | 第 0/30000 步即崩 | `_RasterizeToPixelsBackward` returned nan（anomaly 定位） | scales/opacity 先 NaN（低——上游未定位） | 每 N 步 `torch.isfinite` 检查 scales/opacities |
| gsplat #660 | opacities 一维 (N,) 时 `np.isnan(...).any(axis=0)` 误过滤全部 | 导出空 .ply | **维护者确认** + PR #663 合并（**高**） | 导出前 mask 过滤比例 ≥90% 告警 |
| 官方 #174 | 训练渲染正常，metrics 全 NaN | 指标全 NaN | 未加 `--eval` 致 test 目录空（评论建议 8 赞，中） | 指标 NaN 时先查数据路径/空目录 |

### F3. 伪影类（floater）

| Issue | 触发条件 | 错误信号 | 根因/解法（置信） | 可观测信号 |
|---|---|---|---|---|
| 官方 #1089 | 2024 新 commit 重建 NeRF-synthetic lego（旧版无 floaters） | 大量 floater（截图） | 疑回归，指向外部 issue（低——链条不完整） | 版本/commit 前后 floater 数 A/B 对比 |
| 官方 #941 | 新深度图功能 + Depth-Anything-v2，15k-3M splats | floaters 增多但 PSNR 反升 | 三方独立复现、根因未明（低）；**质量与指标背离** | PSNR↑ 与主观质量↓ 背离检测；深度损失开关的 floater 计数对比 |
| 官方 #852 | 360° 环绕拍摄，主物体悬浮物 | 空中 floaters | 信息不足（仅 1 评论推荐 Vast Gaussian 论文） | — |

### F4. CUDA 崩溃类（IMA/配置错误）

| Issue | 触发条件 | 错误信号 | 根因/解法（置信） | 可观测信号 |
|---|---|---|---|---|
| gsplat #1040 | gsplat 1.5.3，Blackwell sm_120 96GB，cap 8-15M，单轮 relocate 约 8M+ 高斯后 1-2 步崩 | `CUDA error: an illegal memory access`，紧跟 "Relocated ~X" 日志 | 三候选并陈：规模触发（10.7M relocated 附近普适，楼主跨架构证据，中）；PR #1054 int32→int64 交点溢出（2^31 交点 ≈ 11.46M 高斯，中）；fast-math `pow(-1.0f,k)` sm_120 NaN（已开 #1060，中）。workaround：`refine-stop-iter` 设崩溃区下 + retry | 单轮 relocated/alive 比值（>51 逼近 51×51 ratio 钳制）；relocated 计数 cap 满后续增（runaway）；tile 交点总数逼近 2^31；WDDM 显存溢出系统内存 |
| gsplat #1027 | 所有高斯 sigmoid(opacity)≤min_opacity → 空张量 multinomial | `CUDA error: invalid configuration argument` | 楼主定位明确：guard 缺失 + 死亡级联（opacity_reg 单调压低 → "出生即死" → dead fraction 73→91→99.1→100%）（**高**——含最小 repro） | 逐轮 dead fraction 曲线（连续上行逼近 100% 是明确前兆）；alive 数低水位告警 |
| 官方 #41 | RTX 4090 第 0 步即崩（3090 同环境正常） | ssim 处 IMA（异步上报，栈不可信） | 双路径：setup.py 加 `-Xcompiler -fno-gnu-unique`（评论 57 赞多人验证，中**高**）；COLMAP 初始点与相机中心重合 → 极小 scale 退化 → NaN→IMA（2026-09 补充，中）。矛盾并陈：gnu-unique 对部分环境无效 | 编译标志检查；初始点到相机中心最小距离（<1e-2 过滤）；NaN→IMA 链路标记 |

## 二、六条横向发现（产品架构依据）

1. **"OOM" 报错多数不是真 OOM**：#1221（1281 点）、#961（48G 卡）、#1040（96G 用 30-60G）显存远未耗尽——共同上游是**环境层**（CUDA arch 覆盖、编译标志）。→ 检测规则：显存占用 <70% 时报 OOM/IMA → 判环境类签名，优先查 `TORCH_CUDA_ARCH_LIST` 与编译 flags，不建议减载。
2. **规模门控是独立失败模式**：#1040（8M+ relocate）、#1027（51×51 ratio 钳制）、#41（极端退化高斯）均小规模不可复现。→ 监控须含规模外推指标：relocated/alive 比值、tile 交点总数、单点 tile 交点数。
3. **refine 事件计数曲线是最具预警价值的监控面**：#1027 的 dead fraction（73→91→99.1→100%）与 #1040 的 relocated runaway 是同构信号——**比 loss 曲线早数百步发现失败**。
4. **异步上报掩盖真实故障点**：IMA 均报在 loss.item()/backward/相邻算子，栈不可信。→ 归因锚点：记录"最近一次 densify/relocate 事件与崩溃的步距"（本组案例均 ≤2 步）。
5. **NaN 上游集中在退化参数值**：极小 scale/opacity（#248/#455）、近相机点（#41 路径二）。→ 首个检查点即做参数分布体检，不等梯度 NaN。
6. **两仓库结论置信度显著不同**：官方仓库 7 例中 6 例无维护者技术结论，有效解法来自评论且无共识收敛；gsplat 侧 collaborator 参与度高（#660 14 天修复、#248 社区定位根因）。→ 规则库按来源分级置信，gsplat 侧结论高置信优先。

## 三、P0 检测规则清单（v0.2 更新，8 条）

| 规则 | 依据 | 触发/告警条件 |
|---|---|---|
| R1 环境类 OOM 签名 | 横向发现 1 | 显存占用 <70% 报 OOM/IMA → 输出环境检查清单（arch 覆盖、编译 flags） |
| R2 规模外推三指标 | #1040 | relocated/alive > 51 预警；tile 交点总数 > 2^31×0.8 预警；cap 满后 relocated 续增 → runaway 告警 |
| R3 refine 事件曲线 | #1027/#1040 | dead fraction 连续 3 轮上行且 >80% 告警；relocated 计数 runaway 告警 |
| R4 归因锚点 | 横向发现 4 | 记录 densify/relocate 事件步距；崩溃步距 ≤2 → 归因该事件 |
| R5 参数体检 | 横向发现 5 | 首检查点：极小 scale/opacity 分布统计、近相机点（<1e-2）计数 |
| R6 NaN 前兆 | #248/#455 | 每 N 步 isfinite(scales/opacities)；grad norm 突变监控；"正常→2 步全崩"突变特征标记 |
| R7 导出保护 | #660 | 导出前 mask 过滤比例 ≥90% 告警；opacities 维度检查 |
| R8 环境预检 | #961/#41 | 启动时：GPU arch ∈ TORCH_CUDA_ARCH_LIST；编译 flags 含 -fno-gnu-unique（对官方 rasterizer）|

（原 v0.1 P0 的 floater 检测与 densification 停滞保留；R8 质量背离检测（#941）移入 P2——需要视觉评估，超出规则引擎范围。）

## 四、v0.3 待办

1. 扩展关键词普查：converge、quality、blur、densify、opacity（官方仓库 open+closed）
2. 扩展来源：diff-gaussian-rasterization fork、StackOverflow 3dgs 标签、3DGS Discord
3. 与 3dgs-training-debugger 技能的 60 种失败模式对齐映射（导出后合并进规则库）
4. 规则 R1-R8 的伪代码化与单元测试用例（用各 issue 的复现配置做测试数据）

---
*普查执行：星辰超级智能体（子代理逐条阅读 + 主会话审核）· 2026-10-30 · 所有 issue 引用均可回溯官方页面；置信分级见文首纪律*
