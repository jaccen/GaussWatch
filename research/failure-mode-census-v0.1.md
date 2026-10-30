# GaussWatch — 失败模式普查 v0.1

> 普查日期：2026-10-30 · 方法：GitHub issues 标题级搜索（官方 3DGS 仓库 + gsplat）
> 用途：GaussWatch 检测规则引擎的知识来源（P0 功能优先级依据）
> 口径限制：仅标题级搜索，未逐条阅读 issue 正文；官方仓库搜索覆盖 open+closed 全部；频次为"标题命中数"，实际相关 issue 可能更多（部分以其他措辞描述）

## 一、数据来源与搜索结果

| 来源 | 仓库状态 | 查询 | 命中 |
|---|---|---|---|
| graphdeco-inria/gaussian-splatting（官方，667 open issues） | open+closed | `is:issue nan` | 6 |
| graphdeco-inria/gaussian-splatting | open+closed | `is:issue floater` | 9 |
| graphdeco-inria/gaussian-splatting | open+closed | `is:issue out of memory` | **44** |
| nerfstudio-project/gsplat（284 open issues） | open+closed | `is:issue nan OR diverge` | 10 |

**核心发现：显存（OOM）是 3DGS 训练第一高频失败类别**（44 个，占三类合计的 74%），远超视觉伪影（9）和数值错误（6）。

## 二、失败模式分类 v0.1（按频次排序）

### F1. 显存耗尽类（OOM）— 44 例，最高频
代表性 issue：
- [#1221](https://github.com/graphdeco-inria/gaussian-splatting/issues/1221) A100 80GB 加载点云仍 OOM
- [#961](https://github.com/graphdeco-inria/gaussian-splatting/issues/961) / [#740](https://github.com/graphdeco-inria/gaussian-splatting/issues/740) 调用 distCUDA2 时 OOM
- [#1146](https://github.com/graphdeco-inria/gaussian-splatting/issues/1146) 高分辨率图像训练 OOM
- [#1120](https://github.com/graphdeco-inria/gaussian-splatting/issues/1120) RTX 2060 12G 训练 OOM
- [#1107](https://github.com/graphdeco-inria/gaussian-splatting/issues/1107) rasterizer 运行中显存增长 4GiB（泄漏型）

子模式（初判）：
1. 数据加载阶段 OOM（全量图像驻留 GPU，[#723](https://github.com/graphdeco-inria/gaussian-splatting/issues/723) 印证"所有图像必须驻留 GPU"的设计）
2. distCUDA2 等初始化算子 OOM（KNN 邻居计算的临时显存峰值）
3. 高分辨率/大场景训练 OOM（rasterizer 中间缓冲与分辨率成正比）
4. densification 后显存阶梯式增长（高斯数量膨胀）
5. 运行时显存缓慢增长（疑似泄漏型）

### F2. 视觉伪影类（floater/artifact）— 9 例
- [#1089](https://github.com/graphdeco-inria/gaussian-splatting/issues/1089) NeRF 合成数据集大量 floaters
- [#941](https://github.com/graphdeco-inria/gaussian-splatting/issues/941) 深度图导致更多 floaters
- [#852](https://github.com/graphdeco-inria/gaussian-splatting/issues/852) 空中 floaters/artifacts 如何减少
- [#800](https://github.com/graphdeco-inria/gaussian-splatting/issues/800) 室内场景大量 floaters
- [#26](https://github.com/graphdeco-inria/gaussian-splatting/issues/26) 雾状浮动伪影

子模式（初判）：
1. 空中/雾状 floaters（欠收敛或初始化敏感）
2. 深度先验引入的伪影
3. 场景类型相关（室内/合成数据/前向场景 LLFF）
4. 随机性敏感（[#89](https://github.com/graphdeco-inria/gaussian-splatting/issues/89) 优化过程的随机性，19 条讨论）

### F3. 数值稳定类（NaN/发散）— 官方 6 + gsplat 10
- gsplat [#455](https://github.com/nerfstudio-project/gsplat/issues/455) backward 返回 NaN
- gsplat [#660](https://github.com/nerfstudio-project/gsplat/issues/660) opacities NaN 导致导出空 .ply
- gsplat [#1081](https://github.com/nerfstudio-project/gsplat/issues/1081) masked loss 在退化分支返回 NaN（2026-10，最新）
- gsplat [#248](https://github.com/nerfstudio-project/gsplat/issues/248) 数值稳定性问题（17 条讨论）
- 官方 [#174](https://github.com/graphdeco-inria/gaussian-splatting/issues/174) / [#481](https://github.com/graphdeco-inria/gaussian-splatting/issues/481) metrics 阶段 NaN

子模式（初判）：
1. 训练中 opacity/scale 产生 NaN → 传播至导出
2. backward 链路 NaN（梯度爆炸类）
3. masked loss 等自定义损失在退化分支产生 NaN
4. 评测阶段 NaN（模型已收敛但指标计算崩溃）

### F4. CUDA 崩溃类（非法内存访问/配置错误）— 交叉出现
- 官方 [#41](https://github.com/graphdeco-inria/gaussian-splatting/issues/41) / [#781](https://github.com/graphdeco-inria/gaussian-splatting/issues/781) illegal memory access
- gsplat [#1040](https://github.com/nerfstudio-project/gsplat/issues/1040) MCMC relocation 在高斯数 ≥8M 且 Blackwell（sm_120）架构下非法内存访问（2026-08）
- gsplat [#1027](https://github.com/nerfstudio-project/gsplat/issues/1027) 所有高斯低于 min_opacity（空存活集）时 relocate() 崩溃

子模式（初判）：
1. 高斯数量规模阈值触发（≥8M）
2. 架构相关（新 GPU 架构 sm_120）
3. 边界条件（空集合/退化输入）

### F5. 环境与非训练类 — 记录但不进 P0
安装/CUDA 版本（#778/#748/#1303）、多 GPU 支持（#937）、CUDA 调试方法（#827）等。属于 setup 摩擦，非训练过程诊断范畴。

## 三、对 MVP P0 的影响（普查修正）

原 one-pager P0：NaN 预警、floater 检测、densification 停滞。
**普查后修正 P0**：
1. **显存监控与 OOM 预警**（新增，证据：44 例，第一高频——显存趋势预测 + 峰值算子预警）
2. NaN 预警（训练中 + 导出前双重检查）
3. floater 聚类检测
4. densification 停滞检测（原 P0 保留）

理由：痛点频次应决定功能优先级——OOM 类用户最多，且"显存趋势预测"技术上是规则引擎可直接实现的（监控 reserved/allocated 曲线 + densification 事件计数）。

## 四、下一轮普查计划

1. **逐条阅读本轮代表性 issue 正文**（提取触发条件、复现配置、解法——转成检测规则的规格）
2. 扩展关键词：converge、quality、blur、sharp、scale、opacity、densify（官方仓库）
3. 扩展来源：diff-gaussian-rasterization fork 仓库、StackOverflow 3DGS 标签、Reddit r/computervision、3DGS Discord
4. 与 3dgs-training-debugger 系统技能的 60 种失败模式对齐映射（该技能知识已在 TeleAgent 环境，需要导出对齐）
5. 输出 v0.2：每类失败模式的"触发条件 → 可观测信号 → 检测规则草案"三段式规格

---
*普查执行：星辰超级智能体 · 2026-10-30 · 数据为当日 GitHub 页面快照*
