# GaussWatch

[![CI](https://github.com/jaccen/GaussWatch/actions/workflows/ci.yml/badge.svg)](https://github.com/jaccen/GaussWatch/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> Automatic diagnosis and monitoring for 3D Gaussian Splatting training.
> 把 3DGS 训练失败从"重跑+肉眼排查"变成"实时检测+自动归因+修复建议"。

**状态**：Level-A + Level-B GPU 盒验证全通过（2026-10-30），82 项测试全绿 + 真实 gsplat 训练闭环验证，规则引擎、采集层、Web 面板、双框架接入层与挂接注入器就绪

> **Benchmark**：Class-A 运行时注入 + Class-B 日志回放，合计 **13 个故障场景 100% 检出、6 个健康基线零误报**，覆盖 12 条规则，R4 归因准确率 100%——见 [benchmarks/results.md](benchmarks/results.md)，`python benchmarks/runtime_inject.py && python benchmarks/log_replay.py` 一键复现

## 它是什么

GaussWatch 是 3DGS（3D Gaussian Splatting）训练过程的自动诊断与监控工具：监控训练全过程，实时检测失败模式，输出归因与修复建议。

## 它解决什么问题

训练 3DGS 时最常见的失败有显存耗尽、NaN、floaters、densification 异常，目前只能靠"重跑 + 肉眼看图"排查。3DGS 原论文自述 80% 训练时间花在 Python 层，达到 Mip-NeRF360 质量需 48 小时训练，一次失败就是天级损失。官方仓库 667 个 open issues 里，仅标题级命中 OOM 的就有 44 个，失败模式知识散落在各 issue 区和论文附录，没有系统化的诊断工具。GaussWatch 把这些散落的经验沉淀成 16 条可运行规则，每条阈值绑定公开 issue 出处。

## 为什么选 GaussWatch

3DGS 工具链目前缺一层"训练诊断"：

| 层 | 玩家 | Stars | 覆盖 | 空白 |
|---|---|---|---|---|
| 光栅化库 | gsplat（Berkeley+NVIDIA，JMLR 2025） | 5.8k | CUDA kernel、性能 profiling | 训练质量诊断 |
| 工作流 | Nerfstudio（SIGGRAPH 2023） | 12.1k | 端到端流程、viewer 可视化 | 诊断（viewer 只能"看"不能"判"） |
| **训练诊断** | **GaussWatch** | — | 失败模式实时检测 + 归因 + 修复建议 | 本项目所在的层 |

具体优势：

- **阈值可溯源**：16 条规则（R1-R16）的每个判定阈值都绑定公开 GitHub issue 或失败模式编号，不靠经验拍脑袋（见 `gausswatch/rules.py` 常量注释与 `research/rule-specs-v0.4.md`）
- **零依赖、不改计算图**：规则引擎与面板只用 Python 标准库；训练侧只上报已有数值（PSNR、显存、高斯计数），不上报任何张量
- **与框架解耦**：gsplat、官方 3DGS、Nerfstudio 均可接入，事件 API 五个口子（on_step / on_refine / on_crash / on_export / preflight）
- **失败模式知识库做底座**：60 种失败模式 + 104 种 bug 模式知识库对齐，规则从真实 issue 正文证据提炼
- **学术信誉背书**：维护者为 CVPR 2025 论文 Speedy-Splat 作者，诊断规则可信度有出处

## 什么情况下用

**适用：**

- 长时间训练（小时到天级），想在中途发现失败而不是等崩溃后回看
- 批量训练（论文实验、955 方法库复现），想给每条失败自动归因
- 落地团队（数字孪生 / SLAM / 自动驾驶）没有 3DGS 调参经验，需要把调参黑盒变成可观测过程
- 训练前环境预检（GPU 架构匹配 R8、初始化自检 R5），避免起跑就错

**不适用：**

- 不做训练加速：性能 profiling 是 gsplat 的领域，GaussWatch 只做质量诊断
- 不替代可视化 viewer：看渲染图仍用 Nerfstudio viewer，GaussWatch 只判断"训练是否健康"
- 不做姿态估计 / 重建本身：只在训练过程中做诊断，不替代 COLMAP 或重建算法

## 带来什么收益

- **时间收益**：从"崩溃后回看 / 训练结束看图"到"实时检出（≤ `check_every`，默认 100 步）"，避免 48 小时级训练的整次重跑
- **诊断收益**：从"肉眼 + 重跑"到"自动归因 + 修复建议"，每条告警自带 `source` 字段指向触发它的公开 issue，可直接照着修
- **工业化收益**：落地团队无需 3DGS 经验即可定位失败，降低调参门槛，把个人经验变成可复用的规则
- **可验证**：公开 benchmark 自证，Class-A 运行时注入 + Class-B 日志回放合计 13 个故障场景 100% 检出、6 个健康基线零误报、R4 归因准确率 100%（见上方 Benchmark 引用块）

## 当前进展

- [x] 2026-10-30 项目决策（one-pager 见知识库仓库 docs/3dgs-tool-onepager-2026-10-30.md）
- [x] 命名核实：GitHub 组织名与 PyPI 包名 `gausswatch` 均未被占用
- [x] 失败模式普查 v0.1-v0.3：标题级（69 例分布）→ 正文级（14 issue 四段式规格）→ 扩展关键词 + 60 模式知识库对齐
- [x] 规则库 v2 定稿：R1-R16（P0 六条 / P1 六条 / P2 四条），全部阈值绑定公开出处
- [x] 规则引擎参考实现：`gausswatch/rules.py`（纯数值、与训练框架解耦），测试数据全部来自真实 issue 证据
- [x] 采集层：`gausswatch/collector.py`（TrainingMonitor 事件驱动 API，线程安全，与 torch 解耦）
- [x] Web 监控面板：`gausswatch/panel.py`（纯标准库 http.server，单页轮询，告警时间线）
- [x] R10/R11/R15 实现 + 全部 16 条规则可运行，测试套件 65 passed（`tests/`）
- [x] 框架接入层：`gausswatch/integrations/gsplat.py`（gsplat 适配器：strategy/eval 自动包装，refine 事件与 val PSNR 自动上报）与 `official_3dgs.py`（官方仓库适配器），API 逐行核实自两仓库 main 快照（2026-10-30）
- [x] 接入文档：`examples/integrate_gsplat_simple_trainer.md`、`examples/integrate_official_3dgs.md`（逐挂接点行号 + 代码）
- [x] 端到端演示：`examples/demo.py`（模拟训练触发 11 条告警，`--panel` 可选网页模式）
- [x] tiger 集群实测（Level-A）：torch 提取器 2 项测试 + 端到端冒烟通过（V100-32GB，torch 2.7.1+cu118，2026-10-30，`scripts/cluster_verify.sh`）
- [x] Level-B 实战验证：gsplat v1.5.3 + tandt/truck 场景 1500 步真实训练闭环（V100，2026-10-30），wrap_strategy refine 透传 + wrap_eval 双轨 PSNR 采集（17.97→21.37）均生效，0 alerts（健康训练无误报）；注入器 `scripts/inject_gausswatch.py` 5 锚点幂等挂接 + finally 报告块
- [x] 公开 benchmark：Class-A 运行时注入（7 场景）+ Class-B 日志回放（6 场景），合计 13/13 检出、0 误报，覆盖 12 条规则（见 [benchmarks/results.md](benchmarks/results.md)）

## Quick Start

零依赖（Python ≥ 3.10，纯标准库运行）：

```bash
git clone https://github.com/jaccen/GaussWatch.git
cd GaussWatch
pip install -e .[test]             # 纯标准库，秒装；[test] 仅添 pytest
python examples/demo.py            # 控制台报告：模拟训练触发 11 条告警
python examples/demo.py --panel    # 另加 Web 面板 http://127.0.0.1:8420
python -m pytest tests/ -q         # 80 passed, 2 skipped（torch 提取器测试留 GPU 盒跑，见 scripts/cluster_verify.sh）
```

在自己的训练循环中接入（只上报已有数字，无需改计算图）：

```python
from gausswatch.collector import TrainingMonitor
from gausswatch.panel import PanelServer

m = TrainingMonitor(check_every=100)
m.preflight(gpu_arch="8.9", torch_cuda_arch_list=["8.6", "8.9"])   # R8
for step, batch in train_loop:
    ...
    m.on_step(step, vram_allocated=..., num_gaussians=..., grad_norm=...,
              train_psnr=..., test_psnr=...)                          # R6/R9/R10/R12
    if densify:
        m.on_refine(dead_fraction=..., pruned_count=..., total_gaussians=...)  # R2/R3/R13
    ...
m.on_crash(step, "OOM", vram_allocated=..., vram_total=...)          # R1/R4
m.on_export(mask_filtered_ratio=...)                                 # R7

srv = PanelServer(m)             # 可选：实时面板
srv.start(block=False)
print(m.alerts)                  # [RuleAlert(rule='R3', severity='critical', ...)]
```

### 接入真实训练（gsplat / 官方仓库）

```python
# gsplat examples/simple_trainer.py，三行自动接入：
from gausswatch.integrations.gsplat import GsplatAdapter
gw = GsplatAdapter()
gw.wrap_strategy(trainer.strategy)   # densify 事件自动上报（R2/R3/R13）
gw.wrap_eval(trainer)                # val PSNR 自动上报（R10）
```

逐步挂接（每步信号、导出保护、崩溃归因）见：

- `examples/integrate_gsplat_simple_trainer.md` — gsplat main 快照逐行挂接说明
- `examples/integrate_official_3dgs.md` — 官方 graphdeco-inria 仓库挂接说明

API 核实声明：两份文档的行号与行为均逐行核对自 2026-10-30 拉取的仓库快照
（`simple_trainer.py`、`strategy/mcmc.py`、`strategy/default.py`、`exporter.py`、
官方 `train.py`、`scene/gaussian_model.py`），上游改动会导致行号漂移，请以引文代码为准。

每条告警自带 `source` 字段，指向触发它的公开 issue / 失败模式编号。

## 设计原则

- **零依赖**：规则引擎与面板只用标准库；训练侧不上报任何张量，只传数值
- **与框架解耦**：gsplat、官方 3DGS、Nerfstudio 均可接入，事件 API 五个口子
- **阈值可溯源**：每条规则的判定阈值绑定 GitHub issue 或失败模式编号（见 `research/`）
- **缺字段静默跳过**：部分上报不会误报，崩溃不会因监控本身雪崩

## 文档

- `research/rule-specs-v0.4.md` — 规则规格定稿（伪代码 + 测试映射 + B 阶段集成说明）
- `research/failure-mode-census-v0.3.md` — 扩展普查 + 知识库对齐 + 规则库 v2
- `research/failure-mode-census-v0.2.md` — 14 issue 正文级规格
- `research/failure-mode-census-v0.1.md` — 标题级普查（频次分布）
- 决策与竞品分析：Awesome-Agent-Engineering 仓库 `docs/3dgs-tool-onepager-2026-10-30.md`、`docs/oss-growth-anatomy-2026-10-30.md`

## License

Apache-2.0（见 [LICENSE](LICENSE)）
