# GaussWatch

> Automatic diagnosis and monitoring for 3D Gaussian Splatting training.
> 把 3DGS 训练失败从"重跑+肉眼排查"变成"实时检测+自动归因+修复建议"。

**状态**：MVP 开发完成（2026-10-30），规则引擎 + 采集层 + Web 面板已实现并通过 65 项测试

## 它是什么

训练 3DGS 时最常见的失败——显存耗尽、NaN、floaters、densification 异常——目前只能靠"重跑 + 肉眼看图"排查（官方仓库 667 个 open issues 中仅标题级命中 OOM 就有 44 个）。GaussWatch 监控训练过程，实时检测失败模式，输出归因与修复建议。

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
- [ ] tiger 集群实测：torch 提取器 2 项测试（`test_signals_from_splats_shapes` / `test_export_filtered_ratio_matches_manual_mask`）需真实 torch/GPU 验证，公开 benchmark 前置
- [ ] 公开 benchmark（tiger 集群，gsplat/官方实现对比）

## Quick Start

零依赖（Python ≥ 3.10，纯标准库运行）：

```bash
git clone https://github.com/jaccen/GaussWatch.git
cd GaussWatch
python examples/demo.py            # 控制台报告：模拟训练触发 11 条告警
python examples/demo.py --panel    # 另加 Web 面板 http://127.0.0.1:8420
python -m pytest tests/ -q         # 65 passed
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
