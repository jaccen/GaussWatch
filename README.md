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
- [x] 端到端演示：`examples/demo.py`（模拟训练触发 11 条告警，`--panel` 可选网页模式）
- [ ] torch 侧接入示例（gsplat / 官方 3DGS 训练循环的真实挂接）
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
