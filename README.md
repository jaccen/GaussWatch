# GaussWatch

> Automatic diagnosis and monitoring for 3D Gaussian Splatting training.
> 把 3DGS 训练失败从"重跑+肉眼排查"变成"实时检测+自动归因+修复建议"。

**状态**：预研完成（2026-10-30），规则引擎规格定稿，进入 MVP 开发阶段

## 它是什么

训练 3DGS 时最常见的失败——显存耗尽、NaN、floaters、densification 异常——目前只能靠"重跑 + 肉眼看图"排查（官方仓库 667 个 open issues 中仅标题级命中 OOM 就有 44 个）。GaussWatch 监控训练过程，实时检测失败模式，输出归因与修复建议。

## 当前进展

- [x] 2026-10-30 项目决策（one-pager 见知识库仓库 docs/3dgs-tool-onepager-2026-10-30.md）
- [x] 命名核实：GitHub 组织名与 PyPI 包名 `gausswatch` 均未被占用
- [x] 失败模式普查 v0.1-v0.3：标题级（69 例分布）→ 正文级（14 issue 四段式规格）→ 扩展关键词 + 60 模式知识库对齐
- [x] 规则库 v2 定稿：R1-R16（P0 六条 / P1 六条 / P2 四条），全部阈值绑定公开出处
- [x] 规则引擎参考实现：`gausswatch/rules.py`（纯数值、与训练框架解耦）+ 测试套件 35 passed + 3 xfailed（`tests/test_rules.py`，测试数据全部来自真实 issue 证据）
- [ ] MVP：采集层（torch 侧 TrainingSnapshot 填充）+ Web 监控面板 + R10/R11/R15 实现
- [ ] 公开 benchmark（tiger 集群，gsplat/官方实现对比）

## 文档

- `research/rule-specs-v0.4.md` — 规则规格定稿（伪代码 + 测试映射 + B 阶段集成说明）
- `research/failure-mode-census-v0.3.md` — 扩展普查 + 知识库对齐 + 规则库 v2
- `research/failure-mode-census-v0.2.md` — 14 issue 正文级规格
- `research/failure-mode-census-v0.1.md` — 标题级普查（频次分布）
- 决策与竞品分析：Awesome-Agent-Engineering 仓库 `docs/3dgs-tool-onepager-2026-10-30.md`、`docs/oss-growth-anatomy-2026-10-30.md`

## License

待定（倾向 Apache-2.0 / MIT，启动开发时确定）
