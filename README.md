# GaussWatch

> Automatic diagnosis and monitoring for 3D Gaussian Splatting training.
> 把 3DGS 训练失败从"重跑+肉眼排查"变成"实时检测+自动归因+修复建议"。

**状态**：启动于 2026-10-30 · 预研阶段（未开始写代码）

## 它是什么

训练 3DGS 时最常见的失败——显存耗尽、NaN、floaters、densification 异常——目前只能靠"重跑 + 肉眼看图"排查（官方仓库 667 个 open issues 中仅标题级命中 OOM 就有 44 个）。GaussWatch 监控训练过程，实时检测失败模式，输出归因与修复建议。

## 当前进展

- [x] 2026-10-30 项目决策（one-pager 见知识库仓库 docs/3dgs-tool-onepager-2026-10-30.md）
- [x] 命名核实：GitHub 组织名与 PyPI 包名 `gausswatch` 均未被占用
- [x] 失败模式普查 v0.1（`research/failure-mode-census-v0.1.md`）：OOM 44 例 > floater 9 例 > NaN 6 例（官方仓库）+ gsplat 10 例
- [ ] 失败模式普查 v0.2（逐条读 issue 正文 → 检测规则规格）
- [ ] MVP P0：显存监控与 OOM 预警、NaN 双重检查、floater 检测、densification 停滞检测

## 文档

- `research/failure-mode-census-v0.1.md` — 失败模式普查（检测规则知识来源）
- 决策与竞品分析：Awesome-Agent-Engineering 仓库 `docs/3dgs-tool-onepager-2026-10-30.md`、`docs/oss-growth-anatomy-2026-10-30.md`

## License

待定（倾向 Apache-2.0 / MIT，启动开发时确定）
