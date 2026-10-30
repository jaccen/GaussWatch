# Integrating GaussWatch into the official `gaussian-splatting` repo

Line numbers refer to `graphdeco-inria/gaussian-splatting` `main`, snapshot
fetched 2026-10-30 (files verified: `train.py`,
`scene/gaussian_model.py`). Anchor on quoted code, not line numbers.

Rule-coverage note: the official repo has no MCMC teleport, so R2
(relocation clamp) cannot fire here. The densify hook feeds R3
(dead-fraction cascade) and R13 (prune ratio); R10/R11/R12/R6/R7/R8/R1/R4
work as in the gsplat integration.

## 1. Adapter + preflight + init check (after `gaussians.training_setup(opt)`, L52)

```python
from gausswatch.integrations.official_3dgs import (make_adapter,
                                                  signals_from_gaussian_model)

gw = make_adapter()
gw.preflight(gpu_arch="<your cc>", torch_cuda_arch_list=["<built-for>"])
```

## 2. Densify events (R3/R13) — wrap once after model setup

```python
from gausswatch.integrations.official_3dgs import wrap_densify_and_prune
wrap_densify_and_prune(gaussians, gw)
```

`densify_and_prune` returns nothing (`gaussian_model.py` L452-469), so the
wrapper recomputes the opacity-dead mask before the call (same 0.005
threshold as L460) and reports it as `pruned_count` — an approximation that
excludes the oversized-points branch (L461-464). The step number comes from
`gw.monitor`, fed by the on_step call below.

## 3. Per-step signals — after `loss.backward()` (L142)

```python
sig = signals_from_gaussian_model(gaussians)
gw.on_step(
    iteration,
    grad_norm=sig["grad_norm"],               # R6 / R12
    opacities_finite=sig["opacities_finite"],
    num_gaussians=sig["num_gaussians"],       # R9
    vram_allocated=torch.cuda.memory_allocated(),
    vram_total=torch.cuda.get_device_properties(0).total_memory,
)
```

## 4. Eval PSNR (R10) — inside `training_report` (L221-252)

The report loops test and train configs; forward each averaged PSNR:

```python
gw.on_eval(config["name"], psnr_test)   # 'test' -> test series, 'train' -> train series
```

## 5. Export guard (R7) — before `scene.save(iteration)` (L161)

The official `.ply` writer keeps all gaussians, so R7's mask rule applies
only if you post-filter exports. If you do, pass the filtered ratio:

```python
gw.on_export_ratio(filtered_ratio)
```

## 6. Crash attribution (R1/R4) — wrap the `training(...)` call (L282)

```python
try:
    training(lp.extract(args), op.extract(args), pp.extract(args), ...)
except (RuntimeError, MemoryError) as e:
    gw.on_crash(last_iteration, type(e).__name__,
                vram_allocated=torch.cuda.memory_allocated(),
                vram_total=torch.cuda.get_device_properties(0).total_memory)
    raise
```

`last_iteration` is the loop variable if you hoist `iteration` out of the
train loop (one line: initialize `iteration = first_iter` before L73).
