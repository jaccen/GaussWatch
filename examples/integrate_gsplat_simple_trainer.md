# Integrating GaussWatch into gsplat's `simple_trainer.py`

Line numbers refer to `nerfstudio-project/gsplat` `main`, snapshot fetched
2026-10-30 (files verified: `examples/simple_trainer.py`,
`gsplat/strategy/mcmc.py`, `gsplat/strategy/default.py`,
`gsplat/exporter.py`). Upstream edits will shift lines — anchor on the
quoted code, not the numbers.

Net effect: ~15 lines added, zero training-code semantics changed.

## 1. Wrap strategy + eval (automatic: R2/R3/R13 refine events, R10 val PSNR)

After `trainer = Trainer(cfg)` in `simple_trainer.py` (near L800):

```python
from gausswatch.integrations.gsplat import GsplatAdapter

gw = GsplatAdapter()
gw.wrap_strategy(trainer.strategy)   # refine events, dead-fraction cascade
gw.wrap_eval(trainer)                # val PSNR -> R10 decline detection
gw.preflight(gpu_arch="<your cc>", torch_cuda_arch_list=["<built-for>"])
```

How it works (all verified against source):

- `wrap_strategy` wraps `strategy.step_post_backward` — gsplat runs all
  densification inside it (`mcmc.py` L159-177 relocate+add, `default.py`
  L188-216 grow+prune) and exposes no external refine callback. The wrapper
  detects a refine by the gaussian count changing, gated on the strategy's
  own `refine_every`, and recomputes the dead mask with the strategies'
  public formula `sigmoid(opacities) <= 0.005` (`mcmc.py` L203-204,
  `default.py` L351). For MCMC (`cap_max` present), a filled cap with
  continuing relocation raises the #1040 runaway flag (R2/R3).
- `wrap_eval` wraps `Trainer.eval`. gsplat's eval returns nothing; it
  writes `{stats_dir}/{stage}_step{step:04d}.json` (`simple_trainer.py`
  L1305), so the wrapper reads PSNR back from that file.

## 2. Per-step signals (R6 NaN / R9 leak / R11 loss balance / R12 zero grad)

In `train()`, right after `loss.backward()` (L1008), before the optimizer
step (gradients are zeroed at L1148, so this is the last moment):

```python
from gausswatch.integrations.gsplat import signals_from_splats

sig = signals_from_splats(self.splats)
gw.on_step(
    step,
    grad_norm=sig["grad_norm"],               # R6 stable-then-collapse, R12
    scales_finite=sig["scales_finite"],       # R6 direct NaN check
    opacities_finite=sig["opacities_finite"],
    num_gaussians=sig["num_gaussians"],       # R9 count series
    vram_allocated=torch.cuda.memory_allocated(),   # R9 VRAM series
    vram_total=torch.cuda.get_device_properties(0).total_memory,
    reg_loss=...,    # optional: opacity_reg/scale_reg terms if enabled (L1003-1006)
    render_loss=..., # optional: l1loss.item()
)
```

Cost note: `signals_from_splats` runs a handful of reductions on existing
tensors (~sub-millisecond at 1M gaussians). Call it every step, or gate it
with `if step % 100 == 0` if profiling demands.

## 3. Export guard (R7) — before `export_splats(...)` (L1106)

```python
from gausswatch.integrations.gsplat import export_filtered_ratio

ratio = export_filtered_ratio(opacities)   # same 1/255 rule as exporter.py L217/236
gw.on_export_ratio(ratio)                  # critical alert if >= 90% filtered
```

## 4. Crash attribution (R1/R4) — wrap the training call

In the `__main__` block (L1620-ish, `trainer.train()` call):

```python
try:
    trainer.train()
except (RuntimeError, MemoryError) as e:
    gw.on_crash(trainer.step if hasattr(trainer, "step") else -1,
                type(e).__name__,
                vram_allocated=torch.cuda.memory_allocated(),
                vram_total=torch.cuda.get_device_properties(0).total_memory)
    raise
```

R4 attributes the crash to the last densify/relocate event (within 2 steps)
— async CUDA error reporting otherwise hides this.

## 5. Train PSNR for the R10 scissors mode (optional)

`simple_trainer.py` only evaluates the val split, so wrap_eval feeds the
test series only. For the train/test scissors check (gap > 3 dB), add a
cheap train-set probe alongside eval, e.g. right before `self.eval(step)`
(L1189):

```python
with torch.no_grad():
    colors_tr, _, _ = self.stage.render(
        self.scene.id, camtoworlds=camtoworlds, Ks=Ks,
        width=width, height=height, sh_degree=sh_degree_to_use,
        near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        image_ids=image_ids, render_mode="RGB",
    )
    from gausswatch.integrations.gsplat import _to_float
    from torchmetrics.image import PeakSignalNoiseRatio
    psnr_train = _to_float(PeakSignalNoiseRatio(data_range=1.0).to(device)(
        colors_tr.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)))
    gw.on_eval("train", psnr_train)
```

## 6. Reading the results

```python
print(gw.monitor.alerts)      # list[RuleAlert] — rule, severity, message, source
gw.monitor.stats()            # step, counts, history sizes
# live dashboard (zero-dependency):
from gausswatch.panel import PanelServer
PanelServer(gw.monitor, port=8420).start(block=False)
```

Every alert carries its `source` (issue id / KB pattern) — thresholds are
never magic numbers.
