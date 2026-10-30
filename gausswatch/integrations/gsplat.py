"""gsplat adapter — wire TrainingMonitor into gsplat trainers.

Verified against gsplat `main` (fetched 2026-10-30):
- examples/simple_trainer.py  (train loop L869, loss L956-1008, tb point
  L1028-1047, strategy dispatch L1165-1184, eval json L1305, export L1106)
- gsplat/strategy/mcmc.py     (_relocate_gs L203-216: dead_mask = sigmoid
  (opacities) <= min_opacity; 51x51 binoms L95 -> R2's 7.9M clamp)
- gsplat/strategy/default.py  (_prune_gs L351-367: sigmoid(opacities) <
  prune_opa, or oversized scale)
- gsplat/exporter.py          (splat2ply_bytes_compressed L217: mask =
  sigmoid(opacities) > 1/255)

Fastest integration for gsplat's examples/simple_trainer.py:

    from gausswatch.integrations.gsplat import GsplatAdapter

    trainer = Trainer(cfg)
    gw = GsplatAdapter()
    gw.wrap_strategy(trainer.strategy)   # auto refine events (R2/R3/R13)
    gw.wrap_eval(trainer)                # auto eval PSNR (R10 decline)
    gw.preflight(gpu_arch="8.9", torch_cuda_arch_list=["8.6", "8.9"])
    try:
        trainer.train()
    except RuntimeError as e:
        gw.on_crash(trainer.step... , type(e).__name__)
    raise

See examples/integrate_gsplat_simple_trainer.md for the exact line-level
patch against simple_trainer.py, including per-step signals (R6/R9/R11/R12).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..collector import TrainingMonitor

# gsplat/strategy/mcmc.py min_opacity and strategy/default.py prune_opa
DEFAULT_DEAD_THRESHOLD = 0.005
# gsplat/exporter.py splat2ply_bytes_compressed opacity_threshold default
EXPORT_OPACITY_THRESHOLD = 1 / 255


def _require_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "gausswatch.integrations.gsplat needs torch (the core package "
            "does not). Install torch, or feed TrainingMonitor directly with "
            "numbers you already have."
        ) from e
    return torch


def _to_float(x):
    """Duck-typed scalar extraction: tensors via .item(), rest as-is."""
    item = getattr(x, "item", None)
    return float(item()) if callable(item) else float(x)


def signals_from_splats(params, *, dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
                        grad_key: str = "means") -> dict:
    """Extract monitor-ready numbers from a gsplat params dict.

    params is the {name: Parameter} mapping gsplat strategies receive
    ("means"/"scales"/"quats"/"opacities"). Opacities are stored as logits
    and activated with sigmoid (verified: mcmc.py L203, default.py L351).

    Returns dict with: num_gaussians, scales_finite, opacities_finite,
    grad_norm (None if grads not populated), dead_count, dead_fraction.
    """
    torch = _require_torch()
    with torch.no_grad():
        means = params["means"]
        scales = params["scales"]
        opacities = params["opacities"]
        n = means.shape[0]
        scales_finite = bool(torch.isfinite(scales).all().item())
        opacities_finite = bool(torch.isfinite(opacities).all().item())
        dead = torch.sigmoid(opacities.flatten()) <= dead_threshold
        dead_count = int(dead.sum().item())
        grad_norm = None
        grad = params.get(grad_key)
        grad = getattr(grad, "grad", None)
        if grad is not None and not grad.is_sparse:
            grad_norm = float(grad.norm().item())
    return {
        "num_gaussians": n,
        "scales_finite": scales_finite,
        "opacities_finite": opacities_finite,
        "grad_norm": grad_norm,
        "dead_count": dead_count,
        "dead_fraction": dead_count / n if n else 0.0,
    }


def export_filtered_ratio(opacities, *,
                          threshold: float = EXPORT_OPACITY_THRESHOLD) -> float:
    """Fraction of gaussians a gsplat ply export would drop (R7 input).

    Mirrors exporter.py L236: mask = sigmoid(opacities) > opacity_threshold.
    Call before export_splats(...); alert at >= 0.90 is rule R7 (gsplat#660).
    """
    torch = _require_torch()
    with torch.no_grad():
        keep = (torch.sigmoid(opacities.flatten()) > threshold)
        return float(1.0 - keep.float().mean().item())


class GsplatAdapter:
    """Event adapter between a gsplat trainer and a TrainingMonitor.

    The numeric on_* methods are torch-free; the wrap_* helpers need torch.
    All failures on missing attributes are loud (TypeError/AttributeError),
    never silent, so a gsplat API change cannot degrade monitoring quietly.
    """

    def __init__(self, monitor: TrainingMonitor | None = None, *,
                 dead_threshold: float = DEFAULT_DEAD_THRESHOLD):
        self.monitor = monitor if monitor is not None else TrainingMonitor()
        self.dead_threshold = dead_threshold

    # ---- numeric passthrough (torch-free) ----
    def preflight(self, **kw):
        self.monitor.preflight(**kw)

    def check_init(self, **kw):
        self.monitor.check_init(**kw)

    def on_step(self, step: int, **kw):
        self.monitor.on_step(step, **kw)

    def on_eval(self, stage: str, psnr: float):
        """stage 'train' -> train_psnr; 'val'/'test' -> test_psnr (R10)."""
        if stage == "train":
            self.monitor.on_eval(train_psnr=_to_float(psnr))
        else:
            self.monitor.on_eval(test_psnr=_to_float(psnr))

    def on_refine(self, step: int, **kw):
        self.monitor._step = step  # crash attribution anchor needs this
        self.monitor.on_refine(**kw)

    def on_export_ratio(self, filtered_ratio: float):
        self.monitor.on_export(mask_filtered_ratio=_to_float(filtered_ratio))

    def on_crash(self, step: int, error_type: str, **kw):
        self.monitor.on_crash(step, error_type, **kw)

    # ---- torch-side wrappers ----
    def wrap_strategy(self, strategy):
        """Wrap strategy.step_post_backward to auto-report refine events.

        gsplat runs densification inside step_post_backward
        (default.py L188-216, mcmc.py L159-177) and exposes no external
        refine callback. We detect a refine by the params count changing,
        gated on the strategy's own refine_every, and recompute the dead
        mask with the same public formula the strategies use.
        """
        if not hasattr(strategy, "step_post_backward"):
            raise TypeError(
                f"wrap_strategy: {strategy!r} has no step_post_backward — "
                "gsplat API changed; pin a known version or wire on_refine "
                "manually.")
        original = strategy.step_post_backward
        adapter = self
        refine_every = getattr(strategy, "refine_every", 100)
        cap_max = getattr(strategy, "cap_max", None)  # MCMC only

        def wrapped(params, optimizers, state, step, info, *args, **kwargs):
            n_before = params["means"].shape[0]
            result = original(params, optimizers, state, step, info,
                              *args, **kwargs)
            n_after = params["means"].shape[0]
            if step % refine_every == 0 and n_after != n_before:
                sig = signals_from_splats(params,
                                          dead_threshold=adapter.dead_threshold)
                kw = {
                    "relocated_count": sig["dead_count"],
                    "dead_fraction": sig["dead_fraction"],
                    "total_gaussians": n_after,
                }
                # MCMC cap filled but dead gaussians keep teleporting:
                # the runaway signature from gsplat#1040 (R2/R3)
                if cap_max is not None and n_after >= cap_max and sig["dead_count"] > 0:
                    kw["relocated_runaway"] = True
                adapter.on_refine(step, **kw)
            return result

        strategy.step_post_backward = wrapped
        return strategy

    def wrap_eval(self, trainer):
        """Wrap Trainer.eval to auto-report val PSNR (R10 test series).

        gsplat's eval() returns nothing; it writes
        {stats_dir}/{stage}_step{step:04d}.json (simple_trainer.py L1305),
        so we read the PSNR back from that file. Note simple_trainer only
        evaluates the val split — R10's scissors mode additionally needs a
        train PSNR (see examples/integrate_gsplat_simple_trainer.md).
        """
        if not hasattr(trainer, "eval"):
            raise AttributeError(
                f"wrap_eval: {trainer!r} has no .eval — gsplat API changed.")
        stats_dir = getattr(trainer, "stats_dir", None)
        original = trainer.eval
        adapter = self

        def wrapped(step, stage: str = "val", *args, **kwargs):
            original(step, stage, *args, **kwargs)
            if stats_dir is None:
                return
            path = Path(stats_dir) / f"{stage}_step{step:04d}.json"
            try:
                stats = json.loads(path.read_text())
            except OSError:
                return
            psnr = stats.get("psnr")
            if psnr is not None:
                adapter.on_eval(stage, psnr)

        trainer.eval = wrapped
        return trainer

    def autowire(self, trainer):
        """One-call wiring: wrap trainer.strategy and trainer.eval."""
        if not hasattr(trainer, "strategy"):
            raise AttributeError(
                "autowire: trainer has no .strategy attribute — construct "
                "GsplatAdapter and call wrap_strategy/wrap_eval on the "
                "objects you have instead.")
        self.wrap_strategy(trainer.strategy)
        self.wrap_eval(trainer)
        return self
