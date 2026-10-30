"""Official 3DGS adapter (graphdeco-inria/gaussian-splatting).

Verified against `main` (fetched 2026-10-30):
- train.py                    (loop L73, loss L126, densify_and_prune call
  L169-171, eval report L221-252)
- scene/gaussian_model.py     (get_xyz L111 -> raw _xyz; get_opacity L130 ->
  sigmoid-activated; densify_and_prune L452-469: prune mask = get_opacity <
  min_opacity, no return value)

The official training script is procedural, so the integration is lighter
than the gsplat one: extract signals and feed an adapter at four points
(exact patch: examples/integrate_official_3dgs.md).

Note on rule semantics: the official repo has no teleport/MCMC relocation,
so R2 (relocation clamp) cannot fire here. The densify hook feeds
R3 (dead-fraction cascade) and R13 (prune ratio) instead.
"""
from __future__ import annotations

from ..integrations.gsplat import DEFAULT_DEAD_THRESHOLD, GsplatAdapter

__all__ = ["signals_from_gaussian_model", "wrap_densify_and_prune",
           "make_adapter", "DEFAULT_DEAD_THRESHOLD"]


def _require_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "gausswatch.integrations.official_3dgs needs torch (the core "
            "package does not). Install torch, or feed TrainingMonitor "
            "directly with numbers you already have.") from e
    return torch


def signals_from_gaussian_model(model, *,
                                dead_threshold: float = DEFAULT_DEAD_THRESHOLD) -> dict:
    """Extract monitor-ready numbers from a GaussianModel.

    get_opacity is sigmoid-activated (gaussian_model.py L129-130), so the
    dead threshold compares directly, matching densify_and_prune L460
    (prune mask = get_opacity < min_opacity).
    """
    torch = _require_torch()
    with torch.no_grad():
        xyz = model.get_xyz
        opacities = model.get_opacity
        n = xyz.shape[0]
        dead = opacities.flatten() <= dead_threshold
        dead_count = int(dead.sum().item())
        finite = bool(torch.isfinite(xyz).all().item())
        grad = getattr(model._xyz, "grad", None)
        grad_norm = None
        if grad is not None:
            grad_norm = float(grad.norm().item())
    return {
        "num_gaussians": n,
        "opacities_finite": finite,
        "grad_norm": grad_norm,
        "dead_count": dead_count,
        "dead_fraction": dead_count / n if n else 0.0,
    }


def wrap_densify_and_prune(model, adapter: GsplatAdapter):
    """Wrap GaussianModel.densify_and_prune to report refine events.

    densify_and_prune returns nothing (L452-469), so we recompute the
    opacity-dead mask before the call (same 0.005 threshold, L460) and
    report it as pruned_count — an approximation that excludes the
    oversized-points prune branch (big_points_vs / big_points_ws, L461-464).

    The step number comes from the adapter's monitor, so call
    adapter.on_step(iteration, ...) in the training loop as usual.
    """
    if not hasattr(model, "densify_and_prune"):
        raise TypeError(
            f"wrap_densify_and_prune: {model!r} has no densify_and_prune — "
            "official repo API changed; wire on_refine manually instead.")
    original = model.densify_and_prune

    def wrapped(*args, **kwargs):
        torch = _require_torch()
        with torch.no_grad():
            dead = model.get_opacity.flatten() <= DEFAULT_DEAD_THRESHOLD
            dead_count = int(dead.sum().item())
            n_before = model.get_xyz.shape[0]
            dead_fraction = dead_count / n_before if n_before else 0.0
        step = adapter.monitor.stats()["step"]
        original(*args, **kwargs)
        adapter.on_refine(step, pruned_count=dead_count,
                          dead_fraction=dead_fraction,
                          total_gaussians=model.get_xyz.shape[0])

    model.densify_and_prune = wrapped
    return model


def make_adapter(monitor=None, **kw) -> GsplatAdapter:
    """Adapter whose event API is shared with the gsplat integration."""
    return GsplatAdapter(monitor, **kw)
