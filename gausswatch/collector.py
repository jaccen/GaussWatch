"""GaussWatch collector — event-driven bridge between a 3DGS training loop and the rule engine.

Usage (zero torch dependency on this side; the caller passes numbers it already has):

    from gausswatch.collector import TrainingMonitor

    m = TrainingMonitor()
    m.preflight(gpu_arch="8.9", torch_cuda_arch_list=["8.9"])     # R8, once
    for step, batch in train_loop:
        ...
        m.on_step(step, vram_allocated=..., vram_total=..., num_gaussians=...,
                  grad_norm=..., scales_finite=..., opacities_finite=...)
        if densify_step:
            m.on_refine(relocated_count=..., alive_count=..., dead_fraction=...,
                        pruned_count=..., total_gaussians=...)
    try:
        ...
    except RuntimeError:
        m.on_crash(step, "IMA")
    m.on_export(mask_filtered_ratio=..., opacities_ndim=...)

    print(m.alerts)
"""
from __future__ import annotations

import threading
import time

from .rules import (
    RuleAlert, TrainingSnapshot,
    check_r1, check_r2, check_r3, check_r4, check_r5, check_r6, check_r7,
    check_r8, check_r9, check_r10, check_r11, check_r12, check_r13, check_r14,
    check_r16,
)

MAX_ALERTS = 1000


class TrainingMonitor:
    """Feed training events; collect RuleAlerts. Thread-safe."""

    def __init__(self, check_every: int = 100, vram_check_every: int = 500,
                 on_alert=None):
        self.check_every = max(1, int(check_every))
        self.vram_check_every = max(1, int(vram_check_every))
        self._on_alert = on_alert
        self._lock = threading.Lock()
        self.alerts: list[RuleAlert] = []
        self._alert_keys: set[tuple] = set()
        self._alert_times: list[float] = []  # first-seen epoch time, parallel to alerts
        self._step = 0
        # histories
        self.grad_norm_history: list[float] = []
        self.dead_fraction_history: list[float] = []
        self.vram_history: list[int] = []
        self.num_gaussians_history: list[int] = []
        # train/test kept separate: R10's post-peak decline check must see a
        # homogeneous test-PSNR series, not an interleaved one
        self.train_psnr_history: list[float] = []
        self.test_psnr_history: list[float] = []
        self._last_relocate_step: int | None = None
        self._prev_render_metric: float | None = None

    # ---- internals ----
    def _emit(self, alert: RuleAlert | None):
        if alert is None:
            return
        with self._lock:
            key = (alert.rule, alert.message)
            if key in self._alert_keys:
                return
            self._alert_keys.add(key)
            if len(self.alerts) < MAX_ALERTS:
                self.alerts.append(alert)
                self._alert_times.append(time.time())
        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception:
                pass

    @staticmethod
    def _series_ok(series, min_len=3):
        return series is not None and len(series) >= min_len

    # ---- introspection (for the panel and tests) ----
    def stats(self) -> dict:
        """Cheap summary for the web panel: step, alert count, history sizes."""
        with self._lock:
            return {
                "step": self._step,
                "n_alerts": len(self.alerts),
                "n_critical": sum(1 for a in self.alerts if a.severity == "critical"),
                "n_warning": sum(1 for a in self.alerts if a.severity == "warning"),
                "history_sizes": {
                    "grad_norm": len(self.grad_norm_history),
                    "dead_fraction": len(self.dead_fraction_history),
                    "vram": len(self.vram_history),
                    "num_gaussians": len(self.num_gaussians_history),
                    "train_psnr": len(self.train_psnr_history),
                    "test_psnr": len(self.test_psnr_history),
                },
            }

    def alert_records(self) -> list[dict]:
        """Alerts as dicts with first-seen wall-clock time, newest last."""
        with self._lock:
            return [dict(rule=a.rule, severity=a.severity, message=a.message,
                         source=a.source, first_seen=t)
                    for a, t in zip(self.alerts, self._alert_times)]

    # ---- events ----
    def preflight(self, gpu_arch: str | None = None,
                  torch_cuda_arch_list: list | None = None):
        """Call once before training starts (R8)."""
        s = TrainingSnapshot(gpu_arch=gpu_arch,
                             torch_cuda_arch_list=torch_cuda_arch_list or [])
        self._emit(check_r8(s))

    def check_init(self, init_min_camera_dist: float | None = None,
                   init_zero_scale_fraction: float | None = None,
                   init_max_scale: float | None = None,
                   init_max_scale_bound: float | None = None,
                   render_metric: float | None = None,
                   input_metric: float | None = None):
        """Call once around iter<100 (R5 + R14)."""
        s = TrainingSnapshot(init_min_camera_dist=init_min_camera_dist,
                             init_zero_scale_fraction=init_zero_scale_fraction,
                             init_max_scale=init_max_scale,
                             init_max_scale_bound=init_max_scale_bound)
        self._emit(check_r5(s))
        self._emit(check_r14(render_metric, input_metric))

    def on_step(self, step: int, *, vram_allocated: int | None = None,
                vram_total: int | None = None, num_gaussians: int | None = None,
                grad_norm: float | None = None, scales_finite: bool | None = None,
                opacities_finite: bool | None = None, train_psnr: float | None = None,
                test_psnr: float | None = None, reg_loss: float | None = None,
                render_loss: float | None = None,
                loss_mask_coverage: float | None = None):
        self._step = step
        s = TrainingSnapshot(step=step, grad_norm=grad_norm,
                             scales_finite=scales_finite, opacities_finite=opacities_finite)
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
            s.grad_norm_history = self.grad_norm_history
        # periodic NaN-precursor check (R6)
        if step % self.check_every == 0:
            self._emit(check_r6(s))
            self._emit(check_r12(s))
        # VRAM/count series (R9)
        if vram_allocated is not None:
            self.vram_history.append(vram_allocated)
        if num_gaussians is not None:
            self.num_gaussians_history.append(num_gaussians)
        if (vram_allocated is not None and num_gaussians is not None
                and step % self.vram_check_every == 0
                and self._series_ok(self.vram_history) and self._series_ok(self.num_gaussians_history)):
            s9 = TrainingSnapshot(vram_history=list(self.vram_history),
                                  num_gaussians_history=list(self.num_gaussians_history))
            self._emit(check_r9(s9))
        # eval scissors (R10) and loss balance (R11) when eval numbers arrive
        if train_psnr is not None:
            self.train_psnr_history.append(train_psnr)
        if test_psnr is not None:
            self.test_psnr_history.append(test_psnr)
        if train_psnr is not None and test_psnr is not None:
            self._emit(check_r10(train_psnr, test_psnr, list(self.test_psnr_history)))
        if reg_loss is not None and render_loss is not None:
            self._emit(check_r11(reg_loss, render_loss, loss_mask_coverage))

    def on_refine(self, *, relocated_count: int | None = None,
                  alive_count: int | None = None, pruned_count: int | None = None,
                  total_gaussians: int | None = None,
                  dead_fraction: float | None = None,
                  tile_intersections: int | None = None,
                  relocated_runaway: bool = False):
        """Call on every densify/relocate event (R2/R3/R13/R16 proxy)."""
        if dead_fraction is not None:
            self.dead_fraction_history.append(dead_fraction)
        s = TrainingSnapshot(relocated_count=relocated_count,
                             alive_count=alive_count,
                             tile_intersections=tile_intersections,
                             relocated_runaway=relocated_runaway,
                             dead_fraction=dead_fraction,
                             dead_fraction_history=list(self.dead_fraction_history),
                             pruned_count=pruned_count,
                             total_gaussians=total_gaussians)
        self._emit(check_r2(s))
        self._emit(check_r3(s))
        self._emit(check_r13(s))
        if relocated_count is not None:
            self._last_relocate_step = self._step

    def on_render_metric(self, metric: float):
        """Optional: feed a proxy metric each densify cycle (R16 config-effectiveness)."""
        s = TrainingSnapshot(render_metric=metric, render_metric_prev=self._prev_render_metric)
        self._emit(check_r16(s))
        self._prev_render_metric = metric

    def on_crash(self, step: int, error_type: str, *,
                 vram_allocated: int | None = None,
                 vram_total: int | None = None):
        """Call in the exception handler of a training crash (R1 + R4)."""
        s = TrainingSnapshot(step=step, error_type=error_type,
                             crash_step=step, last_event_step=self._last_relocate_step,
                             crash_error_type=error_type,
                             vram_allocated_bytes=vram_allocated,
                             vram_total_bytes=vram_total)
        self._emit(check_r1(s))
        self._emit(check_r4(s))

    def on_export(self, *, mask_filtered_ratio: float | None = None,
                  opacities_ndim: int | None = None):
        """Call before writing .ply (R7)."""
        s = TrainingSnapshot(export_mask_filtered_ratio=mask_filtered_ratio,
                             export_opacities_ndim=opacities_ndim)
        self._emit(check_r7(s))
