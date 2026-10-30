"""GaussWatch rule engine — pure-numerical detection rules (R1-R16).

Design contract:
- Rules consume a snapshot of training state (plain Python numerics), never touch torch.
- Each rule returns a RuleAlert or None. No rule raises on degenerate input
  (missing fields -> skip), so the collector layer can feed partial snapshots.
- Every threshold traces to a public source (GitHub issue / 3dgs-training-debugger
  KB pattern id), documented in research/failure-mode-census-v0.2.md and -v0.3.md.

P0 rules ship runnable reference implementations; P1/P2 rules raise
NotImplementedError until B-phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuleAlert:
    rule: str
    severity: str  # 'warning' | 'critical'
    message: str
    source: str  # evidence provenance


# ---- shared constants (provenance in comments) ----
RATIO_CLAMP = 51.0            # 51x51 binomial ratio table overflows, gsplat#1040
TILE_INTERSECT_LIMIT = 2**31  # int32 intersection offsets overflow, gsplat#1040 (PR #1054)
LOW_VRAM_UTIL = 0.70          # env-OOM signature threshold, census v0.2 finding 1
NEAR_CAMERA_DIST = 1e-2       # degenerate init distance, official #41 (2026-09 comment)
MASK_FILTER_RATIO = 0.90      # export protection, gsplat#660
PRUNE_RATIO_BAND = 0.05       # normal prune only removes transparent gaussians, KB DF-05
TRAIN_TEST_GAP_DB = 3.0       # train/test scissors, KB CF-03
CROSS_RUN_VAR_DB = 0.5        # nondeterminism flag, KB CF-07
REG_LOSS_RATIO = 10.0         # regularization dominates, KB OF-06
MASK_COVER_HIGH = 0.90        # loss mask over-coverage, KB OF-04


@dataclass
class TrainingSnapshot:
    """Partial snapshot of one training step. None = not collected this step."""
    step: int = 0
    vram_allocated_bytes: int | None = None
    vram_total_bytes: int | None = None
    error_type: str | None = None          # 'OOM' | 'IMA' | None (R1)
    relocated_count: int | None = None     # last refine relocation size (R2/R3)
    alive_count: int | None = None         # R2
    tile_intersections: int | None = None  # R2
    relocated_runaway: bool = False        # counter grows after cap filled (R2/R3)
    dead_fraction: float | None = None     # per-refine dead ratio, R3 (#1027)
    dead_fraction_history: list = field(default_factory=list)
    crash_step: int | None = None          # R4
    last_event_step: int | None = None     # last densify/relocate step (R4)
    crash_error_type: str | None = None    # R4
    init_min_camera_dist: float | None = None      # R5 (#41 path 2)
    init_zero_scale_fraction: float | None = None  # R5 (IF-02)
    init_max_scale: float | None = None    # R5 (IF-03)
    init_max_scale_bound: float | None = None
    scales_finite: bool | None = None      # R6 (#248/#455)
    opacities_finite: bool | None = None
    grad_norm: float | None = None         # R6 spike / R12 zero
    grad_norm_history: list = field(default_factory=list)
    export_mask_filtered_ratio: float | None = None  # R7 (#660)
    export_opacities_ndim: int | None = None
    gpu_arch: str | None = None            # R8 (#961)
    torch_cuda_arch_list: list = field(default_factory=list)
    num_gaussians_history: list = field(default_factory=list)  # R9 (#1107)
    vram_history: list = field(default_factory=list)
    train_psnr: float | None = None        # R10 (CF-03/05)
    test_psnr: float | None = None
    psnr_history: list = field(default_factory=list)
    reg_loss: float | None = None          # R11 (OF-06)
    render_loss: float | None = None
    loss_mask_coverage: float | None = None
    pruned_count: int | None = None        # R13 (DF-05)
    total_gaussians: int | None = None
    render_metric: float | None = None     # R16 (#1108 radii distribution proxy)
    render_metric_prev: float | None = None


def _alert(rule, severity, message, source):
    return RuleAlert(rule=rule, severity=severity, message=message, source=source)


# ---- R1: env-type OOM signature (census v0.2, horizontal finding 1) ----
def check_r1(snap: TrainingSnapshot) -> RuleAlert | None:
    """OOM/IMA at low VRAM utilization -> environment signature, not capacity."""
    if snap.error_type not in ("OOM", "IMA"):
        return None
    util = None
    if snap.vram_allocated_bytes and snap.vram_total_bytes:
        util = snap.vram_allocated_bytes / snap.vram_total_bytes
    if util is not None and util < LOW_VRAM_UTIL:
        return _alert("R1", "critical",
                      f"{snap.error_type} at {util:.0%} VRAM utilization — environment "
                      "signature: check TORCH_CUDA_ARCH_LIST / build flags before reducing load",
                      "census v0.2 finding 1; issues #1221 #961 #1040")
    return _alert("R1", "warning",
                  f"{snap.error_type} at high VRAM utilization — capacity-type",
                  "census v0.2 finding 1")


# ---- R2: scale-extrapolation triple (gsplat#1040) ----
def check_r2(snap: TrainingSnapshot) -> RuleAlert | None:
    msgs = []
    if snap.relocated_count and snap.relocated_count >= 7_900_000:
        msgs.append(f"relocation size {snap.relocated_count} at 51x51 ratio-clamp boundary (~7.9M)")
    if snap.tile_intersections and snap.tile_intersections > 0.8 * TILE_INTERSECT_LIMIT:
        msgs.append(f"tile intersections {snap.tile_intersections} approaching 2^31 int32 overflow")
    if snap.relocated_runaway:
        msgs.append("relocated count keeps growing after cap filled (runaway)")
    if msgs:
        return _alert("R2", "critical", "; ".join(msgs), "gsplat#1040 (PR #1054, #1060)")
    return None


# ---- R3: refine-event curve family (gsplat#1027 cascade + KB DF-07) ----
def check_r3(snap: TrainingSnapshot) -> RuleAlert | None:
    h = snap.dead_fraction_history
    if len(h) >= 3:
        last3 = h[-3:]
        if all(last3[i] < last3[i + 1] for i in range(2)) and last3[-1] > 0.8:
            return _alert("R3", "critical",
                          f"dead fraction rising 3 consecutive refines to {last3[-1]:.1%} — "
                          "death-cascade precursor", "gsplat#1027 (73→91→99.1→100%)")
    if snap.relocated_runaway:
        return _alert("R3", "critical", "relocated runaway", "gsplat#1040")
    return None


# ---- R4: crash attribution anchor (census v0.2 finding 4) ----
def check_r4(snap: TrainingSnapshot) -> RuleAlert | None:
    if snap.crash_step is None or snap.last_event_step is None:
        return None
    if snap.crash_error_type not in ("OOM", "IMA"):
        return None
    gap = snap.crash_step - snap.last_event_step
    if 0 <= gap <= 2:
        return _alert("R4", "critical",
                      f"crash {gap} step(s) after densify/relocate event at step "
                      f"{snap.last_event_step} — attribute to that event (async error "
                      "reporting misleads)", "census v0.2 finding 4; #41 #1040")
    return None


# ---- R5: initialization self-check, part 1 (official #41 path 2; KB IF-02/03) ----
def check_r5(snap: TrainingSnapshot) -> RuleAlert | None:
    msgs = []
    if snap.init_min_camera_dist is not None and snap.init_min_camera_dist < NEAR_CAMERA_DIST:
        msgs.append(f"point {snap.init_min_camera_dist:.1e} from camera center — filter before init")
    if snap.init_zero_scale_fraction is not None and snap.init_zero_scale_fraction > 0:
        msgs.append(f"{snap.init_zero_scale_fraction:.1%} zero-scale gaussians at init")
    if (snap.init_max_scale is not None and snap.init_max_scale_bound is not None
            and snap.init_max_scale > snap.init_max_scale_bound):
        msgs.append(f"max scale {snap.init_max_scale:.3g} exceeds bound {snap.init_max_scale_bound:.3g}")
    if msgs:
        return _alert("R5", "critical", "; ".join(msgs),
                      "official #41 (2026-09 comment); KB IF-02/IF-03")
    return None


# ---- R6: NaN precursor (gsplat#248/#455) ----
def check_r6(snap: TrainingSnapshot) -> RuleAlert | None:
    if snap.scales_finite is False or snap.opacities_finite is False:
        which = "scales" if snap.scales_finite is False else "opacities"
        return _alert("R6", "critical", f"{which} contain non-finite values",
                      "gsplat#248/#455")
    h = snap.grad_norm_history
    if len(h) >= 3:
        last = h[-1]
        nonfinite = last != last or last == float("inf")  # NaN != NaN, so use self-inequality
        if nonfinite:
            normal = [x for x in h[:-1] if x == x and 0 < x < float("inf")]
            if len(normal) >= 2:
                return _alert("R6", "critical",
                              f"grad norm went non-finite after {len(normal)} stable steps — "
                              "'stable-then-collapse' signature", "gsplat#248")
    return None


# ---- R7: export protection (gsplat#660, maintainer-confirmed) ----
def check_r7(snap: TrainingSnapshot) -> RuleAlert | None:
    msgs = []
    if snap.export_mask_filtered_ratio is not None and snap.export_mask_filtered_ratio >= MASK_FILTER_RATIO:
        msgs.append(f"export mask filters {snap.export_mask_filtered_ratio:.0%} of gaussians")
    if snap.export_opacities_ndim == 1:
        msgs.append("opacities is 1-D — known any(axis=0) over-filtering bug shape")
    if msgs:
        return _alert("R7", "critical", "; ".join(msgs), "gsplat#660 (PR #663)")
    return None


# ---- R8: environment preflight (official #961/#41; KB 7.3) ----
def check_r8(snap: TrainingSnapshot) -> RuleAlert | None:
    if snap.gpu_arch and snap.torch_cuda_arch_list:
        if snap.gpu_arch not in snap.torch_cuda_arch_list:
            return _alert("R8", "critical",
                          f"GPU arch {snap.gpu_arch} not in TORCH_CUDA_ARCH_LIST "
                          f"{snap.torch_cuda_arch_list} — build/runtime mismatch",
                          "official #961")
    return None


# ---- R9: VRAM-count decoupling (official #1107; KB MF-05/MF-03) ----
def check_r9(snap: TrainingSnapshot) -> RuleAlert | None:
    v, g = snap.vram_history, snap.num_gaussians_history
    if len(v) >= 3 and len(g) >= 3:
        count_stable = (max(g) - min(g)) / max(g[-1], 1) < 0.01
        vram_monotonic = all(v[i] < v[i + 1] for i in range(len(v) - 1))
        if count_stable and vram_monotonic:
            return _alert("R9", "critical",
                          "VRAM grows monotonically while gaussian count is flat — "
                          "buffer leak / optimizer bloat", "official #1107; KB MF-05+MF-03")
    return None


# ---- R12: silent gradient zeroing (KB OF-03; census v0.3 top-10 #6) ----
def check_r12(snap: TrainingSnapshot) -> RuleAlert | None:
    if snap.grad_norm == 0.0:
        return _alert("R12", "critical",
                      "grad norm exactly 0.0 — silent failure (detached graph / broken "
                      "computation path)", "KB OF-03")
    return None


# ---- R13: prune ratio band (KB DF-05) ----
def check_r13(snap: TrainingSnapshot) -> RuleAlert | None:
    if snap.pruned_count is None or not snap.total_gaussians:
        return None
    ratio = snap.pruned_count / snap.total_gaussians
    if ratio > PRUNE_RATIO_BAND:
        return _alert("R13", "warning",
                      f"single prune removed {ratio:.1%} (>5% band) — over-pruning",
                      "KB DF-05")
    return None


# ---- R14: initialization self-check, part 2 — pose-render parity (KB IF-04) ----
def check_r14(render_metric: float | None, input_metric: float | None) -> RuleAlert | None:
    """Compare an early pose-render metric against the input reference metric.

    Divergence >50% flags wrong poses/initialization (KB IF-04).
    """
    if render_metric is None or input_metric is None or input_metric == 0:
        return None
    div = abs(render_metric - input_metric) / abs(input_metric)
    if div > 0.5:
        return _alert("R14", "critical",
                      f"pose-render metric {render_metric:.3g} diverges {div:.0%} from "
                      f"input reference {input_metric:.3g} — check poses/initialization",
                      "KB IF-04")
    return None


# ---- R16: densify parameter effectiveness (official #1108/#741/#544) ----
def check_r16(snap: TrainingSnapshot) -> RuleAlert | None:
    if snap.render_metric is None or snap.render_metric_prev is None:
        return None
    prev = snap.render_metric_prev
    if prev == 0:
        return None
    if abs(snap.render_metric - prev) / abs(prev) < 0.01:
        return _alert("R16", "warning",
                      "densification parameter shows no behavioral effect — "
                      "config-code consistency check", "official #1108/#741/#544")
    return None


# ---- R10: eval scissors (KB CF-03/CF-05) ----
def check_r10(train_psnr, test_psnr, psnr_history):
    """Train/test PSNR scissors gap, or post-peak decline of test PSNR."""
    msgs = []
    if train_psnr is not None and test_psnr is not None:
        gap = train_psnr - test_psnr
        if gap > TRAIN_TEST_GAP_DB:
            msgs.append(f"train/test gap {gap:.1f}dB > {TRAIN_TEST_GAP_DB}dB (overfitting/data issue)")
    if psnr_history and len(psnr_history) >= 3:
        # peak followed by two consecutive declines
        peak = max(psnr_history)
        peak_idx = psnr_history.index(peak)
        tail = psnr_history[peak_idx:]
        if len(tail) >= 3 and tail[-1] < tail[-2] < tail[-2 - 1]:
            msgs.append(f"test PSNR declining after peak {peak:.2f}")
    if msgs:
        return _alert("R10", "warning", "; ".join(msgs), "KB CF-03+CF-05")
    return None


# ---- R11: loss-component imbalance (KB OF-04/OF-06) ----
def check_r11(reg_loss, render_loss, mask_coverage):
    """Regularization dominating render loss, or loss mask over-coverage."""
    msgs = []
    if reg_loss is not None and render_loss is not None and render_loss > 0:
        if reg_loss > REG_LOSS_RATIO * render_loss:
            msgs.append(f"reg loss {reg_loss:.3g} > {REG_LOSS_RATIO:.0f}x render loss {render_loss:.3g}")
    if mask_coverage is not None and mask_coverage > MASK_COVER_HIGH:
        msgs.append(f"loss mask covers {mask_coverage:.0%} of pixels (>90%)")
    if msgs:
        return _alert("R11", "warning", "; ".join(msgs), "KB OF-04+OF-06")
    return None


# ---- R15: cross-run variance (KB CF-07) ----
def check_r15(run_psnrs):
    """Same-config multi-seed PSNR std > 0.5dB -> nondeterministic ops."""
    import statistics
    if not run_psnrs or len(run_psnrs) < 2:
        return None
    std = statistics.stdev(run_psnrs)
    if std > CROSS_RUN_VAR_DB:
        return _alert("R15", "warning",
                      f"cross-run PSNR std {std:.2f}dB > {CROSS_RUN_VAR_DB}dB — "
                      "nondeterministic operators suspected", "KB CF-07")
    return None
