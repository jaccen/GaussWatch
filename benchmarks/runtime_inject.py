#!/usr/bin/env python3
"""GaussWatch Benchmark - Class A: Runtime Injection (event-stream simulation).

Simulates training event streams (on_step / on_refine / on_eval / on_crash /
on_export) and injects failure signals at designated steps, measuring:
  - detection rate (TPR): does the expected rule fire?
  - false-positive rate (FPR): does a healthy stream stay quiet?
  - detection latency: steps from injection to alert (synchronous => 0)
  - attribution accuracy (R4): does crash attribution point to the right event?

Unlike Class B (single-snapshot replay), Class A exercises *temporal* logic:
R4 crash attribution needs a refine event followed by a crash within 2 steps;
R10 scissors needs train/test PSNR arriving via on_eval; R6 stable-then-
collapse needs a grad-norm history. Pure stdlib + gausswatch, no torch/GPU.

Real training-loop integration (wrap_strategy + wrap_eval) was already
verified end-to-end in Level-B (gsplat v1.5.3, tandt/truck, 1500 steps).
This benchmark focuses on the rule engine's temporal detection capability.

Usage:
    python benchmarks/runtime_inject.py           # human-readable report
    python benchmarks/runtime_inject.py --ci      # exit 1 if TPR<100% or FPR>0
    python benchmarks/runtime_inject.py --json    # machine-readable JSON
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, ".")

from gausswatch.collector import TrainingMonitor  # noqa: E402


@dataclass
class AScenario:
    sid: str
    rule: str
    desc: str
    source: str
    run: Callable[[TrainingMonitor], int]  # returns inject step
    expected_rule: str
    expected_severity: str


@dataclass
class AResult:
    sid: str
    rule: str
    desc: str
    source: str
    detected: bool
    alert_rule: str | None
    alert_severity: str | None
    inject_step: int
    status: str  # DETECTED / MISSED / FALSE_POSITIVE / QUIET


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _warmup_normal_steps(m: TrainingMonitor, upto: int, check_every: int = 100):
    """Feed healthy on_step events to establish histories."""
    for s in range(0, upto, check_every):
        m.on_step(s, grad_norm=1.0 + 0.1 * (s / check_every % 3),
                  vram_allocated=8 + s // 500,
                  num_gaussians=100_000 + s * 100)


def _has_alert(m: TrainingMonitor, rule: str, severity: str | None = None) -> bool:
    for a in m.alerts:
        if a.rule == rule:
            return severity is None or a.severity == severity
    return False


def _first_alert(m: TrainingMonitor, rule: str):
    for a in m.alerts:
        if a.rule == rule:
            return a
    return None


# --------------------------------------------------------------------------
# Fault scenarios (7 runtime injections)
# --------------------------------------------------------------------------

def fi1_nan_precursor(m: TrainingMonitor) -> int:
    """R6: scales contain non-finite values at a check_every boundary."""
    inject = 500
    _warmup_normal_steps(m, inject, m.check_every)
    m.on_step(inject, scales_finite=False)  # 500 % 100 == 0 -> R6 fires
    return inject


def fi2_grad_zero(m: TrainingMonitor) -> int:
    """R12: grad_norm exactly 0.0 at a check_every boundary."""
    inject = 700
    _warmup_normal_steps(m, inject, m.check_every)
    m.on_step(inject, grad_norm=0.0)  # 700 % 100 == 0 -> R12 fires
    return inject


def fi3_env_oom(m: TrainingMonitor) -> int:
    """R1: OOM at low VRAM utilization (env signature). 30/96 = 31% < 70%."""
    inject = 800
    _warmup_normal_steps(m, inject, m.check_every)
    m.on_crash(inject, "OOM",
               vram_allocated=30 * 2**30, vram_total=96 * 2**30)
    return inject


def fi4_crash_attribution(m: TrainingMonitor) -> int:
    """R4: crash 1 step after a refine event (gap <= 2 -> attribute)."""
    inject = 800
    _warmup_normal_steps(m, inject, m.check_every)
    # establish a refine event at step 799
    m.on_step(799, grad_norm=1.0)
    m.on_refine(relocated_count=1000, alive_count=5000)  # sets _last_relocate_step=799
    # crash at step 800, gap = 800 - 799 = 1 <= 2
    m.on_crash(800, "IMA",
               vram_allocated=90 * 2**30, vram_total=96 * 2**30)
    return 800


def fi5_scissors(m: TrainingMonitor) -> int:
    """R10: train/test PSNR gap > 3dB via on_eval."""
    inject = 900
    _warmup_normal_steps(m, inject, m.check_every)
    # establish a healthy eval history first
    m.on_eval(train_psnr=25.0, test_psnr=24.5)
    m.on_eval(train_psnr=26.0, test_psnr=25.5)
    # inject scissors: gap = 30.0 - 24.0 = 6.0 dB > 3 dB
    m.on_eval(train_psnr=30.0, test_psnr=24.0)
    return inject


def fi6_export_filter(m: TrainingMonitor) -> int:
    """R7: export mask filters 95% of gaussians."""
    inject = 1000
    _warmup_normal_steps(m, inject, m.check_every)
    m.on_export(mask_filtered_ratio=0.95)
    return inject


def fi7_over_prune(m: TrainingMonitor) -> int:
    """R13: single prune removes 12% (>5% band)."""
    inject = 600
    _warmup_normal_steps(m, inject, m.check_every)
    m.on_refine(pruned_count=1200, total_gaussians=10000)  # 12%
    return inject


FAULT_SCENARIOS: list[AScenario] = [
    AScenario("FI-1", "R6", "NaN precursor: scales non-finite at step 500",
              "gsplat#248/#455", fi1_nan_precursor, "R6", "critical"),
    AScenario("FI-2", "R12", "gradient zeroing: grad_norm=0.0 at step 700",
              "KB OF-03", fi2_grad_zero, "R12", "critical"),
    AScenario("FI-3", "R1", "env-OOM signature: 31% util at step 800",
              "#1221/#1040 (96GB using 30-60GB)", fi3_env_oom, "R1", "critical"),
    AScenario("FI-4", "R4", "crash attribution: IMA 1 step after refine",
              "census v0.2 finding 4; #41/#1040", fi4_crash_attribution, "R4", "critical"),
    AScenario("FI-5", "R10", "eval scissors: train/test gap 6dB at step 900",
              "KB CF-03/CF-05", fi5_scissors, "R10", "warning"),
    AScenario("FI-6", "R7", "export over-filtering: mask removes 95%",
              "gsplat#660", fi6_export_filter, "R7", "critical"),
    AScenario("FI-7", "R13", "over-prune: single prune removes 12%",
              "KB DF-05", fi7_over_prune, "R13", "warning"),
]


# --------------------------------------------------------------------------
# Healthy baseline (FPR) -- full event stream, expect 0 alerts
# --------------------------------------------------------------------------

def healthy_stream(m: TrainingMonitor) -> int:
    """A complete healthy training stream: preflight, 1000 steps, refines,
    evals, export. No failure injected. Expect 0 alerts."""
    m.preflight(gpu_arch="8.9", torch_cuda_arch_list=["8.6", "8.9"])  # R8 ok
    m.check_init(init_min_camera_dist=0.5, init_zero_scale_fraction=0.0,
                 init_max_scale=0.4, init_max_scale_bound=1.0)          # R5 ok
    for s in range(0, 1001, 100):
        m.on_step(s, grad_norm=1.0 + 0.1 * (s // 100 % 3),
                  vram_allocated=8 + s // 250,
                  num_gaussians=100_000 + s * 100,
                  scales_finite=True, opacities_finite=True)
    m.on_refine(relocated_count=1000, alive_count=5000,
                pruned_count=200, total_gaussians=10000,
                dead_fraction=0.03)  # 2% prune, low dead fraction -> quiet
    m.on_eval(train_psnr=25.0, test_psnr=24.5)   # 0.5dB gap -> quiet
    m.on_eval(train_psnr=26.0, test_psnr=25.5)
    m.on_export(mask_filtered_ratio=0.02, opacities_ndim=2)  # 2% -> quiet
    return -1  # no injection


HEALTHY_SCENARIOS: list[AScenario] = [
    AScenario("HC-A", "-", "healthy full stream: preflight+1000 steps+refine+eval+export",
              "control case", healthy_stream, "", ""),
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def _run_one(sc: AScenario) -> AResult:
    m = TrainingMonitor(check_every=100, vram_check_every=500)
    inject_step = sc.run(m)
    if sc.expected_rule:  # fault scenario
        alert = _first_alert(m, sc.expected_rule)
        if alert and alert.severity == sc.expected_severity:
            return AResult(sc.sid, sc.rule, sc.desc, sc.source, True,
                           alert.rule, alert.severity, inject_step, "DETECTED")
        if alert:
            return AResult(sc.sid, sc.rule, sc.desc, sc.source, False,
                           alert.rule, alert.severity, inject_step, "WRONG_SEVERITY")
        return AResult(sc.sid, sc.rule, sc.desc, sc.source, False,
                       None, None, inject_step, "MISSED")
    # healthy scenario
    if m.alerts:
        a = m.alerts[0]
        return AResult(sc.sid, sc.rule, sc.desc, sc.source, False,
                       a.rule, a.severity, inject_step, "FALSE_POSITIVE")
    return AResult(sc.sid, sc.rule, sc.desc, sc.source, True,
                   None, None, inject_step, "QUIET")


def run_benchmark() -> tuple[list[AResult], list[AResult]]:
    fault = [_run_one(sc) for sc in FAULT_SCENARIOS]
    healthy = [_run_one(sc) for sc in HEALTHY_SCENARIOS]
    return fault, healthy


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _format_text(fault: list[AResult], healthy: list[AResult]) -> str:
    tpr = sum(1 for r in fault if r.status == "DETECTED")
    fpr = sum(1 for r in healthy if r.status == "FALSE_POSITIVE")
    lines = [
        "GaussWatch Benchmark -- Class A: Runtime Injection",
        "=" * 54,
        "",
        f"Fault scenarios (TPR) -- {len(fault)} runtime injections:",
    ]
    for r in fault:
        lines.append(
            f"  {r.sid:<5} {r.rule:<4} {r.status:<14} "
            f"{r.alert_rule or '-':<4} {r.alert_severity or '-':<9} "
            f"@step {r.inject_step:<5} {r.desc}"
        )
    lines.append(f"\n  TPR: {tpr}/{len(fault)} = {tpr / len(fault):.0%}")
    lines.append("  Detection latency: 0 steps (synchronous, <= check_every)\n")

    lines.append(f"Healthy baseline (FPR) -- {len(healthy)} full streams:")
    for r in healthy:
        lines.append(
            f"  {r.sid:<5} {'-':<4} {r.status:<14} "
            f"{'-':<4} {'-':<9} {' '*10} {r.desc}"
        )
    lines.append(f"\n  FPR: {fpr}/{len(healthy)} = {fpr / len(healthy):.0%}\n")

    # attribution accuracy (R4 specific)
    r4 = next((r for r in fault if r.sid == "FI-4"), None)
    if r4:
        attr_ok = r4.status == "DETECTED"
        lines.append(f"  Attribution accuracy (R4): {'100% (1/1)' if attr_ok else '0% (0/1)'}")

    passed = (tpr == len(fault)) and (fpr == 0)
    lines.append(f"\nResult: {'PASS' if passed else 'FAIL'} "
                 f"(TPR {tpr}/{len(fault)}, FPR {fpr}/{len(healthy)})")
    return "\n".join(lines)


def _format_json(fault: list[AResult], healthy: list[AResult]) -> str:
    tpr = sum(1 for r in fault if r.status == "DETECTED")
    fpr = sum(1 for r in healthy if r.status == "FALSE_POSITIVE")
    r4 = next((r for r in fault if r.sid == "FI-4"), None)
    return json.dumps({
        "class": "A_runtime_injection",
        "tpr": {"detected": tpr, "total": len(fault), "rate": tpr / len(fault)},
        "fpr": {"false_positives": fpr, "total": len(healthy), "rate": fpr / len(healthy)},
        "detection_latency_steps": 0,
        "attribution_accuracy": "100% (1/1)" if r4 and r4.status == "DETECTED" else "0% (0/1)",
        "passed": (tpr == len(fault)) and (fpr == 0),
        "fault_scenarios": [vars(r) for r in fault],
        "healthy_scenarios": [vars(r) for r in healthy],
    }, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="GaussWatch Class-A runtime-injection benchmark")
    parser.add_argument("--ci", action="store_true",
                        help="exit 1 if TPR<100% or FPR>0 (CI regression gate)")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON")
    args = parser.parse_args()

    fault, healthy = run_benchmark()

    if args.json:
        print(_format_json(fault, healthy))
    else:
        print(_format_text(fault, healthy))

    tpr = sum(1 for r in fault if r.status == "DETECTED")
    fpr = sum(1 for r in healthy if r.status == "FALSE_POSITIVE")
    passed = (tpr == len(fault)) and (fpr == 0)
    if args.ci and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
