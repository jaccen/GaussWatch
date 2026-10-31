#!/usr/bin/env python3
"""GaussWatch Benchmark - Class B: Log Replay.

Replays real-issue numeric sequences through the rule engine to measure
detection rate (TPR) and false-positive rate (FPR). Pure stdlib + gausswatch,
no torch, no GPU -- runs in CI and on any laptop.

Every replay sequence traces to a public GitHub issue (see source field),
mirroring the evidence already used in tests/test_rules.py. No fabricated data.

Usage:
    python benchmarks/log_replay.py           # print human-readable report
    python benchmarks/log_replay.py --ci      # same, but exit 1 if TPR<100% or FPR>0
    python benchmarks/log_replay.py --json    # machine-readable JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

# Allow running from repo root without install (CI does pip install -e . first).
sys.path.insert(0, ".")

from gausswatch.rules import (  # noqa: E402
    TrainingSnapshot,
    check_r2,
    check_r3,
    check_r5,
    check_r8,
    check_r9,
)


@dataclass
class Scenario:
    sid: str                   # RP-1
    rule: str                  # R3
    desc: str                  # short description
    source: str                # issue provenance
    make_snapshot: Callable[[], TrainingSnapshot]
    check: Callable[[TrainingSnapshot], Any]
    expected_rule: str         # expected alert rule id
    expected_severity: str     # 'critical' | 'warning'


@dataclass
class Result:
    sid: str
    rule: str
    desc: str
    source: str
    detected: bool
    alert_rule: str | None
    alert_severity: str | None
    status: str                # DETECTED / MISSED / FALSE_POSITIVE / QUIET


# --------------------------------------------------------------------------
# Fault scenarios (TPR) -- each injects a real failure signature.
# --------------------------------------------------------------------------

FAULT_SCENARIOS: list[Scenario] = [
    Scenario(
        sid="RP-1", rule="R3",
        desc="death cascade: dead fraction 0.73 -> 0.91 -> 0.991 (3 consecutive rises >0.8)",
        source="gsplat#1027 (real recorded sequence 73->91->99.1->100%)",
        make_snapshot=lambda: TrainingSnapshot(dead_fraction_history=[0.73, 0.91, 0.991]),
        check=check_r3, expected_rule="R3", expected_severity="critical",
    ),
    Scenario(
        sid="RP-2", rule="R2",
        desc="ratio-clamp breach: relocated count 7.9M at 51x51 binomial boundary",
        source="gsplat#1040 (7.9M/8.0M relocation, ratio mean ~79 > 51)",
        make_snapshot=lambda: TrainingSnapshot(relocated_count=7_900_000, alive_count=8_000_000),
        check=check_r2, expected_rule="R2", expected_severity="critical",
    ),
    Scenario(
        sid="RP-3", rule="R2",
        desc="int32 overflow: tile intersections at 85% of 2^31",
        source="gsplat#1040 (int32 intersection offset overflow, PR #1054)",
        make_snapshot=lambda: TrainingSnapshot(tile_intersections=int(0.85 * 2**31)),
        check=check_r2, expected_rule="R2", expected_severity="critical",
    ),
    Scenario(
        sid="RP-4", rule="R9",
        desc="VRAM leak: gaussian count flat (50200) while VRAM monotonic +4GiB/call",
        source="official#1107 (count constant 50200, +4GiB per rasterizer call)",
        make_snapshot=lambda: TrainingSnapshot(
            num_gaussians_history=[50200, 50200, 50200, 50200],
            vram_history=[8, 12, 16, 20],
        ),
        check=check_r9, expected_rule="R9", expected_severity="critical",
    ),
    Scenario(
        sid="RP-5", rule="R5",
        desc="near-camera point: init point 1e-3 from camera center -> degenerate scale",
        source="official#41 (2026-09 comment: COLMAP point on camera center -> NaN->IMA)",
        make_snapshot=lambda: TrainingSnapshot(init_min_camera_dist=1e-3),
        check=check_r5, expected_rule="R5", expected_severity="critical",
    ),
    Scenario(
        sid="RP-6", rule="R8",
        desc="arch mismatch: GPU arch 8.0 not in TORCH_CUDA_ARCH_LIST [8.6, 8.9]",
        source="official#961 (Docker build missing local arch -> runtime mismatch)",
        make_snapshot=lambda: TrainingSnapshot(gpu_arch="8.0", torch_cuda_arch_list=["8.6", "8.9"]),
        check=check_r8, expected_rule="R8", expected_severity="critical",
    ),
]


# --------------------------------------------------------------------------
# Healthy baseline (FPR) -- each is a known-good training signature that
# must produce NO alert. Mirrors the *_quiet tests in tests/test_rules.py.
# --------------------------------------------------------------------------

HEALTHY_SCENARIOS: list[Scenario] = [
    Scenario(
        sid="HC-2", rule="R2",
        desc="healthy refine: 1M relocated, 1e8 tile intersections (well within limits)",
        source="control case (test_r2_healthy_quiet)",
        make_snapshot=lambda: TrainingSnapshot(
            relocated_count=1_000_000, alive_count=5_000_000, tile_intersections=10**8,
        ),
        check=check_r2, expected_rule="", expected_severity="",
    ),
    Scenario(
        sid="HC-3", rule="R3",
        desc="healthy fluctuation: dead fraction 0.10 -> 0.08 -> 0.12 (no cascade)",
        source="control case (test_r3_healthy_fluctuation_quiet)",
        make_snapshot=lambda: TrainingSnapshot(dead_fraction_history=[0.10, 0.08, 0.12]),
        check=check_r3, expected_rule="", expected_severity="",
    ),
    Scenario(
        sid="HC-5", rule="R5",
        desc="healthy init: dist 0.5, no zero-scale, scale within bound",
        source="control case (test_r5_clean_init_quiet)",
        make_snapshot=lambda: TrainingSnapshot(
            init_min_camera_dist=0.5, init_zero_scale_fraction=0.0,
            init_max_scale=0.4, init_max_scale_bound=1.0,
        ),
        check=check_r5, expected_rule="", expected_severity="",
    ),
    Scenario(
        sid="HC-8", rule="R8",
        desc="healthy preflight: GPU arch 8.9 present in ARCH_LIST",
        source="control case (test_r8_arch_present_quiet)",
        make_snapshot=lambda: TrainingSnapshot(gpu_arch="8.9", torch_cuda_arch_list=["8.0", "8.9"]),
        check=check_r8, expected_rule="", expected_severity="",
    ),
    Scenario(
        sid="HC-9", rule="R9",
        desc="healthy densify: count growing (100k->900k) with VRAM rise (not a leak)",
        source="control case (test_r9_count_growing_quiet)",
        make_snapshot=lambda: TrainingSnapshot(
            num_gaussians_history=[100_000, 300_000, 600_000, 900_000],
            vram_history=[8, 12, 16, 20],
        ),
        check=check_r9, expected_rule="", expected_severity="",
    ),
]


def _run_one(sc: Scenario) -> Result:
    snap = sc.make_snapshot()
    alert = sc.check(snap)
    if alert is None:
        if sc.expected_rule:  # fault scenario that was NOT detected
            return Result(sc.sid, sc.rule, sc.desc, sc.source, False, None, None, "MISSED")
        return Result(sc.sid, sc.rule, sc.desc, sc.source, True, None, None, "QUIET")
    # alert produced
    if sc.expected_rule:  # fault scenario detected
        ok = alert.rule == sc.expected_rule and alert.severity == sc.expected_severity
        return Result(sc.sid, sc.rule, sc.desc, sc.source, ok, alert.rule, alert.severity,
                      "DETECTED" if ok else "WRONG_RULE")
    # healthy scenario that fired -> false positive
    return Result(sc.sid, sc.rule, sc.desc, sc.source, False, alert.rule, alert.severity,
                  "FALSE_POSITIVE")


def run_benchmark() -> tuple[list[Result], list[Result]]:
    fault_results = [_run_one(sc) for sc in FAULT_SCENARIOS]
    healthy_results = [_run_one(sc) for sc in HEALTHY_SCENARIOS]
    return fault_results, healthy_results


def _format_text(fault: list[Result], healthy: list[Result]) -> str:
    tpr = sum(1 for r in fault if r.status == "DETECTED")
    fpr = sum(1 for r in healthy if r.status == "FALSE_POSITIVE")
    lines = [
        "GaussWatch Benchmark -- Class B: Log Replay",
        "=" * 52,
        "",
        f"Fault scenarios (TPR) -- {len(fault)} real-issue signatures:",
    ]
    for r in fault:
        lines.append(
            f"  {r.sid:<5} {r.rule:<3} {r.status:<12} "
            f"{r.alert_rule or '-':<4} {r.alert_severity or '-':<9} {r.desc}"
        )
    lines.append(f"\n  TPR: {tpr}/{len(fault)} = {tpr / len(fault):.0%}\n")

    lines.append(f"Healthy baseline (FPR) -- {len(healthy)} control cases:")
    for r in healthy:
        lines.append(
            f"  {r.sid:<5} {r.rule:<3} {r.status:<12} "
            f"{'-':<4} {'-':<9} {r.desc}"
        )
    lines.append(f"\n  FPR: {fpr}/{len(healthy)} = {fpr / len(healthy):.0%}\n")

    passed = (tpr == len(fault)) and (fpr == 0)
    lines.append(f"Result: {'PASS' if passed else 'FAIL'} "
                 f"(TPR {tpr}/{len(fault)}, FPR {fpr}/{len(healthy)})")
    return "\n".join(lines)


def _format_json(fault: list[Result], healthy: list[Result]) -> str:
    tpr = sum(1 for r in fault if r.status == "DETECTED")
    fpr = sum(1 for r in healthy if r.status == "FALSE_POSITIVE")
    return json.dumps({
        "class": "B_log_replay",
        "tpr": {"detected": tpr, "total": len(fault), "rate": tpr / len(fault)},
        "fpr": {"false_positives": fpr, "total": len(healthy), "rate": fpr / len(healthy)},
        "passed": (tpr == len(fault)) and (fpr == 0),
        "fault_scenarios": [vars(r) for r in fault],
        "healthy_scenarios": [vars(r) for r in healthy],
    }, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="GaussWatch Class-B log-replay benchmark")
    parser.add_argument("--ci", action="store_true",
                        help="exit 1 if TPR<100% or FPR>0 (for CI regression gate)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
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
