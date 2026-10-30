"""GaussWatch end-to-end demo: a simulated 3DGS run with known failure signatures.

Walks a fake 4500-step training loop whose numbers reproduce real issue
evidence (census v0.2/v0.3), fires every wired rule, then prints the alerts.
Pass --panel to also serve the web dashboard.

    python examples/demo.py            # console report
    python examples/demo.py --panel    # console report + http://127.0.0.1:8420

Zero dependencies beyond the standard library; no torch, no GPU needed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gausswatch.collector import TrainingMonitor  # noqa: E402


def banner(text: str):
    print(f"\n--- {text} " + "-" * max(0, 58 - len(text)))


def simulate(m: TrainingMonitor):
    GiB = 2**30

    banner("preflight (R8)")
    # GPU is arch 8.9 but the installed wheel was built for 8.6 only -> the
    # mismatch is caught before a single training step (official #961 class)
    m.preflight(gpu_arch="8.9", torch_cuda_arch_list=["8.6"])

    banner("training loop (events only; real code forwards the numbers it has)")
    timeline = [
        # (step, event kwargs for on_step)
        (500,  dict(vram_allocated=8 * GiB, num_gaussians=50_000, grad_norm=1.1)),
        (1000, dict(vram_allocated=9 * GiB, num_gaussians=60_000, grad_norm=1.15)),
        (1500, dict(vram_allocated=10 * GiB, num_gaussians=80_000, grad_norm=1.2)),
        (2000, dict(vram_allocated=11 * GiB, num_gaussians=110_000)),
    ]
    for step, kw in timeline:
        m.on_step(step, **kw)

    # eval passes are separate events (like gsplat Trainer.eval)
    m.on_eval(train_psnr=29.8, test_psnr=28.5)

    m.on_refine(dead_fraction=0.42, pruned_count=200, total_gaussians=10_000)

    # over-pruning: a single prune removed 8% of gaussians (KB DF-05 band is 5%)
    m.on_refine(dead_fraction=0.68, pruned_count=800, total_gaussians=10_000)

    # eval scissors: train 31.2 vs test 27.9 -> 3.3dB gap (KB CF-03 threshold 3dB)
    m.on_step(3000, vram_allocated=12 * GiB, num_gaussians=140_000, grad_norm=1.18)
    m.on_eval(train_psnr=30.4, test_psnr=28.9)   # still climbing

    banner("densify events (R3 death cascade)")

    # death cascade: dead fraction 0.42 -> 0.68 -> 0.91 over three refines
    # (#1027 real sequence shape: 73 -> 91 -> 99.1 -> 100%)
    m.on_refine(dead_fraction=0.91)

    # loss-component imbalance: reg loss 20x render loss; mask covers 95% of
    # pixels (KB OF-06 + OF-04); silent zero grad (KB OF-03)
    m.on_step(4000, grad_norm=0.0, reg_loss=14.0, render_loss=0.7,
              loss_mask_coverage=0.95)
    m.on_eval(train_psnr=31.2, test_psnr=27.9)   # 3.3dB scissors gap

    # NaN collapse after stable steps (gsplat#248 signature) + post-peak
    # test-PSNR decline (KB CF-05)
    m.on_step(4500, grad_norm=float("nan"))
    m.on_eval(train_psnr=31.5, test_psnr=27.5)   # two declines after peak 28.9
    # a relocate event lands exactly at step 4500 -> becomes the attribution
    # anchor for the crash one step later (R4)
    m.on_refine(relocated_count=30_000, alive_count=140_000)

    banner("crash (R1 + R4)")
    # IMA one step after the last relocate event, at 47% VRAM utilization ->
    # environment signature, and attribution anchors to the step-4500 refine
    m.on_crash(4501, "IMA", vram_allocated=45 * GiB, vram_total=96 * GiB)

    banner("export guard (R7)")
    # export mask would silently drop 95% of gaussians (gsplat#660 class)
    m.on_export(mask_filtered_ratio=0.95)


def report(m: TrainingMonitor):
    stats = m.stats()
    print(f"\nstep reached : {stats['step']}")
    print(f"alerts       : {stats['n_alerts']} "
          f"({stats['n_critical']} critical / {stats['n_warning']} warning)\n")
    for a in m.alert_records():
        t = time.strftime("%H:%M:%S", time.localtime(a["first_seen"]))
        print(f"  [{t}] {a['rule']:<4} {a['severity']:<8} {a['message']}")
        print(f"        source: {a['source']}")


def main():
    ap = argparse.ArgumentParser(description="GaussWatch end-to-end demo")
    ap.add_argument("--panel", action="store_true",
                    help="also serve the web dashboard on port 8420")
    args = ap.parse_args()

    m = TrainingMonitor()
    simulate(m)
    report(m)

    if args.panel:
        from gausswatch.panel import serve
        print("\nserving panel — open the URL below (Ctrl+C to stop)")
        serve(m, port=8420)


if __name__ == "__main__":
    main()
