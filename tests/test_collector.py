"""Tests for the event-driven collector (TrainingMonitor).

Each test drives the public event API (preflight/check_init/on_step/on_refine/
on_crash/on_export) the way a training loop would, then asserts on alerts.
"""
import threading

from gausswatch.collector import TrainingMonitor
from gausswatch.rules import RuleAlert


def test_preflight_r8_mismatch():
    m = TrainingMonitor()
    m.preflight(gpu_arch="8.0", torch_cuda_arch_list=["8.6", "8.9"])
    assert [a.rule for a in m.alerts] == ["R8"]


def test_preflight_clean_quiet():
    m = TrainingMonitor()
    m.preflight(gpu_arch="8.9", torch_cuda_arch_list=["8.6", "8.9"])
    assert m.alerts == []


def test_end_to_end_crash_attribution():
    """relocate at step N, IMA crash at N+1 with low VRAM -> R4 anchor + R1 env signature."""
    m = TrainingMonitor()
    m.on_refine(relocated_count=12_000)          # step 0 context; sets event anchor
    m._step = 4500                                # training reached step 4500
    m.on_refine(relocated_count=30_000)           # event at 4500
    m.on_crash(4501, "IMA", vram_allocated=45 * 2**30, vram_total=96 * 2**30)
    rules = [a.rule for a in m.alerts]
    assert "R4" in rules and "R1" in rules
    r1 = next(a for a in m.alerts if a.rule == "R1")
    assert r1.severity == "critical" and "environment" in r1.message


def test_r3_cascade_via_monitor():
    m = TrainingMonitor()
    for d in (0.42, 0.68, 0.91):
        m.on_refine(dead_fraction=d)
    assert any(a.rule == "R3" and "death-cascade" in a.message for a in m.alerts)


def test_r9_leak_via_monitor():
    m = TrainingMonitor(vram_check_every=1)
    for step, (v, g) in enumerate([(8, 50200), (12, 50200), (16, 50200)], start=1):
        m.on_step(step * m.vram_check_every, vram_allocated=v * 2**30,
                  num_gaussians=g)
    assert any(a.rule == "R9" and "leak" in a.message for a in m.alerts)


def test_r10_uses_test_history_only():
    """Post-peak decline must be detected on the test-PSNR series, even though
    train PSNR keeps rising (regression guard for the interleaved-history bug)."""
    m = TrainingMonitor()
    evals = [(29.8, 28.5), (30.4, 28.9), (31.2, 27.9), (31.5, 27.5)]
    for i, (tr, te) in enumerate(evals, start=1):
        m.on_step(i * 100, train_psnr=tr, test_psnr=te)
    r10 = [a for a in m.alerts if a.rule == "R10"]
    assert any("declining" in a.message for a in r10)


def test_r10_scissors_via_monitor():
    m = TrainingMonitor()
    m.on_step(100, train_psnr=30.0, test_psnr=24.0)
    assert any(a.rule == "R10" and "overfitting" in a.message for a in m.alerts)


def test_r11_loss_imbalance_via_monitor():
    m = TrainingMonitor()
    m.on_step(100, reg_loss=14.0, render_loss=0.7, loss_mask_coverage=0.95)
    r11 = [a for a in m.alerts if a.rule == "R11"]
    assert any("reg loss" in a.message for a in r11)
    assert any("95%" in a.message for a in r11)


def test_r6_nan_and_r12_zero_grad():
    m = TrainingMonitor(check_every=1)
    m.on_step(1, grad_norm=1.1)
    m.on_step(2, grad_norm=1.2)
    m.on_step(3, grad_norm=float("nan"))
    assert any(a.rule == "R6" and "stable-then-collapse" in a.message for a in m.alerts)
    m2 = TrainingMonitor(check_every=1)
    m2.on_step(1, grad_norm=0.0)
    assert any(a.rule == "R12" for a in m2.alerts)


def test_r7_export_via_monitor():
    m = TrainingMonitor()
    m.on_export(mask_filtered_ratio=0.95)
    assert any(a.rule == "R7" and "95%" in a.message for a in m.alerts)


def test_alert_dedup():
    m = TrainingMonitor()
    for _ in range(5):
        m.on_step(100, grad_norm=0.0)  # same R12 message every time
    assert len([a for a in m.alerts if a.rule == "R12"]) == 1


def test_on_alert_callback():
    seen = []
    m = TrainingMonitor(on_alert=seen.append)
    m.on_step(100, grad_norm=0.0)
    assert len(seen) == 1 and isinstance(seen[0], RuleAlert)


def test_callback_exception_swallowed():
    def boom(a):
        raise RuntimeError("user callback bug")
    m = TrainingMonitor(on_alert=boom)
    m.on_step(100, grad_norm=0.0)  # must not raise
    assert len(m.alerts) == 1


def test_partial_events_never_crash():
    m = TrainingMonitor()
    m.on_step(0)
    m.on_refine()
    m.on_crash(1, "RuntimeError")
    m.on_export()
    m.check_init()
    assert m.alerts == []


def test_max_alerts_cap():
    m = TrainingMonitor()
    for i in range(1001):  # unique prune ratios -> unique messages
        m.on_refine(pruned_count=6 + i, total_gaussians=100)
    assert len(m.alerts) == 1000


def test_stats_and_alert_records():
    m = TrainingMonitor()
    m.on_step(700, grad_norm=0.0, vram_allocated=8 * 2**30, num_gaussians=50000)
    s = m.stats()
    assert s["step"] == 700 and s["n_alerts"] == 1 and s["n_critical"] == 1
    assert s["history_sizes"]["vram"] == 1
    recs = m.alert_records()
    assert recs[0]["rule"] == "R12" and recs[0]["first_seen"] > 0


def test_thread_safety_smoke():
    m = TrainingMonitor()
    def worker(k):
        for i in range(200):
            m.on_step(k * 100000 + i, grad_norm=1.0 + i * 1e-9)
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # grad norms are all distinct positives -> no rule fires, but no corruption either
    assert len(m.grad_norm_history) == 800 and m.alerts == []
