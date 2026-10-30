"""Unit tests for GaussWatch detection rules R1-R16.

Test data derives from real issue evidence (census v0.2/v0.3):
- #1027 death cascade: dead fraction 0.73 -> 0.91 -> 0.991 -> 1.00
- #1040: relocated/alive ~79 at 7.9M/8.0M (51x51 ratio clamp); 96GB card using 30-60GB
- #1107: gaussian count flat, VRAM +4GiB per rasterizer call
- #1221: OOM with only 1281 init points (env signature)
- #41: point 1e-3 from camera center; -fno-gnu-unique fix
- #660: 1-D opacities over-filtered by any(axis=0)
- #248: hundreds of stable steps then NaN collapse
- KB patterns: OF-03 zero grad, DF-05 prune band, IF-02 zero scale, CF-07 seed variance

All 16 rules (P0/P1/P2) are runnable reference implementations -> tests must PASS.
"""
from gausswatch.rules import (
    TrainingSnapshot, check_r1, check_r2, check_r3, check_r4, check_r5,
    check_r6, check_r7, check_r8, check_r9, check_r12, check_r13, check_r14,
    check_r16, check_r10, check_r11, check_r15,
)


def snap(**kw):
    return TrainingSnapshot(**kw)


# ---- R1 env-type OOM signature ----
def test_r1_env_oom_signature():
    # #1040: 96GB card using only 30-60GB -> IMA classified as env-type
    a = check_r1(snap(error_type="IMA", vram_allocated_bytes=45 * 2**30, vram_total_bytes=96 * 2**30))
    assert a and a.severity == "critical" and "environment" in a.message

def test_r1_capacity_oom():
    # 95% utilization -> capacity-type warning, not env signature
    a = check_r1(snap(error_type="OOM", vram_allocated_bytes=95 * 2**30, vram_total_bytes=100 * 2**30))
    assert a and a.severity == "warning" and "capacity" in a.message

def test_r1_no_error_no_alert():
    assert check_r1(snap(error_type=None)) is None


# ---- R2 scale-extrapolation triple (gsplat#1040) ----
def test_r2_ratio_clamp_breach():
    # #1040 evidence: relocation at ~7.9M hits the 51x51 binomial ratio clamp (mean ~79 > 51)
    a = check_r2(snap(relocated_count=7_900_000, alive_count=8_000_000))
    assert a and "ratio-clamp" in a.message

def test_r2_tile_int32_overflow_warning():
    # int32 limit 2^31; 85% of it -> warning
    a = check_r2(snap(tile_intersections=int(0.85 * 2**31)))
    assert a and "int32" in a.message

def test_r2_runaway():
    a = check_r2(snap(relocated_runaway=True))
    assert a and "runaway" in a.message

def test_r2_healthy_quiet():
    assert check_r2(snap(relocated_count=1_000_000, alive_count=5_000_000,
                         tile_intersections=10**8)) is None


# ---- R3 refine-event death cascade (gsplat#1027) ----
def test_r3_death_cascade_real_sequence():
    # #1027 real sequence: 73% -> 91% -> 99.1% -> 100%
    s = snap(dead_fraction_history=[0.73, 0.91, 0.991])
    a = check_r3(s)
    assert a and "death-cascade" in a.message

def test_r3_healthy_fluctuation_quiet():
    assert check_r3(snap(dead_fraction_history=[0.10, 0.08, 0.12])) is None

def test_r3_insufficient_history_quiet():
    assert check_r3(snap(dead_fraction_history=[0.9, 0.95])) is None


# ---- R4 crash attribution anchor ----
def test_r4_crash_one_step_after_relocate():
    a = check_r4(snap(crash_step=6002, last_event_step=6001, crash_error_type="IMA"))
    assert a and "1 step" in a.message and "attribute" in a.message

def test_r4_far_from_event_quiet():
    assert check_r4(snap(crash_step=6100, last_event_step=6001, crash_error_type="IMA")) is None

def test_r4_non_ima_quiet():
    assert check_r4(snap(crash_step=6002, last_event_step=6001, crash_error_type="ValueError")) is None


# ---- R5 initialization self-check (official #41 path 2; KB IF-02/03) ----
def test_r5_near_camera_point():
    # #41: COLMAP point 1e-3 from camera center -> degenerate scale -> NaN -> IMA
    a = check_r5(snap(init_min_camera_dist=1e-3))
    assert a and "filter before init" in a.message

def test_r5_zero_scale_fraction():
    a = check_r5(snap(init_zero_scale_fraction=0.10))
    assert a and "zero-scale" in a.message

def test_r5_max_scale_over_bound():
    a = check_r5(snap(init_max_scale=2.5, init_max_scale_bound=1.0))
    assert a and "exceeds bound" in a.message

def test_r5_clean_init_quiet():
    assert check_r5(snap(init_min_camera_dist=0.5, init_zero_scale_fraction=0.0,
                         init_max_scale=0.4, init_max_scale_bound=1.0)) is None


# ---- R6 NaN precursor (gsplat#248/#455) ----
def test_r6_scales_nonfinite():
    a = check_r6(snap(scales_finite=False))
    assert a and "scales" in a.message

def test_r6_stable_then_collapse():
    # #248: hundreds of stable steps then NaN within 1-2 steps
    a = check_r6(snap(grad_norm_history=[1.2, 1.1, 1.3, 1.15, float("nan")]))
    assert a and "stable-then-collapse" in a.message

def test_r6_grad_normal_quiet():
    assert check_r6(snap(grad_norm_history=[1.2, 1.1, 1.3], scales_finite=True)) is None


# ---- R7 export protection (gsplat#660 maintainer-confirmed) ----
def test_r7_mask_over_filtering():
    a = check_r7(snap(export_mask_filtered_ratio=0.95))
    assert a and "95%" in a.message

def test_r7_1d_opacities_bug_shape():
    a = check_r7(snap(export_opacities_ndim=1))
    assert a and "1-D" in a.message

def test_r7_normal_export_quiet():
    assert check_r7(snap(export_mask_filtered_ratio=0.02, export_opacities_ndim=2)) is None


# ---- R8 environment preflight (official #961) ----
def test_r8_arch_missing():
    a = check_r8(snap(gpu_arch="8.0", torch_cuda_arch_list=["8.6", "8.9"]))
    assert a and "8.0" in a.message and "TORCH_CUDA_ARCH_LIST" in a.message

def test_r8_arch_present_quiet():
    assert check_r8(snap(gpu_arch="8.9", torch_cuda_arch_list=["8.0", "8.9"])) is None


# ---- R9 VRAM-count decoupling (official #1107) ----
def test_r9_leak_signature():
    # #1107: count flat (50200), VRAM +4GiB per call, monotonic
    s = snap(num_gaussians_history=[50200, 50200, 50200, 50200],
             vram_history=[8, 12, 16, 20])
    a = check_r9(s)
    assert a and "leak" in a.message

def test_r9_count_growing_quiet():
    # densification growing VRAM is expected -> no leak alert
    s = snap(num_gaussians_history=[100_000, 300_000, 600_000, 900_000],
             vram_history=[8, 12, 16, 20])
    assert check_r9(s) is None


# ---- R12 silent gradient zeroing (KB OF-03) ----
def test_r12_exact_zero():
    a = check_r12(snap(grad_norm=0.0))
    assert a and "exactly 0.0" in a.message

def test_r12_normal_grad_quiet():
    assert check_r12(snap(grad_norm=1.15)) is None


# ---- R13 prune ratio band (KB DF-05) ----
def test_r13_over_prune():
    a = check_r13(snap(pruned_count=1200, total_gaussians=10000))  # 12% > 5%
    assert a and "12.0%" in a.message and "over-pruning" in a.message

def test_r13_normal_prune_quiet():
    assert check_r13(snap(pruned_count=300, total_gaussians=10000)) is None


# ---- R14 pose-render parity (KB IF-04) ----
def test_r14_parity_divergence():
    a = check_r14(render_metric=0.3, input_metric=1.0)  # 70% divergence
    assert a and "check poses" in a.message

def test_r14_parity_ok_quiet():
    assert check_r14(render_metric=0.95, input_metric=1.0) is None


# ---- R16 densify parameter effectiveness (official #1108) ----
def test_r16_param_ineffective():
    a = check_r16(snap(render_metric=100.0, render_metric_prev=100.2))  # 0.2% < 1%
    assert a and "no behavioral effect" in a.message

def test_r16_param_effective_quiet():
    assert check_r16(snap(render_metric=110.0, render_metric_prev=100.0)) is None


# ---- R10 eval scissors (KB CF-03/CF-05) ----
def test_r10_scissors_gap():
    # train 30.0 vs test 24.0 -> 6dB gap > 3dB threshold (overfitting/data issue)
    a = check_r10(30.0, 24.0, [24.0, 24.1])
    assert a and "6.0dB" in a.message and "overfitting" in a.message

def test_r10_post_peak_decline():
    # test PSNR peaked 29.5 then two consecutive declines (CF-05)
    a = check_r10(29.0, 29.2, [28.0, 29.0, 29.5, 29.4, 29.2])
    assert a and "declining" in a.message and "29.50" in a.message

def test_r10_healthy_quiet():
    # 1dB gap and still rising -> quiet
    assert check_r10(29.0, 28.0, [27.5, 27.8, 28.0]) is None


# ---- R11 loss-component imbalance (KB OF-04/OF-06) ----
def test_r11_regularization_dominates():
    # reg 15.0 vs render 0.8 -> 18.75x > 10x threshold (OF-06)
    a = check_r11(15.0, 0.8, 0.5)
    assert a and "reg loss 15" in a.message and "10x" in a.message

def test_r11_mask_overcoverage():
    # mask covers 95% of pixels > 90% (OF-04)
    a = check_r11(None, None, 0.95)
    assert a and "95%" in a.message

def test_r11_healthy_quiet():
    assert check_r11(0.5, 1.0, 0.3) is None


# ---- R15 cross-run variance (KB CF-07) ----
def test_r15_high_seed_variance():
    # 3 seeds, std ~0.80dB > 0.5dB -> nondeterministic operators suspected
    a = check_r15([30.1, 29.2, 30.8])
    assert a and "0.80dB" in a.message and "nondeterministic" in a.message

def test_r15_stable_seeds_quiet():
    assert check_r15([30.0, 30.1, 29.95]) is None

def test_r15_insufficient_runs_quiet():
    # need >= 2 runs for a meaningful std
    assert check_r15([30.0]) is None
