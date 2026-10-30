"""Tests for framework integrations.

The adapter orchestration logic (event wiring, refine detection, eval-json
reading, cap/runaway semantics) is tested with fakes — no torch required.
The torch-side extractors (signals_from_splats / export_filtered_ratio /
signals_from_gaussian_model) are skipped unless torch is installed; run
them on the GPU box (tiger) before tagging a release.
"""
import json
from pathlib import Path

import pytest

from gausswatch.collector import TrainingMonitor
from gausswatch.integrations import gsplat as gw_gsplat
from gausswatch.integrations.gsplat import GsplatAdapter, _to_float


# ---- fakes ----
class FakeMeans:
    def __init__(self, n):
        self.shape = (n,)


class FakeStrategy:
    """Mimics gsplat strategies' public surface used by wrap_strategy."""
    refine_every = 100

    def __init__(self, cap_max=None):
        self.cap_max = cap_max
        self.calls = []

    def step_post_backward(self, params, optimizers, state, step, info, **kw):
        self.calls.append(step)
        params["means"] = FakeMeans(params["means"].shape[0] + 10)  # refine grew


class FakeFlatStrategy(FakeStrategy):
    def step_post_backward(self, params, optimizers, state, step, info, **kw):
        self.calls.append(step)  # no count change (non-refine step)


class FakeTrainer:
    """Mimics gsplat Trainer: eval writes {stage}_step{step:04d}.json.
    writes_json=False simulates a missing/unreadable stats file."""
    def __init__(self, stats_dir=None, writes_json: bool = True):
        self.stats_dir = str(stats_dir) if stats_dir else None
        self.writes_json = writes_json

    def eval(self, step, stage="val"):
        if self.stats_dir and self.writes_json:
            p = Path(self.stats_dir) / f"{stage}_step{step:04d}.json"
            p.write_text(json.dumps({"psnr": 28.5, "ssim": 0.9}))


# ---- _to_float ----
def test_to_float_tensor_like():
    class T:
        def item(self):
            return 3.5
    assert _to_float(T()) == 3.5


def test_to_float_plain_number():
    assert _to_float(2) == 2.0


# ---- numeric passthrough ----
def test_adapter_eval_routes_by_stage():
    a = GsplatAdapter()
    a.on_eval("train", 29.8)
    a.on_eval("val", 28.5)
    assert a.monitor.train_psnr_history == [29.8]
    assert a.monitor.test_psnr_history == [28.5]


def test_adapter_step_refine_export_crash_end_to_end():
    a = GsplatAdapter()
    a.preflight(gpu_arch="8.9", torch_cuda_arch_list=["8.6"])   # R8
    a.on_step(3000, reg_loss=14.0, render_loss=0.7)             # R11
    a.on_eval("train", 31.2)
    a.on_eval("val", 27.9)                                      # R10 scissors
    a.on_refine(4500, relocated_count=30_000)                   # anchor
    a.on_crash(4501, "IMA", vram_allocated=45 * 2**30,
               vram_total=96 * 2**30)                           # R1 + R4
    a.on_export_ratio(0.95)                                     # R7
    rules = {r for r in (al.rule for al in a.monitor.alerts)}
    assert {"R8", "R11", "R10", "R1", "R4", "R7"} <= rules


# ---- wrap_strategy (orchestration via monkeypatched extractor) ----
@pytest.fixture()
def fake_signals(monkeypatch):
    captured = {}

    def fake(params, *, dead_threshold=gw_gsplat.DEFAULT_DEAD_THRESHOLD):
        captured["dead_threshold"] = dead_threshold
        return {"num_gaussians": params["means"].shape[0],
                "scales_finite": True, "opacities_finite": True,
                "grad_norm": None, "dead_count": 5, "dead_fraction": 0.01}

    monkeypatch.setattr(gw_gsplat, "signals_from_splats", fake)
    return captured


def test_wrap_strategy_reports_refine_on_count_change(fake_signals):
    a = GsplatAdapter()
    s = FakeStrategy()
    a.wrap_strategy(s)
    params = {"means": FakeMeans(100)}
    for step in (100, 200):
        s.step_post_backward(params, {}, {}, step, {})
    # one refine event per strategy call, dead fraction flows through
    assert a.monitor.dead_fraction_history == [0.01, 0.01]


def test_wrap_strategy_quiet_when_count_unchanged():
    a = GsplatAdapter()
    s = FakeFlatStrategy()  # never changes the gaussian count
    a.wrap_strategy(s)
    params = {"means": FakeMeans(100)}
    for step in (100, 200, 300):
        s.step_post_backward(params, {}, {}, step, {})
    assert a.monitor.dead_fraction_history == []


def test_wrap_strategy_mcmc_runaway_at_cap(fake_signals):
    """MCMC cap filled while dead gaussians keep teleporting = #1040 runaway."""
    a = GsplatAdapter()
    s = FakeStrategy(cap_max=1000)
    a.wrap_strategy(s)
    params = {"means": FakeMeans(990)}
    s.step_post_backward(params, {}, {}, 100, {})  # 990 -> 1000: hits cap, dead>0
    s.step_post_backward(params, {}, {}, 200, {})  # >= cap again (alert deduped)
    alerts = [al for al in a.monitor.alerts if al.rule in ("R2", "R3")]
    assert any("runaway" in al.message for al in alerts)


def test_wrap_strategy_missing_method_is_loud():
    a = GsplatAdapter()
    with pytest.raises(TypeError, match="step_post_backward"):
        a.wrap_strategy(object())


def test_wrap_strategy_preserves_original_call(fake_signals):
    a = GsplatAdapter()
    s = FakeStrategy()
    a.wrap_strategy(s)
    params = {"means": FakeMeans(100)}
    s.step_post_backward(params, {}, {}, 100, {}, packed=True)
    assert s.calls == [100]  # original body executed


# ---- wrap_eval ----
def test_wrap_eval_reads_val_psnr_json(tmp_path):
    (tmp_path / "val_step0100.json").write_text(json.dumps({"psnr": 28.5}))
    a = GsplatAdapter()
    t = FakeTrainer(stats_dir=tmp_path)
    a.wrap_eval(t)
    t.eval(100)
    assert a.monitor.test_psnr_history == [28.5]


def test_wrap_eval_train_stage_routes_to_train(tmp_path):
    (tmp_path / "train_step0100.json").write_text(json.dumps({"psnr": 29.8}))
    a = GsplatAdapter()
    t = FakeTrainer(stats_dir=tmp_path, writes_json=False)
    a.wrap_eval(t)
    t.eval(100, stage="train")
    assert a.monitor.train_psnr_history == [29.8]


def test_wrap_eval_no_stats_dir_never_crashes():
    a = GsplatAdapter()
    t = FakeTrainer(stats_dir=None)
    a.wrap_eval(t)
    t.eval(100)
    assert a.monitor.test_psnr_history == []


def test_wrap_eval_missing_file_never_crashes(tmp_path):
    a = GsplatAdapter()
    t = FakeTrainer(stats_dir=tmp_path, writes_json=False)
    a.wrap_eval(t)
    t.eval(100)  # stats file absent -> silent skip, no exception
    assert a.monitor.test_psnr_history == []


# ---- autowire ----
def test_autowire_both_wraps(tmp_path, fake_signals):
    (tmp_path / "val_step0100.json").write_text(json.dumps({"psnr": 28.5}))
    a = GsplatAdapter()
    t = FakeTrainer(stats_dir=tmp_path)
    t.strategy = FakeStrategy()
    a.autowire(t)
    params = {"means": FakeMeans(100)}
    t.strategy.step_post_backward(params, {}, {}, 100, {})
    t.eval(100)
    assert a.monitor.dead_fraction_history == [0.01]
    assert a.monitor.test_psnr_history == [28.5]


def test_autowire_missing_strategy_is_loud():
    with pytest.raises(AttributeError, match="strategy"):
        GsplatAdapter().autowire(FakeTrainer())


# ---- torch-side extractors (need real torch; run on GPU box) ----
def test_signals_from_splats_shapes():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    params = {
        "means": torch.nn.Parameter(torch.randn(10, 3)),
        "scales": torch.nn.Parameter(torch.randn(10, 3)),
        "quats": torch.nn.Parameter(torch.randn(10, 4)),
        "opacities": torch.nn.Parameter(torch.randn(10, 1)),
    }
    sig = gw_gsplat.signals_from_splats(params)
    assert sig["num_gaussians"] == 10
    assert sig["scales_finite"] and sig["opacities_finite"]
    assert sig["grad_norm"] is None  # no backward yet
    assert 0.0 <= sig["dead_fraction"] <= 1.0


def test_export_filtered_ratio_matches_manual_mask():
    torch = pytest.importorskip("torch")
    opacities = torch.randn(1000)
    ratio = gw_gsplat.export_filtered_ratio(opacities)
    manual = 1.0 - (torch.sigmoid(opacities) > 1 / 255).float().mean().item()
    assert abs(ratio - manual) < 1e-6
