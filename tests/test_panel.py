"""Tests for the zero-dependency web panel (PanelServer)."""
import json
import threading
import time
import urllib.request

from gausswatch.collector import TrainingMonitor
from gausswatch.panel import PanelServer


def _get(path: str, port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.read().decode("utf-8")


def _wait_ready(port: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _get("/api/state", port)
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("panel did not become ready")


def _make():
    m = TrainingMonitor()
    srv = PanelServer(m, port=0)  # OS-assigned port: no collisions in CI
    srv.start(block=False)
    _wait_ready(srv.port)
    return m, srv


def test_index_serves_dashboard_html():
    _, srv = _make()
    try:
        html = _get("/", srv.port)
        assert "GaussWatch" in html and "/api/state" in html
    finally:
        srv.shutdown()


def test_api_state_json_shape():
    m, srv = _make()
    try:
        m.on_step(500, grad_norm=0.0, vram_allocated=8 * 2**30,
                  num_gaussians=50000)
        d = json.loads(_get("/api/state", srv.port))
        assert d["stats"]["step"] == 500 and d["stats"]["n_alerts"] == 1
        assert len(d["alerts"]) == 1
        a = d["alerts"][0]
        assert a["rule"] == "R12" and a["severity"] == "critical"
        assert "first_seen" in a and "source" in a
    finally:
        srv.shutdown()


def test_empty_state_ok():
    _, srv = _make()
    try:
        d = json.loads(_get("/api/state", srv.port))
        assert d["stats"]["step"] == 0 and d["alerts"] == []
    finally:
        srv.shutdown()


def test_shown_alerts_reflect_monitor_within_locks():
    """Concurrent training-thread writes + panel reads must not corrupt state."""
    m, srv = _make()
    try:
        def feed():
            for i in range(50):
                m.on_step(i * 100 + 100, grad_norm=0.0 if i == 49 else 1.0)
        t = threading.Thread(target=feed)
        t.start()
        for _ in range(20):  # poll while the trainer thread is writing
            json.loads(_get("/api/state", srv.port))
        t.join()
        d = json.loads(_get("/api/state", srv.port))
        assert d["stats"]["n_alerts"] == 1
    finally:
        srv.shutdown()
