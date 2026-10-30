#!/usr/bin/env bash
# GaussWatch cluster verification — run on the GPU box (tiger) before release.
#
# Verifies what the local CI cannot: the two torch-side extractor tests and
# a Level-A end-to-end numeric smoke (torch tensors through the full chain
# monitor -> adapter -> panel). No dataset needed.
#
# Usage:
#     bash scripts/cluster_verify.sh
#
# Level B (real gsplat training loop, needs a dataset + built gsplat) is a
# separate step: see examples/integrate_gsplat_simple_trainer.md, then run
# simple_trainer.py with the patch applied and check gw.monitor.alerts.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "======================================================"
echo "== GaussWatch cluster verification =="
echo "======================================================"
python - <<'EOF'
import sys
print(f"python      : {sys.version.split()[0]}")
try:
    import torch
    print(f"torch       : {torch.__version__}  (cuda available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
except ImportError:
    print("torch       : NOT INSTALLED — extractor tests will fail; install torch first")
    sys.exit(1)
EOF

echo
echo "== [1/2] torch extractor tests (skipped on CI, must PASS here) ========"
python -m pytest tests/test_integrations.py -v -k "signals_from_splats_shapes or export_filtered_ratio" 2>&1 | tail -6

echo
echo "== [2/2] Level A end-to-end numeric smoke (no dataset required) ======="
python - <<'EOF'
import json
import urllib.request

import torch

from gausswatch.collector import TrainingMonitor
from gausswatch.integrations.gsplat import (
    GsplatAdapter, export_filtered_ratio, signals_from_splats,
)
from gausswatch.panel import PanelServer

torch.manual_seed(0)
n = 50_000
params = {
    "means":    torch.nn.Parameter(torch.randn(n, 3, device="cuda")),
    "scales":   torch.nn.Parameter(torch.randn(n, 3, device="cuda")),
    "quats":    torch.nn.Parameter(torch.randn(n, 4, device="cuda")),
    "opacities": torch.nn.Parameter(torch.randn(n, 1, device="cuda")),
}
params["scales"].data[0] = float("nan")  # force one non-finite value

sig = signals_from_splats(params)
assert sig["num_gaussians"] == n
assert sig["scales_finite"] is False, "NaN injection must be detected"
assert sig["grad_norm"] is None  # no backward yet
print(f"signals_from_splats        OK  (n={n}, scales_finite={sig['scales_finite']})")

ratio = export_filtered_ratio(params["opacities"])
manual = 1.0 - (torch.sigmoid(params["opacities"]) > 1/255).float().mean().item()
assert abs(ratio - manual) < 1e-6
print(f"export_filtered_ratio      OK  (ratio={ratio:.4f})")

gw = GsplatAdapter()
gw.preflight(gpu_arch="7.0", torch_cuda_arch_list=["8.6"])            # R8 (V100=7.0)
gw.on_step(3000, grad_norm=0.0)                                       # R12
gw.on_eval("train", 31.2); gw.on_eval("val", 27.9)                    # R10
gw.on_refine(4500, relocated_count=30_000)                            # anchor
gw.on_crash(4501, "IMA", vram_allocated=45 * 2**30, vram_total=96 * 2**30)  # R1+R4
gw.on_export_ratio(ratio)                    # R7 stays silent: ratio ~0.5 << 0.90
                                             # (no false alarm is also a check)
rules = {a.rule for a in gw.monitor.alerts}
expected = {"R8", "R12", "R10", "R1", "R4"}
assert expected <= rules, f"missing rules: {expected - rules}"
print(f"adapter event chain        OK  (rules fired: {sorted(rules)})")

srv = PanelServer(gw.monitor, port=0)
srv.start(block=False)
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/api/state", timeout=5) as r:
        payload = json.loads(r.read())
    assert payload["stats"]["n_alerts"] == len(gw.monitor.alerts)
    print(f"panel HTTP probe           OK  (port {srv.port}, {payload['stats']['n_alerts']} alerts)")
finally:
    srv.shutdown()

print()
print("ALL CLUSTER CHECKS PASSED")
EOF

echo
echo "======================================================"
echo "== cluster verification complete =="
echo "======================================================"
