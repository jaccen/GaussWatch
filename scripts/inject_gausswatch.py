#!/usr/bin/env python
"""Inject GaussWatch hooks into gsplat's examples/simple_trainer.py.

Anchor-based (matches code content, not line numbers), idempotent, and loud:
if any anchor is missing or appears more than once, the injector aborts
without writing — upstream drift must be resolved by a human, never silently
skipped.

Usage:
    python scripts/inject_gausswatch.py <path-to-simple_trainer.py> <output.py>

Verified against gsplat v1.5.3 examples/simple_trainer.py (2026-10-30):
the strategy lives at runner.cfg.strategy in both v1.5.3 and main.
"""
from __future__ import annotations

import sys
from pathlib import Path

# (name, anchor, mode, payload)
# mode: "after" inserts payload after the anchor line,
#       "before" inserts before it, "replace" swaps the anchor for the payload.
INJECTIONS: list[tuple[str, str, str, str]] = [
    (
        "imports",
        "import torch.nn.functional as F",
        "after",
        """
# --- GaussWatch integration (injected by scripts/inject_gausswatch.py) ---
from gausswatch.integrations.gsplat import (
    GsplatAdapter,
    export_filtered_ratio,
    signals_from_splats,
)""",
    ),
    (
        "wire-strategy-eval-preflight",
        "    runner = Runner(local_rank, world_rank, world_size, cfg)",
        "after",
        """

    # --- GaussWatch: wrap strategy + eval, run preflight (injected) ---
    runner.gw = GsplatAdapter()
    runner.gw.wrap_strategy(runner.cfg.strategy)   # gsplat keeps strategy on cfg
    runner.gw.wrap_eval(runner)
    _gw_cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
    runner.gw.preflight(gpu_arch=_gw_cc, torch_cuda_arch_list=[_gw_cc])
    print(f"[GaussWatch] wired (strategy/eval), preflight cc={_gw_cc}")""",
    ),
    (
        "per-step-signals",
        "            loss.backward()",
        "after",
        """

            # --- GaussWatch: per-step signals (injected) ---
            _gw_sig = signals_from_splats(self.splats)
            self.gw.on_step(
                step,
                grad_norm=_gw_sig["grad_norm"],
                scales_finite=_gw_sig["scales_finite"],
                opacities_finite=_gw_sig["opacities_finite"],
                num_gaussians=_gw_sig["num_gaussians"],
                vram_allocated=torch.cuda.memory_allocated(),
                vram_total=torch.cuda.get_device_properties(0).total_memory,
            )""",
    ),
    (
        "export-guard",
        "                export_splats(",
        "before",
        """                # --- GaussWatch: export guard (injected) ---
                self.gw.on_export_ratio(export_filtered_ratio(opacities))
""",
    ),
    (
        "crash-attribution",
        "        runner.train()",
        "replace",
        """        try:
            runner.train()
        except (RuntimeError, MemoryError) as _gw_err:
            runner.gw.on_crash(
                None,  # falls back to the last step seen by on_step
                type(_gw_err).__name__,
                vram_allocated=torch.cuda.memory_allocated(),
                vram_total=torch.cuda.get_device_properties(0).total_memory,
            )
            raise
        finally:
            _gw_stats = runner.gw.monitor.stats()
            print(f"[GaussWatch] done @ step {_gw_stats['step']}: "
                  f"{_gw_stats['n_alerts']} alerts "
                  f"({_gw_stats['n_critical']} critical / "
                  f"{_gw_stats['n_warning']} warning)")
            for _a in runner.gw.monitor.alert_records():
                print(f"[GaussWatch]   {_a['rule']}/{_a['severity']}: {_a['message']}")""",
    ),
]


def inject(src: str) -> str:
    lines = src.splitlines(keepends=True)
    for name, anchor, mode, payload in INJECTIONS:
        hits = [i for i, ln in enumerate(lines) if ln.rstrip("\n") == anchor]
        if len(hits) != 1:
            sys.exit(
                f"inject_gausswatch: anchor {name!r} matched {len(hits)} times "
                f"(expected exactly 1) — upstream code drifted, resolve manually.\n"
                f"    anchor: {anchor!r}"
            )
        i = hits[0]
        if mode == "after":
            lines.insert(i + 1, payload + "\n")
        elif mode == "before":
            lines.insert(i, payload)
        elif mode == "replace":
            lines[i] = payload + "\n"
        else:
            sys.exit(f"unknown mode {mode!r}")
        print(f"  injected: {name}")
    return "".join(lines)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src_path, dst_path = Path(sys.argv[1]), Path(sys.argv[2])
    src = src_path.read_text(encoding="utf-8")
    if "gausswatch" in src:
        print(f"{src_path} already contains gausswatch hooks — skip (idempotent)")
        return
    out = inject(src)
    compile(out, str(dst_path), "exec")  # syntax gate before writing
    dst_path.write_text(out, encoding="utf-8")
    print(f"wrote {dst_path} ({len(out.splitlines())} lines, syntax-checked)")


if __name__ == "__main__":
    main()
