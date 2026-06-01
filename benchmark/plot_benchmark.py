#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-render benchmark plots from saved data — no solver required.

Reads ``benchmark/results/<cpu>/benchmark.json`` plus the corresponding
``benchmark/results/<cpu>/data/*.npz`` files (produced by
``benchmark_traj_opt.py``) and writes:

    benchmark/results/<cpu>/benchmark.png                       # summary bar chart
    benchmark/results/<cpu>/trajectories/traj_method<m>_N<n>.png # per-trial trajectory

The plotting logic is imported from ``benchmark_traj_opt.py``; this script is
just a data-driven thin wrapper. Edit the plotting functions in the main
module if you want to change figure layout.

Usage
-----

    # All CPUs / methods / Ns found in benchmark/results/
    python benchmark/plot_benchmark.py

    # Only one CPU's results:
    python benchmark/plot_benchmark.py --cpu Intel_Core_Ultra_7_265K

    # Restrict to a method or N subset:
    python benchmark/plot_benchmark.py --methods 3 --nodes 100,200

    # Custom results directory:
    python benchmark/plot_benchmark.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# This script lives at ``<repo>/benchmark/plot_benchmark.py``. Add the
# benchmark folder (for ``benchmark_traj_opt``) and the project's
# ``scripts/`` (for ``tvc_common``) to ``sys.path`` so imports work without
# env tweaks.
_BENCH_DIR = Path(__file__).resolve().parent
_PROJ_DIR = _BENCH_DIR.parent
for _p in (_BENCH_DIR, _PROJ_DIR / "scripts"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import numpy as np  # noqa: E402

from benchmark_traj_opt import (  # noqa: E402
    TrialResult,
    load_trajectory_data,
    plot_summary,
    plot_trajectory,
    slugify_cpu,
)


def _parse_int_list(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in s.replace(";", ",").split(",") if p.strip())


def _load_run_bundle(json_path: Path) -> dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def _trial_from_dict(d: dict[str, Any]) -> TrialResult:
    """Recreate a :class:`TrialResult` from JSON. Ignores unknown keys."""
    allowed = {
        "method", "method_name", "N", "dt", "duration",
        "total_time_s", "setup_time_s", "solve_time_s",
        "total_iters", "avg_time_per_iter_ms",
        "final_cost", "status", "error_msg", "extra",
    }
    # Back-fill missing timing fields for legacy JSONs (pre-split).
    if "setup_time_s" not in d:
        d = {**d, "setup_time_s": 0.0}
    if "solve_time_s" not in d:
        d = {**d, "solve_time_s": d.get("total_time_s", float("nan"))}
    kwargs = {k: v for k, v in d.items() if k in allowed}
    return TrialResult(**kwargs)


def _waypoints_from_bundle(bundle: dict[str, Any]) -> list[list[float]] | None:
    """Best-effort reconstruction of the original waypoint list."""
    cfg = bundle.get("config", {})
    start = cfg.get("start")
    goal = cfg.get("goal")
    duration = cfg.get("duration")
    if not (start and goal and duration is not None):
        return None
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    syaw = float(start[3]) if len(start) > 3 else 0.0
    gx, gy, gz = float(goal[0]), float(goal[1]), float(goal[2])
    gyaw = float(goal[3]) if len(goal) > 3 else 0.0
    return [[sx, sy, sz, syaw, 0.0], [gx, gy, gz, gyaw, float(duration)]]


def render_one_cpu(json_path: Path, results_dir: Path,
                   methods_filter: set[int] | None = None,
                   nodes_filter: set[int] | None = None,
                   summary: bool = True,
                   trajectories: bool = True) -> list[Path]:
    """Render plots for a single ``<cpu_slug>/benchmark.json`` bundle.

    Returns the list of PNGs that were written.
    """
    bundle = _load_run_bundle(json_path)
    cpu_info = bundle.get("cpu_info", {})
    # The CPU folder name is the source of truth; fall back to the JSON slug
    # only if the file happens to sit directly under results_dir (legacy).
    cpu_dir = json_path.parent
    cpu_slug = cpu_dir.name if cpu_dir != results_dir else slugify_cpu(cpu_info.get("cpu_model"))

    raw_results = bundle.get("results", [])
    results: list[TrialResult] = []
    for r in raw_results:
        results.append(_trial_from_dict(r))

    if methods_filter is not None:
        results = [r for r in results if r.method in methods_filter]
    if nodes_filter is not None:
        results = [r for r in results if r.N in nodes_filter]

    written: list[Path] = []

    if summary and results:
        p = plot_summary(results_dir, cpu_info, results, cpu_slug)
        if p is not None:
            written.append(p)

    if trajectories:
        data_dir = cpu_dir / "data"
        waypoints_default = _waypoints_from_bundle(bundle)
        for r in results:
            if r.status != "ok":
                continue
            npz_path = data_dir / f"method{r.method}_N{r.N}.npz"
            if not npz_path.exists():
                print(f"  [skip] missing data file: {npz_path}")
                continue
            d = load_trajectory_data(npz_path)
            waypoints = d.get("waypoints", waypoints_default)
            png = plot_trajectory(
                results_dir, cpu_slug, r.method, r.N, d["dt"],
                d["xs"].tolist(), d["us"].tolist(),
                waypoints=waypoints,
            )
            if png is not None:
                written.append(png)

    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Re-render benchmark plots from saved data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default=str(_BENCH_DIR / "results"),
                   help="Directory containing benchmark_*.json + data/*.npz")
    p.add_argument("--cpu", default=None,
                   help="CPU slug (e.g. Intel_Core_Ultra_7_265K). Default: all "
                        "benchmark_*.json files found.")
    p.add_argument("--methods", type=_parse_int_list, default=None,
                   help="Restrict to a subset of method ids (comma-separated)")
    p.add_argument("--nodes", type=_parse_int_list, default=None,
                   help="Restrict to a subset of N values (comma-separated)")
    p.add_argument("--no-summary", action="store_true",
                   help="Skip the benchmark summary bar chart")
    p.add_argument("--no-trajectories", action="store_true",
                   help="Skip per-trial trajectory plots")
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        print(f"[plot] no such directory: {results_dir}", file=sys.stderr)
        return 2

    if args.cpu:
        candidates = [results_dir / args.cpu / "benchmark.json"]
        if not candidates[0].exists():
            print(f"[plot] file not found: {candidates[0]}", file=sys.stderr)
            return 2
    else:
        candidates = sorted(results_dir.glob("*/benchmark.json"))

    if not candidates:
        print(f"[plot] no <cpu>/benchmark.json found under {results_dir}", file=sys.stderr)
        return 2

    m_filter = set(args.methods) if args.methods else None
    n_filter = set(args.nodes) if args.nodes else None

    total_written: list[Path] = []
    for jp in candidates:
        print(f"[plot] processing CPU '{jp.parent.name}' ({jp})")
        try:
            written = render_one_cpu(
                jp, results_dir,
                methods_filter=m_filter,
                nodes_filter=n_filter,
                summary=not args.no_summary,
                trajectories=not args.no_trajectories,
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for w in written:
            print(f"  wrote {w.relative_to(results_dir)}")
            total_written.append(w)

    print(f"[plot] done — {len(total_written)} file(s) written under {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
