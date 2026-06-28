#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVC Trajectory Optimization Benchmark
=====================================

Cross-platform / cross-method benchmark harness for the TVC trajectory
optimization stack. Runs a fixed start-to-end OCP across a sweep of node
counts ``N`` and records timing/iteration stats per (method, N) cell.

Supported methods (numbers match GUI labels):
    - Method 3: Pinocchio + Crocoddyl ``SolverBoxFDDP``
    - Method 4: Acados (tracking cost)
    - Method 5: Acados (minimum-time per segment)

Typical use:

    # Only Method 3, single N=20, no plot - good first smoke test:
    python benchmark_traj_opt.py --methods 3 --nodes 20

    # Full sweep (Methods 3,4,5 x N in {20,50,100,200}):
    python benchmark_traj_opt.py

Outputs (under ``benchmark/results/<cpu_slug>/`` by default):
    - ``benchmark.csv``  : one row per (method, N) trial
    - ``benchmark.json`` : same data + full input config / CPU info
    - ``benchmark.png``  : summary bar chart
    - ``data/method<m>_N<n>.npz``        : raw xs/us for re-plotting later
    - ``trajectories/traj_method<m>_N<n>.png`` : per-trial trajectory plot

Each result row records: cpu_model, cpu_mhz_max, total_time, setup_time,
solve_time, total_iters, avg_time_per_iter_ms (solve-only),
final_cost, dt, N, method, status.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import socket
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Path / acados setup. This file lives at ``<repo>/benchmark/benchmark_traj_opt.py``;
# we expose the project's ``scripts/`` and root directory on ``sys.path`` so the
# solver modules can be imported, and preload acados libs via ``tvc_runtime``
# (avoids LD_LIBRARY_PATH gymnastics).
# ---------------------------------------------------------------------------
_BENCH_DIR = Path(__file__).resolve().parent
_PROJ_DIR = _BENCH_DIR.parent
_SCRIPTS_DIR = _PROJ_DIR / "scripts"
for _p in (_SCRIPTS_DIR, _PROJ_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

try:
    from tvc_runtime import preload_acados_libs as _preload_acados_libs  # type: ignore
except Exception:
    _preload_acados_libs = None


# ---------------------------------------------------------------------------
# Default parameters – mirror the GUI defaults for Method 3/4/5 so results are
# comparable to what a user would get from the GUI without manual tweaks.
# ---------------------------------------------------------------------------
DEFAULT_PHYS = {
    "m": 0.6,
    "I": [0.02, 0.02, 0.01],  # diagonal [Ixx, Iyy, Izz]
    "r_thrust": [0.0, 0.0, -0.2],
    "g": 9.81,
}

# Running cost weights – same shape used by both pinocchio and acados solvers.
DEFAULT_WEIGHTS_RUNNING = {
    "p": 1.0,
    "v": 0.2,
    "R": 0.5,
    "yaw": 0.5,
    "w": 0.1,
    "u": 0.5,
    "du": 0.5,
}
TERMINAL_COST_MULTIPLIER = 200.0

# Method-specific overrides on top of DEFAULT_WEIGHTS_RUNNING.
# (Method 3 wants stronger orientation tracking to suppress yaw drift; this
# mirrors GUI DEFAULT_PARAMS[2].)
WEIGHT_OVERRIDES_BY_METHOD = {
    3: {"R": 2.0, "yaw": 2.0},
    4: {},
    5: {},
}

# Constraint bounds – control + state. Control bounds passed in *radians*.
DEFAULT_BOUNDS_HUMAN = {
    "th_p_max_deg": 10.0,
    "th_r_max_deg": 10.0,
    "T_max": 25.0,
    "tau_yaw_max": 1.0,
    "k_bound": 200.0,
    "v_horizontal_max": 1.0,
    "v_vertical_max": 0.5,
    "roll_max_deg": 10.0,
    "pitch_max_deg": 10.0,
    "yaw_max_deg": 30.0,
    "w_max": 2.0,
    "k_state_bound": 20.0,  # Method 3 default in GUI is 200, but 20 is the
                            # acados default; we'll override for M3 below.
}

# Method-specific bound tweaks.
BOUND_OVERRIDES_BY_METHOD = {
    3: {"k_state_bound": 200.0, "v_horizontal_max": 1.0, "v_vertical_max": 3.0},
    4: {},
    5: {},
}

# Method 5 (min-time) needs a couple of extra knobs.
MIN_TIME_DEFAULTS = {
    "min_time_weight": 1.0,
    "min_time_T_min": 0.15,
    "min_time_T_max_scale": 1.0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    """Inputs that fully determine a benchmark sweep."""

    start: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x,y,z,yaw_deg
    goal: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0)
    duration: float = 5.0
    nodes: tuple[int, ...] = (20, 50, 100, 200)
    methods: tuple[int, ...] = (3, 4, 5)
    max_iter: int = 100
    warmup: bool = False
    output_dir: str = ""  # filled in main()
    config_path: str | None = None
    bounds_human: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BOUNDS_HUMAN))
    phys: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PHYS))
    plot: bool = True


@dataclass
class TrialResult:
    """Per-(method, N) trial outcome.

    Timing fields are split so that fair cross-method / cross-CPU comparisons
    can be made:

    - ``total_time_s``: end-to-end wall clock of the trial, **includes**
      Acados C-code generation, OCP / solver instantiation, initial guess
      setup, NLP solve and post-processing. Best reflects what a user sees
      when they click "Optimize" in the GUI for the first time.
    - ``solve_time_s``: pure ``solver.solve()`` time summed across segments
      (from ``meta['total_solve_time']`` reported by Acados). For Method 3
      (Crocoddyl) we approximate ``solve_time = total_time`` because there
      is no codegen; the Pinocchio/Crocoddyl model construction is sub-ms.
    - ``setup_time_s``: ``total_time_s - solve_time_s``. Captures
      first-run codegen cost for Acados; ~0 for Method 3.
    - ``avg_time_per_iter_ms``: ``solve_time_s / total_iters * 1000``,
      i.e. average **NLP-iteration** time (does not include initialization
      / codegen). This is the right metric to compare CPUs.
    """

    method: int
    method_name: str
    N: int
    dt: float
    duration: float
    total_time_s: float
    setup_time_s: float
    solve_time_s: float
    total_iters: int
    avg_time_per_iter_ms: float
    final_cost: float
    status: str  # "ok" / "error" / "skipped"
    error_msg: str = ""
    # Extra acados metadata (only for Method 5 / free-tf etc.)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Platform / CPU info
# ---------------------------------------------------------------------------
def collect_cpu_info() -> dict[str, Any]:
    """Best-effort CPU / OS / Python info collection (Linux-friendly)."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_model": "unknown",
        "cpu_mhz_current_avg": None,
        "cpu_mhz_max": None,
        "cpu_mhz_min": None,
    }

    try:
        model_name: str | None = None
        cpu_mhz: list[float] = []
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and model_name is None:
                    model_name = line.split(":", 1)[1].strip()
                elif line.startswith("cpu MHz"):
                    try:
                        cpu_mhz.append(float(line.split(":", 1)[1].strip()))
                    except Exception:
                        pass
        if model_name:
            info["cpu_model"] = model_name
        if cpu_mhz:
            info["cpu_mhz_current_avg"] = float(np.mean(cpu_mhz))
    except Exception:
        pass

    for path, key in (
        ("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "cpu_mhz_max"),
        ("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq", "cpu_mhz_min"),
    ):
        try:
            with open(path) as f:
                info[key] = int(f.read().strip()) / 1000.0  # kHz -> MHz
        except Exception:
            pass

    return info


# ---------------------------------------------------------------------------
# Build solver inputs (weights / bounds / waypoints / m / I / r_thrust)
# ---------------------------------------------------------------------------
def _build_weights(method: int, terminal_multiplier: float = TERMINAL_COST_MULTIPLIER) -> dict[str, Any]:
    w = dict(DEFAULT_WEIGHTS_RUNNING)
    w.update(WEIGHT_OVERRIDES_BY_METHOD.get(method, {}))
    w["terminal_cost_multiplier"] = terminal_multiplier
    # Flags expected by acados solver but harmless for pinocchio.
    w["schedule_ref"] = False
    w["terminal_constraint"] = False
    w["waypoint_terminal_cost"] = True
    w["unified_interp_initial_guess"] = False
    w["actuator_dynamics"] = False
    w["actuator_tau"] = [0.05, 0.05, 0.05, 0.05]

    if method == 4:
        w["acados_objective"] = "tracking"
    elif method == 5:
        w["acados_objective"] = "min_time"
        w.update(MIN_TIME_DEFAULTS)
    else:
        w["acados_objective"] = "tracking"  # unused for pinocchio
    return w


def _build_terminal_weights(method: int, running: dict[str, Any]) -> dict[str, Any] | None:
    """Mirror GUI's terminal_weights = running * terminal_cost_multiplier."""
    k = running.get("terminal_cost_multiplier", TERMINAL_COST_MULTIPLIER)
    keys = ("p", "v", "R", "yaw", "w", "u", "du")
    return {k_: running[k_] * k for k_ in keys}


def _build_bounds(method: int, human: dict[str, float]) -> dict[str, Any]:
    b = dict(DEFAULT_BOUNDS_HUMAN)
    b.update(human)
    b.update(BOUND_OVERRIDES_BY_METHOD.get(method, {}))

    th_p_rad = float(np.radians(b["th_p_max_deg"]))
    th_r_rad = float(np.radians(b["th_r_max_deg"]))
    tau_yaw = float(b["tau_yaw_max"])

    return {
        "th_p": (-th_p_rad, th_p_rad),
        "th_r": (-th_r_rad, th_r_rad),
        "T": (0.0, float(b["T_max"])),
        "tau_yaw": (-tau_yaw, tau_yaw),
        "k_bound": float(b["k_bound"]),
        "state_v_horizontal_max": float(b["v_horizontal_max"]),
        "state_v_vertical_max": float(b["v_vertical_max"]),
        "state_roll_max": float(np.radians(b["roll_max_deg"])),
        "state_pitch_max": float(np.radians(b["pitch_max_deg"])),
        "state_yaw_max": float(np.radians(b["yaw_max_deg"])),
        "state_w_max": float(b["w_max"]),
        "state_k_state_bound": float(b["k_state_bound"]),
        "state_constraint_lxx_scale": 0.0,
    }


def _build_waypoints(start: tuple, goal: tuple, duration: float) -> list[list[float]]:
    """Build a 2-waypoint plan: ``[start@0, goal@duration]``.

    Waypoint format expected by the solvers: ``[x, y, z, yaw_deg, time]``.
    """
    sx, sy, sz, syaw = (float(v) for v in start)
    gx, gy, gz, gyaw = (float(v) for v in goal)
    return [
        [sx, sy, sz, syaw, 0.0],
        [gx, gy, gz, gyaw, float(duration)],
    ]


# ---------------------------------------------------------------------------
# Method dispatchers
# ---------------------------------------------------------------------------
def _method_name(method: int) -> str:
    return {
        3: "Method 3 (BoxFDDP)",
        4: "Method 4 (Acados)",
        5: "Method 5 (Acados min-time)",
    }.get(method, f"Method {method}")


def _run_method3(cfg: BenchmarkConfig, N: int, dt: float, waypoints: list,
                 weights: dict, terminal_weights: dict, bounds: dict
                 ) -> tuple[TrialResult, list, list]:
    """Method 3: Pinocchio + Crocoddyl BoxFDDP."""
    from tvc_traj_opt_pinocchio import solve_with_pinocchio_waypoints  # type: ignore

    m = float(cfg.phys["m"])
    I = np.diag(cfg.phys["I"])
    r_thrust = np.array(cfg.phys["r_thrust"], dtype=float)

    t0 = time.perf_counter()
    combined_xs, combined_us, all_loggers = solve_with_pinocchio_waypoints(
        dt=dt,
        waypoints=waypoints,
        m=m,
        I=I,
        r_thrust=r_thrust,
        weights=weights,
        terminal_weights=terminal_weights,
        bounds=bounds,
        max_iter=cfg.max_iter,
        use_box_solver=True,
        callback=None,
        running_flag=None,
    )
    total_time = time.perf_counter() - t0

    total_iters = sum(len(lg.costs) for lg in all_loggers) if all_loggers else 0
    final_cost = float(all_loggers[-1].costs[-1]) if (all_loggers and all_loggers[-1].costs) else float("nan")
    # Crocoddyl has no codegen; Pinocchio / problem construction is sub-ms.
    # Treat solve_time ≈ total_time so the per-iter metric is comparable.
    solve_time = total_time
    setup_time = 0.0
    avg_ms = (solve_time / total_iters * 1000.0) if total_iters > 0 else float("nan")

    result = TrialResult(
        method=3,
        method_name=_method_name(3),
        N=N,
        dt=dt,
        duration=cfg.duration,
        total_time_s=total_time,
        setup_time_s=setup_time,
        solve_time_s=solve_time,
        total_iters=total_iters,
        avg_time_per_iter_ms=avg_ms,
        final_cost=final_cost,
        status="ok",
    )
    return result, list(combined_xs), list(combined_us)


def _run_method_acados(cfg: BenchmarkConfig, method: int, N: int, dt: float, waypoints: list,
                       weights: dict, terminal_weights: dict, bounds: dict
                       ) -> tuple[TrialResult, list, list]:
    """Method 4 / 5: Acados (tracking / min-time)."""
    from tvc_traj_opt_acados import solve_with_acados_waypoints, ACADOS_AVAILABLE  # type: ignore

    if not ACADOS_AVAILABLE:
        raise RuntimeError(
            "Acados is not available in this environment. "
            "Install casadi and build acados; or restrict --methods to 3."
        )

    m = float(cfg.phys["m"])
    I = np.diag(cfg.phys["I"])
    r_thrust = np.array(cfg.phys["r_thrust"], dtype=float)

    t0 = time.perf_counter()
    pack = solve_with_acados_waypoints(
        dt=dt,
        waypoints=waypoints,
        m=m,
        I=I,
        r_thrust=r_thrust,
        weights=weights,
        terminal_weights=terminal_weights,
        bounds=bounds,
        max_iter=cfg.max_iter,
        use_box_solver=False,
        callback=None,
        running_flag=None,
        iteration_callback=None,
        verbose_solve=False,
    )
    total_time = time.perf_counter() - t0

    combined_xs, combined_us, all_loggers, _us_actual = pack[:4]
    extra = pack[4] if len(pack) >= 5 and isinstance(pack[4], dict) else {}

    # Acados-specific iteration accounting: prefer the solver-reported SQP iter
    # count from the meta dict (set by our patched solver). The Crocoddyl-style
    # ``len(logger.costs)`` only yields 1 when ``verbose_solve=False`` because
    # acados writes a single final cost into the logger.
    if isinstance(extra.get("total_sqp_iters"), (int, float)) and int(extra["total_sqp_iters"]) > 0:
        total_iters = int(extra["total_sqp_iters"])
    else:
        total_iters = sum(len(lg.costs) for lg in all_loggers) if all_loggers else 0

    # For Method 5 (min-time) the actual segment durations are decided by the
    # solver; ``plot_dt`` in the meta is the correct time step for plotting and
    # interpreting the trajectory. Method 4 (tracking) uses the user dt.
    effective_dt = float(extra["plot_dt"]) if extra.get("plot_dt") else dt
    effective_duration = effective_dt * (len(combined_xs) - 1) if combined_xs else cfg.duration

    # Split timing: ``meta['total_solve_time']`` is the pure summed
    # ``solver.solve()`` time (no codegen, no OCP construction). Setup is
    # whatever the wall-clock measurement of the whole solve_with_acados_*
    # call captured on top of that.
    solve_time = float(extra.get("total_solve_time")) if extra.get("total_solve_time") else float("nan")
    if np.isfinite(solve_time) and solve_time > 0.0:
        setup_time = max(total_time - solve_time, 0.0)
    else:
        # Fallback if meta is missing - assume everything was solve.
        solve_time = total_time
        setup_time = 0.0

    final_cost = float(all_loggers[-1].costs[-1]) if (all_loggers and all_loggers[-1].costs) else float("nan")
    avg_ms = (solve_time / total_iters * 1000.0) if total_iters > 0 else float("nan")

    result = TrialResult(
        method=method,
        method_name=_method_name(method),
        N=N,
        dt=effective_dt,
        duration=effective_duration,
        total_time_s=total_time,
        setup_time_s=setup_time,
        solve_time_s=solve_time,
        total_iters=total_iters,
        avg_time_per_iter_ms=avg_ms,
        final_cost=final_cost,
        status="ok",
        extra={k: v for k, v in extra.items() if isinstance(v, (int, float, str, bool, list))},
    )
    return result, list(combined_xs), list(combined_us)


def run_trial(cfg: BenchmarkConfig, method: int, N: int
              ) -> tuple[TrialResult, list | None, list | None]:
    """Single (method, N) trial. Returns ``(result, xs, us)``; ``xs``/``us`` are
    ``None`` if the trial errored or is skipped. Never raises.
    """
    dt = cfg.duration / float(N)
    waypoints = _build_waypoints(cfg.start, cfg.goal, cfg.duration)
    weights = _build_weights(method)
    terminal_weights = _build_terminal_weights(method, weights)
    bounds = _build_bounds(method, cfg.bounds_human)

    try:
        if method == 3:
            return _run_method3(cfg, N, dt, waypoints, weights, terminal_weights, bounds)
        if method in (4, 5):
            return _run_method_acados(cfg, method, N, dt, waypoints, weights, terminal_weights, bounds)
        return (
            TrialResult(
                method=method, method_name=_method_name(method), N=N, dt=dt, duration=cfg.duration,
                total_time_s=float("nan"), setup_time_s=float("nan"), solve_time_s=float("nan"),
                total_iters=0, avg_time_per_iter_ms=float("nan"),
                final_cost=float("nan"), status="skipped",
                error_msg=f"Method {method} not implemented in benchmark.",
            ),
            None,
            None,
        )
    except Exception as e:
        traceback.print_exc()
        return (
            TrialResult(
                method=method, method_name=_method_name(method), N=N, dt=dt, duration=cfg.duration,
                total_time_s=float("nan"), setup_time_s=float("nan"), solve_time_s=float("nan"),
                total_iters=0, avg_time_per_iter_ms=float("nan"),
                final_cost=float("nan"), status="error", error_msg=f"{type(e).__name__}: {e}",
            ),
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
METHOD_COLORS = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c"}


def _xs_to_state_arrays(xs: list, us: list, dt: float) -> dict[str, np.ndarray]:
    """Decode a Method-1 / 17-dim state trajectory into named state arrays.

    Returns dict with: ``t``, ``t_u``, ``p`` (N+1, 3), ``v`` (N+1, 3),
    ``euler_deg`` (N+1, 3), ``w`` (N+1, 3), ``u`` (N, 4).
    """
    # Lazy import - tvc_common already on sys.path via _SCRIPTS_DIR.
    from tvc_common import quat_to_euler_deg  # type: ignore

    xs_arr = np.array([np.asarray(x, dtype=float).flatten() for x in xs])
    us_arr = np.array([np.asarray(u, dtype=float).flatten() for u in us])
    if us_arr.ndim == 1:
        us_arr = us_arr.reshape(-1, 4)
    n_steps = xs_arr.shape[0]
    n_u = us_arr.shape[0]

    p = xs_arr[:, 0:3]
    v = xs_arr[:, 3:6]
    quat = xs_arr[:, 6:10]  # [w, x, y, z]
    w = xs_arr[:, 10:13]
    euler_deg = np.array([quat_to_euler_deg(quat[i], "wxyz") for i in range(n_steps)])

    return {
        "t": np.arange(n_steps) * dt,
        "t_u": np.arange(n_u) * dt,
        "p": p,
        "v": v,
        "euler_deg": euler_deg,
        "w": w,
        "u": us_arr,
    }


def plot_trajectory(out_dir: Path, cpu_slug: str, method: int, N: int, dt: float,
                    xs: list, us: list, waypoints: list[list[float]] | None = None) -> Path | None:
    """Render a 4-panel figure: 3D path (left) + position / velocity / euler /
    angular-velocity / control (right, 3 rows).

    Saves under ``<out_dir>/trajectories/traj_<cpu_slug>_method<m>_N<n>.png``
    and returns the file path. Returns ``None`` if matplotlib is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    except Exception as e:
        print(f"[traj-plot] matplotlib unavailable: {e}")
        return None

    if not xs or not us:
        return None

    s = _xs_to_state_arrays(xs, us, dt)
    p, v = s["p"], s["v"]
    euler, w_ang = s["euler_deg"], s["w"]
    u = s["u"]
    t, t_u = s["t"], s["t_u"]

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(
        3, 4, hspace=0.45, wspace=0.4,
        left=0.05, right=0.97, top=0.92, bottom=0.08,
    )
    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    ax_pos = fig.add_subplot(gs[0, 2])
    ax_vel = fig.add_subplot(gs[0, 3])
    ax_eul = fig.add_subplot(gs[1, 2])
    ax_w = fig.add_subplot(gs[1, 3])
    ax_u = fig.add_subplot(gs[2, 2:4])

    # ----- 3D trajectory -----
    color = METHOD_COLORS.get(method, "#1f77b4")
    ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color=color, lw=1.6, label="trajectory")
    ax3d.scatter(p[0, 0], p[0, 1], p[0, 2], c="g", marker="o", s=100,
                 depthshade=False, label="start", zorder=10)
    ax3d.scatter(p[-1, 0], p[-1, 1], p[-1, 2], c="r", marker="*", s=180,
                 depthshade=False, label="end", zorder=10)
    if waypoints:
        wps = np.array([[wp[0], wp[1], wp[2]] for wp in waypoints], dtype=float)
        ax3d.scatter(wps[:, 0], wps[:, 1], wps[:, 2], facecolors="none",
                     edgecolors="k", s=140, linewidths=1.2, depthshade=False,
                     label="waypoints", zorder=9)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("3D trajectory")
    # Use bounding-box midpoint (NOT mean) so start/end stay symmetric in view.
    # The trajectory often dwells near the goal, biasing the mean and pushing
    # the start marker off-axis.
    xmin, xmax = float(p[:, 0].min()), float(p[:, 0].max())
    ymin, ymax = float(p[:, 1].min()), float(p[:, 1].max())
    zmin, zmax = float(p[:, 2].min()), float(p[:, 2].max())
    span = max((xmax - xmin), (ymax - ymin), (zmax - zmin), 1e-3)
    half = span * 0.55  # ~10% padding around bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    ax3d.set_xlim(cx - half, cx + half)
    ax3d.set_ylim(cy - half, cy + half)
    ax3d.set_zlim(cz - half, cz + half)
    try:
        ax3d.set_box_aspect((1, 1, 1))  # matplotlib >= 3.3
    except Exception:
        pass
    ax3d.legend(loc="upper left", fontsize=8)

    # ----- Position -----
    for col, lbl in enumerate(("x", "y", "z")):
        ax_pos.plot(t, p[:, col], label=lbl)
    ax_pos.set_xlabel("t (s)"); ax_pos.set_ylabel("position (m)")
    ax_pos.set_title("Position"); ax_pos.grid(True, alpha=0.3); ax_pos.legend(fontsize=8)

    # ----- Velocity -----
    for col, lbl in enumerate(("vx", "vy", "vz")):
        ax_vel.plot(t, v[:, col], label=lbl)
    ax_vel.set_xlabel("t (s)"); ax_vel.set_ylabel("velocity (m/s)")
    ax_vel.set_title("Velocity"); ax_vel.grid(True, alpha=0.3); ax_vel.legend(fontsize=8)

    # ----- Euler angles -----
    for col, lbl in enumerate(("roll", "pitch", "yaw")):
        ax_eul.plot(t, euler[:, col], label=lbl)
    ax_eul.set_xlabel("t (s)"); ax_eul.set_ylabel("euler (deg)")
    ax_eul.set_title("Attitude (ZYX Euler)"); ax_eul.grid(True, alpha=0.3); ax_eul.legend(fontsize=8)

    # ----- Angular velocity -----
    for col, lbl in enumerate(("wx", "wy", "wz")):
        ax_w.plot(t, w_ang[:, col], label=lbl)
    ax_w.set_xlabel("t (s)"); ax_w.set_ylabel("angular vel (rad/s)")
    ax_w.set_title("Angular velocity"); ax_w.grid(True, alpha=0.3); ax_w.legend(fontsize=8)

    # ----- Control (twin-y: angles in deg on left, thrust/torque on right) -----
    if u.shape[1] >= 4:
        th_p_deg = np.degrees(u[:, 0])
        th_r_deg = np.degrees(u[:, 1])
        T_N = u[:, 2]
        tau_yaw = u[:, 3]
        ax_u.step(t_u, th_p_deg, where="post", label="th_p (deg)", color="#1f77b4")
        ax_u.step(t_u, th_r_deg, where="post", label="th_r (deg)", color="#2ca02c")
        ax_u.set_xlabel("t (s)"); ax_u.set_ylabel("TVC angle (deg)")
        ax_u.grid(True, alpha=0.3)
        ax_u_right = ax_u.twinx()
        ax_u_right.step(t_u, T_N, where="post", label="T (N)", color="#d62728", lw=1.2)
        ax_u_right.step(t_u, tau_yaw, where="post", label="tau_yaw (N·m)", color="#9467bd", lw=1.2)
        ax_u_right.set_ylabel("thrust / yaw torque")
        # Combined legend
        h1, l1 = ax_u.get_legend_handles_labels()
        h2, l2 = ax_u_right.get_legend_handles_labels()
        ax_u.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, ncol=2)
        ax_u.set_title("Control inputs")

    total_T = float(t[-1]) if len(t) > 0 else 0.0
    # Trajectory length may exceed the requested ``N`` (e.g. acados min-time
    # enforces ``N_internal >= 50``); show both when they differ.
    N_actual = int(s["p"].shape[0] - 1)
    if N_actual == int(N):
        title_N = f"N={N}"
    else:
        title_N = f"N_req={N}, N_actual={N_actual}"
    fig.suptitle(
        f"TVC trajectory — {_method_name(method)}, {title_N}, dt={dt:.4f}s, "
        f"total T={total_T:.3f}s",
        fontsize=13,
    )

    traj_dir = out_dir / cpu_slug / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    out_path = traj_dir / f"traj_method{method}_N{N}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_summary_table(results: list[TrialResult]) -> None:
    """Pretty-print a tab-aligned summary of all trials to stdout."""
    if not results:
        print("[summary] no results to summarize.")
        return
    header = (
        f"{'method':<28s}{'N':>6s}{'dt(s)':>10s}{'total(s)':>10s}{'setup(s)':>10s}"
        f"{'solve(s)':>10s}{'iters':>8s}{'ms/iter':>10s}{'final_cost':>14s}  status"
    )
    print()
    print("=" * len(header))
    print("Benchmark summary (ms/iter is solve-only, excludes setup/codegen)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    # Sort by method, then N for readable layout
    for r in sorted(results, key=lambda x: (x.method, x.N)):
        cost = f"{r.final_cost:.3e}" if np.isfinite(r.final_cost) else "  nan"
        total = f"{r.total_time_s:.3f}" if np.isfinite(r.total_time_s) else "  nan"
        setup = f"{r.setup_time_s:.3f}" if np.isfinite(r.setup_time_s) else "  nan"
        solve = f"{r.solve_time_s:.3f}" if np.isfinite(r.solve_time_s) else "  nan"
        avg = f"{r.avg_time_per_iter_ms:.2f}" if np.isfinite(r.avg_time_per_iter_ms) else " nan"
        print(
            f"{r.method_name:<28s}{r.N:>6d}{r.dt:>10.4f}{total:>10s}{setup:>10s}"
            f"{solve:>10s}{r.total_iters:>8d}{avg:>10s}{cost:>14s}  {r.status}"
        )
    print("=" * len(header))


def plot_summary(out_dir: Path, cpu_info: dict, results: list[TrialResult], cpu_slug: str) -> Path | None:
    """Save a 4-panel summary figure by (method, N).

    Panels:
        1) Total time stacked as setup + solve (codegen exposed)
        2) Pure solve time (NLP only)
        3) Total iterations
        4) ms/iter (solve-only, excludes setup/codegen)

    Returns the path to the saved PNG, or ``None`` if matplotlib is unavailable
    or no successful trials are present.
    """
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        print("[plot] no successful trials to plot.")
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return None

    methods = sorted({r.method for r in ok})
    Ns = sorted({r.N for r in ok})
    n_methods = len(methods)
    n_N = len(Ns)

    table_setup = np.full((n_methods, n_N), np.nan)
    table_solve = np.full((n_methods, n_N), np.nan)
    table_iters = np.full((n_methods, n_N), np.nan)
    table_avg = np.full((n_methods, n_N), np.nan)
    for r in ok:
        i = methods.index(r.method)
        j = Ns.index(r.N)
        table_setup[i, j] = r.setup_time_s if np.isfinite(r.setup_time_s) else 0.0
        table_solve[i, j] = r.solve_time_s
        table_iters[i, j] = r.total_iters
        table_avg[i, j] = r.avg_time_per_iter_ms

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x_idx = np.arange(n_N, dtype=float)
    bar_w = 0.8 / max(n_methods, 1)

    # --- Panel 1: stacked total = setup + solve ---
    ax = axes[0]
    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2.0) * bar_w
        color = METHOD_COLORS.get(method, None)
        # Solve part (bottom), setup hatched on top
        solve_bars = ax.bar(
            x_idx + offset, table_solve[i], width=bar_w,
            color=color, edgecolor="white", linewidth=0.6,
            label=f"{_method_name(method)} (solve)" if i == 0 else None,
        )
        setup_bars = ax.bar(
            x_idx + offset, table_setup[i], width=bar_w,
            bottom=table_solve[i],
            color=color, edgecolor="white", linewidth=0.6,
            alpha=0.35, hatch="//",
            label=f"{_method_name(method)} (setup)" if i == 0 else None,
        )
        # Annotate total on top of stacked bar
        for k in range(n_N):
            tot = (table_setup[i, k] if np.isfinite(table_setup[i, k]) else 0.0) + (
                table_solve[i, k] if np.isfinite(table_solve[i, k]) else 0.0)
            if np.isfinite(tot) and tot > 0.0:
                ax.text(x_idx[k] + offset, tot,
                        f"{tot:.2f}" if tot < 100 else f"{tot:.0f}",
                        ha="center", va="bottom", fontsize=7)
        # Inline legend per-method
        if i == 0:
            ax.bar([], [], color="gray", label="solve")
            ax.bar([], [], color="gray", alpha=0.35, hatch="//", label="setup")
    ax.set_xticks(x_idx); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of nodes N"); ax.set_ylabel("seconds")
    ax.set_title("Total time = solve + setup (stacked)")
    ax.grid(True, axis="y", alpha=0.3)
    # Build a single combined legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS.get(m, "gray")) for m in methods]
    labels = [_method_name(m) for m in methods]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.35, hatch="//"))
    labels.append("setup (hatched)")
    ax.legend(handles, labels, fontsize=7, loc="best")

    # --- Panels 2-4: simple grouped bars ---
    panel_specs = (
        (axes[1], table_solve, "Solve time (NLP only)", "seconds"),
        (axes[2], table_iters, "Total iterations", "iters"),
        (axes[3], table_avg, "Avg time per iter (solve only)", "ms / iter"),
    )
    for ax, table, title, ylabel in panel_specs:
        for i, method in enumerate(methods):
            offset = (i - (n_methods - 1) / 2.0) * bar_w
            bars = ax.bar(
                x_idx + offset,
                table[i],
                width=bar_w,
                color=METHOD_COLORS.get(method, None),
                label=_method_name(method),
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, val in zip(bars, table[i]):
                if np.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, val,
                            f"{val:.2f}" if val < 100 else f"{val:.0f}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("Number of nodes N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    cpu_label = cpu_info.get("cpu_model") or "unknown CPU"
    fmax = cpu_info.get("cpu_mhz_max")
    freq_label = f", max {fmax/1000.0:.2f} GHz" if isinstance(fmax, (int, float)) else ""
    fig.suptitle(f"TVC traj-opt benchmark — {cpu_label}{freq_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    cpu_dir = out_dir / cpu_slug
    cpu_dir.mkdir(parents=True, exist_ok=True)
    out_path = cpu_dir / "benchmark.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved figure: {out_path}")
    return out_path


def save_trajectory_data(out_dir: Path, cpu_slug: str, method: int, N: int,
                         dt: float, duration: float, xs: list, us: list,
                         waypoints: list[list[float]] | None = None,
                         cfg: BenchmarkConfig | None = None) -> Path:
    """Persist the raw optimization output for later re-plotting.

    Files are written under ``<out_dir>/<cpu_slug>/data/method<m>_N<n>.npz`` and
    contain ``xs``, ``us``, ``dt``, ``duration``, ``method``, ``N``, plus the
    waypoints / start / goal used to build the problem (for plot annotations).
    """
    data_dir = out_dir / cpu_slug / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"method{method}_N{N}.npz"
    xs_arr = np.array([np.asarray(x, dtype=float).flatten() for x in xs])
    us_arr = np.array([np.asarray(u, dtype=float).flatten() for u in us])
    payload: dict[str, np.ndarray] = {
        "xs": xs_arr,
        "us": us_arr,
        "dt": np.array(dt, dtype=float),
        "duration": np.array(duration, dtype=float),
        "method": np.array(method, dtype=np.int32),
        "N": np.array(N, dtype=np.int32),
    }
    if waypoints is not None:
        payload["waypoints"] = np.asarray(waypoints, dtype=float)
    if cfg is not None:
        payload["start"] = np.asarray(cfg.start, dtype=float)
        payload["goal"] = np.asarray(cfg.goal, dtype=float)
    np.savez_compressed(path, **payload)
    return path


def load_trajectory_data(path: Path) -> dict[str, Any]:
    """Load a ``.npz`` written by :func:`save_trajectory_data`.

    Returns a dict with the same keys as the payload (arrays for ``xs``/``us``/
    ``waypoints``, scalars otherwise).
    """
    with np.load(path, allow_pickle=False) as z:
        out: dict[str, Any] = {
            "xs": z["xs"],
            "us": z["us"],
            "dt": float(z["dt"]),
            "duration": float(z["duration"]),
            "method": int(z["method"]),
            "N": int(z["N"]),
        }
        if "waypoints" in z.files:
            out["waypoints"] = z["waypoints"].tolist()
        if "start" in z.files:
            out["start"] = tuple(float(v) for v in z["start"].tolist())
        if "goal" in z.files:
            out["goal"] = tuple(float(v) for v in z["goal"].tolist())
    return out


CSV_FIELDS = (
    "timestamp", "cpu_model", "cpu_mhz_max", "cpu_mhz_current_avg", "cpu_count_logical",
    "system", "release", "python_version",
    "method", "method_name", "N", "dt", "duration",
    "total_time_s", "setup_time_s", "solve_time_s",
    "total_iters", "avg_time_per_iter_ms",
    "final_cost", "status", "error_msg",
)


def slugify_cpu(model: str | None) -> str:
    """Turn a CPU model string into a filesystem-friendly slug.

    Example: ``Intel(R) Core(TM) Ultra 7 265K`` -> ``Intel_Core_Ultra_7_265K``.
    Frequency suffixes such as ``@ 3.50GHz`` are dropped so the same chip
    yields the same slug regardless of governor state.
    """
    if not model:
        return "unknown_cpu"
    cleaned = re.sub(r"\(R\)|\(TM\)|\(C\)|@.*", "", model)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown_cpu"


def write_results(out_dir: Path, cfg: BenchmarkConfig, cpu_info: dict, results: list[TrialResult]) -> tuple[Path, Path]:
    cpu_slug = slugify_cpu(cpu_info.get("cpu_model"))
    cpu_dir = out_dir / cpu_slug
    cpu_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = cpu_dir / "benchmark.csv"
    json_path = cpu_dir / "benchmark.json"
    if csv_path.exists() or json_path.exists():
        print(f"[benchmark] overwriting existing files for CPU '{cpu_slug}'")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "timestamp": stamp,
                "cpu_model": cpu_info.get("cpu_model"),
                "cpu_mhz_max": cpu_info.get("cpu_mhz_max"),
                "cpu_mhz_current_avg": cpu_info.get("cpu_mhz_current_avg"),
                "cpu_count_logical": cpu_info.get("cpu_count_logical"),
                "system": cpu_info.get("system"),
                "release": cpu_info.get("release"),
                "python_version": cpu_info.get("python_version"),
                "method": r.method,
                "method_name": r.method_name,
                "N": r.N,
                "dt": f"{r.dt:.6f}",
                "duration": f"{r.duration:.4f}",
                "total_time_s": f"{r.total_time_s:.6f}",
                "setup_time_s": f"{r.setup_time_s:.6f}",
                "solve_time_s": f"{r.solve_time_s:.6f}",
                "total_iters": r.total_iters,
                "avg_time_per_iter_ms": f"{r.avg_time_per_iter_ms:.4f}",
                "final_cost": f"{r.final_cost:.6e}",
                "status": r.status,
                "error_msg": r.error_msg,
            })

    bundle = {
        "timestamp": stamp,
        "cpu_info": cpu_info,
        "config": {
            "start": list(cfg.start),
            "goal": list(cfg.goal),
            "duration": cfg.duration,
            "nodes": list(cfg.nodes),
            "methods": list(cfg.methods),
            "max_iter": cfg.max_iter,
            "warmup": cfg.warmup,
            "bounds_human": cfg.bounds_human,
            "phys": cfg.phys,
            "config_path": cfg.config_path,
        },
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(bundle, f, indent=2, default=str)

    return csv_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_xyzyaw(s: str) -> tuple[float, float, float, float]:
    parts = [p for p in s.replace(";", ",").split(",") if p.strip()]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(f"expected 3 or 4 comma-separated values, got: {s!r}")
    vals = [float(p) for p in parts]
    if len(vals) == 3:
        vals.append(0.0)
    return (vals[0], vals[1], vals[2], vals[3])


def _parse_int_list(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in s.replace(";", ",").split(",") if p.strip())


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    p = argparse.ArgumentParser(
        description="TVC trajectory optimization benchmark (Methods 3/4/5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", type=_parse_xyzyaw, default="0,0,0,0",
                   help="Start waypoint x,y,z[,yaw_deg]")
    p.add_argument("--goal", type=_parse_xyzyaw, default="1,1,1,0",
                   help="Goal waypoint x,y,z[,yaw_deg]")
    p.add_argument("--duration", type=float, default=5.0, help="Total trajectory time [s]")
    p.add_argument("--nodes", type=_parse_int_list, default="20,50,100,200",
                   help="Comma-separated list of N (node) values to sweep")
    p.add_argument("--methods", type=_parse_int_list, default="3,4,5",
                   help="Comma-separated method ids to benchmark (subset of {3,4,5})")
    p.add_argument("--max-iter", type=int, default=100, help="Solver iteration cap per segment")
    p.add_argument("--warmup", action="store_true",
                   help="Run one extra trial per (method,N) cell first and discard its timing")
    p.add_argument("--output-dir", type=str, default="",
                   help="Where to write CSV/JSON/NPZ/PNG (default: <repo>/benchmark/results)")
    p.add_argument("--config", type=str, default=None,
                   help="Optional JSON file that overrides bounds_human / phys / weights")
    p.add_argument("--no-plot", action="store_true",
                   help="Disable summary PNG plot (text summary is always printed)")
    args = p.parse_args(argv)

    cfg = BenchmarkConfig(
        start=tuple(args.start),
        goal=tuple(args.goal),
        duration=float(args.duration),
        nodes=tuple(int(n) for n in args.nodes),
        methods=tuple(int(m) for m in args.methods),
        max_iter=int(args.max_iter),
        warmup=bool(args.warmup),
        output_dir=args.output_dir or str(_BENCH_DIR / "results"),
        config_path=args.config,
    )
    cfg.plot = not bool(args.no_plot)

    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        with open(cfg_path) as f:
            override = json.load(f)
        if "bounds_human" in override and isinstance(override["bounds_human"], dict):
            cfg.bounds_human.update(override["bounds_human"])
        if "phys" in override and isinstance(override["phys"], dict):
            cfg.phys.update(override["phys"])
        cfg.config_path = str(cfg_path)

    bad = [m for m in cfg.methods if m not in (3, 4, 5)]
    if bad:
        raise SystemExit(f"Unsupported methods: {bad}. Supported: 3,4,5.")
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)

    if _preload_acados_libs is not None and any(m in (4, 5) for m in cfg.methods):
        try:
            _preload_acados_libs()
        except Exception as e:
            print(f"[benchmark] WARNING: acados lib preload failed: {e}", file=sys.stderr)

    cpu_info = collect_cpu_info()

    print("=" * 70)
    print("TVC trajectory optimization benchmark")
    print("=" * 70)
    print(f"  CPU         : {cpu_info.get('cpu_model')}")
    print(f"  CPU MHz max : {cpu_info.get('cpu_mhz_max')}")
    print(f"  CPU MHz cur : {cpu_info.get('cpu_mhz_current_avg')}")
    print(f"  Host / OS   : {cpu_info.get('hostname')} / {cpu_info.get('system')} {cpu_info.get('release')}")
    print(f"  Python      : {cpu_info.get('python_version')}")
    print(f"  start       : {cfg.start}")
    print(f"  goal        : {cfg.goal}")
    print(f"  duration    : {cfg.duration} s")
    print(f"  nodes       : {list(cfg.nodes)}")
    print(f"  methods     : {list(cfg.methods)}")
    print(f"  max_iter    : {cfg.max_iter}")
    print(f"  warmup      : {cfg.warmup}")
    print("=" * 70)

    results: list[TrialResult] = []
    trajectories: dict[tuple[int, int], tuple[list, list, float]] = {}
    waypoints = _build_waypoints(cfg.start, cfg.goal, cfg.duration)
    out_dir = Path(cfg.output_dir)
    cpu_slug = slugify_cpu(cpu_info.get("cpu_model"))
    data_paths: list[Path] = []
    for method in cfg.methods:
        for N in cfg.nodes:
            if cfg.warmup:
                print(f"[warmup ] method={method} N={N} ...", flush=True)
                _ = run_trial(cfg, method, N)
            print(f"[bench  ] method={method} N={N} ...", flush=True)
            r, xs, us = run_trial(cfg, method, N)
            results.append(r)
            print(
                f"  -> status={r.status:<7s}  total={r.total_time_s:>8.3f}s  "
                f"setup={r.setup_time_s:>7.3f}s  solve={r.solve_time_s:>7.3f}s  "
                f"iters={r.total_iters:<5d}  avg={r.avg_time_per_iter_ms:>8.2f} ms/iter  "
                f"cost={r.final_cost:.3e}"
            )
            if r.status == "error":
                print(f"     error: {r.error_msg}")
            elif xs is not None and us is not None:
                trajectories[(method, N)] = (xs, us, r.dt)
                # Always persist raw optimization output so plot_benchmark.py
                # can re-render plots without re-running the solver. For
                # Method 5 (min-time), ``r.dt`` / ``r.duration`` already reflect
                # the optimized segment times (via meta['plot_dt']).
                npz_path = save_trajectory_data(
                    out_dir, cpu_slug, method, N, r.dt, r.duration,
                    xs, us, waypoints=waypoints, cfg=cfg,
                )
                data_paths.append(npz_path)

    csv_path, json_path = write_results(out_dir, cfg, cpu_info, results)
    print_summary_table(results)
    png_path = None
    traj_paths: list[Path] = []
    if cfg.plot:
        png_path = plot_summary(out_dir, cpu_info, results, cpu_slug)
        for (method, N), (xs, us, dt) in trajectories.items():
            p = plot_trajectory(out_dir, cpu_slug, method, N, dt,
                                xs, us, waypoints=waypoints)
            if p is not None:
                traj_paths.append(p)
    print("=" * 70)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    for p in data_paths:
        print(f"Wrote: {p}")
    if png_path is not None:
        print(f"Wrote: {png_path}")
    for p in traj_paths:
        print(f"Wrote: {p}")
    print("=" * 70)
    print(
        "Hint: rerun plotting only (no solver) via "
        f"`python {_BENCH_DIR / 'plot_benchmark.py'} --cpu {cpu_slug}`"
    )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
