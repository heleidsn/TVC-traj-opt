# -*- coding: utf-8 -*-
"""
Differential-flatness trajectory planner (Method 8).

Plans smooth rest-to-rest segments on the center-of-oscillation flat outputs
(xi_x, xi_y) plus altitude z and yaw, then reconstructs feasible states/controls.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLERS_DIR = os.path.join(ROOT_DIR, 'controllers')
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT_DIR, CONTROLLERS_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from controllers.flatness import (  # noqa: E402
    FlatnessParams,
    clip_controls,
    method1_from_flat,
    min_snap_1d,
)

class _FlatnessLogger:
    """Minimal logger placeholder for GUI cost plots."""

    def __init__(self):
        self.costs = [0.0]


def _inertia_diagonal(I) -> Tuple[float, float, float]:
    """Extract (Ixx, Iyy, Izz) from a 3-vector or 3×3 inertia matrix."""
    arr = np.asarray(I, dtype=float)
    if arr.ndim == 2:
        return float(arr[0, 0]), float(arr[1, 1]), float(arr[2, 2])
    flat = arr.reshape(-1)
    if flat.size < 3:
        raise ValueError('Inertia must provide at least Ixx, Iyy, Izz')
    return float(flat[0]), float(flat[1]), float(flat[2])


def _thrust_position(r_thrust) -> Tuple[float, float, float]:
    r = np.asarray(r_thrust, dtype=float).reshape(-1)
    if r.size < 3:
        raise ValueError('r_thrust must have 3 components')
    return float(r[0]), float(r[1]), float(r[2])


def _segment_samples(t0: float, t1: float, dt: float) -> np.ndarray:
    n = max(int(np.ceil((t1 - t0) / dt)) + 1, 2)
    return np.linspace(t0, t1, n)


def _eval_segment_flat(
    fp: FlatnessParams,
    t: float,
    t0: float,
    t1: float,
    x0: float,
    x1: float,
) -> Tuple[float, float, float, float, float]:
    return min_snap_1d(t, t0, t1, x0, x1)


def solve_with_flatness_waypoints(
    dt: float,
    waypoints: List[List[float]],
    m: float,
    I: Tuple[float, float, float],
    r_thrust: Tuple[float, float, float],
    weights: Optional[Dict[str, Any]] = None,
    bounds: Optional[Dict[str, Any]] = None,
    max_iter: int = 1,
    callback: Optional[Callable] = None,
    running_flag: Optional[Callable[[], bool]] = None,
    iteration_callback: Optional[Callable] = None,
    verbose_solve: bool = False,
    unified: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[_FlatnessLogger], List[np.ndarray], Dict[str, Any]]:
    """
    Build a dynamically feasible trajectory via differential flatness.

    Returns (combined_xs, combined_us, loggers, us_actual, meta) matching Acados solvers.
    States are Method-1 17-dim; controls are [th_p, th_r, T, tau_yaw].
    """
    del max_iter, verbose_solve, unified
    weights = weights or {}
    bounds = bounds or {}
    if running_flag is None:
        running_flag = lambda: True

    if len(waypoints) < 2:
        raise ValueError('Need at least 2 waypoints')

    ixx, iyy, izz = _inertia_diagonal(I)
    _, _, r_thrust_z = _thrust_position(r_thrust)
    fp = FlatnessParams.from_gui(m, ixx, iyy, izz, r_thrust_z, g=float(bounds.get('g', 9.81)))

    th_p_max = float(bounds.get('th_p_max', 10.0))
    th_r_max = float(bounds.get('th_r_max', 10.0))
    t_bounds = bounds.get('T')
    t_min = float(bounds.get('T_min', t_bounds[0] if t_bounds else 0.0))
    t_max = float(bounds.get('T_max', t_bounds[1] if t_bounds else m * fp.g * 3.0))
    tau_max = float(bounds.get('tau_yaw_max', 1.0))

    combined_xs: List[np.ndarray] = []
    combined_us: List[np.ndarray] = []
    flat_samples: Dict[str, List[float]] = {
        't': [], 'xi_x': [], 'xi_y': [], 'z': [], 'psi': [],
    }
    segment_boundaries = [0]

    for seg_idx in range(len(waypoints) - 1):
        if not running_flag():
            break
        wp0 = waypoints[seg_idx]
        wp1 = waypoints[seg_idx + 1]
        t0 = float(wp0[4])
        t1 = float(wp1[4])
        if t1 <= t0:
            raise ValueError(f'Segment {seg_idx}: arrival time must increase')

        times = _segment_samples(t0, t1, dt)
        seg_xs: List[np.ndarray] = []
        seg_us: List[np.ndarray] = []
        seg_flat: Dict[str, List[float]] = {
            't': [], 'xi_x': [], 'xi_y': [], 'z': [], 'psi': [],
        }

        x0, y0, z0 = float(wp0[0]), float(wp0[1]), float(wp0[2])
        x1, y1, z1 = float(wp1[0]), float(wp1[1]), float(wp1[2])
        psi0 = np.radians(float(wp0[3]) if len(wp0) > 3 else 0.0)
        psi1 = np.radians(float(wp1[3]) if len(wp1) > 3 else 0.0)

        for i, t in enumerate(times):
            if i > 0 and t == times[i - 1]:
                continue
            xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x = _eval_segment_flat(fp, t, t0, t1, x0, x1)
            xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y = _eval_segment_flat(fp, t, t0, t1, y0, y1)
            z, dz, ddz, _, _ = _eval_segment_flat(fp, t, t0, t1, z0, z1)
            psi, dpsi, ddpsi, _, _ = _eval_segment_flat(fp, t, t0, t1, psi0, psi1)

            x17, u_opt = method1_from_flat(
                fp,
                xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x,
                xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y,
                z, dz, ddz,
                psi, dpsi, ddpsi,
            )
            u_opt = clip_controls(u_opt, th_p_max, th_r_max, t_min, t_max, tau_max)
            seg_xs.append(x17)
            seg_us.append(u_opt)
            seg_flat['t'].append(float(t))
            seg_flat['xi_x'].append(float(xi_x))
            seg_flat['xi_y'].append(float(xi_y))
            seg_flat['z'].append(float(z))
            seg_flat['psi'].append(float(psi))

        if iteration_callback is not None:
            iteration_callback(1, 0.0, 0.0, seg_idx)

        if callback is not None:
            completed_xs = [combined_xs] if combined_xs else []
            completed_us = [combined_us] if combined_us else []
            callback(None, seg_idx, seg_xs, seg_us, completed_xs, completed_us)

        if seg_idx == 0:
            combined_xs.extend(seg_xs)
            combined_us.extend(seg_us)
            for key in flat_samples:
                flat_samples[key].extend(seg_flat[key])
        else:
            combined_xs.extend(seg_xs[1:])
            combined_us.extend(seg_us[1:] if seg_us else [])
            for key in flat_samples:
                flat_samples[key].extend(seg_flat[key][1:])
        segment_boundaries.append(len(combined_xs) - 1)

    if not combined_xs:
        raise RuntimeError('Flatness planner produced no samples')

    us_actual = [u.copy() for u in combined_us]
    loggers = [_FlatnessLogger() for _ in waypoints[:-1]]

    meta = {
        'method': 'Method 8 (Differential flatness)',
        'plot_dt': float(dt),
        'segment_boundary_indices': segment_boundaries,
        'flatness': True,
        'flat_outputs': {k: np.asarray(v, dtype=float) for k, v in flat_samples.items()},
        'time_states': np.asarray(flat_samples['t'], dtype=float),
        'flatness_physics': {
            'mass': float(m),
            'Ixx': ixx,
            'Iyy': iyy,
            'Izz': izz,
            'r_thrust_z': r_thrust_z,
            'g': float(bounds.get('g', 9.81)),
        },
    }
    if len(meta['time_states']) != len(combined_xs):
        raise RuntimeError(
            f'Flatness planner time/state length mismatch: '
            f'{len(meta["time_states"])} vs {len(combined_xs)}'
        )
    return combined_xs, combined_us, loggers, us_actual, meta


def solve_with_flatness_waypoints_unified(*args, **kwargs):
    """Alias: flatness planner always builds one concatenated trajectory."""
    kwargs['unified'] = True
    return solve_with_flatness_waypoints(*args, **kwargs)
