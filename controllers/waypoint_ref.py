# -*- coding: utf-8 -*-
"""Piecewise-hold reference from NMP waypoints (no min-snap planning)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .flatness import FlatnessParams, state13_from_flat
from .linear_model import state13_to_state12


def _normalize_waypoint(wp: Sequence[float]) -> Tuple[float, float, float, float, float]:
    row = [float(wp[i]) for i in range(min(len(wp), 5))]
    while len(row) < 5:
        row.append(0.0)
    return tuple(row)  # type: ignore[return-value]


def waypoint_target_index(t: float, waypoints: Sequence[Sequence[float]]) -> int:
    """
    Active GotoSetpoint target at time *t*.

    Waypoints are ``[ξ_x, ξ_y, z, yaw_deg, arrival_time]``. Before the first
    interior arrival time the target is waypoint 1; after the last arrival the
    target is the final waypoint.
    """
    n = len(waypoints)
    if n <= 1:
        return 0
    t = float(t)
    for i in range(1, n):
        if t < float(_normalize_waypoint(waypoints[i])[4]) - 1e-9:
            return i
    return n - 1


def _hover_state13_from_waypoint(
    fp: FlatnessParams,
    wp: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Equilibrium 13-state + plant control at a ξ waypoint (zero motion)."""
    xi_x, xi_y, z, yaw_deg, _t = _normalize_waypoint(wp)
    psi = np.radians(yaw_deg)
    x13, u_opt = state13_from_flat(
        fp,
        xi_x, 0.0, 0.0, 0.0, 0.0,
        xi_y, 0.0, 0.0, 0.0, 0.0,
        z, 0.0, 0.0,
        psi, 0.0, 0.0,
    )
    return x13, u_opt


def build_waypoint_flight_reference(
    waypoints: Sequence[Sequence[float]],
    flatness_physics: Dict[str, float],
    dt: float = 0.05,
    terminal_hold_s: float = 3.0,
) -> Dict[str, Any]:
    """
    Build a step-hold trajectory reference for direct waypoint flight.

    Returns a dict compatible with NMP tracking / plotting:
    ``xs``, ``us``, ``time_states``, ``flat_outputs``, ``flatness_physics``,
    ``x0`` (12-state at the first waypoint), ``waypoint_mode``.
    """
    if len(waypoints) < 2:
        raise ValueError('Need at least 2 waypoints for direct flight.')

    wps = [_normalize_waypoint(w) for w in waypoints]
    for i in range(len(wps) - 1):
        if wps[i][4] >= wps[i + 1][4]:
            raise ValueError(
                f'Waypoint arrival times must increase (leg {i} → {i + 1}).',
            )

    fp = FlatnessParams.from_gui(
        flatness_physics['mass'],
        flatness_physics['Ixx'],
        flatness_physics['Iyy'],
        flatness_physics['Izz'],
        flatness_physics['r_thrust_z'],
        g=float(flatness_physics.get('g', 9.81)),
    )

    t_end = float(wps[-1][4]) + max(float(terminal_hold_s), 0.0)
    dt = max(float(dt), 0.01)
    n = max(int(np.ceil(t_end / dt)) + 1, 2)
    times = np.linspace(0.0, t_end, n, dtype=float)

    xs = np.zeros((n, 13), dtype=float)
    us = np.zeros((n - 1, 4), dtype=float)
    xi_x = np.zeros(n)
    xi_y = np.zeros(n)
    z_arr = np.zeros(n)
    psi_arr = np.zeros(n)

    for k, t in enumerate(times):
        wi = waypoint_target_index(t, wps)
        x13, u_opt = _hover_state13_from_waypoint(fp, wps[wi])
        xs[k] = x13
        if k < n - 1:
            us[k] = u_opt
        xi_x[k], xi_y[k], z_arr[k], yaw_deg, _ = wps[wi]
        psi_arr[k] = np.radians(yaw_deg)

    x0_13, _ = _hover_state13_from_waypoint(fp, wps[0])
    flat_outputs = {
        't': times.copy(),
        'xi_x': xi_x,
        'xi_y': xi_y,
        'z': z_arr,
        'psi': psi_arr,
        'piecewise_constant': True,
    }

    return {
        'xs': xs.tolist(),
        'us': us.tolist(),
        'time_states': times,
        'dt': dt,
        'flat_outputs': flat_outputs,
        'flatness_physics': dict(flatness_physics),
        'x0': state13_to_state12(x0_13),
        'waypoint_mode': True,
        'method_name': 'Direct waypoint flight',
    }
