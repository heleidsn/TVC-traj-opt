# -*- coding: utf-8 -*-
"""NMP / differential-flatness visualization: flat output ξ vs COM state."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from controllers.flatness import (
    FlatnessParams,
    estimate_flat_state_from_state12,
    lateral_flat_reconstruct,
)
from controllers.linear_model import state13_to_state12


def nmp_axes_dict(window) -> Dict[str, Any]:
    """Matplotlib axes for the 3×2 NMP plot tab."""
    return {
        'position': window.ax_nmp_position,
        'velocity': window.ax_nmp_velocity,
        'angle': window.ax_nmp_angle,
        'angvel': window.ax_nmp_angvel,
        'gimbal': window.ax_nmp_gimbal,
        'thrust': window.ax_nmp_thrust,
    }


def _physics_to_flatness_params(phy: Dict[str, float]) -> FlatnessParams:
    return FlatnessParams.from_gui(
        phy['mass'], phy['Ixx'], phy['Iyy'], phy['Izz'],
        phy['r_thrust_z'], g=float(phy.get('g', 9.81)),
    )


def _interp_us_to_t(t, us, n_state):
    """Interpolate plant controls [th_p, th_r, T, tau_yaw] onto state time grid."""
    us = np.asarray(us, dtype=float)
    if us.ndim != 2 or us.shape[1] < 4:
        return None
    nu = us.shape[0]
    if nu == n_state:
        t_u = t
    elif nu == n_state - 1:
        t_u = t[:-1] if len(t) > 1 else t
        if len(t_u) != nu:
            t_u = np.linspace(t[0], t[-1], nu)
    else:
        t_u = np.linspace(t[0], t[-1], nu)
    out = np.zeros((n_state, 4), dtype=float)
    for j in range(4):
        out[:, j] = np.interp(t, t_u, us[:, j])
    return out


def _flat_derivatives_on_grid(t, t_flat, flat: Dict[str, np.ndarray], key: str):
    """ξ and derivatives up to 4th order on the state time grid."""
    raw = np.asarray(flat[key], dtype=float)
    d1 = np.gradient(raw, t_flat, edge_order=2)
    d2 = np.gradient(d1, t_flat, edge_order=2)
    d3 = np.gradient(d2, t_flat, edge_order=2)
    d4 = np.gradient(d3, t_flat, edge_order=2)
    return (
        np.interp(t, t_flat, raw),
        np.interp(t, t_flat, d1),
        np.interp(t, t_flat, d2),
        np.interp(t, t_flat, d3),
        np.interp(t, t_flat, d4),
    )


def build_nmp_series(
    traj: Optional[Dict[str, Any]],
    phy: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Build ξ vs COM series plus flat-derived states and controls."""
    if not traj or traj.get('xs') is None:
        return None

    xs = np.asarray(traj['xs'], dtype=float)
    if xs.ndim != 2 or xs.shape[0] < 2:
        return None

    x12 = state13_to_state12(xs) if xs.shape[1] >= 13 else xs[:, :12]
    n = x12.shape[0]

    ts = traj.get('time_states')
    if ts is not None and len(ts) == n:
        t = np.asarray(ts, dtype=float)
    else:
        dt = float(traj.get('dt', 0.05))
        t = np.arange(n, dtype=float) * dt

    fp_phy = traj.get('flatness_physics') or phy
    if fp_phy is None:
        return None
    fp = _physics_to_flatness_params(fp_phy)
    g = fp.g

    flat = traj.get('flat_outputs') or {}
    has_stored = (
        isinstance(flat, dict)
        and flat.get('t') is not None
        and len(flat['t']) >= 2
    )

    xi_x = np.zeros(n)
    xi_y = np.zeros(n)
    dxi_x = np.zeros(n)
    dxi_y = np.zeros(n)
    ddxi_x = np.zeros(n)
    ddxi_y = np.zeros(n)
    dddxi_x = np.zeros(n)
    dddxi_y = np.zeros(n)
    ddddxi_x = np.zeros(n)
    ddddxi_y = np.zeros(n)
    dpsi = np.zeros(n)

    if has_stored:
        t_flat = np.asarray(flat['t'], dtype=float)
        if 'xi_x' in flat:
            xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x = _flat_derivatives_on_grid(
                t, t_flat, flat, 'xi_x',
            )
        if 'xi_y' in flat:
            xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y = _flat_derivatives_on_grid(
                t, t_flat, flat, 'xi_y',
            )
        z_flat = (
            np.interp(t, t_flat, np.asarray(flat['z'], dtype=float))
            if 'z' in flat else x12[:, 2].copy()
        )
        if 'psi' in flat:
            psi_flat, dpsi, _, _, _ = _flat_derivatives_on_grid(t, t_flat, flat, 'psi')
        else:
            psi_flat = 2.0 * np.arcsin(np.clip(x12[:, 8], -1.0, 1.0))
            dpsi = x12[:, 11]
        if 'z' in flat:
            _, dz, ddz = _flat_derivatives_on_grid(t, t_flat, flat, 'z')[:3]
        else:
            dz = x12[:, 5]
            ddz = np.gradient(x12[:, 5], t, edge_order=2)
    else:
        z_flat = x12[:, 2].copy()
        psi_flat = 2.0 * np.arcsin(np.clip(x12[:, 8], -1.0, 1.0))
        dz = x12[:, 5]
        ddz = np.gradient(x12[:, 5], t, edge_order=2)
        dpsi = x12[:, 11]
        for i in range(n):
            est = estimate_flat_state_from_state12(fp, x12[i])
            xi_x[i] = est['xi_x']
            xi_y[i] = est['xi_y']
            dxi_x[i] = est['dxi_x']
            dxi_y[i] = est['dxi_y']

    x_com = x12[:, 0]
    y_com = x12[:, 1]
    z_com = x12[:, 2]
    vx, vy, vz = x12[:, 3], x12[:, 4], x12[:, 5]
    p, q_rate, r = x12[:, 9], x12[:, 10], x12[:, 11]
    roll = 2.0 * x12[:, 6]
    pitch = 2.0 * x12[:, 7]
    yaw = 2.0 * np.arcsin(np.clip(x12[:, 8], -1.0, 1.0))

    pitch_flat = np.zeros(n)
    roll_flat = np.zeros(n)
    q_flat = np.zeros(n)
    p_flat = np.zeros(n)
    th_p_flat = np.zeros(n)
    th_r_flat = np.zeros(n)
    for i in range(n):
        fxi = lateral_flat_reconstruct(
            fp, xi_x[i], dxi_x[i], ddxi_x[i], dddxi_x[i], ddddxi_x[i], channel='pitch',
        )
        fyi = lateral_flat_reconstruct(
            fp, xi_y[i], dxi_y[i], ddxi_y[i], dddxi_y[i], ddddxi_y[i], channel='roll',
        )
        pitch_flat[i] = 2.0 * fxi['q_comp']
        roll_flat[i] = 2.0 * fyi['q_comp']
        q_flat[i] = fxi['rate_comp']
        p_flat[i] = fyi['rate_comp']
        th_p_flat[i] = fxi['delta']
        th_r_flat[i] = fyi['delta']

    thrust_flat = fp.mass * (g + ddz)
    offset_x_meas = x_com - xi_x
    offset_y_meas = y_com - xi_y

    us_on_t = None
    if traj.get('us') is not None:
        us_on_t = _interp_us_to_t(t, traj['us'], n)
    elif xs.shape[1] >= 17:
        us_on_t = xs[:, 13:17].copy()

    return {
        't': t,
        'xi_x': xi_x, 'xi_y': xi_y,
        'x_com': x_com, 'y_com': y_com, 'z_com': z_com, 'z_flat': z_flat,
        'vx': vx, 'vy': vy, 'vz': vz,
        'dxi_x': dxi_x, 'dxi_y': dxi_y, 'dz': dz,
        'roll_deg': np.degrees(roll), 'pitch_deg': np.degrees(pitch),
        'yaw_deg': np.degrees(yaw), 'psi_flat_deg': np.degrees(psi_flat),
        'roll_flat_deg': np.degrees(roll_flat), 'pitch_flat_deg': np.degrees(pitch_flat),
        'p_deg_s': np.degrees(p), 'q_deg_s': np.degrees(q_rate), 'r_deg_s': np.degrees(r),
        'p_flat_deg_s': np.degrees(p_flat), 'q_flat_deg_s': np.degrees(q_flat),
        'dpsi_deg_s': np.degrees(dpsi),
        'th_p_flat_deg': np.degrees(th_p_flat), 'th_r_flat_deg': np.degrees(th_r_flat),
        'thrust_flat': thrust_flat,
        'offset_x_meas': offset_x_meas, 'offset_y_meas': offset_y_meas,
        'us': us_on_t,
        'omega_z_pitch': fp.omega_z_pitch,
        'flat_offset_pitch': fp.flat_offset_pitch,
        'mass': fp.mass, 'g': g,
    }


def _clear_ax(ax):
    ax.clear()
    ax.grid(True, alpha=0.3)
    twin = getattr(ax, '_nmp_twin', None)
    if twin is not None:
        try:
            twin.remove()
        except Exception:
            pass
        ax._nmp_twin = None


def _ensure_twin(ax):
    twin = getattr(ax, '_nmp_twin', None)
    if twin is None:
        twin = ax.twinx()
        ax._nmp_twin = twin
    else:
        twin.clear()
    return twin


def draw_nmp_panels(axes: Dict[str, Any], series: Optional[Dict[str, Any]], summary: str = ''):
    """Redraw 3×2 NMP panels: position | velocity / angle | angvel / gimbal | thrust."""
    fig = axes['position'].figure
  # all axes share figure

    if series is None:
        fig.suptitle(
            summary or 'NMP — run flatness planning',
            fontsize=11, fontweight='bold', y=0.98,
        )
        for ax in axes.values():
            _clear_ax(ax)
            ax.set_xlabel('Time (s)')
        return

    t = series['t']
    max_dx = float(np.max(np.abs(series['offset_x_meas']))) * 1000
    max_dy = float(np.max(np.abs(series['offset_y_meas']))) * 1000
    fig.suptitle(
        f'NMP flat planning  |  peak |x−ξ_x|={max_dx:.1f} mm  |y−ξ_y|={max_dy:.1f} mm'
        + (f'  |  {summary}' if summary else ''),
        fontsize=10, fontweight='bold', y=0.98,
    )

    # ── Position: ξ vs COM (+ Δ on right axis, mm) ──
    ax = axes['position']
    _clear_ax(ax)
    ax.plot(t, series['xi_x'], 'C0-', lw=1.8, label='ξ_x')
    ax.plot(t, series['x_com'], 'C0--', lw=1.2, label='x COM')
    ax.plot(t, series['xi_y'], 'C1-', lw=1.8, label='ξ_y')
    ax.plot(t, series['y_com'], 'C1--', lw=1.2, label='y COM')
    ax.plot(t, series['z_flat'], 'C2-', lw=1.4, label='z (flat)')
    ax.plot(t, series['z_com'], 'C2:', lw=1.0, alpha=0.8, label='z COM')
    ax.set_ylabel('m')
    ax.set_title('Position')
    ax2 = _ensure_twin(ax)
    ax2.plot(t, series['offset_x_meas'] * 1000, 'C3-', lw=1.0, alpha=0.75, label='Δx mm')
    ax2.plot(t, series['offset_y_meas'] * 1000, 'C4-', lw=1.0, alpha=0.75, label='Δy mm')
    ax2.set_ylabel('COM−ξ (mm)', fontsize=8)
    ax2.tick_params(labelsize=7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=5.5, ncol=2)

    # ── Velocity ──
    ax = axes['velocity']
    _clear_ax(ax)
    ax.plot(t, series['dxi_x'], 'C0-', lw=1.6, label='ξ̇_x')
    ax.plot(t, series['vx'], 'C0--', lw=1.1, label='v_x')
    ax.plot(t, series['dxi_y'], 'C1-', lw=1.6, label='ξ̇_y')
    ax.plot(t, series['vy'], 'C1--', lw=1.1, label='v_y')
    ax.plot(t, series['dz'], 'C2-', lw=1.3, label='ż (flat)')
    ax.plot(t, series['vz'], 'C2:', lw=1.0, alpha=0.8, label='v_z')
    ax.set_ylabel('m/s')
    ax.set_title('Velocity')
    ax.legend(loc='best', fontsize=6)

    # ── Angle (deg) ──
    ax = axes['angle']
    _clear_ax(ax)
    ax.plot(t, series['pitch_flat_deg'], 'C0-', lw=1.6, label='θ pitch (ξ)')
    ax.plot(t, series['pitch_deg'], 'C0--', lw=1.1, label='θ pitch (state)')
    ax.plot(t, series['roll_flat_deg'], 'C1-', lw=1.6, label='φ roll (ξ)')
    ax.plot(t, series['roll_deg'], 'C1--', lw=1.1, label='φ roll (state)')
    ax.plot(t, series['psi_flat_deg'], 'C2-', lw=1.3, label='ψ yaw (flat)')
    ax.plot(t, series['yaw_deg'], 'C2:', lw=1.0, alpha=0.8, label='ψ yaw (state)')
    ax.set_ylabel('deg')
    ax.set_title('Angle')
    ax.legend(loc='best', fontsize=6)

    # ── Angular velocity (deg/s) ──
    ax = axes['angvel']
    _clear_ax(ax)
    ax.plot(t, series['q_flat_deg_s'], 'C0-', lw=1.6, label='q̇ from ξ (pitch)')
    ax.plot(t, series['q_deg_s'], 'C0--', lw=1.1, label='q (state)')
    ax.plot(t, series['p_flat_deg_s'], 'C1-', lw=1.6, label='ṗ from ξ (roll)')
    ax.plot(t, series['p_deg_s'], 'C1--', lw=1.1, label='p (state)')
    ax.plot(t, series['dpsi_deg_s'], 'C2-', lw=1.3, label='ψ̇ (flat)')
    ax.plot(t, series['r_deg_s'], 'C2:', lw=1.0, alpha=0.8, label='r (state)')
    ax.set_ylabel('deg/s')
    ax.set_title('Angular velocity')
    ax.legend(loc='best', fontsize=6)

    # ── Gimbal ──
    ax = axes['gimbal']
    _clear_ax(ax)
    ax.plot(t, series['th_p_flat_deg'], 'C0-', lw=1.6, label='δ_pitch (ξ⁽⁴⁾)')
    ax.plot(t, series['th_r_flat_deg'], 'C1-', lw=1.6, label='δ_roll (ξ⁽⁴⁾)')
    us = series.get('us')
    if us is not None:
        ax.plot(t, np.degrees(us[:, 0]), 'C0--', lw=1.1, label='th_p plan')
        ax.plot(t, np.degrees(us[:, 1]), 'C1--', lw=1.1, label='th_r plan')
    ax.set_ylabel('deg')
    ax.set_title('Gimbal angles')
    ax.legend(loc='best', fontsize=6)

    # ── Thrust & yaw torque ──
    ax = axes['thrust']
    _clear_ax(ax)
    ax.plot(t, series['thrust_flat'], 'C0-', lw=1.6, label='T (m(g+z̈))')
    hover = series['mass'] * series['g']
    ax.axhline(hover, color='#888', ls=':', lw=0.7)
    if us is not None:
        ax.plot(t, us[:, 2], 'C0--', lw=1.1, label='T plan')
    ax.set_ylabel('Thrust (N)')
    ax.set_title('Thrust & yaw torque')
    ax2 = _ensure_twin(ax)
    if us is not None:
        ax2.plot(t, us[:, 3], 'C3-', lw=1.4, label='τ_yaw plan')
    ax2.set_ylabel('τ_yaw (N·m)', fontsize=8)
    ax2.tick_params(labelsize=7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=6)

    for ax in axes.values():
        ax.set_xlabel('Time (s)')
