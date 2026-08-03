# -*- coding: utf-8 -*-
"""Plot panels for numerical tracking simulation."""

from __future__ import annotations

import numpy as np

from tvc_traj_gui_plots import draw_cost_panel
from tvc_traj_gui_plot_layout import apply_responsive_layout, place_legend
from tvc_traj_gui_3d import apply_equal_aspect_3d, equal_axis_cube_from_points
from controllers.params import CONTROLLER_FLATNESS
from controllers.flatness import FlatnessParams, estimate_flat_state_from_state12

_REF_STYLE = dict(linestyle=':', linewidth=1.2, alpha=0.65)
_SIM_STYLE = dict(linestyle='-', linewidth=1.6, alpha=0.9)
_COLORS = ('tab:blue', 'tab:orange', 'tab:green')
_LABELS3 = ('x', 'y', 'z')
_LABELS3_XI = ('ξ_x', 'ξ_y', 'z')


def _flatness_params_from_plan(plan):
    fp_phy = (plan or {}).get('flatness_physics') if isinstance(plan, dict) else None
    if not isinstance(fp_phy, dict):
        return None
    req = ('mass', 'Ixx', 'Iyy', 'Izz', 'r_thrust_z')
    if not all(k in fp_phy for k in req):
        return None
    return FlatnessParams.from_gui(
        fp_phy['mass'], fp_phy['Ixx'], fp_phy['Iyy'], fp_phy['Izz'], fp_phy['r_thrust_z'],
        g=float(fp_phy.get('g', 9.81)),
    )


def _xi_pos_vel_arrays(x_arr, fp):
    """(pos[N,3], vel[N,3]) = (ξ_x, ξ_y, z) and their rates from a 12-state history."""
    n = x_arr.shape[0]
    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)
    for i in range(n):
        est = estimate_flat_state_from_state12(fp, x_arr[i, :12])
        pos[i] = (est['xi_x'], est['xi_y'], est['z'])
        vel[i] = (est['dxi_x'], est['dxi_y'], est['dz'])
    return pos, vel


def _quat12_to_euler_deg(qx, qy, qz, quat_to_euler_fn=None):
    if quat_to_euler_fn is not None:
        q = np.array([1.0, qx, qy, qz], dtype=float)
        return np.degrees(quat_to_euler_fn(q, format='wxyz'))
    return np.array([2.0 * qx, 2.0 * qy, 2.0 * qz]) * (180.0 / np.pi)


def _numerical_derivative(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    out = np.zeros_like(y)
    if n < 2:
        return out
    for i in range(n):
        if i == 0:
            dt = max(t[1] - t[0], 1e-9)
            out[i] = (y[1] - y[0]) / dt
        elif i == n - 1:
            dt = max(t[-1] - t[-2], 1e-9)
            out[i] = (y[-1] - y[-2]) / dt
        else:
            dt = max(t[i + 1] - t[i - 1], 1e-9)
            out[i] = (y[i + 1] - y[i - 1]) / dt
    return out


def _plan_controls_at_t(plan, t_arr):
    """Interpolate planned [th_p, th_r, T, tau_yaw] along t."""
    if not plan or plan.get('xs') is None:
        return None
    xs = np.asarray(plan['xs'], dtype=float)
    us = plan.get('us')
    time_states = plan.get('time_states')
    if time_states is not None:
        tp = np.asarray(time_states, dtype=float).reshape(-1)
    else:
        dt = plan.get('plot_dt') or plan.get('dt') or 0.05
        tp = np.arange(len(xs), dtype=float) * float(dt)
    if us is None or len(us) == 0:
        return None
    us = np.asarray(us, dtype=float)
    tu = tp[: len(us)]
    out = np.zeros((len(t_arr), min(4, us.shape[1])), dtype=float)
    for j in range(out.shape[1]):
        out[:, j] = np.interp(t_arr, tu, us[:, j])
    if out.shape[1] < 4:
        out = np.hstack([out, np.zeros((len(t_arr), 4 - out.shape[1]))])
    return out


def _align_control_hist(arr, n):
    u = np.asarray(arr, dtype=float) if arr is not None else np.zeros((0, 4))
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if len(u) == 0:
        return np.zeros((n, 4), dtype=float)
    if len(u) >= n:
        return u[:n]
    pad = np.repeat(u[-1:, :], n - len(u), axis=0)
    return np.vstack([u, pad])


def _align_cascade_series(arr, n, result=None):
    """Match cascade setpoint history length to simulation time vector."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if len(a) == n:
        return a
    if len(a) == 0:
        return None
    ratio = None
    if result is not None:
        sim_dt = float(result.get('sim_dt', 0.0) or 0.0)
        control_dt = float(result.get('control_dt', 0.0) or 0.0)
        if sim_dt > 0 and control_dt >= sim_dt:
            ratio = max(1, int(round(control_dt / sim_dt)))
    if ratio and ratio > 1 and len(a) * ratio >= n - ratio:
        expanded = np.repeat(a, ratio, axis=0)
        if len(expanded) >= n:
            return expanded[:n]
    if len(a) < n:
        pad = np.repeat(a[-1:], n - len(a), axis=0)
        return np.vstack([a, pad])
    return a[:n]


def _ensure_twin_axis(ax):
    """Reuse a cached twin y-axis instead of stacking a new one on every redraw.

    ``ax.clear()`` does not remove a previously created ``ax.twinx()`` axes —
    it's a separate Axes sharing the x-axis, so calling ``twinx()`` again on
    every redraw leaves the old one's stale data (and x-limits, from whatever
    duration that earlier run was) still in the shared x-axis group, stretching
    the panel's visible time range past the current data.
    """
    twin = getattr(ax, '_tvc_twin', None)
    if twin is None:
        twin = ax.twinx()
        ax._tvc_twin = twin
    else:
        twin.clear()
    return twin


def _style_axis(ax, title, xlabel='Time (s)', ylabel=''):
    ax.set_title(title, fontsize=9, fontweight='bold', pad=2)
    ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)


_SP_STYLE = dict(linestyle='-.', linewidth=1.5, alpha=0.9)


def _plot_xyz(
    ax, t, ref_arr, sim_arr, ylabel, title, deg=False, cmd_arr=None, canvas_width_px=800,
    result=None, labels=_LABELS3, show_plan=True, show_sim=True, show_cascade=True,
):
    """Plot plan (dotted), cascade setpoint (dash-dot), and simulation (solid)."""
    _style_axis(ax, title, ylabel=ylabel)
    n = len(t)
    if cmd_arr is not None and show_cascade:
        cmd_arr = _align_cascade_series(cmd_arr, n, result)
    else:
        cmd_arr = None
    for i, c in enumerate(_COLORS):
        lbl = labels[i]
        r = ref_arr[:, i]
        s = sim_arr[:, i]
        if deg:
            r = np.degrees(r) if ref_arr.shape[1] == 3 and np.max(np.abs(r)) < 10 else r
            s = np.degrees(s) if sim_arr.shape[1] == 3 and np.max(np.abs(s)) < 10 else s
        if show_plan:
            ax.plot(t, r, color=c, label=f'{lbl} plan', **_REF_STYLE)
        if cmd_arr is not None and cmd_arr.shape[0] == n:
            cmd_i = cmd_arr[:, i]
            if np.any(np.isfinite(cmd_i)):
                if deg and cmd_arr.shape[1] == 3 and np.max(np.abs(cmd_i)) < 10:
                    cmd_i = np.degrees(cmd_i)
                ax.plot(t, cmd_i, color=c, label=f'{lbl} cascade', **_SP_STYLE)
        if show_sim:
            ax.plot(t, s, color=c, label=f'{lbl} sim', **_SIM_STYLE)
    n_leg = len(ax.get_legend_handles_labels()[0])
    place_legend(ax, canvas_width_px, n_items=max(n_leg, 1))


def tracking_state_axes_dict(window):
    """Collect state-tab axes from MainWindow."""
    return {
        'ax_pos': window.ax_trk_pos,
        'ax_att': window.ax_trk_att,
        'ax_vel': window.ax_trk_vel,
        'ax_angvel': window.ax_trk_angvel,
        'ax_acc': window.ax_trk_acc,
        'ax_angacc': window.ax_trk_angacc,
        'ax_gimbal': window.ax_trk_gimbal,
        'ax_thrust': window.ax_trk_thrust,
    }


def tracking_metrics_axes_dict(window):
    return {
        'ax_cost': window.ax_metrics_cost,
        'ax_pos_err': window.ax_metrics_pos_err,
        'ax_vel_err': window.ax_metrics_vel_err,
        'ax_info': window.ax_metrics_info,
    }


def _panel_width_px(axes):
    ax = next((axes[k] for k in axes if axes.get(k) is not None), None)
    if ax is None:
        return 800
    fig = ax.figure
    if fig is not None and fig.canvas is not None:
        return fig.canvas.get_width_height()[0]
    return 800


def draw_tracking_state_panels(
    axes, result, plan=None, quat_to_euler_fn=None,
    show_plan=True, show_sim=True, show_cascade=True,
):
    """Draw 4×2 state/control panels (reference vs simulation)."""
    if result is None or 't' not in result:
        return

    width_px = _panel_width_px(axes)
    t = np.asarray(result['t'], dtype=float)
    x_sim = np.asarray(result['x_sim'], dtype=float)
    x_ref = np.asarray(result['x_ref'], dtype=float)

    euler_ref = np.array([
        _quat12_to_euler_deg(x_ref[i, 6], x_ref[i, 7], x_ref[i, 8], quat_to_euler_fn)
        for i in range(len(t))
    ])
    euler_sim = np.array([
        _quat12_to_euler_deg(x_sim[i, 6], x_sim[i, 7], x_sim[i, 8], quat_to_euler_fn)
        for i in range(len(t))
    ])

    # When ξ was the control point, Position/Velocity/Acceleration show what was
    # actually tracked (ξ_x, ξ_y, z) instead of COM — otherwise the panel can't
    # reveal any difference from COM-mode tracking.
    fp = None
    if result.get('controller_id') == CONTROLLER_FLATNESS:
        fp = _flatness_params_from_plan(plan)
    if fp is not None:
        pos_ref, vel_ref = _xi_pos_vel_arrays(x_ref, fp)
        pos_sim, vel_sim = _xi_pos_vel_arrays(x_sim, fp)
        pos_label, vel_label, acc_label = 'ξ position (m)', 'ξ velocity (m/s)', 'ξ acceleration (m/s²)'
        pos_title, vel_title, acc_title = 'Position (ξ)', 'Velocity (ξ̇)', 'Acceleration (ξ̈)'
        xyz_labels = _LABELS3_XI
    else:
        pos_ref, vel_ref = x_ref[:, 0:3], x_ref[:, 3:6]
        pos_sim, vel_sim = x_sim[:, 0:3], x_sim[:, 3:6]
        pos_label, vel_label, acc_label = 'Position (m)', 'Velocity (m/s)', 'Acceleration (m/s²)'
        pos_title, vel_title, acc_title = 'Position', 'Velocity', 'Acceleration'
        xyz_labels = _LABELS3

    acc_ref = np.column_stack([
        _numerical_derivative(vel_ref[:, i], t) for i in range(3)
    ])
    acc_sim = np.column_stack([
        _numerical_derivative(vel_sim[:, i], t) for i in range(3)
    ])
    angacc_ref = np.column_stack([
        _numerical_derivative(x_ref[:, 9 + i], t) for i in range(3)
    ])
    angacc_sim = np.column_stack([
        _numerical_derivative(x_sim[:, 9 + i], t) for i in range(3)
    ])

    u_sim = np.asarray(result.get('u_hist'), dtype=float)
    if u_sim.ndim == 1:
        u_sim = u_sim.reshape(-1, 1)
    if u_sim.shape[0] != len(t):
        if u_sim.shape[0] > len(t):
            u_sim = u_sim[: len(t)]
        else:
            pad = np.repeat(u_sim[-1:], len(t) - u_sim.shape[0], axis=0)
            u_sim = np.vstack([u_sim, pad])

    u_ref = _plan_controls_at_t(plan, t)
    if u_ref is None:
        u_ref = np.zeros_like(u_sim)

    act_on = bool(result.get('actuator_dynamics_enabled'))

    u_cascade = None
    if result.get('u_cmd_hist') is not None:
        u_cascade = _align_control_hist(result.get('u_cmd_hist'), len(t))

    cascade = result.get('cascade_sp')
    sp_vel = sp_att = sp_rate = None
    if cascade:
        sp_vel = _align_cascade_series(cascade.get('vel'), len(t), result)
        sp_att = _align_cascade_series(cascade.get('att_rad'), len(t), result)
        sp_rate = _align_cascade_series(cascade.get('rate_rad_s'), len(t), result)

    mapping = [
        ('ax_pos', pos_ref, pos_sim, pos_label, pos_title, False, None, xyz_labels),
        ('ax_att', euler_ref, euler_sim, 'Euler (deg)', 'Attitude', False,
         np.degrees(sp_att) if sp_att is not None else None, _LABELS3),
        ('ax_vel', vel_ref, vel_sim, vel_label, vel_title, False, sp_vel, xyz_labels),
        ('ax_angvel', np.degrees(x_ref[:, 9:12]), np.degrees(x_sim[:, 9:12]),
         'Angular vel (°/s)', 'Angular velocity', False,
         np.degrees(sp_rate) if sp_rate is not None else None, _LABELS3),
        ('ax_acc', acc_ref, acc_sim, acc_label, acc_title, False, None, xyz_labels),
        ('ax_angacc', angacc_ref, angacc_sim, 'Angular acc (rad/s²)', 'Angular acceleration', False, None, _LABELS3),
    ]
    for key, ref_arr, sim_arr, ylab, title, deg, cmd_arr, lbls in mapping:
        ax = axes.get(key)
        if ax is None:
            continue
        ax.clear()
        _plot_xyz(
            ax, t, ref_arr, sim_arr, ylab, title, deg=deg, cmd_arr=cmd_arr,
            canvas_width_px=width_px, result=result, labels=lbls,
            show_plan=show_plan, show_sim=show_sim, show_cascade=show_cascade,
        )

    ax_g = axes.get('ax_gimbal')
    if ax_g is not None:
        ax_g.clear()
        _style_axis(ax_g, 'Gimbal pitch / roll', ylabel='Angle (deg)')
        th_p_sim, th_r_sim = np.degrees(u_sim[:, 0]), np.degrees(u_sim[:, 1])
        th_p_plan, th_r_plan = np.degrees(u_ref[:, 0]), np.degrees(u_ref[:, 1])
        sim_tag = 'act' if act_on else 'sim'
        n_items = 0
        if show_plan:
            ax_g.plot(t, th_p_plan, color='tab:blue', label='pitch plan', **_REF_STYLE)
            ax_g.plot(t, th_r_plan, color='tab:orange', label='roll plan', **_REF_STYLE)
            n_items += 2
        if show_sim:
            ax_g.plot(t, th_p_sim, color='tab:blue', label=f'pitch {sim_tag}', **_SIM_STYLE)
            ax_g.plot(t, th_r_sim, color='tab:orange', label=f'roll {sim_tag}', **_SIM_STYLE)
            n_items += 2
        if show_cascade and u_cascade is not None:
            th_p_cascade = np.degrees(u_cascade[:, 0])
            th_r_cascade = np.degrees(u_cascade[:, 1])
            ax_g.plot(t, th_p_cascade, color='tab:blue', label='pitch cascade', **_SP_STYLE)
            ax_g.plot(t, th_r_cascade, color='tab:orange', label='roll cascade', **_SP_STYLE)
            n_items += 2
        place_legend(ax_g, width_px, n_items=max(n_items, 1))

    ax_t = axes.get('ax_thrust')
    if ax_t is not None:
        ax_t.clear()
        _style_axis(ax_t, 'Thrust & yaw torque', ylabel='Thrust (N)')
        ax_t2 = _ensure_twin_axis(ax_t)
        ax_t2.tick_params(axis='y', labelsize=7)
        sim_tag = 'act' if act_on else 'sim'
        thrust_lines = yaw_lines = 0
        if show_plan:
            ax_t.plot(t, u_ref[:, 2], color='tab:blue', label='T plan', **_REF_STYLE)
            ax_t2.plot(t, u_ref[:, 3], color='tab:red', label='τ plan', **_REF_STYLE)
            thrust_lines += 1
            yaw_lines += 1
        if show_sim:
            ax_t.plot(t, u_sim[:, 2], color='tab:blue', label=f'T {sim_tag}', **_SIM_STYLE)
            ax_t2.plot(t, u_sim[:, 3], color='tab:red', label=f'τ {sim_tag}', **_SIM_STYLE)
            thrust_lines += 1
            yaw_lines += 1
        if show_cascade and u_cascade is not None:
            ax_t.plot(t, u_cascade[:, 2], color='tab:blue', label='T cascade', **_SP_STYLE)
            ax_t2.plot(t, u_cascade[:, 3], color='tab:red', label='τ cascade', **_SP_STYLE)
            thrust_lines += 1
            yaw_lines += 1
        ax_t2.set_ylabel('Yaw torque (N·m)', fontsize=8)
        lines1, lab1 = ax_t.get_legend_handles_labels()
        lines2, lab2 = ax_t2.get_legend_handles_labels()
        place_legend(
            ax_t, width_px, n_items=max(thrust_lines + yaw_lines, 1),
            handles=lines1 + lines2, labels=lab1 + lab2,
        )


def draw_tracking_3d_panel(ax, result, plan=None):
    """Dedicated 3D tab: planned path, reference, and simulated trajectory."""
    if ax is None:
        return
    ax.clear()
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title('3D Trajectory (plan / ref / sim)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    width_px = 800
    if ax.figure is not None and ax.figure.canvas is not None:
        width_px = ax.figure.canvas.get_width_height()[0]

    point_sets = []
    if plan and plan.get('xs') is not None:
        xs = np.asarray(plan['xs'], dtype=float)
        point_sets.append(xs[:, 0:3])
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], 'b-', linewidth=1.5, alpha=0.5, label='Planned')

    if result is None or 'x_sim' not in result:
        if point_sets:
            lo, hi = equal_axis_cube_from_points(np.vstack(point_sets))
            apply_equal_aspect_3d(ax, lo, hi)
        if plan and plan.get('xs') is not None:
            place_legend(ax, width_px, n_items=1)
        return

    x_ref = np.asarray(result['x_ref'], dtype=float)
    x_sim = np.asarray(result['x_sim'], dtype=float)
    point_sets.extend([x_ref[:, 0:3], x_sim[:, 0:3]])
    lo, hi = equal_axis_cube_from_points(np.vstack(point_sets))
    apply_equal_aspect_3d(ax, lo, hi)

    ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], 'k--', linewidth=1.2, alpha=0.7, label='Ref (interp)')
    ax.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], 'r-', linewidth=2.0, alpha=0.9, label='Simulation')
    ax.scatter(x_sim[0, 0], x_sim[0, 1], x_sim[0, 2], c='g', s=40, label='Start')
    ax.scatter(x_sim[-1, 0], x_sim[-1, 1], x_sim[-1, 2], c='r', s=40, label='End')
    place_legend(ax, width_px, n_items=5)


def draw_tracking_metrics_panels(axes, result, opt_summary=None, cost_loggers=None):
    """
    Metrics tab: optimization cost, tracking errors, summary text.
    """
    ax_cost = axes.get('ax_cost')
    ax_pe = axes.get('ax_pos_err')
    ax_ve = axes.get('ax_vel_err')
    ax_info = axes.get('ax_info')

    width_px = _panel_width_px(axes)

    if ax_cost is not None:
        ax_cost.clear()
        if cost_loggers:
            draw_cost_panel(ax_cost, cost_loggers)
        else:
            _style_axis(ax_cost, 'Optimization cost', xlabel='Iteration', ylabel='Cost (log)')
            ax_cost.text(
                0.5, 0.5, 'No optimization cost data',
                ha='center', va='center', transform=ax_cost.transAxes, fontsize=10,
            )

    if result is None or 't' not in result:
        for ax, title in ((ax_pe, 'Position error'), (ax_ve, 'Velocity error')):
            if ax is not None:
                ax.clear()
                _style_axis(ax, title, ylabel='Error')
                ax.text(0.5, 0.5, 'Run numerical tracking simulation',
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
        if ax_info is not None:
            ax_info.clear()
            ax_info.axis('off')
            _write_metrics_info(ax_info, result, opt_summary)
        return

    t = np.asarray(result['t'], dtype=float)
    pos_err = np.asarray(result.get('pos_err'), dtype=float)
    if pos_err.size == 0:
        pos_err = np.asarray(result['x_sim'][:, 0:3]) - np.asarray(result['x_ref'][:, 0:3])
    vel_err = np.asarray(result['x_sim'][:, 3:6]) - np.asarray(result['x_ref'][:, 3:6])
    err_norm = np.linalg.norm(pos_err, axis=1)
    vel_norm = np.linalg.norm(vel_err, axis=1)

    if ax_pe is not None:
        ax_pe.clear()
        _style_axis(ax_pe, 'Position tracking error', ylabel='Error (m)')
        ax_pe.plot(t, err_norm, 'k-', linewidth=1.5, label='|Δp|')
        for i, c in enumerate(_COLORS):
            ax_pe.plot(t, pos_err[:, i], color=c, linewidth=1.0, alpha=0.8, label=f'Δ{_LABELS3[i]}')
        place_legend(ax_pe, width_px, n_items=4)

    if ax_ve is not None:
        ax_ve.clear()
        _style_axis(ax_ve, 'Velocity tracking error', ylabel='Error (m/s)')
        ax_ve.plot(t, vel_norm, 'k-', linewidth=1.5, label='|Δv|')
        for i, c in enumerate(_COLORS):
            ax_ve.plot(t, vel_err[:, i], color=c, linewidth=1.0, alpha=0.8, label=f'Δv{_LABELS3[i]}')
        place_legend(ax_ve, width_px, n_items=4)

    if ax_info is not None:
        ax_info.clear()
        ax_info.axis('off')
        _write_metrics_info(ax_info, result, opt_summary)


def _write_metrics_info(ax, result, opt_summary):
    lines = ['── Tracking ──']
    if result:
        lines.extend([
            f"Controller : {result.get('controller_id', '?')}",
            f"Duration   : {float(result['t'][-1] - result['t'][0]):.3f} s",
            f"Samples    : {len(result['t'])}",
            f"Max |Δp|   : {result['max_pos_err_m'] * 1000:.2f} mm",
            f"RMSE |Δp|  : {result['rmse_pos_m'] * 1000:.2f} mm",
        ])
    else:
        lines.append('(no tracking simulation yet)')

    lines.append('')
    lines.append('── Optimization ──')
    if opt_summary:
        lines.extend([
            f"Method     : {opt_summary.get('method', '?')}",
            f"Iterations : {opt_summary.get('total_iters', '?')}",
            f"Total time : {opt_summary.get('total_time_s', opt_summary.get('total_time', '?'))} s",
            f"Path length: {opt_summary.get('path_length_m', 0):.3f} m",
            f"Traj dur.  : {opt_summary.get('trajectory_duration_s', '?')} s",
        ])
        seg_costs = opt_summary.get('segment_final_costs') or []
        if seg_costs:
            lines.append(f"Final cost : {seg_costs[-1]:.4e}")
    else:
        lines.append('(no optimization summary)')

    ax.text(
        0.02, 0.98, '\n'.join(lines),
        ha='left', va='top', fontsize=9.5, family='monospace',
        transform=ax.transAxes,
    )


def tracking_summary_text(result):
    if not result:
        return ''
    if result.get('is_cascade_tune'):
        level = result.get('tune_level', '?')
        sp = result.get('tune_setpoints') or {}
        sp_txt = ', '.join(f'{k}={v:g}' for k, v in sp.items())
        return (
            f"PX4 cascade tune ({level}): duration {result.get('tune_duration_s', 0):.1f} s — "
            f"setpoints: {sp_txt}"
        )
    txt = (
        f"Tracking ({result.get('controller_id', '?')}): "
        f"max |Δp| = {result['max_pos_err_m'] * 1000:.1f} mm, "
        f"RMSE = {result['rmse_pos_m'] * 1000:.1f} mm"
    )
    sim_dur = result.get('sim_total_duration_s')
    if sim_dur is not None:
        txt += f", sim {float(sim_dur):.2f} s"
    r_scale = float(result.get('r_gimbal_scale', 1.0))
    if r_scale > 1.5:
        txt += f" (gimbal R ×{r_scale:.0f} for {result.get('platform_id', 'platform')} mass)"
    return txt
