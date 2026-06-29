# -*- coding: utf-8 -*-
"""
GUI-style trajectory dashboard plots (3x4 GridSpec: 3D + cost, states, controls).
Shared by tvc_traj_opt_gui.py and scripts like tvc_traj_opt_acados.py.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tvc_common import quat_to_euler, SEGMENT_COLORS, segment_boundaries_from_waypoints


# Align with matplotlib non-interactive backends (avoids rcsetup.non_interactive_bk on 3.9+).
_MPL_NONINTERACTIVE_BACKENDS = frozenset(
    {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
)


def _mpl_backend_is_interactive():
    """True if current backend can open a GUI window (not Agg/pdf/svg…)."""
    try:
        bk = str(matplotlib.get_backend()).lower()
    except Exception:
        return False
    if bk in _MPL_NONINTERACTIVE_BACKENDS:
        return False
    if "backend_agg" in bk and "tk" not in bk and "qt" not in bk and "wx" not in bk:
        return False
    return True


def mpl_show_nonblocking_if_interactive(show=True):
    """
    On interactive backends: ``plt.show(block=False)`` plus a short pause so the window
    refreshes before ``savefig``. Returns False on non-interactive backends or if ``show`` is False.
    """
    if not show or not _mpl_backend_is_interactive():
        return False
    plt.show(block=False)
    try:
        plt.pause(0.08)
    except Exception:
        pass
    return True


def mpl_show_if_interactive(show=True):
    """
    Call ``plt.show()`` only on interactive backends.
    ``Agg`` / ``pdf`` backends are non-interactive (no window); skip to avoid warnings.
    """
    if not show:
        return
    if not _mpl_backend_is_interactive():
        return
    plt.show()

# Default limit lines if optimization bounds omit keys (matches typical GUI defaults)
DEFAULT_BOUNDS_DISPLAY = {
    'v_horizontal_max': 1.0,
    'v_vertical_max': 3.0,
    'roll_max': 10.0,
    'pitch_max': 10.0,
    'yaw_max': 180.0,
    'w_max': 2.0,
    'th_p_max': 10.0,
    'th_r_max': 10.0,
    'T_min': 0.0,
    'T_max': 25.0,
    'tau_yaw_max': 1.0,
}


def bounds_display_from_optimization_bounds(bounds):
    """Convert solver ``bounds`` dict to GUI-style display limits (degrees for TVC/Euler)."""
    d = DEFAULT_BOUNDS_DISPLAY.copy()
    if not bounds:
        return d
    if bounds.get('th_p') is not None:
        d['th_p_max'] = float(np.degrees(abs(bounds['th_p'][1])))
    if bounds.get('th_r') is not None:
        d['th_r_max'] = float(np.degrees(abs(bounds['th_r'][1])))
    if bounds.get('T') is not None:
        d['T_min'] = float(bounds['T'][0])
        d['T_max'] = float(bounds['T'][1])
    if bounds.get('tau_yaw') is not None:
        d['tau_yaw_max'] = float(abs(bounds['tau_yaw'][1]))
    if 'state_v_horizontal_max' in bounds:
        d['v_horizontal_max'] = float(bounds['state_v_horizontal_max'])
    if 'state_v_vertical_max' in bounds:
        d['v_vertical_max'] = float(bounds['state_v_vertical_max'])
    if 'state_roll_max' in bounds:
        d['roll_max'] = float(np.degrees(bounds['state_roll_max']))
    if 'state_pitch_max' in bounds:
        d['pitch_max'] = float(np.degrees(bounds['state_pitch_max']))
    if 'state_yaw_max' in bounds:
        d['yaw_max'] = float(np.degrees(bounds['state_yaw_max']))
    if 'state_w_max' in bounds:
        d['w_max'] = float(bounds['state_w_max'])
    return d


def draw_cost_panel(ax_cost, all_loggers, solve_meta=None):
    """Semilogy cost per segment-like histories (IPOPT ``obj`` list, Acados SQP, etc.)."""
    ax_cost.clear()
    colors = ['b', 'r', 'g', 'm', 'c', 'orange', 'purple', 'brown']
    if all_loggers and len(all_loggers) > 0:
        cumulative_iter = 0
        n_nonempty = sum(1 for lg in all_loggers if lg and len(lg.costs) > 0)
        for seg_idx, logger in enumerate(all_loggers):
            if logger and len(logger.costs) > 0:
                color = colors[seg_idx % len(colors)]
                label = (f'NLP cost ({len(logger.costs)} iterates)'
                         if n_nonempty == 1 else f'Segment {seg_idx + 1}')
                seg_iterations = np.arange(len(logger.costs)) + cumulative_iter
                y = np.maximum(np.asarray(logger.costs, dtype=float), 1e-300)
                ax_cost.semilogy(seg_iterations, y,
                                color=color, linewidth=2.5,
                                marker='o', markersize=3, label=label)
                cumulative_iter += len(logger.costs)
    ax_cost.set_xlabel('Iteration', fontsize=10)
    ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
    ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(fontsize=8, loc='best')
    if all_loggers and len(all_loggers) > 0:
        last_logger = all_loggers[-1]
        if last_logger and len(last_logger.costs) > 0:
            final_cost = last_logger.costs[-1]
            ax_cost.text(0.02, 0.98, f'Final Cost: {final_cost:.4e}',
                        transform=ax_cost.transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if solve_meta:
        lines = []
        if solve_meta.get('iter_count') is not None:
            lines.append(f"Iterations: {int(solve_meta['iter_count'])}")
        if solve_meta.get('wall_time_s') is not None:
            wt = float(solve_meta['wall_time_s'])
            lines.append(f"Solver wall: {wt:.4f} s")
        if solve_meta.get('time_per_iter_s') is not None:
            ti = float(solve_meta['time_per_iter_s']) * 1000.0
            lines.append(f"≈ {ti:.2f} ms / iter")
        if solve_meta.get('dt_grid') is not None:
            lines.append(rf"Grid $h=t_f/N$: {float(solve_meta['dt_grid']):.6f} s")
        if lines:
            ax_cost.text(
                0.02, 0.02, '\n'.join(lines),
                transform=ax_cost.transAxes, fontsize=8,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.65),
            )


def draw_trajectory_panels(axes, xs, us, dt, waypoints, bounds_display, quat_to_euler_fn=None,
                           us_actual=None, time_step_override=None,
                           segment_boundaries_override=None, time_states=None):
    """
    Draw 3D path, position, velocity, Euler, angular velocity, TVC, thrust, yaw torque.

    Parameters
    ----------
    axes : dict with keys ax_3d, ax_pos, ax_vel, ax_angvel, ax_euler, ax_pitch, ax_roll,
           ax_thrust, ax_yaw
    us_actual : array-like, optional
        Shape (N+1, 4): actuator **actual** inputs [th_p, th_r, T, tau_yaw] at each state time
        (e.g. nx=16 acados state x[12:16]) when actuator dynamics are used. Plotted as **red dashed**
        curves on a higher z-order than cmd and limit lines.
    time_step_override : float, optional
        If set, use this Δt for the time axis instead of ``dt`` (e.g. Acados min-time total duration / (N-1)).
        Ignored when ``time_states`` is given.
    segment_boundaries_override : list of int, optional
        State indices marking segment ends; overrides ``segment_boundaries_from_waypoints`` when given.
    time_states : array-like, optional
        Physical time [s] at each state, length ``len(xs)``. Use for multi-segment min-time where each
        segment has its own ``T_seg`` and grid spacing (see ``physical_time_grid_per_shooting_segment``).
    """
    if xs is None or len(xs) == 0:
        return
    q2e = quat_to_euler_fn or (lambda q: quat_to_euler(q, format='wxyz'))
    bd = bounds_display or DEFAULT_BOUNDS_DISPLAY

    dt_vis = float(time_step_override) if time_step_override is not None else float(dt)
    if time_states is not None:
        time_states = np.asarray(time_states, dtype=float).reshape(-1)
        if time_states.size != len(xs):
            raise ValueError(
                f"time_states length {time_states.size} != len(xs) {len(xs)}"
            )
    else:
        time_states = np.arange(len(xs), dtype=float) * dt_vis
    nu = len(us)
    ns = len(time_states)
    if nu == ns - 1:
        time_controls = time_states[:nu]
    elif nu == ns:
        time_controls = np.asarray(time_states, dtype=float).copy()
    else:
        t0, t1 = float(time_states[0]), float(time_states[-1])
        time_controls = (
            np.linspace(t0, t1, nu, dtype=float)
            if nu > 1
            else np.array([t0], dtype=float)
        )
    xs_array = np.array(xs)
    us_array = np.array(us)
    positions = xs_array[:, 0:3]
    velocities = xs_array[:, 3:6]
    quaternions = xs_array[:, 6:10]
    angular_velocities = xs_array[:, 10:13]
    th_p = us_array[:, 0]
    th_r = us_array[:, 1]
    T = us_array[:, 2]
    tau_yaw = us_array[:, 3]
    euler_angles = np.array([q2e(q) for q in quaternions])

    ua = None
    if us_actual is not None:
        ua = np.asarray(us_actual, dtype=float)
        if ua.ndim == 2 and ua.shape[1] >= 4:
            pass  # use ua
        else:
            ua = None
    cmd_suffix = ' (cmd)' if ua is not None else ''
    if ua is not None:
        na = ua.shape[0]
        if na == len(time_states):
            time_u_actual = time_states
        elif na == len(time_controls):
            time_u_actual = time_controls
        else:
            time_u_actual = np.arange(na, dtype=float) * dt_vis
    else:
        time_u_actual = None

    ax_3d = axes['ax_3d']
    ax_pos = axes['ax_pos']
    ax_vel = axes['ax_vel']
    ax_angvel = axes['ax_angvel']
    ax_euler = axes['ax_euler']
    ax_pitch = axes['ax_pitch']
    ax_roll = axes['ax_roll']
    ax_thrust = axes['ax_thrust']
    ax_yaw = axes['ax_yaw']

    # 1. 3D
    ax_3d.clear()
    if segment_boundaries_override is not None:
        boundaries = [min(int(b), len(positions) - 1) for b in segment_boundaries_override]
    else:
        boundaries = [
            min(b, len(positions) - 1)
            for b in segment_boundaries_from_waypoints(waypoints or [], dt_vis)
        ]
    if boundaries:
        idx = 0
        for i, end_idx in enumerate(boundaries):
            if idx <= end_idx:
                seg_pos = positions[idx:end_idx + 1]
                c = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
                ax_3d.plot(seg_pos[:, 0], seg_pos[:, 1], seg_pos[:, 2],
                          color=c, linewidth=2.5, label=f'Segment {i + 1}')
                if idx < end_idx:
                    ax_3d.scatter(seg_pos[-1, 0], seg_pos[-1, 1], seg_pos[-1, 2],
                                 color=c, s=60, marker='o', edgecolors='black', linewidths=1, zorder=5)
            idx = end_idx
        if idx < len(positions) - 1:
            seg_pos = positions[idx:]
            c = SEGMENT_COLORS[len(boundaries) % len(SEGMENT_COLORS)]
            ax_3d.plot(seg_pos[:, 0], seg_pos[:, 1], seg_pos[:, 2],
                      color=c, linewidth=2.5, label=f'Segment {len(boundaries) + 1}')
    else:
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                  'b-', linewidth=2, label='Trajectory')
    ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                 color='green', s=100, marker='o', label='Start')
    ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                 color='red', s=100, marker='*', label='End')
    if waypoints is not None and len(waypoints) > 0:
        for i, wp in enumerate(waypoints):
            if len(wp) >= 3:
                if i == 0:
                    continue
                ax_3d.scatter(wp[0], wp[1], wp[2],
                             color='orange', s=50, marker='^',
                             edgecolors='darkorange', linewidths=1.5,
                             label=f'WP {i}', zorder=5, alpha=0.8)
                ax_3d.text(wp[0], wp[1], wp[2], f' {i}',
                          fontsize=9, color='darkorange',
                          fontweight='bold', zorder=6)

    all_x = positions[:, 0].tolist()
    all_y = positions[:, 1].tolist()
    all_z = positions[:, 2].tolist()
    if waypoints is not None and len(waypoints) > 0:
        for wp in waypoints:
            if len(wp) >= 3:
                all_x.append(wp[0])
                all_y.append(wp[1])
                all_z.append(wp[2])
    x_range = max(all_x) - min(all_x) if len(all_x) > 0 else 1.0
    y_range = max(all_y) - min(all_y) if len(all_y) > 0 else 1.0
    z_range = max(all_z) - min(all_z) if len(all_z) > 0 else 1.0
    max_range = max(x_range, y_range, z_range)
    if max_range == 0:
        max_range = 1.0
    x_center = (max(all_x) + min(all_x)) / 2 if len(all_x) > 0 else 0.0
    y_center = (max(all_y) + min(all_y)) / 2 if len(all_y) > 0 else 0.0
    z_center = (max(all_z) + min(all_z)) / 2 if len(all_z) > 0 else 0.0
    half_range = max_range / 2.0
    ax_3d.set_xlim([x_center - half_range, x_center + half_range])
    ax_3d.set_ylim([y_center - half_range, y_center + half_range])
    ax_3d.set_zlim([z_center - half_range, z_center + half_range])
    ax_3d.set_xlabel('X (m)', fontsize=10)
    ax_3d.set_ylabel('Y (m)', fontsize=10)
    ax_3d.set_zlabel('Z (m)', fontsize=10)
    ax_3d.set_title('3D Position Trajectory', fontsize=11, fontweight='bold')
    ax_3d.grid(True, alpha=0.3)

    # 2. Position
    ax_pos.clear()
    ax_pos.plot(time_states, positions[:, 0], 'r-', label='x', linewidth=2)
    ax_pos.plot(time_states, positions[:, 1], 'g-', label='y', linewidth=2)
    ax_pos.plot(time_states, positions[:, 2], 'b-', label='z', linewidth=2)
    if waypoints is not None and len(waypoints) > 0:
        last_wp = waypoints[-1]
        ax_pos.axhline(y=last_wp[0], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
        ax_pos.axhline(y=last_wp[1], color='g', linestyle='--', alpha=0.5, linewidth=1.5)
        ax_pos.axhline(y=last_wp[2], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
    ax_pos.set_xlabel('Time (s)', fontsize=9)
    ax_pos.set_ylabel('Position (m)', fontsize=9)
    ax_pos.set_title('Position', fontsize=10, fontweight='bold')
    ax_pos.legend(fontsize=8, loc='best')
    ax_pos.grid(True, alpha=0.3)

    # 3. Velocity
    v_horizontal_max_val = bd['v_horizontal_max']
    v_vertical_max_val = bd['v_vertical_max']
    ax_vel.clear()
    ax_vel.plot(time_states, velocities[:, 0], 'r-', label='vx', linewidth=2)
    ax_vel.plot(time_states, velocities[:, 1], 'g-', label='vy', linewidth=2)
    ax_vel.plot(time_states, velocities[:, 2], 'b-', label='vz', linewidth=2)
    ax_vel.axhline(y=v_horizontal_max_val, color='purple', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Max V_h ({v_horizontal_max_val:.1f} m/s)')
    ax_vel.axhline(y=-v_horizontal_max_val, color='purple', linestyle='--',
                  linewidth=1.5, alpha=0.7)
    ax_vel.axhline(y=v_vertical_max_val, color='orange', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Max V_z ({v_vertical_max_val:.1f} m/s)')
    ax_vel.axhline(y=-v_vertical_max_val, color='orange', linestyle='--',
                  linewidth=1.5, alpha=0.7)
    ax_vel.set_xlabel('Time (s)', fontsize=9)
    ax_vel.set_ylabel('Velocity (m/s)', fontsize=9)
    ax_vel.set_title('Linear Velocity', fontsize=10, fontweight='bold')
    ax_vel.legend(fontsize=7, loc='best')
    ax_vel.grid(True, alpha=0.3)

    # 4. Euler (layout column before angular velocity)
    roll_max_deg = bd['roll_max']
    pitch_max_deg = bd['pitch_max']
    euler_deg = np.degrees(euler_angles)
    ax_euler.clear()
    ax_euler.plot(time_states, euler_deg[:, 0], 'r-', label='Roll', linewidth=2)
    ax_euler.plot(time_states, euler_deg[:, 1], 'g-', label='Pitch', linewidth=2)
    ax_euler.plot(time_states, euler_deg[:, 2], 'b-', label='Yaw', linewidth=2)
    ax_euler.axhline(y=roll_max_deg, color='r', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Roll Max ({roll_max_deg:.1f}°)')
    ax_euler.axhline(y=-roll_max_deg, color='r', linestyle='--',
                    linewidth=1.5, alpha=0.7)
    ax_euler.axhline(y=pitch_max_deg, color='g', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Pitch Max ({pitch_max_deg:.1f}°)')
    ax_euler.axhline(y=-pitch_max_deg, color='g', linestyle='--',
                    linewidth=1.5, alpha=0.7)
    ax_euler.set_xlabel('Time (s)', fontsize=9)
    ax_euler.set_ylabel('Euler Angles (deg)', fontsize=9)
    ax_euler.set_title('Attitude (Euler)', fontsize=10, fontweight='bold')
    ax_euler.legend(fontsize=7, loc='best')
    ax_euler.grid(True, alpha=0.3)

    # 5. Angular velocity (rad/s in state → plot in deg/s)
    w_max_val = bd['w_max']
    w_max_deg = np.degrees(w_max_val)
    omega_deg_s = np.degrees(angular_velocities)
    ax_angvel.clear()
    ax_angvel.plot(time_states, omega_deg_s[:, 0], 'r-', label='ωx', linewidth=2)
    ax_angvel.plot(time_states, omega_deg_s[:, 1], 'g-', label='ωy', linewidth=2)
    ax_angvel.plot(time_states, omega_deg_s[:, 2], 'b-', label='ωz', linewidth=2)
    ax_angvel.axhline(y=w_max_deg, color='r', linestyle='--',
                     linewidth=1.5, alpha=0.7, label=f'Max (±{w_max_deg:.1f} °/s)')
    ax_angvel.axhline(y=-w_max_deg, color='r', linestyle='--',
                     linewidth=1.5, alpha=0.7)
    ax_angvel.set_xlabel('Time (s)', fontsize=9)
    ax_angvel.set_ylabel('Angular Vel (°/s)', fontsize=9)
    ax_angvel.set_title('Angular Velocity', fontsize=10, fontweight='bold')
    ax_angvel.legend(fontsize=7, loc='best')
    ax_angvel.grid(True, alpha=0.3)

    # 6–9. Controls
    th_p_max_deg = bd['th_p_max']
    th_r_max_deg = bd['th_r_max']
    T_min_val = bd.get('T_min', 0.0)
    T_max_val = bd['T_max']
    tau_yaw_max = bd['tau_yaw_max']

    # Actuator actual overlay: one style (red dashed) on top of cmd / limit lines
    _z_cmd, _z_lim, _z_act = 2, 1, 8
    _c_act = '#c62828'  # red, distinct from cmd channel colors

    ax_pitch.clear()
    th_p_deg = np.degrees(th_p)
    ax_pitch.plot(time_controls, th_p_deg, 'b-', linewidth=2,
                 label=f'θ_pitch{cmd_suffix}', marker='o', markersize=2, zorder=_z_cmd)
    if ua is not None:
        ax_pitch.plot(
            time_u_actual, np.degrees(ua[:, 0]), color=_c_act, linestyle='--',
            linewidth=2, label='θ_pitch (actual)', zorder=_z_act,
        )
    ax_pitch.axhline(y=th_p_max_deg, color='r', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Max ({th_p_max_deg:.1f}°)', zorder=_z_lim)
    ax_pitch.axhline(y=-th_p_max_deg, color='r', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Min (-{th_p_max_deg:.1f}°)', zorder=_z_lim)
    ax_pitch.set_xlabel('Time (s)', fontsize=9)
    ax_pitch.set_ylabel('Angle (deg)', fontsize=9)
    ax_pitch.set_title('TVC Pitch Angle', fontsize=10, fontweight='bold')
    ax_pitch.legend(fontsize=7, loc='best')
    ax_pitch.grid(True, alpha=0.3)

    ax_roll.clear()
    th_r_deg = np.degrees(th_r)
    ax_roll.plot(time_controls, th_r_deg, 'r-', linewidth=2,
                label=f'θ_roll{cmd_suffix}', marker='o', markersize=2, zorder=_z_cmd)
    if ua is not None:
        ax_roll.plot(
            time_u_actual, np.degrees(ua[:, 1]), color=_c_act, linestyle='--',
            linewidth=2, label='θ_roll (actual)', zorder=_z_act,
        )
    ax_roll.axhline(y=th_r_max_deg, color='b', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Max ({th_r_max_deg:.1f}°)', zorder=_z_lim)
    ax_roll.axhline(y=-th_r_max_deg, color='b', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Min (-{th_r_max_deg:.1f}°)', zorder=_z_lim)
    ax_roll.set_xlabel('Time (s)', fontsize=9)
    ax_roll.set_ylabel('Angle (deg)', fontsize=9)
    ax_roll.set_title('TVC Roll Angle', fontsize=10, fontweight='bold')
    ax_roll.legend(fontsize=7, loc='best')
    ax_roll.grid(True, alpha=0.3)

    ax_thrust.clear()
    ax_thrust.plot(time_controls, T, 'g-', linewidth=2,
                  label=f'Thrust{cmd_suffix}', marker='o', markersize=2, zorder=_z_cmd)
    if ua is not None:
        ax_thrust.plot(
            time_u_actual, ua[:, 2], color=_c_act, linestyle='--',
            linewidth=2, label='Thrust (actual)', zorder=_z_act,
        )
    ax_thrust.axhline(y=T_max_val, color='r', linestyle='--',
                     linewidth=1.5, alpha=0.7, label=f'Max ({T_max_val:.1f} N)', zorder=_z_lim)
    ax_thrust.axhline(y=T_min_val, color='r', linestyle='--',
                     linewidth=1.5, alpha=0.7, label=f'Min ({T_min_val:.1f} N)', zorder=_z_lim)
    ax_thrust.set_xlabel('Time (s)', fontsize=9)
    ax_thrust.set_ylabel('Thrust (N)', fontsize=9)
    ax_thrust.set_title('Thrust', fontsize=10, fontweight='bold')
    ax_thrust.legend(fontsize=7, loc='best')
    ax_thrust.grid(True, alpha=0.3)

    ax_yaw.clear()
    ax_yaw.plot(time_controls, tau_yaw, 'm-', linewidth=2,
               label=f'τ_yaw{cmd_suffix}', marker='o', markersize=2, zorder=_z_cmd)
    if ua is not None:
        ax_yaw.plot(
            time_u_actual, ua[:, 3], color=_c_act, linestyle='--',
            linewidth=2, label='τ_yaw (actual)', zorder=_z_act,
        )
    ax_yaw.axhline(y=tau_yaw_max, color='r', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Max ({tau_yaw_max:.2f} N·m)', zorder=_z_lim)
    ax_yaw.axhline(y=-tau_yaw_max, color='r', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Min (-{tau_yaw_max:.2f} N·m)', zorder=_z_lim)
    ax_yaw.set_xlabel('Time (s)', fontsize=9)
    ax_yaw.set_ylabel('Torque (N·m)', fontsize=9)
    ax_yaw.set_title('Yaw Torque', fontsize=10, fontweight='bold')
    ax_yaw.legend(fontsize=7, loc='best')
    ax_yaw.grid(True, alpha=0.3)


def build_gui_style_figure(suptitle='TVC Rocket Trajectory Optimization'):
    """Create Figure + axes dict matching tvc_traj_opt_gui layout."""
    fig = plt.figure(figsize=(20, 10.5))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)
    ax_3d = fig.add_subplot(gs[0, 0:2], projection='3d')
    ax_3d.set_xlabel('X (m)', fontsize=10)
    ax_3d.set_ylabel('Y (m)', fontsize=10)
    ax_3d.set_zlabel('Z (m)', fontsize=10)
    ax_3d.set_title('3D Position Trajectory', fontsize=11, fontweight='bold')
    ax_3d.grid(True, alpha=0.3)
    ax_cost = fig.add_subplot(gs[0, 2:4])
    ax_cost.set_xlabel('Iteration', fontsize=10)
    ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
    ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
    ax_cost.grid(True, alpha=0.3)
    ax_pos = fig.add_subplot(gs[1, 0])
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_euler = fig.add_subplot(gs[1, 2])
    ax_angvel = fig.add_subplot(gs[1, 3])
    for ax, label, title in [
        (ax_pos, 'Time (s)', 'Position'), (ax_vel, 'Time (s)', 'Linear Velocity'),
        (ax_euler, 'Time (s)', 'Attitude (Euler)'), (ax_angvel, 'Time (s)', 'Angular Velocity'),
    ]:
        ax.set_xlabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax_pos.set_ylabel('Position (m)', fontsize=9)
    ax_vel.set_ylabel('Velocity (m/s)', fontsize=9)
    ax_euler.set_ylabel('Euler Angles (deg)', fontsize=9)
    ax_angvel.set_ylabel('Angular Vel (°/s)', fontsize=9)
    ax_pitch = fig.add_subplot(gs[2, 0])
    ax_roll = fig.add_subplot(gs[2, 1])
    ax_thrust = fig.add_subplot(gs[2, 2])
    ax_yaw = fig.add_subplot(gs[2, 3])
    for ax, ylab, ttl in [
        (ax_pitch, 'Angle (deg)', 'TVC Pitch Angle'), (ax_roll, 'Angle (deg)', 'TVC Roll Angle'),
        (ax_thrust, 'Thrust (N)', 'Thrust'), (ax_yaw, 'Torque (N·m)', 'Yaw Torque'),
    ]:
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(ttl, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    axes = {
        'ax_3d': ax_3d, 'ax_cost': ax_cost, 'ax_pos': ax_pos, 'ax_vel': ax_vel,
        'ax_angvel': ax_angvel, 'ax_euler': ax_euler, 'ax_pitch': ax_pitch,
        'ax_roll': ax_roll, 'ax_thrust': ax_thrust, 'ax_yaw': ax_yaw,
    }
    return fig, axes


def plot_gui_style_results(xs, us, dt, waypoints=None, optimization_bounds=None,
                          all_loggers=None, suptitle='TVC Rocket Trajectory Optimization',
                          show=True, us_actual=None, solve_meta=None,
                          segment_boundaries_override=None, time_states=None):
    """
    Full GUI-style dashboard (cost + trajectory panels) for standalone scripts.
    Returns matplotlib figure.

    solve_meta : optional dict with iter_count, wall_time_s, time_per_iter_s, dt_grid for cost panel.
    segment_boundaries_override / time_states : same as ``draw_trajectory_panels`` (multi-segment min-time).
    """
    bounds_disp = bounds_display_from_optimization_bounds(optimization_bounds)
    fig, axes = build_gui_style_figure(suptitle=suptitle)
    draw_cost_panel(axes['ax_cost'], all_loggers, solve_meta=solve_meta)
    traj_axes = {k: v for k, v in axes.items() if k != 'ax_cost'}
    draw_trajectory_panels(
        traj_axes,
        xs,
        us,
        dt,
        waypoints,
        bounds_disp,
        us_actual=us_actual,
        time_step_override=solve_meta.get("plot_dt") if solve_meta and time_states is None else None,
        segment_boundaries_override=segment_boundaries_override,
        time_states=time_states,
    )
    mpl_show_if_interactive(show)
    return fig
