# -*- coding: utf-8 -*-
"""Generate 3D tracking animation GIF with attitude-aware rocket arrow and thrust vector."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D

from controllers.flatness import FlatnessParams, estimate_flat_state_from_state12
from tvc_common import R_from_quat, quat_norm
from tvc_rocket_platforms import normalize_platform_id, rocket_visual_geometry
from tvc_traj_gui_3d import (
    apply_equal_aspect_3d,
    equal_axis_cube_from_points,
    thrust_application_point_world,
    thrust_direction_world,
)

# Target display rate when encoding GIFs; long sims may drop fps to keep duration.
_GIF_TARGET_FPS = 12.0
_GIF_MAX_FRAMES = 480


def gif_frame_schedule(
    t,
    playback_speed: float = 1.0,
    target_fps: float = _GIF_TARGET_FPS,
    max_frames: int = _GIF_MAX_FRAMES,
):
    """
    Choose frame indices and encode fps so playback lasts ``sim_duration / speed``.

    ``playback_speed=1`` → real-time; ``>1`` → accelerated (shorter GIF).
    """
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n < 2:
        return np.array([0], dtype=int), float(target_fps)
    duration = max(float(t[-1] - t[0]), 1e-9)
    speed = max(float(playback_speed), 1e-6)
    play_dur = duration / speed
    n_ideal = play_dur * float(target_fps)
    n_frames = int(round(n_ideal))
    n_frames = max(2, min(n_frames, int(max_frames), n))
    fps = max(n_frames / play_dur, 0.5)
    frame_idx = np.linspace(0, n - 1, n_frames, dtype=int)
    return frame_idx, float(fps)


def _state12_to_quat_wxyz(x12):
    """Rebuild unit quaternion [w,x,y,z] from linear-model state qx,qy,qz."""
    qx, qy, qz = float(x12[6]), float(x12[7]), float(x12[8])
    s = max(1.0 - qx * qx - qy * qy - qz * qz, 0.0)
    qw = float(np.sqrt(s))
    return quat_norm(np.array([qw, qx, qy, qz], dtype=float))


def _r_thrust_body_from_result(result):
    if result is None:
        return np.array([0.0, 0.0, -0.2], dtype=float)
    r = result.get('r_thrust_body')
    if r is not None:
        return np.asarray(r, dtype=float).reshape(3)
    return np.array([0.0, 0.0, -0.2], dtype=float)


def _platform_id_from_result(result):
    if result is None:
        return 'proxy'
    return normalize_platform_id(result.get('platform_id'))


def _rocket_geometry_from_result(result):
    """Custom Model-tab geometry from ``result``, else the platform default."""
    geo = result.get('rocket_geometry') if isinstance(result, dict) else None
    if isinstance(geo, dict) and all(k in geo for k in ('body_bottom_z', 'nose_tip_z', 'fin_z')):
        out = dict(geo)
        out.setdefault('shaft_lw', 4.5)
        out.setdefault('fin_lw', 2.6)
        return out
    return rocket_visual_geometry(_platform_id_from_result(result))


def _flatness_params_from_plan(plan):
    if not plan:
        return None
    fp = plan.get('flatness_physics') if isinstance(plan, dict) else None
    if not isinstance(fp, dict):
        return None
    req = ('mass', 'Ixx', 'Iyy', 'Izz', 'r_thrust_z')
    if not all(k in fp for k in req):
        return None
    return FlatnessParams.from_gui(
        fp['mass'], fp['Ixx'], fp['Iyy'], fp['Izz'], fp['r_thrust_z'],
        g=float(fp.get('g', 9.81)),
    )


def _xi_positions_from_state_history(x_hist, fp):
    """Reconstruct the ξ (center-of-oscillation) point in world frame from state history.

    ξ_x/ξ_y use the small-angle flat-output formula (matches the flatness
    controller). The height can't just reuse COM z: ξ is a body-fixed point
    offset by ``d`` along +body Z from the COM (the classic pendulum
    center-of-oscillation distance I_cm/(m*l)), so even at zero tilt it sits
    ``d`` meters above the COM, and its height changes with attitude like any
    other rigid body-fixed point (same convention as the nose/base markers).
    """
    if fp is None:
        return None
    x_hist = np.asarray(x_hist, dtype=float)
    if x_hist.ndim != 2 or x_hist.shape[1] < 12:
        return None
    d = 0.5 * (fp.flat_offset_pitch + fp.flat_offset_roll)
    out = np.zeros((x_hist.shape[0], 3), dtype=float)
    for i in range(x_hist.shape[0]):
        est = estimate_flat_state_from_state12(fp, x_hist[i, :12])
        q = _state12_to_quat_wxyz(x_hist[i])
        R = R_from_quat(q)
        out[i, 0] = est['xi_x']
        out[i, 1] = est['xi_y']
        out[i, 2] = float(x_hist[i, 2]) + d * R[2, 2]
    return out


def _draw_static_paths(ax, x_ref, x_sim, plan_xs=None):
    if plan_xs is not None and len(plan_xs) > 1:
        ax.plot(
            plan_xs[:, 0], plan_xs[:, 1], plan_xs[:, 2],
            color='tab:blue', linewidth=1.0, alpha=0.35, label='Planned',
        )
    ax.plot(
        x_ref[:, 0], x_ref[:, 1], x_ref[:, 2],
        color='0.35', linewidth=1.0, linestyle='--', alpha=0.7, label='Ref',
    )
    ax.plot(
        x_sim[:, 0], x_sim[:, 1], x_sim[:, 2],
        color='0.75', linewidth=1.0, alpha=0.6, label='Sim trail',
    )


def _align_controls(u_hist, n):
    u = np.asarray(u_hist, dtype=float) if u_hist is not None else np.zeros((0, 4))
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if len(u) == 0:
        return np.zeros((n, 4), dtype=float)
    if len(u) >= n:
        return u[:n]
    pad = np.repeat(u[-1:, :], n - len(u), axis=0)
    return np.vstack([u, pad])


def _draw_arrow3d(ax, tail, head, color='darkorange', lw=2.5, head_frac=0.22, head_width_frac=0.10):
    """Draw a 3D arrow with the tip at *head*."""
    tail = np.asarray(tail, dtype=float).reshape(3)
    head = np.asarray(head, dtype=float).reshape(3)
    vec = head - tail
    length = float(np.linalg.norm(vec))
    artists = []
    if length < 1e-9:
        return artists
    u = vec / length
    shaft, = ax.plot(
        [tail[0], head[0]], [tail[1], head[1]], [tail[2], head[2]],
        color=color, linewidth=lw, solid_capstyle='round', zorder=7,
    )
    artists.append(shaft)
    head_len = head_frac * length
    head_w = head_width_frac * length
    ref = np.array([0.0, 0.0, 1.0]) if abs(float(u[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(u, ref)
    v /= max(float(np.linalg.norm(v)), 1e-9)
    base = head - u * head_len
    for sign in (-1.0, 1.0):
        wing = base + v * sign * head_w
        wing_line, = ax.plot(
            [head[0], wing[0]], [head[1], wing[1]], [head[2], wing[2]],
            color=color, linewidth=lw, zorder=7,
        )
        artists.append(wing_line)
    return artists


def _thrust_label_offset(thrust_dir, span):
    """Small offset beside the thrust arrow so the label does not cover the shaft."""
    u = np.asarray(thrust_dir, dtype=float).reshape(3)
    u /= max(float(np.linalg.norm(u)), 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(float(u[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp = np.cross(u, ref)
    perp /= max(float(np.linalg.norm(perp)), 1e-9)
    return perp * max(0.04 * span, 0.02)


def _body_point_world(com_pos, q_wxyz, z_body):
    R = R_from_quat(np.asarray(q_wxyz, dtype=float))
    return np.asarray(com_pos, dtype=float).reshape(3) + R @ np.array([0.0, 0.0, float(z_body)])


def _rocket_body_endpoints(com_pos, q_wxyz, geom):
    """
    Rocket stick model in world frame using platform geometry (meters, COM at origin).

    Returns (base, tip, mid, com).
    """
    base = _body_point_world(com_pos, q_wxyz, geom['body_bottom_z'])
    tip = _body_point_world(com_pos, q_wxyz, geom['nose_tip_z'])
    mid = _body_point_world(com_pos, q_wxyz, geom['fin_z'])
    com = np.asarray(com_pos, dtype=float).reshape(3)
    return base, tip, mid, com


def generate_tracking_gif(
    result,
    plan=None,
    output_path=None,
    fps=None,
    max_frames=None,
    figsize=(8.0, 8.0),
    dpi=120,
    view_mode: str = '3d',
    playback_speed: float = 1.0,
) -> str:
    """
    Render tracking simulation to an animated GIF.

    ``view_mode`` is ``'3d'`` (default) or ``'2d'`` (side view, XZ).
    ``playback_speed`` is 1.0 for real-time, or >1 to accelerate.

    Returns absolute path to the written GIF file.
    """
    mode = str(view_mode or '3d').strip().lower()
    kwargs = dict(
        result=result,
        plan=plan,
        output_path=output_path,
        fps=fps,
        max_frames=max_frames,
        figsize=figsize,
        dpi=dpi,
        playback_speed=playback_speed,
    )
    if mode == '2d':
        return _generate_tracking_gif_2d(**kwargs)
    return _generate_tracking_gif_3d(**kwargs)


def _generate_tracking_gif_3d(
    result,
    plan=None,
    output_path=None,
    fps=None,
    max_frames=None,
    figsize=(8.0, 8.0),
    dpi=120,
    playback_speed: float = 1.0,
) -> str:
    if result is None or 'x_sim' not in result:
        raise ValueError('No tracking simulation result to animate.')

    t = np.asarray(result['t'], dtype=float)
    x_sim = np.asarray(result['x_sim'], dtype=float)
    x_ref = np.asarray(result['x_ref'], dtype=float)
    n = len(t)
    if n < 2:
        raise ValueError('Tracking trajectory too short for animation.')

    plan_xs = None
    if plan and plan.get('xs') is not None:
        plan_xs = np.asarray(plan['xs'], dtype=float)[:, 0:3]
    fp = _flatness_params_from_plan(plan)
    xi_sim = _xi_positions_from_state_history(x_sim, fp)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'trajs', 'tracking_anim.gif',
        )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target_fps = float(fps) if fps is not None else _GIF_TARGET_FPS
    frame_cap = int(max_frames) if max_frames is not None else _GIF_MAX_FRAMES
    frame_idx, fps = gif_frame_schedule(
        t,
        playback_speed=playback_speed,
        target_fps=target_fps,
        max_frames=frame_cap,
    )
    quats = np.array([_state12_to_quat_wxyz(x_sim[i]) for i in range(n)])
    u_hist = _align_controls(result.get('u_hist'), n)
    r_thrust_body = _r_thrust_body_from_result(result)
    rocket_geom = _rocket_geometry_from_result(result)

    all_pts = [x_sim[:, 0:3], x_ref[:, 0:3]]
    if xi_sim is not None:
        all_pts.append(xi_sim)
    if plan_xs is not None:
        all_pts.append(plan_xs)

    hover_T = float(np.median(u_hist[:, 2])) if len(u_hist) else 1.0
    thrust_scale = max(hover_T, 1e-3)
    for i in range(n):
        pos = x_sim[i, 0:3]
        base_pt, tip_pt, _, _ = _rocket_body_endpoints(pos, quats[i], rocket_geom)
        all_pts.append(np.vstack([base_pt, tip_pt]))
        th_p, th_r, T = u_hist[i, 0], u_hist[i, 1], u_hist[i, 2]
        if T > 1e-6:
            app_pt = thrust_application_point_world(pos, quats[i], r_thrust_body)
            thrust_dir = thrust_direction_world(quats[i], th_p, th_r)
            thrust_tip = app_pt + thrust_dir * 0.35 * (T / thrust_scale)
            all_pts.append(np.vstack([app_pt, thrust_tip]))

    lo, hi = equal_axis_cube_from_points(np.vstack(all_pts))
    span = float(hi[0] - lo[0])
    thrust_len_base = max(0.18, 0.22 * span)
    shaft_lw = float(rocket_geom.get('shaft_lw', 4.5))
    fin_lw = float(rocket_geom.get('fin_lw', 2.6))
    fin_len = 0.16

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.06)

    trail_line, = ax.plot([], [], [], color='tab:red', linewidth=1.6, alpha=0.85)
    xi_trail_line, = ax.plot([], [], [], color='tab:cyan', linewidth=1.4, alpha=0.80)
    rocket_parts = {
        'shaft': None, 'nose': None, 'fins': [], 'thrust_lines': [],
        'thrust_label': None, 'com': None, 'xi': None,
    }

    def _com_legend_handle():
        return Line2D(
            [0], [0], linestyle='None', marker='o', markersize=7,
            markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.6,
            label='COM',
        )

    def _xi_legend_handle():
        return Line2D(
            [0], [0], linestyle='None', marker='o', markersize=6,
            markerfacecolor='cyan', markeredgecolor='black', markeredgewidth=0.6,
            label='Xi',
        )

    def _clear_rocket_artists():
        for key in ('shaft', 'nose', 'thrust_label', 'com', 'xi'):
            artist = rocket_parts.get(key)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                rocket_parts[key] = None
        for fin in rocket_parts.get('fins') or []:
            try:
                fin.remove()
            except Exception:
                pass
        rocket_parts['fins'] = []
        for line in rocket_parts.get('thrust_lines') or []:
            try:
                line.remove()
            except Exception:
                pass
        rocket_parts['thrust_lines'] = []

    def _setup_axes(title):
        apply_equal_aspect_3d(ax, lo, hi)
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.view_init(elev=24, azim=-58)
        ax.grid(True, alpha=0.25)

    def _init():
        ax.cla()
        _setup_axes('Tracking animation')
        _draw_static_paths(ax, x_ref, x_sim, plan_xs)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(_com_legend_handle())
        labels.append('COM')
        if xi_sim is not None:
            handles.append(_xi_legend_handle())
            labels.append('Xi')
        ax.legend(handles, labels, loc='upper left', fontsize=7, framealpha=0.88)
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        xi_trail_line.set_data([], [])
        xi_trail_line.set_3d_properties([])
        return (trail_line, xi_trail_line)

    def _update(k):
        i = int(frame_idx[k])
        pos = x_sim[i, 0:3]
        trail_line.set_data(x_sim[: i + 1, 0], x_sim[: i + 1, 1])
        trail_line.set_3d_properties(x_sim[: i + 1, 2])
        if xi_sim is not None:
            xi_trail_line.set_data(xi_sim[: i + 1, 0], xi_sim[: i + 1, 1])
            xi_trail_line.set_3d_properties(xi_sim[: i + 1, 2])
        else:
            xi_trail_line.set_data([], [])
            xi_trail_line.set_3d_properties([])

        _clear_rocket_artists()
        q = quats[i]
        R = R_from_quat(q)
        wing = R[:, 0]
        base, tip, mid, com = _rocket_body_endpoints(pos, q, rocket_geom)
        rocket_parts['shaft'], = ax.plot(
            [base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
            color='crimson', linewidth=shaft_lw, solid_capstyle='round', zorder=5,
        )
        rocket_parts['nose'] = ax.scatter(
            [tip[0]], [tip[1]], [tip[2]], c='crimson', s=72, zorder=6,
        )
        for sign in (-1.0, 1.0):
            fin_tip = mid + wing * sign * fin_len
            fin_line, = ax.plot(
                [mid[0], fin_tip[0]], [mid[1], fin_tip[1]], [mid[2], fin_tip[2]],
                color='darkred', linewidth=fin_lw, zorder=4,
            )
            rocket_parts['fins'].append(fin_line)

        rocket_parts['com'] = ax.scatter(
            [com[0]], [com[1]], [com[2]],
            c='gold', s=42, marker='o', edgecolors='black', linewidths=0.6, zorder=10,
        )
        if xi_sim is not None:
            xi = xi_sim[i]
            rocket_parts['xi'] = ax.scatter(
                [xi[0]], [xi[1]], [xi[2]],
                c='cyan', s=36, marker='o', edgecolors='black', linewidths=0.6, zorder=10,
            )

        th_p, th_r, T = u_hist[i, 0], u_hist[i, 1], u_hist[i, 2]
        T_disp = float(T) if np.isfinite(T) else 0.0
        if T_disp > 1e-6:
            app_pt = thrust_application_point_world(pos, q, r_thrust_body)
            thrust_dir = thrust_direction_world(q, th_p, th_r)
            thrust_len = thrust_len_base * (T_disp / thrust_scale)
            tail = app_pt - thrust_dir * thrust_len
            rocket_parts['thrust_lines'].extend(
                _draw_arrow3d(ax, tail, app_pt, color='darkorange', lw=3.0),
            )
            tip_mark = ax.scatter(
                [app_pt[0]], [app_pt[1]], [app_pt[2]],
                c='darkorange', s=40, edgecolors='saddlebrown', linewidths=0.6, zorder=8,
            )
            rocket_parts['thrust_lines'].append(tip_mark)
            label_pos = 0.5 * (tail + app_pt) + _thrust_label_offset(thrust_dir, span)
            rocket_parts['thrust_label'] = ax.text(
                label_pos[0], label_pos[1], label_pos[2],
                f'{T_disp:.1f} N',
                fontsize=8, color='saddlebrown', fontweight='bold',
                ha='center', va='center', zorder=9,
                bbox=dict(
                    boxstyle='round,pad=0.25', facecolor='white',
                    edgecolor='darkorange', alpha=0.88,
                ),
            )
        else:
            rocket_parts['thrust_label'] = ax.text(
                pos[0], pos[1], pos[2] + 0.06 * span,
                'T ≈ 0 N',
                fontsize=8, color='0.45', ha='center', va='center', zorder=9,
            )

        apply_equal_aspect_3d(ax, lo, hi)
        ax.set_title(
            f'Tracking  t = {t[i]:.2f} s',
            fontsize=10, fontweight='bold',
        )
        artists = [trail_line, xi_trail_line, rocket_parts['shaft'], rocket_parts['nose'], rocket_parts['com']]
        if rocket_parts['xi'] is not None:
            artists.append(rocket_parts['xi'])
        artists.extend(rocket_parts['fins'])
        artists.extend(rocket_parts['thrust_lines'])
        if rocket_parts['thrust_label'] is not None:
            artists.append(rocket_parts['thrust_label'])
        return tuple(artists)

    _init()
    anim = animation.FuncAnimation(
        fig, _update, frames=len(frame_idx), init_func=_init, blit=False,
        interval=1000.0 / fps,
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def _draw_arrow2d(ax, tail, head, color='darkorange', lw=2.5, head_frac=0.22, head_width_frac=0.12):
    """Draw a 2D arrow (V head) with the tip at *head*, mirroring ``_draw_arrow3d``."""
    tail = np.asarray(tail, dtype=float).reshape(2)
    head = np.asarray(head, dtype=float).reshape(2)
    vec = head - tail
    length = float(np.linalg.norm(vec))
    artists = []
    if length < 1e-9:
        return artists
    u = vec / length
    shaft, = ax.plot(
        [tail[0], head[0]], [tail[1], head[1]],
        color=color, linewidth=lw, solid_capstyle='round', zorder=7,
    )
    artists.append(shaft)
    head_len = head_frac * length
    head_w = head_width_frac * length
    perp = np.array([-u[1], u[0]])
    base = head - u * head_len
    for sign in (-1.0, 1.0):
        wing = base + perp * sign * head_w
        wing_line, = ax.plot(
            [head[0], wing[0]], [head[1], wing[1]], color=color, linewidth=lw, zorder=7,
        )
        artists.append(wing_line)
    return artists


def _generate_tracking_gif_2d(
    result,
    plan=None,
    output_path=None,
    fps=None,
    max_frames=None,
    figsize=(8.0, 8.0),
    dpi=120,
    playback_speed: float = 1.0,
) -> str:
    """Side-view XZ tracking animation: same rocket stick model + thrust arrow as 3D."""
    if result is None or 'x_sim' not in result:
        raise ValueError('No tracking simulation result to animate.')

    XZ = [0, 2]

    t = np.asarray(result['t'], dtype=float)
    x_sim = np.asarray(result['x_sim'], dtype=float)
    x_ref = np.asarray(result['x_ref'], dtype=float)
    n = len(t)
    if n < 2:
        raise ValueError('Tracking trajectory too short for animation.')

    plan_xs = None
    if plan and plan.get('xs') is not None:
        plan_xs = np.asarray(plan['xs'], dtype=float)[:, 0:3]
    fp = _flatness_params_from_plan(plan)
    xi_sim = _xi_positions_from_state_history(x_sim, fp)
    # Nominal flat-output position at COM height (pre-correction ξ), for comparison.
    xi_proj = None
    if xi_sim is not None:
        xi_proj = np.column_stack([xi_sim[:, 0], xi_sim[:, 1], x_sim[:, 2]])

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'trajs', 'tracking_anim_2d.gif',
        )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target_fps = float(fps) if fps is not None else _GIF_TARGET_FPS
    frame_cap = int(max_frames) if max_frames is not None else _GIF_MAX_FRAMES
    frame_idx, fps = gif_frame_schedule(
        t,
        playback_speed=playback_speed,
        target_fps=target_fps,
        max_frames=frame_cap,
    )
    quats = np.array([_state12_to_quat_wxyz(x_sim[i]) for i in range(n)])
    u_hist = _align_controls(result.get('u_hist'), n)
    r_thrust_body = _r_thrust_body_from_result(result)
    rocket_geom = _rocket_geometry_from_result(result)

    all_pts = [x_sim[:, XZ], x_ref[:, XZ]]
    if xi_sim is not None:
        all_pts.append(xi_sim[:, XZ])
        all_pts.append(xi_proj[:, XZ])
    if plan_xs is not None:
        all_pts.append(plan_xs[:, XZ])

    hover_T = float(np.median(u_hist[:, 2])) if len(u_hist) else 1.0
    thrust_scale = max(hover_T, 1e-3)
    for i in range(n):
        pos = x_sim[i, 0:3]
        base_pt, tip_pt, _, _ = _rocket_body_endpoints(pos, quats[i], rocket_geom)
        all_pts.append(np.vstack([base_pt[XZ], tip_pt[XZ]]))
        th_p, th_r, T = u_hist[i, 0], u_hist[i, 1], u_hist[i, 2]
        if T > 1e-6:
            app_pt = thrust_application_point_world(pos, quats[i], r_thrust_body)
            thrust_dir = thrust_direction_world(quats[i], th_p, th_r)
            thrust_tip = app_pt + thrust_dir * 0.35 * (T / thrust_scale)
            all_pts.append(np.vstack([app_pt[XZ], thrust_tip[XZ]]))

    pts = np.vstack(all_pts)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 0.5))
    pad = max(0.15 * span, 0.25)
    xlo, xhi = lo[0] - pad, hi[0] + pad
    zlo, zhi = lo[1] - pad, hi[1] + pad
    thrust_len_base = max(0.18, 0.22 * span)
    shaft_lw = float(rocket_geom.get('shaft_lw', 4.5))
    fin_lw = float(rocket_geom.get('fin_lw', 2.6))
    fin_len = 0.16

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.08)

    ax.plot(x_ref[:, 0], x_ref[:, 2], 'b--', lw=1.2, alpha=0.65, label='Ref COM')
    ax.plot(x_sim[:, 0], x_sim[:, 2], color='0.82', lw=0.9, alpha=0.6, label='Sim COM path')
    if xi_sim is not None:
        ax.plot(xi_sim[:, 0], xi_sim[:, 2], color='0.75', lw=0.9, alpha=0.6, label='Sim Xi path')
    if plan_xs is not None:
        ax.plot(plan_xs[:, 0], plan_xs[:, 2], 'g:', lw=1.0, alpha=0.55, label='Plan')
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(zlo, zhi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Z (m)', fontsize=9)
    ax.grid(True, alpha=0.25)

    trail_line, = ax.plot([], [], color='tab:red', linewidth=1.6, alpha=0.85)
    xi_trail_line, = ax.plot([], [], color='tab:cyan', linewidth=1.4, alpha=0.80)
    rocket_parts = {
        'shaft': None, 'nose': None, 'fins': [], 'thrust_lines': [],
        'thrust_label': None, 'com': None, 'xi': None, 'xi_proj': None, 'xi_link': None,
    }

    def _com_legend_handle():
        return Line2D(
            [0], [0], linestyle='None', marker='o', markersize=7,
            markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.6, label='COM',
        )

    def _xi_legend_handle():
        return Line2D(
            [0], [0], linestyle='None', marker='o', markersize=6,
            markerfacecolor='cyan', markeredgecolor='black', markeredgewidth=0.6, label='Xi',
        )

    def _xi_proj_legend_handle():
        return Line2D(
            [0], [0], linestyle='None', marker='o', markersize=5,
            markerfacecolor='white', markeredgecolor='tab:cyan', markeredgewidth=1.0,
            label='Xi proj (COM height)',
        )

    def _clear_rocket_artists():
        for key in ('shaft', 'nose', 'thrust_label', 'com', 'xi', 'xi_proj', 'xi_link'):
            artist = rocket_parts.get(key)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                rocket_parts[key] = None
        for fin in rocket_parts.get('fins') or []:
            try:
                fin.remove()
            except Exception:
                pass
        rocket_parts['fins'] = []
        for line in rocket_parts.get('thrust_lines') or []:
            try:
                line.remove()
            except Exception:
                pass
        rocket_parts['thrust_lines'] = []

    def _init():
        handles, labels = ax.get_legend_handles_labels()
        handles.append(_com_legend_handle())
        labels.append('COM')
        if xi_sim is not None:
            handles.append(_xi_legend_handle())
            labels.append('Xi')
            handles.append(_xi_proj_legend_handle())
            labels.append('Xi proj (COM height)')
        ax.legend(handles, labels, loc='upper left', fontsize=6.5, framealpha=0.88)
        trail_line.set_data([], [])
        xi_trail_line.set_data([], [])
        ax.set_title('Tracking animation (XZ)', fontsize=10, fontweight='bold')
        return (trail_line, xi_trail_line)

    def _update(k):
        i = int(frame_idx[k])
        pos = x_sim[i, 0:3]
        trail_line.set_data(x_sim[: i + 1, 0], x_sim[: i + 1, 2])
        if xi_sim is not None:
            xi_trail_line.set_data(xi_sim[: i + 1, 0], xi_sim[: i + 1, 2])
        else:
            xi_trail_line.set_data([], [])

        _clear_rocket_artists()
        q = quats[i]
        R = R_from_quat(q)
        base, tip, mid, com = _rocket_body_endpoints(pos, q, rocket_geom)
        rocket_parts['shaft'], = ax.plot(
            [base[0], tip[0]], [base[2], tip[2]],
            color='crimson', linewidth=shaft_lw, solid_capstyle='round', zorder=5,
        )
        rocket_parts['nose'] = ax.scatter(
            [tip[0]], [tip[2]], c='crimson', s=72, zorder=6,
        )
        wing = R[:, 0]
        for sign in (-1.0, 1.0):
            fin_tip = mid + wing * sign * fin_len
            fin_line, = ax.plot(
                [mid[0], fin_tip[0]], [mid[2], fin_tip[2]],
                color='darkred', linewidth=fin_lw, zorder=4,
            )
            rocket_parts['fins'].append(fin_line)

        rocket_parts['com'] = ax.scatter(
            [com[0]], [com[2]],
            c='gold', s=42, marker='o', edgecolors='black', linewidths=0.6, zorder=10,
        )
        if xi_sim is not None:
            xi = xi_sim[i]
            xi_p = xi_proj[i]
            rocket_parts['xi_link'], = ax.plot(
                [xi_p[0], xi[0]], [xi_p[2], xi[2]],
                color='tab:cyan', linestyle=':', linewidth=1.1, alpha=0.85, zorder=9,
            )
            rocket_parts['xi_proj'] = ax.scatter(
                [xi_p[0]], [xi_p[2]],
                c='white', s=30, marker='o', edgecolors='tab:cyan', linewidths=1.0, zorder=9,
            )
            rocket_parts['xi'] = ax.scatter(
                [xi[0]], [xi[2]],
                c='cyan', s=36, marker='o', edgecolors='black', linewidths=0.6, zorder=10,
            )

        th_p, th_r, T = u_hist[i, 0], u_hist[i, 1], u_hist[i, 2]
        T_disp = float(T) if np.isfinite(T) else 0.0
        if T_disp > 1e-6:
            app_pt = thrust_application_point_world(pos, q, r_thrust_body)
            thrust_dir = thrust_direction_world(q, th_p, th_r)
            thrust_len = thrust_len_base * (T_disp / thrust_scale)
            tail = app_pt - thrust_dir * thrust_len
            rocket_parts['thrust_lines'].extend(
                _draw_arrow2d(ax, tail[XZ], app_pt[XZ], color='darkorange', lw=3.0),
            )
            tip_mark = ax.scatter(
                [app_pt[0]], [app_pt[2]],
                c='darkorange', s=40, edgecolors='saddlebrown', linewidths=0.6, zorder=8,
            )
            rocket_parts['thrust_lines'].append(tip_mark)
            rocket_parts['thrust_label'] = ax.text(
                0.5 * (tail[0] + app_pt[0]) + 0.04 * span,
                0.5 * (tail[2] + app_pt[2]),
                f'{T_disp:.1f} N',
                fontsize=8, color='saddlebrown', fontweight='bold',
                ha='center', va='center', zorder=9,
                bbox=dict(
                    boxstyle='round,pad=0.25', facecolor='white',
                    edgecolor='darkorange', alpha=0.88,
                ),
            )
        else:
            rocket_parts['thrust_label'] = ax.text(
                pos[0], pos[2] + 0.06 * span,
                'T ≈ 0 N',
                fontsize=8, color='0.45', ha='center', va='center', zorder=9,
            )

        ax.set_title(f'Tracking XZ  t = {t[i]:.2f} s', fontsize=10, fontweight='bold')
        artists = [trail_line, xi_trail_line, rocket_parts['shaft'], rocket_parts['nose'], rocket_parts['com']]
        if rocket_parts['xi'] is not None:
            artists.append(rocket_parts['xi'])
            artists.append(rocket_parts['xi_proj'])
            artists.append(rocket_parts['xi_link'])
        artists.extend(rocket_parts['fins'])
        artists.extend(rocket_parts['thrust_lines'])
        if rocket_parts['thrust_label'] is not None:
            artists.append(rocket_parts['thrust_label'])
        return tuple(artists)

    _init()
    anim = animation.FuncAnimation(
        fig, _update, frames=len(frame_idx), init_func=_init, blit=False,
        interval=1000.0 / fps,
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path
