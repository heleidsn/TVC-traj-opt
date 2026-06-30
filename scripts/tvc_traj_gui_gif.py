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

from tvc_common import R_from_quat, quat_norm
from tvc_rocket_platforms import normalize_platform_id, rocket_visual_geometry
from tvc_traj_gui_3d import (
    apply_equal_aspect_3d,
    equal_axis_cube_from_points,
    thrust_application_point_world,
    thrust_direction_world,
)


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
    fps=12,
    max_frames=120,
    figsize=(8.0, 8.0),
    dpi=120,
) -> str:
    """
    Render tracking simulation to an animated GIF.

    Uses a square figure and equal axis scaling. Thrust is drawn as an arrow
    whose tip sits at the thrust application point (COM + R @ r_thrust).

    Returns absolute path to the written GIF file.
    """
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

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'trajs', 'tracking_anim.gif',
        )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_idx = np.linspace(0, n - 1, min(max_frames, n), dtype=int)
    quats = np.array([_state12_to_quat_wxyz(x_sim[i]) for i in range(n)])
    u_hist = _align_controls(result.get('u_hist'), n)
    r_thrust_body = _r_thrust_body_from_result(result)
    rocket_geom = rocket_visual_geometry(_platform_id_from_result(result))

    all_pts = [x_sim[:, 0:3], x_ref[:, 0:3]]
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
    rocket_parts = {
        'shaft': None, 'nose': None, 'fins': [], 'thrust_lines': [],
        'thrust_label': None, 'com': None,
    }

    def _com_legend_handle():
        return Line2D(
            [0], [0], linestyle='None', marker='o', markersize=7,
            markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.6,
            label='COM',
        )

    def _clear_rocket_artists():
        for key in ('shaft', 'nose', 'thrust_label', 'com'):
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
        ax.legend(handles, labels, loc='upper left', fontsize=7, framealpha=0.88)
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        return (trail_line,)

    def _update(k):
        i = int(frame_idx[k])
        pos = x_sim[i, 0:3]
        trail_line.set_data(x_sim[: i + 1, 0], x_sim[: i + 1, 1])
        trail_line.set_3d_properties(x_sim[: i + 1, 2])

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
        artists = [trail_line, rocket_parts['shaft'], rocket_parts['nose'], rocket_parts['com']]
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
