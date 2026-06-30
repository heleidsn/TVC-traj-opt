# -*- coding: utf-8 -*-
"""Shared 3D plotting helpers for tracking trajectory and GIF animation."""

from __future__ import annotations

import numpy as np

from tvc_common import R_from_quat


def equal_axis_cube_from_points(points, margin=0.15, min_span=0.5):
    """
    Compute equal-length axis limits centered on the point cloud.

    Returns (lo, hi) each shape (3,) with identical edge length on x/y/z.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    lo_raw = pts.min(axis=0)
    hi_raw = pts.max(axis=0)
    center = 0.5 * (lo_raw + hi_raw)
    span = float(max(np.max(hi_raw - lo_raw), min_span))
    half = 0.5 * span * (1.0 + 2.0 * margin)
    lo = center - half
    hi = center + half
    return lo, hi


def apply_equal_aspect_3d(ax, lo, hi):
    """Apply equal data limits and cubic box aspect to a 3D axes."""
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def thrust_direction_body(th_p, th_r):
    """
    Unit thrust force direction in body frame (+Z nominal, gimbal pitch/roll in rad).

    Matches planner controls [th_p, th_r, T, tau_yaw]: Ry(th_p) @ Rx(th_r) @ e_z.
    """
    cp, sp = np.cos(th_p), np.sin(th_p)
    cr, sr = np.cos(th_r), np.sin(th_r)
    direction = np.array([sp * cr, -sr, cp * cr], dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return direction / norm


def thrust_direction_world(q_wxyz, th_p, th_r):
    """Thrust force direction in the inertial frame."""
    R = R_from_quat(np.asarray(q_wxyz, dtype=float))
    return R @ thrust_direction_body(float(th_p), float(th_r))


def thrust_application_point_world(com_pos, q_wxyz, r_thrust_body):
    """World-frame thrust application point: COM + R @ r_thrust (body frame)."""
    R = R_from_quat(np.asarray(q_wxyz, dtype=float))
    r = np.asarray(r_thrust_body, dtype=float).reshape(3)
    return np.asarray(com_pos, dtype=float).reshape(3) + R @ r
