# -*- coding: utf-8 -*-
"""12-state linear TVC model (consistent with deployed lqr.py)."""

from __future__ import annotations

import numpy as np


def physics_from_gui(mass, Ixx, Iyy, Izz, r_thrust_z, g=9.81):
    """Build phy_params dict from GUI physics spinboxes."""
    return {
        'MASS': float(mass),
        'G': float(g),
        'I_XX': float(Ixx),
        'I_YY': float(Iyy),
        'I_ZZ': float(Izz),
        'DIST_COM_2_THRUST': abs(float(r_thrust_z)),
    }


def build_AB(phy_params):
    """Return (A, B) for state [x,y,z,vx,vy,vz,qx,qy,qz,p,q,r] and u [qx,qy,T,r]."""
    g = phy_params['G']
    m = phy_params['MASS']
    l = phy_params['DIST_COM_2_THRUST']
    Ixx = phy_params['I_XX']
    Iyy = phy_params['I_YY']
    Izz = phy_params['I_ZZ']

    A = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2 * g, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -2 * g, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)

    B = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 2 * g, 0, 0],
        [-2 * g, 0, 0, 0],
        [0, 0, 1 / m, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-l * m * g / Ixx, 0, 0, 0],
        [0, -l * m * g / Iyy, 0, 0],
        [0, 0, 0, 1 / Izz],
    ], dtype=float)
    return A, B


def discretize_euler(A, B, dt):
    """First-order Euler discretization."""
    dt = float(dt)
    Ad = np.eye(A.shape[0]) + A * dt
    Bd = B * dt
    return Ad, Bd


def state13_to_state12(xs13):
    """Drop quaternion w component; keep [pos, vel, qx,qy,qz, angvel]."""
    xs13 = np.asarray(xs13, dtype=float)
    if xs13.ndim == 1:
        return np.concatenate([xs13[0:6], xs13[7:10], xs13[10:13]])
    return np.hstack([xs13[:, 0:6], xs13[:, 7:10], xs13[:, 10:13]])


def state12_to_state13(x12):
    """Rebuild planner 13-state [pos, vel, qw,qx,qy,qz, angvel] from 12-state."""
    x12 = np.asarray(x12, dtype=float).reshape(12)
    qx, qy, qz = x12[6], x12[7], x12[8]
    qw = float(np.sqrt(max(1.0 - qx * qx - qy * qy - qz * qz, 0.0)))
    return np.concatenate([x12[0:6], [qw, qx, qy, qz], x12[9:12]])


def control_opt_to_lqr(us4, mass, g=9.81):
    """
    Map planner controls [th_p, th_r, T, tau_yaw] to LQR inputs [qx, qy, T_delta, r].
    Small-angle: qx ≈ th_r/2, qy ≈ th_p/2; thrust delta = T - m*g.
    """
    us4 = np.asarray(us4, dtype=float).reshape(4)
    th_p, th_r, T, tau = us4
    hover = mass * g
    return np.array([th_r / 2.0, th_p / 2.0, T - hover, tau], dtype=float)


def lqr_to_control_opt(u4, mass, g=9.81):
    """Inverse map LQR control to planner [th_p, th_r, T, tau_yaw]."""
    u4 = np.asarray(u4, dtype=float).reshape(4)
    qx, qy, T_delta, tau = u4
    hover = mass * g
    return np.array([2.0 * qy, 2.0 * qx, T_delta + hover, tau], dtype=float)
