# -*- coding: utf-8 -*-
"""Waypoint ↔ Acados state maps and Method-1 conversion (no CasADi)."""

import numpy as np

from tvc_common import euler_to_quat_pinocchio, quat_to_euler


def _acados_x_for_method1(x_full):
    """Drop pseudotime duration state ``T_seg`` (last component) for Method-1 / plotting."""
    x_full = np.asarray(x_full, dtype=float).flatten()
    if x_full.size <= 12:
        return x_full
    return x_full[:-1]


def acados_state_to_method1(x_acados):
    """Convert Acados state [x,y,z,phi,theta,psi,vx,vy,vz,wx,wy,wz(,u_prev)] to Method 1 format (17-dim)"""
    x_acados = np.asarray(x_acados).flatten()
    p = x_acados[:3]
    phi, theta, psi = x_acados[3], x_acados[4], x_acados[5]
    v = x_acados[6:9]
    w = x_acados[9:12]
    q = euler_to_quat_pinocchio(phi, theta, psi)
    # Method 1: [p(3), v(3), q(4) wxyz, w(3), u_prev(4)]
    x_m1 = np.zeros(17)
    x_m1[0:3] = p
    x_m1[3:6] = v
    x_m1[6:10] = [q[3], q[0], q[1], q[2]]  # wxyz from qxyzw
    x_m1[10:13] = w
    return x_m1


def method1_to_acados_state(x_m1):
    """Convert Method 1 state [p,v,q wxyz,w,...] to Acados [x,y,z,phi,theta,psi,vx,vy,vz,wx,wy,wz]"""
    x = np.zeros(12)
    x[0:3] = np.asarray(x_m1[0:3], dtype=float)
    x[3:6] = quat_to_euler(np.asarray(x_m1[6:10], dtype=float), format="wxyz")
    x[6:9] = np.asarray(x_m1[3:6], dtype=float)
    x[9:12] = np.asarray(x_m1[10:13], dtype=float)
    return x


def waypoint_to_acados_state(wp, uref=None):
    """Convert waypoint [x,y,z,yaw_deg,time] to Acados state (goal).
    If uref is given, return augmented 16-dim state [x_12, uref] for control rate model."""
    x = np.zeros(12)
    x[0:3] = [float(wp[0]), float(wp[1]), float(wp[2])]
    yaw_deg = float(wp[3]) if len(wp) > 3 else 0.0
    x[5] = np.radians(yaw_deg)  # psi
    if uref is not None:
        x = np.concatenate([x, np.asarray(uref).flatten()[:4]])
    return x
