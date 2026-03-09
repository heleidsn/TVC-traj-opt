#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization - Shared utilities

Common functions extracted from tvc_traj_opt.py, tvc_traj_opt_acados.py,
tvc_traj_opt_pinocchio.py, tvc_traj_opt_gui.py for shared use.
"""

import numpy as np


# =============================================================================
# Quaternion utilities
# =============================================================================

def quat_mul(q1, q2):
    """Quaternion multiplication, q = [w,x,y,z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conj(q):
    """Quaternion conjugate"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_norm(q):
    """Quaternion normalization"""
    return q / np.linalg.norm(q)


def quat_exp(dtheta):
    """SO(3) exponential map to quaternion, dtheta is 3D rotation vector"""
    a = np.linalg.norm(dtheta)
    if a < 1e-12:
        return np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
    axis = dtheta / a
    s = np.sin(0.5*a)
    return np.array([np.cos(0.5*a), axis[0]*s, axis[1]*s, axis[2]*s])


def so3_log_from_quat(q):
    """Quaternion to rotation vector log map, q must be unit quaternion"""
    q = quat_norm(q)
    w, v = q[0], q[1:]
    nv = np.linalg.norm(v)
    w = np.clip(w, -1.0, 1.0)
    if nv < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(nv, w)
    return angle * (v / nv)


def R_from_quat(q):
    """Quaternion to rotation matrix, q=[w,x,y,z]"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])


def quat_to_euler(q, format='wxyz'):
    """
    Quaternion to Euler angles (ZYX order), radians.
    
    Args:
        q: quaternion
        format: 'wxyz' = [w,x,y,z] (Method 1), 'xyzw' = [qx,qy,qz,qw] (Pinocchio)
    """
    if format == 'xyzw':
        w, x, y, z = q[3], q[0], q[1], q[2]
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def quat_to_euler_deg(q, format='wxyz'):
    """Quaternion to Euler angles (ZYX), degrees"""
    return np.degrees(quat_to_euler(q, format))


def euler_to_quat_pinocchio(phi, theta, psi):
    """Euler ZYX (rad) to quaternion [qx, qy, qz, qw] (Pinocchio format)"""
    cy, sy = np.cos(psi/2), np.sin(psi/2)
    cp, sp = np.cos(theta/2), np.sin(theta/2)
    cr, sr = np.cos(phi/2), np.sin(phi/2)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    q = np.array([qx, qy, qz, qw])
    return q / (np.linalg.norm(q) + 1e-12)


def euler_to_quat_wxyz(phi, theta, psi):
    """Euler ZYX (rad) to quaternion [w, x, y, z] (Method 1 format)"""
    q_xyzw = euler_to_quat_pinocchio(phi, theta, psi)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def yaw_to_quaternion(yaw_deg):
    """Yaw only (deg) to quaternion [w, x, y, z], assume roll=0, pitch=0"""
    yaw_rad = np.radians(yaw_deg)
    w = np.cos(yaw_rad / 2.0)
    z = np.sin(yaw_rad / 2.0)
    return np.array([w, 0.0, 0.0, z])


# =============================================================================
# Rotation matrices (NumPy)
# =============================================================================

def Rx(a):
    """Rotation matrix about x-axis"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def Ry(a):
    """Rotation matrix about y-axis"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])


def Rz(a):
    """Rotation matrix about z-axis"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


# =============================================================================
# Segment boundaries
# =============================================================================

def segment_boundaries_from_waypoints(waypoints, dt):
    """
    Compute segment boundary indices from waypoints [x,y,z,yaw,time].
    Returns [end_idx_seg1, end_idx_seg2, ...]
    """
    if not waypoints or len(waypoints) < 2:
        return []
    t0 = float(waypoints[0][4]) if len(waypoints[0]) >= 5 else 0.0
    return [int((float(wp[4]) - t0) / dt) for wp in waypoints[1:] if len(wp) >= 5]


# =============================================================================
# Segment colors (for plotting)
# =============================================================================

SEGMENT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
