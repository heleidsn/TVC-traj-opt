# -*- coding: utf-8 -*-
"""Nonlinear TVC rigid-body dynamics model (same semi-implicit integration as trajectory optimization)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from tvc_common import R_from_quat, Rx, Ry, quat_exp, quat_mul, quat_norm

from .linear_model import physics_from_gui


class NonlinearTVCPlant:
    """
    13-state TVC rocket: [p, v, qw, qx, qy, qz, wx, wy, wz].
    Control u = [th_p, th_r, T, tau_yaw] (planner / optimizer units).
  """

    def __init__(self, mass, I_body, r_thrust_body, g=9.81, tvc_order='pitch_roll'):
        self.m = float(mass)
        self.I = np.asarray(I_body, dtype=float).reshape(3, 3)
        self.Iinv = np.linalg.inv(self.I)
        self.r = np.asarray(r_thrust_body, dtype=float).reshape(3)
        self.g = float(g)
        self.tvc_order = tvc_order

    def _rtvc(self, th_p, th_r):
        if self.tvc_order == 'pitch_roll':
            return Ry(th_p) @ Rx(th_r)
        if self.tvc_order == 'roll_pitch':
            return Rx(th_r) @ Ry(th_p)
        raise ValueError(f'Bad tvc_order: {self.tvc_order}')

    def step(self, x13, u_opt, dt):
        """Advance one semi-implicit Euler step (matches TVCRocketActionModel._step)."""
        dt = float(dt)
        x13 = np.asarray(x13, dtype=float).reshape(13)
        u = np.asarray(u_opt, dtype=float).reshape(4)
        p = x13[0:3]
        v = x13[3:6]
        q = x13[6:10]
        w = x13[10:13]

        th_p, th_r, T, tau_yaw = u
        rwb = R_from_quat(q)
        rtvc = self._rtvc(th_p, th_r)
        fb = rtvc @ np.array([0.0, 0.0, T], dtype=float)
        fw = rwb @ fb
        tau = np.cross(self.r, fb) + np.array([0.0, 0.0, tau_yaw], dtype=float)

        a = (fw / self.m) + np.array([0.0, 0.0, -self.g], dtype=float)
        v_next = v + dt * a
        p_next = p + dt * v_next
        w_dot = self.Iinv @ (tau - np.cross(w, self.I @ w))
        w_next = w + dt * w_dot
        dq = quat_exp(w_next * dt)
        q_next = quat_norm(quat_mul(dq, q))

        return np.concatenate([p_next, v_next, q_next, w_next])


def plant_from_phy_gui(phy_gui: Dict[str, Any]) -> NonlinearTVCPlant:
    """Build nonlinear dynamics model from GUI Physical parameters."""
    mass = float(phy_gui['mass'])
    I = np.diag([
        float(phy_gui['Ixx']),
        float(phy_gui['Iyy']),
        float(phy_gui['Izz']),
    ])
    r_thrust = np.array([
        float(phy_gui.get('r_thrust_x', 0.0)),
        float(phy_gui.get('r_thrust_y', 0.0)),
        float(phy_gui.get('r_thrust_z', -0.2)),
    ], dtype=float)
    g = float(phy_gui.get('g', 9.81))
    return NonlinearTVCPlant(mass, I, r_thrust, g=g)


def tracker_phy_from_gui(phy_gui: Dict[str, Any]) -> Dict[str, float]:
    """Linear-controller parameters (lever arm uses |r_thrust_z|)."""
    return physics_from_gui(
        phy_gui['mass'], phy_gui['Ixx'], phy_gui['Iyy'], phy_gui['Izz'],
        phy_gui.get('r_thrust_z', -0.2), phy_gui.get('g', 9.81),
    )
