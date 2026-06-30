# -*- coding: utf-8 -*-
"""Full-state LQR trajectory tracker with optional feedforward."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from .linear_model import build_AB


class LQRTracker:
    def __init__(self, phy_params, params):
        self.phy = phy_params
        self.A, self.B = build_AB(phy_params)
        self.K = None
        self.update_gains(params)

    def _Q_from_params(self, params):
        keys = ['Q_x', 'Q_y', 'Q_z', 'Q_vx', 'Q_vy', 'Q_vz',
                'Q_qx', 'Q_qy', 'Q_qz', 'Q_p', 'Q_q', 'Q_r']
        return np.diag([float(params[k]) for k in keys])

    def _R_from_params(self, params):
        return np.diag([
            float(params['R_qx']), float(params['R_qy']),
            float(params['R_T']), float(params['R_r']),
        ])

    def update_gains(self, params):
        Q = self._Q_from_params(params)
        R = self._R_from_params(params)
        P = scipy.linalg.solve_continuous_are(self.A, self.B, Q, R)
        self.K = np.linalg.solve(R, self.B.T @ P)

    def compute(self, x12, ref, u_ff=None, use_ff=True, clip_pos=0.5, clip_vel=0.5):
        e = np.asarray(x12, dtype=float) - np.asarray(ref, dtype=float)
        e[0:3] = np.clip(e[0:3], -clip_pos, clip_pos)
        e[3:6] = np.clip(e[3:6], -clip_vel, clip_vel)
        u_fb = -self.K @ e
        u = u_fb
        if use_ff and u_ff is not None:
            u = u + np.asarray(u_ff, dtype=float)
        return u
