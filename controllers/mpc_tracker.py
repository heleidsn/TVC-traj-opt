# -*- coding: utf-8 -*-
"""Linear receding-horizon MPC tracker (unconstrained QP via batch solution)."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from .linear_model import build_AB, discretize_euler


class MPCTracker:
    def __init__(self, phy_params, params):
        self.phy = phy_params
        self.A, self.B = build_AB(phy_params)
        self.horizon = int(params.get('horizon', 20))
        self.mpc_dt = float(params.get('mpc_dt', 0.05))
        self._build_cost(params)
        self._precompute()

    def _Q_from_params(self, params):
        keys = ['Q_x', 'Q_y', 'Q_z', 'Q_vx', 'Q_vy', 'Q_vz',
                'Q_qx', 'Q_qy', 'Q_qz', 'Q_p', 'Q_q', 'Q_r']
        return np.diag([float(params[k]) for k in keys])

    def _R_from_params(self, params):
        return np.diag([
            float(params['R_qx']), float(params['R_qy']),
            float(params['R_T']), float(params['R_r']),
        ])

    def _build_cost(self, params):
        self.Q = self._Q_from_params(params)
        self.R = self._R_from_params(params)
        Ad, Bd = discretize_euler(self.A, self.B, self.mpc_dt)
        try:
            P = scipy.linalg.solve_discrete_are(Ad, Bd, self.Q, self.R)
        except Exception:
            P = self.Q.copy()
        self.Qf = P
        self.Ad = Ad
        self.Bd = Bd

    def update_params(self, params):
        self.horizon = int(params.get('horizon', self.horizon))
        self.mpc_dt = float(params.get('mpc_dt', self.mpc_dt))
        self._build_cost(params)
        self._precompute()

    def _precompute(self):
        n = self.A.shape[0]
        m = self.B.shape[1]
        N = self.horizon
        Ad, Bd = self.Ad, self.Bd

        self.Phi = np.zeros((N * n, n))
        self.Gamma = np.zeros((N * n, N * m))
        Apow = np.eye(n)
        for k in range(N):
            row = slice(k * n, (k + 1) * n)
            self.Phi[row, :] = Apow @ Ad
            for j in range(k + 1):
                col = slice(j * m, (j + 1) * m)
                Ak = np.linalg.matrix_power(Ad, k - j)
                self.Gamma[row, col] = Ak @ Bd
            Apow = Apow @ Ad

        Qbar = scipy.linalg.block_diag(*([self.Q] * N))
        Rbar = scipy.linalg.block_diag(*([self.R] * N))
        self.H = self.Gamma.T @ Qbar @ self.Gamma + Rbar
        self.H = 0.5 * (self.H + self.H.T) + 1e-8 * np.eye(self.H.shape[0])

    def compute(self, x12, ref_horizon, u_ff=None, use_ff=True, clip_pos=0.5, clip_vel=0.5):
        """
        ref_horizon : (N+1, 12) reference over MPC horizon.
        """
        x0 = np.asarray(x12, dtype=float)
        N = self.horizon
        xref = np.asarray(ref_horizon, dtype=float)
        if xref.shape[0] < N + 1:
            pad = np.repeat(xref[-1:], N + 1 - xref.shape[0], axis=0)
            xref = np.vstack([xref, pad])
        xref = xref[: N + 1]

        x_ref_stack = xref[1: N + 1].reshape(-1)
        x_pred_ref = self.Phi @ x0
        err_free = x_pred_ref - x_ref_stack
        f = self.Gamma.T @ scipy.linalg.block_diag(*([self.Q] * N)) @ err_free

        try:
            U = np.linalg.solve(self.H, -f)
        except np.linalg.LinAlgError:
            U, _, _, _ = np.linalg.lstsq(self.H, -f, rcond=None)

        u = U[: self.B.shape[1]]
        if use_ff and u_ff is not None:
            u = u + np.asarray(u_ff, dtype=float)
        return u
