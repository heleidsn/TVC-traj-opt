# -*- coding: utf-8 -*-
"""Linear receding-horizon MPC tracker (unconstrained QP via batch solution)."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from .linear_model import build_AB, discretize_euler


class MPCTracker:
    """
    Receding-horizon tracker on the 12-state linear TVC model.

    Control law: u = u_ref + du, where u_ref is the planned trajectory control
    and du is the first step of a finite-horizon LQ problem that drives the
    *error state* e = x - x_ref toward zero:

        e_{k+1} = Ad e_k + Bd du_k
        min  sum_k e_k' Q e_k + du_k' R du_k  (+ terminal Qf)

    This matches LQR feedforward + feedback structure and remains stable when the
    reference states come from nonlinear trajectory optimization (they are not
    exactly reachable by the linear prediction model).
    """

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

        Q_blocks = [self.Q] * (N - 1) + [self.Qf]
        self.Qbar = scipy.linalg.block_diag(*Q_blocks)
        self.Rbar = scipy.linalg.block_diag(*([self.R] * N))
        self.H = self.Gamma.T @ self.Qbar @ self.Gamma + self.Rbar
        self.H = 0.5 * (self.H + self.H.T) + 1e-8 * np.eye(self.H.shape[0])

    def compute(
        self,
        x12,
        ref_horizon,
        u_ref_horizon=None,
        u_ff=None,
        use_ff=True,
        clip_pos=0.5,
        clip_vel=0.5,
        ignore_attitude_error=False,
        ignore_rate_error=False,
    ):
        """
        ref_horizon : (N+1, 12) planned states (only ref_horizon[0] used for error).
        u_ref_horizon : (N, 4) planned controls in LQR coords; u_ref_horizon[0] is feedforward.

        Returns u_ref[0] + du[0] in LQR coordinates [qx, qy, T_delta, r].
        """
        del u_ff, use_ff

        x0 = np.asarray(x12, dtype=float)
        N = self.horizon
        m = self.B.shape[1]
        xref = np.asarray(ref_horizon, dtype=float)
        if xref.shape[0] < 1:
            raise ValueError('ref_horizon must contain at least one state sample')
        x_ref0 = xref[0]

        if u_ref_horizon is None or len(u_ref_horizon) == 0:
            u_ref0 = np.zeros(m, dtype=float)
        else:
            u_ref0 = np.asarray(u_ref_horizon[0], dtype=float).reshape(m)

        e0 = x0 - x_ref0
        e0[0:3] = np.clip(e0[0:3], -clip_pos, clip_pos)
        e0[3:6] = np.clip(e0[3:6], -clip_vel, clip_vel)
        if ignore_attitude_error:
            e0[6:9] = 0.0
        if ignore_rate_error:
            e0[9:12] = 0.0

        err_free = self.Phi @ e0
        rhs = -self.Gamma.T @ self.Qbar @ err_free

        try:
            dU = np.linalg.solve(self.H, rhs)
        except np.linalg.LinAlgError:
            dU, _, _, _ = np.linalg.lstsq(self.H, rhs, rcond=None)

        du0 = dU[:m]
        return u_ref0 + du0
