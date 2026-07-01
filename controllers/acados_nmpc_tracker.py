# -*- coding: utf-8 -*-
"""Nonlinear receding-horizon tracking NMPC via acados (direct u output)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from tvc_common import quat_to_euler

from .linear_model import state12_to_state13, state13_to_state12


def state13_to_acados12(x13) -> np.ndarray:
    """Map simulator state [p,v,qwxyz,w] to Acados [p,euler,v,w]."""
    x13 = np.asarray(x13, dtype=float).flatten()
    phi, theta, psi = quat_to_euler(x13[6:10], format="wxyz")
    return np.array([
        x13[0], x13[1], x13[2],
        phi, theta, psi,
        x13[3], x13[4], x13[5],
        x13[10], x13[11], x13[12],
    ], dtype=float)


def state12_to_acados12(x12) -> np.ndarray:
    return state13_to_acados12(state12_to_state13(np.asarray(x12, dtype=float).reshape(12)))


def acados_nmpc_available() -> bool:
    try:
        from tvc_traj_opt_acados_tracking import ACADOS_TRACKING_AVAILABLE
        return bool(ACADOS_TRACKING_AVAILABLE)
    except ImportError:
        return False


class AcadosNmpcTracker:
    """
    Receding-horizon nonlinear MPC tracker.

    Control law: u = u₀* from Acados, tracking planned (x_ref, u_ref) along the
    horizon. No u_ref + Δu decomposition.
    """

    def __init__(self, phy_gui: Dict[str, Any], params: Dict[str, Any]):
        if not acados_nmpc_available():
            raise ImportError(
                "Acados is not available. Install CasADi and build acados; "
                "see TVC-traj-opt/README.md."
            )
        from tvc_traj_opt_acados_tracking import create_tracking_nmpc_solver

        self.phy_gui = dict(phy_gui)
        self.params = dict(params)
        self.mass = float(phy_gui["mass"])
        self.g = float(phy_gui.get("g", 9.81))
        self._solver = None
        self.horizon = int(params.get("horizon", 15))
        self.nmpc_dt = float(params.get("nmpc_dt", params.get("control_dt", 0.05)))
        self._solver_key = None
        self._build_solver(create_tracking_nmpc_solver)

    def _build_solver(self, factory):
        key = (
            self.horizon,
            round(self.nmpc_dt, 6),
            round(self.mass, 4),
            tuple(sorted((k, self.params.get(k)) for k in ("nmpc_nlp_max_iter", "nmpc_terminal_scale", "nmpc_du_weight"))),
        )
        if self._solver is not None and key == self._solver_key:
            return
        if self._solver is not None:
            try:
                self._solver = None
            except Exception:
                pass
        self._solver, self.horizon, self.nmpc_dt = factory(self.phy_gui, self.params)
        self._solver_key = key

    def update_params(self, params: Dict[str, Any]):
        self.params = dict(params)
        self.horizon = int(params.get("horizon", self.horizon))
        self.nmpc_dt = float(params.get("nmpc_dt", params.get("control_dt", self.nmpc_dt)))
        from tvc_traj_opt_acados_tracking import create_tracking_nmpc_solver
        self._solver_key = None
        self._build_solver(create_tracking_nmpc_solver)

    def reset(self):
        if self._solver is not None:
            try:
                self._solver.reset()
            except Exception:
                pass

    @staticmethod
    def _reference_control(ref, t_query: float) -> np.ndarray:
        if ref.in_terminal_hold(t_query):
            return np.array([0.0, 0.0, ref.mass * ref.g, 0.0], dtype=float)
        return ref.control_opt_at(t_query)

    def _set_references(self, ref, t0: float, horizon: int, dt: float):
        solver = self._solver
        for k in range(horizon):
            tk = t0 + k * dt
            x_ref = state12_to_acados12(ref.tracking_state12_at(tk))
            u_ref = self._reference_control(ref, tk)
            solver.set(k, "yref", np.concatenate([x_ref, u_ref]))
        tN = t0 + horizon * dt
        xN = state12_to_acados12(ref.tracking_state12_at(tN))
        solver.set(horizon, "yref", xN)

    def compute(
        self,
        x13,
        ref,
        t_query: float,
        horizon: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Solve one NMPC step.

        Returns plant control [th_p, th_r, T, tau_yaw] (rad, rad, N, N·m).
        """
        horizon = int(horizon if horizon is not None else self.horizon)
        dt = float(dt if dt is not None else self.nmpc_dt)
        x0 = state13_to_acados12(x13)

        solver = self._solver
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        self._set_references(ref, float(t_query), horizon, dt)

        status = solver.solve()
        if status != 0:
            # Fall back to hover trim if the NLP fails.
            return np.array([0.0, 0.0, self.mass * self.g, 0.0], dtype=float)
        return np.asarray(solver.get(0, "u"), dtype=float).reshape(4)
