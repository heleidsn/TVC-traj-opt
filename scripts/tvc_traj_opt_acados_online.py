#!/usr/bin/env python3
"""
Cached single-segment Acados solver for online replanning.

``solve_with_acados_waypoints`` rebuilds ``AcadosOcpSolver`` on every call
(codegen + dlopen), which dominates wall time (~1–2 s) even when SQP solve is
~70 ms. This module keeps one solver alive and only updates x0 / goal / refs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tvc_traj_opt_acados import ACADOS_AVAILABLE
from tvc_traj_opt_acados_cost import build_acados_ocp, _sync_acados_terminal_ls_from_ocp
from tvc_traj_opt_acados_dynamics import export_tvc_ode_model
from tvc_traj_opt_acados_state import acados_state_to_method1, waypoint_to_acados_state

if ACADOS_AVAILABLE:
    from acados_template import AcadosOcpSolver


def _json_key(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


class OnlineReplanSolverCache:
    """Persistent Acados solver for repeated current-state -> goal replans."""

    def __init__(self) -> None:
        self._solver: Any = None
        self._ocp: Any = None
        self._config_key: Optional[str] = None
        self._N: int = 0
        self._Tf: float = 0.0
        self._uref: Optional[np.ndarray] = None
        self._use_control_rate: bool = False
        self._use_actuator_dynamics: bool = False
        self._last_build_s: float = 0.0
        self._last_cache_hit: bool = False

    def invalidate(self) -> None:
        self._solver = None
        self._ocp = None
        self._config_key = None

    def _running_yref(
        self,
        x_ref_12: np.ndarray,
        uref_arr: np.ndarray,
    ) -> np.ndarray:
        x_ref_12 = np.asarray(x_ref_12, dtype=float).flatten()[:12]
        if self._use_control_rate:
            return np.concatenate([x_ref_12, uref_arr, uref_arr, np.zeros(4)])
        if self._use_actuator_dynamics:
            return np.concatenate([x_ref_12, uref_arr, uref_arr])
        return np.concatenate([x_ref_12, uref_arr])

    def _cache_root(self) -> str:
        root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'c_generated_code',
            'online_replan_cache',
        )
        os.makedirs(root, exist_ok=True)
        return root

    def _config_digest(
        self,
        dt: float,
        N: int,
        Tf: float,
        m: float,
        I: np.ndarray,
        r_thrust: np.ndarray,
        weights: dict,
        bounds: dict,
        max_iter: int,
    ) -> str:
        payload = {
            'dt': float(dt),
            'N': int(N),
            'Tf': float(Tf),
            'm': float(m),
            'I': np.asarray(I, dtype=float).reshape(-1).tolist(),
            'r': np.asarray(r_thrust, dtype=float).reshape(-1).tolist(),
            'weights': weights,
            'bounds': bounds,
            'max_iter': int(max_iter),
        }
        return hashlib.sha1(_json_key(payload).encode('utf-8')).hexdigest()[:16]

    def _ensure_solver(
        self,
        dt: float,
        N: int,
        Tf: float,
        m: float,
        I: np.ndarray,
        r_thrust: np.ndarray,
        weights: dict,
        bounds: dict,
        max_iter: int,
        terminal_weights: Optional[dict],
        x0: np.ndarray,
        xg: np.ndarray,
    ) -> None:
        if not ACADOS_AVAILABLE:
            raise ImportError('Acados not available')

        config_key = self._config_digest(dt, N, Tf, m, I, r_thrust, weights, bounds, max_iter)
        if self._solver is not None and self._config_key == config_key:
            self._last_cache_hit = True
            self._last_build_s = 0.0
            return

        t_build0 = time.perf_counter()
        self._last_cache_hit = False

        use_actuator_dynamics = bool(weights.get('actuator_dynamics', False))
        actuator_tau = weights.get('actuator_tau', [0.05, 0.05, 0.05, 0.05])
        use_control_rate = (not use_actuator_dynamics) and float(weights.get('du', 0.0)) > 0
        model_suffix = '_act' if use_actuator_dynamics else ('_du' if use_control_rate else '')
        digest = config_key
        code_export_dir = os.path.join(
            self._cache_root(),
            f'cfg_{digest}_N{N}{model_suffix}',
        )
        os.makedirs(code_export_dir, exist_ok=True)
        json_file = os.path.join(code_export_dir, f'online_replan{model_suffix}.json')
        model_name = f'online_replan_{digest[:8]}{model_suffix}'

        uref = np.array([0.0, 0.0, float(m) * 9.81, 0.0])
        model = export_tvc_ode_model(
            float(m),
            I,
            r_thrust,
            model_name=model_name,
            use_control_rate=use_control_rate,
            use_actuator_dynamics=use_actuator_dynamics,
            actuator_tau=actuator_tau,
        )
        ocp = build_acados_ocp(
            model,
            N,
            Tf,
            x0,
            xg,
            uref,
            weights,
            bounds,
            dt,
            terminal_weights=terminal_weights,
            code_export_dir=code_export_dir,
            json_file=json_file,
            nlp_solver_max_iter=int(max_iter),
            qp_solver=None,
            verbose_solve=False,
        )
        try:
            try:
                solver = AcadosOcpSolver(ocp, json_file=json_file, build=True, verbose=False)
            except TypeError:
                solver = AcadosOcpSolver(ocp, verbose=False)
        except Exception as exc:
            raise RuntimeError(f'Online replan Acados solver creation failed: {exc}') from exc

        self._solver = solver
        self._ocp = ocp
        self._config_key = config_key
        self._N = int(N)
        self._Tf = float(Tf)
        self._uref = uref
        self._use_control_rate = use_control_rate
        self._use_actuator_dynamics = use_actuator_dynamics
        self._last_build_s = time.perf_counter() - t_build0

    def _augment_state(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).flatten()
        uref = np.asarray(self._uref, dtype=float)
        if self._use_control_rate or self._use_actuator_dynamics:
            if x_arr.size >= 16:
                return x_arr[:16]
            if x_arr.size >= 12:
                return np.concatenate([x_arr[:12], uref])
        return x_arr

    def _apply_initial_state(self, solver: Any, x0_seg: np.ndarray) -> None:
        """Pin stage-0 state to measured x0 (guess + equality constraint)."""
        x0_arr = np.asarray(x0_seg, dtype=float).flatten()
        solver.set(0, 'x', x0_arr)
        try:
            solver.constraints_set(0, 'lbx', x0_arr)
            solver.constraints_set(0, 'ubx', x0_arr)
        except Exception:
            pass

    def _apply_runtime_params(self, solver: Any, N: int) -> None:
        if self._use_control_rate:
            p_val = np.array([N / max(self._Tf, 1e-6)])
            for i in range(N):
                solver.set(i, 'p', p_val)
        elif self._use_actuator_dynamics:
            actuator_tau = [0.05, 0.05, 0.05, 0.05]
            tau_arr = np.asarray(actuator_tau, dtype=float).flatten()
            if tau_arr.size < 4:
                tau_arr = np.resize(tau_arr, 4)
            p_val = np.asarray(1.0 / np.maximum(tau_arr, 1e-6), dtype=np.float64)
            for i in range(N):
                solver.set(i, 'p', p_val)

    def _set_stage_refs_and_guess(
        self,
        solver: Any,
        x0_seg: np.ndarray,
        xg_seg: np.ndarray,
        N: int,
        weights: dict,
    ) -> None:
        """Update LS references and SQP initial guess (goal can change every replan)."""
        uref_arr = np.asarray(self._uref, dtype=float)
        x0_12 = np.asarray(x0_seg, dtype=float).flatten()[:12]
        xg_12 = np.asarray(xg_seg, dtype=float).flatten()[:12]
        use_schedule_ref = bool(weights.get('schedule_ref', True))

        for i in range(N):
            if use_schedule_ref:
                alpha = float(i) / max(N, 1)
                x_ref = (1.0 - alpha) * x0_12 + alpha * xg_12
            else:
                x_ref = xg_12
            solver.set(i, 'yref', self._running_yref(x_ref, uref_arr))

        if self._ocp is not None:
            self._ocp.cost.yref_e = xg_12.copy()
            _sync_acados_terminal_ls_from_ocp(solver, self._ocp, N)

        for i in range(1, N + 1):
            alpha = float(i) / max(N, 1)
            x_guess = (1.0 - alpha) * x0_seg + alpha * xg_seg
            solver.set(i, 'x', x_guess)
        for i in range(N):
            solver.set(i, 'u', uref_arr)

    def solve_segment(
        self,
        dt: float,
        waypoints: Sequence[Sequence[float]],
        m: float,
        I: np.ndarray,
        r_thrust: np.ndarray,
        weights: dict,
        bounds: dict,
        max_iter: int = 100,
        terminal_weights: Optional[dict] = None,
        initial_state: Optional[np.ndarray] = None,
        verbose_solve: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], list, Optional[np.ndarray], dict]:
        if len(waypoints) < 2:
            raise ValueError('online replan expects at least two waypoints')

        duration = float(waypoints[1][4]) - float(waypoints[0][4])
        if duration <= 0.0:
            raise ValueError('segment duration must be positive')
        N = max(10, int(round(duration / float(dt))))
        Tf = duration

        if initial_state is not None:
            x0 = np.asarray(initial_state, dtype=float).flatten()
            if x0.size > 12:
                x0 = x0[:12]
        else:
            x0 = waypoint_to_acados_state(waypoints[0])
        xg = waypoint_to_acados_state(waypoints[1])

        t_wall0 = time.perf_counter()
        self._ensure_solver(
            dt, N, Tf, float(m), I, r_thrust, weights, bounds, max_iter,
            terminal_weights, x0, xg,
        )
        build_s = float(self._last_build_s)
        cache_hit = bool(self._last_cache_hit)

        solver = self._solver
        x0_seg = self._augment_state(x0)
        xg_seg = self._augment_state(xg)
        self._apply_initial_state(solver, x0_seg)
        self._apply_runtime_params(solver, N)
        self._set_stage_refs_and_guess(solver, x0_seg, xg_seg, N, weights)

        t0 = time.perf_counter()
        status = solver.solve()
        solve_s = time.perf_counter() - t0
        wall_s = time.perf_counter() - t_wall0

        if status != 0 and verbose_solve:
            print(f'[online replan] Acados status={status}')

        seg_xs = [
            acados_state_to_method1(np.array(solver.get(i, 'x'), copy=True))
            for i in range(N + 1)
        ]
        seg_us = [np.array(solver.get(i, 'u'), copy=True) for i in range(N)]

        class _SimpleLogger:
            def __init__(self, cost: float):
                self.costs = [float(cost)]

        try:
            cost_val = float(solver.get_cost())
        except Exception:
            cost_val = 0.0
        try:
            sqp_iter = int(solver.get_stats('sqp_iter'))
        except Exception:
            sqp_iter = 1

        meta = {
            'total_solve_time': float(solve_s),
            'wall_time': float(wall_s),
            'build_time': float(build_s),
            'cache_hit': cache_hit,
            'sqp_iter': sqp_iter,
            'status': int(status),
        }
        return seg_xs, seg_us, [_SimpleLogger(cost_val)], None, meta


def solve_with_acados_online_cached(
    cache: OnlineReplanSolverCache,
    dt,
    waypoints,
    m,
    I,
    r_thrust,
    weights,
    bounds,
    max_iter=100,
    use_box_solver=False,
    callback=None,
    running_flag=None,
    terminal_weights=None,
    iteration_callback=None,
    verbose_solve=False,
    initial_state=None,
):
    """Drop-in replacement for ``solve_with_acados_waypoints`` (single segment)."""
    del use_box_solver, callback, running_flag, iteration_callback
    return cache.solve_segment(
        float(dt),
        waypoints,
        float(m),
        np.asarray(I, dtype=float),
        np.asarray(r_thrust, dtype=float),
        dict(weights),
        dict(bounds),
        max_iter=int(max_iter),
        terminal_weights=terminal_weights,
        initial_state=initial_state,
        verbose_solve=bool(verbose_solve),
    )
