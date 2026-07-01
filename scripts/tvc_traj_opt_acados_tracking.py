# -*- coding: utf-8 -*-
"""
Receding-horizon **nonlinear tracking NMPC** for closed-loop trajectory following.

Unlike the linear ``MPCTracker`` (u = u_ref + Δu), this formulation outputs the
first optimal control u₀ directly from a nonlinear least-squares Acados OCP that
tracks the planned state / control reference along the horizon.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import tvc_runtime

    tvc_runtime.ensure_scripts_on_path()
    tvc_runtime.preload_acados_libs()
except Exception:
    pass

try:
    import casadi as ca
    from acados_template import AcadosOcp, AcadosOcpSolver
except ImportError:
    ca = None
    AcadosOcp = None
    AcadosOcpSolver = None

try:
    from tvc_traj_opt_acados_dynamics import export_tvc_ode_model
    from tvc_traj_opt_acados_cost import (
        _apply_control_bounds,
        _per_channel_ls_weights,
        _state_tracking_q_diag,
        _u_sigma_from_weights_or_bounds,
    )
except ImportError:
    export_tvc_ode_model = None

ACADOS_TRACKING_AVAILABLE = (
    ca is not None
    and AcadosOcp is not None
    and AcadosOcpSolver is not None
    and export_tvc_ode_model is not None
)


def _require_acados_tracking():
    if not ACADOS_TRACKING_AVAILABLE:
        raise ImportError(
            "Acados nonlinear tracking NMPC is unavailable. "
            "Install CasADi and build acados; set ACADOS_SOURCE_DIR. "
            "See TVC-traj-opt/README.md."
        )


def tracking_params_to_acados_weights(params: Dict[str, Any]) -> Dict[str, Any]:
    """Map Tracking GUI parameters to Acados LS weight keys."""
    return {
        "p": float(np.mean([params.get("Q_x", 1.0), params.get("Q_y", 1.0), params.get("Q_z", 1.0)])),
        "v": float(np.mean([params.get("Q_vx", 1.0), params.get("Q_vy", 1.0), params.get("Q_vz", 1.0)])),
        "roll": float(params.get("Q_qx", 1.0)),
        "pitch": float(params.get("Q_qy", 1.0)),
        "yaw": float(params.get("Q_qz", 0.1)),
        "w": float(np.mean([params.get("Q_p", 1.0), params.get("Q_q", 1.0), params.get("Q_r", 0.01)])),
        "u": np.array([
            1.0 / max(float(params.get("R_qy", 10.0)), 1e-6),
            1.0 / max(float(params.get("R_qx", 10.0)), 1e-6),
            1.0 / max(float(params.get("R_T", 1.0)), 1e-6),
            1.0 / max(float(params.get("R_r", 10.0)), 1e-6),
        ], dtype=float),
        "du": float(params.get("nmpc_du_weight", 0.05)),
        "terminal_cost_multiplier": float(params.get("nmpc_terminal_scale", 10.0)),
    }


def params_to_acados_bounds(params: Dict[str, Any], mass: float, g: float) -> Dict[str, tuple]:
    """Build Acados control bounds from tracking constraint parameters."""
    th_p_max = np.radians(float(params.get("th_p_max", 10.0)))
    th_r_max = np.radians(float(params.get("th_r_max", 10.0)))
    T_min = float(params.get("T_min", 0.0))
    T_max = float(params.get("T_max", max(mass * g * 2.0, 1.0)))
    tau_max = float(params.get("tau_yaw_max", 8.0))
    return {
        "th_p": (-th_p_max, th_p_max),
        "th_r": (-th_r_max, th_r_max),
        "T": (T_min, T_max),
        "tau_yaw": (-tau_max, tau_max),
    }


def build_acados_tracking_ocp(
    model,
    N: int,
    dt: float,
    bounds: Dict[str, tuple],
    weights: Dict[str, Any],
    *,
    code_export_dir: Optional[str] = None,
    json_file: Optional[str] = None,
    nlp_solver_max_iter: int = 10,
    verbose: bool = False,
):
    """
    Nonlinear tracking NMPC OCP on the 12-state Euler TVC model.

    Running cost: || [x; u] - [x_ref; u_ref] ||_W
    Terminal cost: || x - x_ref_N ||_{W_e}
    """
    _require_acados_tracking()
    ocp = AcadosOcp()
    ocp.model = model
    if code_export_dir is not None:
        try:
            ocp.code_gen_opts.code_export_directory = code_export_dir
        except AttributeError:
            ocp.code_export_directory = code_export_dir
    if json_file is not None:
        try:
            ocp.code_gen_opts.json_file = json_file
        except AttributeError:
            ocp.json_file = json_file

    N = int(N)
    dt = float(dt)
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    sigma_u = _u_sigma_from_weights_or_bounds(weights, bounds)
    w_u_diag = _per_channel_ls_weights(weights.get("u", 1e-2), sigma_u)
    w_du_diag = _per_channel_ls_weights(weights.get("du", 0.05), sigma_u)
    Q = np.diag(_state_tracking_q_diag(weights))
    R_mat = np.diag(w_u_diag)

    cost_y_expr = ca.vertcat(model.x[:12], model.u)
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = cost_y_expr
    ocp.cost.yref = np.zeros(16)
    ocp.cost.W = np.block([[Q, np.zeros((12, 4))], [np.zeros((4, 12)), R_mat]])

    Qe = np.diag(np.diag(Q) * float(weights.get("terminal_cost_multiplier", 10.0)))
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = model.x[:12]
    ocp.cost.yref_e = np.zeros(12)
    ocp.cost.W_e = Qe

    _apply_control_bounds(ocp, bounds)

    nx = 12
    ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int32)
    ocp.constraints.lbx_0 = np.zeros(nx)
    ocp.constraints.ubx_0 = np.zeros(nx)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = int(nlp_solver_max_iter)
    ocp.solver_options.print_level = 2 if verbose else 0
    ocp.solver_options.store_iterates = False
    return ocp


def _solver_cache_dir(key: str) -> str:
    cache_root = os.path.join(_ROOT, ".cache", "acados_tracking_nmpc")
    os.makedirs(cache_root, exist_ok=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    path = os.path.join(cache_root, f"nmpc_{digest}")
    os.makedirs(path, exist_ok=True)
    return path


def create_tracking_nmpc_solver(
    phy_gui: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[Any, int, float]:
    """Build (or load cached) ``AcadosOcpSolver`` for tracking NMPC."""
    _require_acados_tracking()

    mass = float(phy_gui["mass"])
    g = float(phy_gui.get("g", 9.81))
    I = np.diag([
        float(phy_gui.get("Ixx", 0.02)),
        float(phy_gui.get("Iyy", 0.02)),
        float(phy_gui.get("Izz", 0.01)),
    ])
    r_thrust = np.array([
        float(phy_gui.get("r_thrust_x", 0.0)),
        float(phy_gui.get("r_thrust_y", 0.0)),
        float(phy_gui.get("r_thrust_z", -0.2)),
    ], dtype=float)

    horizon = int(params.get("horizon", 15))
    dt = float(params.get("nmpc_dt", params.get("control_dt", 0.05)))
    bounds = params_to_acados_bounds(params, mass, g)
    weights = tracking_params_to_acados_weights(params)
    nlp_iter = int(params.get("nmpc_nlp_max_iter", 10))

    cache_key = "|".join([
        f"m={mass:.4f}",
        f"I={I.flat[0]:.5f},{I.flat[4]:.5f},{I.flat[8]:.5f}",
        f"r={r_thrust[2]:.4f}",
        f"N={horizon}",
        f"dt={dt:.4f}",
        f"bounds={bounds}",
        f"w={weights.get('p')},{weights.get('v')}",
    ])
    code_dir = _solver_cache_dir(cache_key)
    json_path = os.path.join(code_dir, "tracking_ocp.json")

    model = export_tvc_ode_model(
        mass, I, r_thrust, g=g,
        model_name=f"tvc_tracking_{hashlib.sha1(cache_key.encode()).hexdigest()[:8]}",
        use_control_rate=False,
        use_actuator_dynamics=False,
    )
    ocp = build_acados_tracking_ocp(
        model, horizon, dt, bounds, weights,
        code_export_dir=code_dir,
        json_file=json_path,
        nlp_solver_max_iter=nlp_iter,
    )
    solver = AcadosOcpSolver(ocp, json_file=json_path, build=True)
    return solver, horizon, dt
