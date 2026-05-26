#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free-final-time minimum-fuel **TVC 6-DoF** guidance using the same continuous dynamics as
``export_tvc_ode_model`` in ``tvc_traj_opt_acados.py`` (ZYX Euler, ``R_y(th_p) R_x(th_r)`` thrust).

Discretization follows the spirit of Spannagl et al., arXiv:2103.04709: pseudo-time τ ∈ [0,1],
optimize scalar ``t_f``, forward Euler on a fixed grid ``N``.

**Objective (discrete):** weighted thrust magnitude (commanded, or **actual** ``T`` if actuators) +
optional quadratic penalty on command increments when **not** using actuator dynamics (otherwise the
lag provides smoothing).

**Velocity:** hard box ``|v_x|,|v_y|,|v_z| ≤ v_max`` at every node (no slack); matches typical GUI axes.

**Optional:** first-order actuator lag on ``[th_p, th_r, T, τ_yaw]`` (same idea as Acados
``use_actuator_dynamics``).

**Numerical note:** avoid ``norm_2`` in constraints where gradients are singular (use squared forms).

Dependencies: numpy, casadi, matplotlib (optional dashboard); **acados** optional for the fast
``--nlp-solver acados`` path (same SQP + HPIPM stack as ``tvc_traj_opt_acados.py``). CasADi IPOPT
/sqpmethod remain available as fallbacks.

Example:
  python scripts/tvc_traj_opt_acados_min_time.py
  python scripts/tvc_traj_opt_acados_min_time.py --nlp-solver ipopt
  python scripts/tvc_traj_opt_acados_min_time.py --soft-terminal --no-actuator
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tvc_common import euler_to_quat_wxyz

try:
    import casadi as ca
except ImportError as e:
    print("CasADi is required: pip install casadi", file=sys.stderr)
    raise SystemExit(1) from e

# Small eps for smooth ‖T‖ in objective (avoids line-search artifacts at ‖T‖→0).
_T_NORM_OBJ_EPS = 1e-8


class _CostLogger:
    """Minimal logger compatible with ``tvc_traj_gui_plots.draw_cost_panel``."""

    __slots__ = ("costs",)

    def __init__(self, costs: List[float]):
        self.costs = list(costs)


def _solver_report_from_opti_sol(
    sol: Any,
    N: int,
    tf_opt: float,
    wall_time_s: float,
    nlp_solver: str = "ipopt",
) -> Dict[str, Any]:
    """Extract iteration objectives / status from CasADi ``Opti.solve()`` ``stats``."""
    st = sol.stats()
    iter_count = int(st.get("iter_count", 0))
    cost_hist: List[float] = []
    it = st.get("iterations")
    if isinstance(it, dict) and "obj" in it:
        cost_hist = [float(x) for x in it["obj"]]
    elif isinstance(it, list):
        for row in it:
            if isinstance(row, dict) and "obj" in row:
                cost_hist.append(float(row["obj"]))
    cost_opt = float(cost_hist[-1]) if cost_hist else float("nan")
    return {
        "nlp_solver": str(nlp_solver),
        "success": bool(st.get("success", False)),
        "return_status": str(st.get("return_status", "")),
        "stats": st,
        "iter_count": iter_count,
        "cost_history": cost_hist,
        "cost_optimal": cost_opt,
        "dt_grid": float(tf_opt) / float(N) if N > 0 else 0.0,
        "wall_time_s": float(wall_time_s),
        "time_per_iter_s": float(wall_time_s) / max(iter_count, 1),
    }


def _configure_opti_nlp_solver(opti: ca.Opti, nlp_solver: str) -> str:
    """
    Attach CasADi NLP backend to ``opti``: ``ipopt`` (interior point) or ``sqpmethod`` (SQP + QP).

    SQP uses bundled ``qrqp`` for QP subproblems (no extra install). For indefinite QPs you may
    switch to ``qpoases`` in code if your CasADi build supports it.
    """
    name = str(nlp_solver).lower().strip()
    if name == "ipopt":
        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 4000,
                "ipopt.bound_relax_factor": 1e-8,
            },
        )
    elif name == "sqpmethod":
        opti.solver(
            "sqpmethod",
            {
                "print_time": 0,
                "max_iter": 4000,
                "qpsol": "qrqp",
            },
        )
    else:
        raise ValueError(
            f"Unknown nlp_solver={nlp_solver!r}; use 'acados', 'ipopt', or 'sqpmethod'"
        )
    return name


def _Rx_fftf(a: ca.SX) -> ca.SX:
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, c, -s),
        ca.horzcat(0, s, c),
    )


def _Ry_fftf(a: ca.SX) -> ca.SX:
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(
        ca.horzcat(c, 0, s),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-s, 0, c),
    )


def _tvc_expl_phys(
    m: float,
    I_m: np.ndarray,
    r_thrust: np.ndarray,
    g: float,
    state_phys: ca.SX,
    th_p: ca.SX,
    th_r: ca.SX,
    T: ca.SX,
    tau_yaw: ca.SX,
) -> ca.SX:
    """Matches ``export_tvc_ode_model`` translational/rotational subset (12 states)."""
    I_m = np.asarray(I_m, dtype=float).reshape(3, 3)
    r_thrust = np.asarray(r_thrust, dtype=float).reshape(3)
    phi = state_phys[3]
    theta = state_phys[4]
    psi = state_phys[5]
    vx, vy, vz = state_phys[6], state_phys[7], state_phys[8]
    wx, wy, wz = state_phys[9], state_phys[10], state_phys[11]
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi), ca.sin(psi)
    Rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi, cphi),
    )
    Ry = ca.vertcat(
        ca.horzcat(cth, 0, sth),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-sth, 0, cth),
    )
    Rz = ca.vertcat(
        ca.horzcat(cpsi, -spsi, 0),
        ca.horzcat(spsi, cpsi, 0),
        ca.horzcat(0, 0, 1),
    )
    R = Rz @ Ry @ Rx
    Rtvc = _Ry_fftf(th_p) @ _Rx_fftf(th_r)
    Fb = Rtvc @ ca.vertcat(0, 0, T)
    Fw = R @ Fb
    Fg = ca.vertcat(0, 0, -g * float(m))
    v_dot = (Fw + Fg) / float(m)
    r_sx = ca.SX(r_thrust)
    tau_b = ca.cross(r_sx, Fb) + ca.vertcat(0, 0, tau_yaw)
    I_sx = ca.SX(I_m)
    Iinv = ca.inv(I_sx)
    w_vec = ca.vertcat(wx, wy, wz)
    w_dot = Iinv @ (tau_b - ca.cross(w_vec, I_sx @ w_vec))
    G = ca.vertcat(
        ca.horzcat(1, sphi * sth / cth, cphi * sth / cth),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi / cth, cphi / cth),
    )
    euler_dot = G @ w_vec
    p_dot = ca.vertcat(vx, vy, vz)
    return ca.vertcat(p_dot, euler_dot, v_dot, w_dot)


def build_tvc_rocket_dynamics_casadi(
    m: float,
    I_m: np.ndarray,
    r_thrust: np.ndarray,
    g: float,
    use_actuator_dynamics: bool,
    actuator_tau: Optional[np.ndarray],
) -> Tuple[ca.Function, int]:
    """
    Same continuous dynamics as ``tvc_traj_opt_acados.export_tvc_ode_model`` (no Acados types).
    Returns ``(f(x,u), nx)`` with ``f`` the right-hand side ``ẋ``.
    """
    if use_actuator_dynamics:
        tau_a = np.asarray(
            actuator_tau if actuator_tau is not None else [0.05, 0.05, 0.05, 0.05],
            dtype=float,
        ).reshape(4)
        inv_tau = 1.0 / np.maximum(tau_a, 1e-6)
        nx = 16
        x = ca.SX.sym("x", nx)
        u_cmd = ca.SX.sym("u", 4)
        phys = x[0:12]
        u_act = x[12:16]
        f12 = _tvc_expl_phys(
            m, I_m, r_thrust, g, phys, u_act[0], u_act[1], u_act[2], u_act[3]
        )
        udot = ca.vertcat(
            *[
                (u_cmd[i] - u_act[i]) * float(inv_tau[i])
                for i in range(4)
            ]
        )
        xdot = ca.vertcat(f12, udot)
    else:
        nx = 12
        x = ca.SX.sym("x", nx)
        u_cmd = ca.SX.sym("u", 4)
        phys = x
        f12 = _tvc_expl_phys(
            m, I_m, r_thrust, g, phys, u_cmd[0], u_cmd[1], u_cmd[2], u_cmd[3]
        )
        xdot = f12
    return ca.Function("f_tvc", [x, u_cmd], [xdot]), nx


def _setup_acados_env_min_time() -> None:
    """Match ``tvc_traj_opt_acados._setup_acados_env`` so HPIPM/QP shared libs resolve."""
    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if not acados_root:
        try:
            import acados_template

            pkg_path = Path(acados_template.__file__).resolve().parent
            for _ in range(3):
                pkg_path = pkg_path.parent
                if (pkg_path / "lib").exists():
                    acados_root = str(pkg_path)
                    break
        except Exception:
            pass
    if acados_root:
        lib_path = os.path.join(acados_root, "lib")
        if os.path.isdir(lib_path):
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_path not in ld_path.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = (
                    lib_path + (os.pathsep + ld_path if ld_path else "")
                )
            if "ACADOS_SOURCE_DIR" not in os.environ:
                os.environ["ACADOS_SOURCE_DIR"] = acados_root


def _solve_fftf_tvc_acados_act(
    N: int,
    m: float,
    p0: np.ndarray,
    v0: np.ndarray,
    pf: np.ndarray,
    vf: np.ndarray,
    ptol: float,
    vtol: float,
    lambda1_control_smooth: float,
    vmax: float,
    T_min: float,
    T_max: float,
    theta_max_deg: float,
    udot_max: float,
    tf_bounds: Tuple[float, float],
    use_glideslope_ascent: bool,
    gamma_glide_deg: float,
    g_mag: float,
    min_tf_regularization: float,
    exact_terminal: bool,
    I_m: np.ndarray,
    r_thrust: np.ndarray,
    actuator_tau: Optional[np.ndarray],
    tvc_euler_max_deg: Tuple[float, float, float],
    tvc_w_max: float,
    tvc_tau_yaw_lim: float,
    x0f: np.ndarray,
    lambda_yaw: float,
    psi_ref: float,
    acados_nlp_iter: int,
    acados_qp_solver: str,
    u_hover: np.ndarray,
    export_subdir: str = "",
    verbose_solve: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """
    Acados FFTF with first-order actuator: state ``[x_phys(12), u_act(4), t_f(1)]`` (nx=17).
    Running cost uses ‖T_act‖ only (no ``u_cmd - u_cmd_prev``); smoothing is implicit in actuator ODE.
    """
    export_subdir = str(export_subdir or "").strip()
    _setup_acados_env_min_time()
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    x0f = np.asarray(x0f, dtype=float).flatten()
    if x0f.size != 16:
        raise ValueError(f"actuator path expects x0f length 16, got {x0f.size}")

    phi_mx = float(np.radians(tvc_euler_max_deg[0]))
    th_mx = float(np.radians(tvc_euler_max_deg[1]))
    psi_mx = float(np.radians(tvc_euler_max_deg[2]))
    th_gimbal = float(np.radians(theta_max_deg))
    T_lo, T_hi = float(T_min), float(T_max)
    tw = float(tvc_tau_yaw_lim)
    wmx = float(tvc_w_max)
    vbx = float(vmax)
    tf_lo, tf_hi = float(tf_bounds[0]), float(tf_bounds[1])
    Nf = float(N)

    tau_a = np.asarray(
        actuator_tau if actuator_tau is not None else [0.05, 0.05, 0.05, 0.05],
        dtype=float,
    ).reshape(4)
    inv_tau = 1.0 / np.maximum(tau_a, 1e-6)
    inv_tau_ca = ca.vertcat(*[float(inv_tau[i]) for i in range(4)])

    x_phys = ca.SX.sym("x_phys", 12)
    u_act = ca.SX.sym("u_act", 4)
    tf_s = ca.SX.sym("tf")
    x_full = ca.vertcat(x_phys, u_act, tf_s)
    u_cmd = ca.SX.sym("u", 4)

    f12 = _tvc_expl_phys(
        m, I_m, r_thrust, g_mag, x_phys, u_act[0], u_act[1], u_act[2], u_act[3]
    )
    tf_safe = ca.fmax(tf_s, 1e-6)
    u_act_dot = tf_s * (u_cmd - u_act) * inv_tau_ca
    f_expl = ca.vertcat(tf_s * f12, u_act_dot, 0)

    psi = x_phys[5]
    wx, wy, wz = x_phys[9], x_phys[10], x_phys[11]
    T_use = u_act[2]
    Tmag = ca.sqrt(T_use**2 + _T_NORM_OBJ_EPS)
    h_step = tf_safe / Nf
    la_y = float(lambda_yaw)
    psr = float(psi_ref)
    running = h_step * Tmag
    if la_y > 0:
        running += h_step * la_y * (psi - psr) ** 2
    term_e = 0
    if min_tf_regularization > 0:
        term_e += float(min_tf_regularization) * tf_s
    if la_y > 0:
        term_e += h_step * la_y * (psi - psr) ** 2

    model = AcadosModel()
    model.name = "tvc_fftf_sqp_act"
    model.x = x_full
    model.xdot = ca.SX.sym("xdot", 17)
    model.u = u_cmd
    model.f_expl_expr = f_expl
    model.cost_expr_ext_cost = running
    model.cost_expr_ext_cost_e = term_e

    vx, vy, vz = x_phys[6], x_phys[7], x_phys[8]
    phi, theta, psi_h = x_phys[3], x_phys[4], x_phys[5]
    w_sq = wx**2 + wy**2 + wz**2 - wmx**2

    h_rows = [vx, vy, vz, phi, theta, psi_h, w_sq]
    lh = np.array(
        [-vbx, -vbx, -vbx, -phi_mx, -th_mx, -psi_mx, -1e15],
        dtype=float,
    )
    uh = np.array(
        [vbx, vbx, vbx, phi_mx, th_mx, psi_mx, 0.0],
        dtype=float,
    )

    if use_glideslope_ascent:
        tan_g = float(np.tan(np.radians(gamma_glide_deg)))
        tan_gsq = tan_g**2
        pk = x_phys[0:3]
        dx = pk[0] - float(p0[0])
        dy = pk[1] - float(p0[1])
        dz = pk[2] - float(p0[2])
        h_rows.append(-dz)
        h_rows.append(tan_gsq * (dx**2 + dy**2) - dz**2)
        lh = np.append(lh, [-1e15, -1e15])
        uh = np.append(uh, [0.0, 0.0])

    model.con_h_expr = ca.vertcat(*h_rows)
    ocp = AcadosOcp()
    ocp.model = model
    script_dir = Path(__file__).resolve().parent
    _sub = f"_{export_subdir}" if export_subdir else ""
    code_export_dir = str(
        script_dir / "c_generated_code" / f"fftf_acados_N{N}_act{_sub}"
    )
    json_path = str(Path(code_export_dir) / f"{model.name}.json")
    try:
        ocp.code_gen_opts.code_export_directory = code_export_dir
    except AttributeError:
        ocp.code_export_directory = code_export_dir
    try:
        ocp.code_gen_opts.json_file = json_path
    except AttributeError:
        ocp.json_file = json_path

    ocp.solver_options.N_horizon = int(N)
    ocp.solver_options.tf = 1.0
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.lbu = np.array([-th_gimbal, -th_gimbal, T_lo, -tw], dtype=float)
    ocp.constraints.ubu = np.array([th_gimbal, th_gimbal, T_hi, tw], dtype=float)
    ocp.constraints.idxbu = np.arange(4, dtype=int)
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh

    ocp.constraints.idxbx_0 = np.arange(16, dtype=int)
    ocp.constraints.lbx_0 = np.concatenate([x0f[:12], x0f[12:16]])
    ocp.constraints.ubx_0 = ocp.constraints.lbx_0.copy()

    ocp.constraints.idxbx = np.array([12, 13, 14, 15, 16], dtype=int)
    ocp.constraints.lbx = np.array(
        [-th_gimbal, -th_gimbal, T_lo, -tw, tf_lo], dtype=float
    )
    ocp.constraints.ubx = np.array(
        [th_gimbal, th_gimbal, T_hi, tw, tf_hi], dtype=float
    )

    xg_e = np.concatenate(
        [pf, [0.0, 0.0, float(psi_ref)], vf.ravel(), np.zeros(3)], dtype=float
    )
    if exact_terminal:
        ocp.constraints.idxbx_e = np.concatenate(
            [np.arange(12, dtype=int), np.array([12, 13, 14, 15, 16], dtype=int)]
        )
        ocp.constraints.lbx_e = np.concatenate(
            [
                xg_e,
                np.array([-th_gimbal, -th_gimbal, T_lo, -tw], dtype=float),
                [tf_lo],
            ]
        )
        ocp.constraints.ubx_e = np.concatenate(
            [
                xg_e,
                np.array([th_gimbal, th_gimbal, T_hi, tw], dtype=float),
                [tf_hi],
            ]
        )
    else:
        ocp.constraints.idxbx_e = np.array(
            [3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16], dtype=int
        )
        ocp.constraints.lbx_e = np.array(
            [
                0.0,
                0.0,
                float(psi_ref),
                0.0,
                0.0,
                0.0,
                -th_gimbal,
                -th_gimbal,
                T_lo,
                -tw,
                tf_lo,
            ],
            dtype=float,
        )
        ocp.constraints.ubx_e = np.array(
            [
                0.0,
                0.0,
                float(psi_ref),
                0.0,
                0.0,
                0.0,
                th_gimbal,
                th_gimbal,
                T_hi,
                tw,
                tf_hi,
            ],
            dtype=float,
        )
        ep = x_phys[0:3] - ca.vertcat(
            float(pf[0]), float(pf[1]), float(pf[2])
        )
        ev = x_phys[6:9] - ca.vertcat(
            float(vf[0]), float(vf[1]), float(vf[2])
        )

    h_e_rows = [
        vx,
        vy,
        vz,
        phi,
        theta,
        psi_h,
        w_sq,
    ]
    lh_e2 = np.array(
        [-vbx, -vbx, -vbx, -phi_mx, -th_mx, -psi_mx, -1e15],
        dtype=float,
    )
    uh_e2 = np.array(
        [vbx, vbx, vbx, phi_mx, th_mx, psi_mx, 0.0],
        dtype=float,
    )
    if use_glideslope_ascent:
        tan_gsq_e = float(np.tan(np.radians(gamma_glide_deg))) ** 2
        pk_e = x_phys[0:3]
        dx_e = pk_e[0] - float(p0[0])
        dy_e = pk_e[1] - float(p0[1])
        dz_e = pk_e[2] - float(p0[2])
        h_e_rows.append(-dz_e)
        h_e_rows.append(tan_gsq_e * (dx_e**2 + dy_e**2) - dz_e**2)
        lh_e2 = np.append(lh_e2, [-1e15, -1e15])
        uh_e2 = np.append(uh_e2, [0.0, 0.0])

    if exact_terminal:
        model.con_h_expr_e = tf_s * 0
        ocp.constraints.lh_e = np.array([0.0], dtype=float)
        ocp.constraints.uh_e = np.array([0.0], dtype=float)
    else:
        model.con_h_expr_e = ca.vertcat(
            ca.dot(ep, ep) - float(ptol) ** 2,
            ca.dot(ev, ev) - float(vtol) ** 2,
            *h_e_rows,
        )
        ocp.constraints.lh_e = np.concatenate(
            [np.array([-1e15, -1e15], dtype=float), lh_e2]
        )
        ocp.constraints.uh_e = np.concatenate(
            [np.array([0.0, 0.0], dtype=float), uh_e2]
        )

    ocp.solver_options.qp_solver = acados_qp_solver
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 2 if verbose_solve else 0
    ocp.solver_options.nlp_solver_max_iter = int(acados_nlp_iter)

    t_build0 = time.perf_counter()
    try:
        try:
            solver = AcadosOcpSolver(ocp, verbose=False, build=True, generate=True)
        except TypeError:
            solver = AcadosOcpSolver(ocp, verbose=False)
    except OSError as e:
        raise RuntimeError(
            f"AcadosOcpSolver failed ({e}). Set LD_LIBRARY_PATH to acados lib "
            "or use ./scripts/run_acados.sh"
        ) from e
    build_wall = time.perf_counter() - t_build0

    tf0 = 0.5 * (tf_lo + tf_hi)
    u_act0 = np.asarray(x0f[12:16], dtype=float).flatten()
    for i in range(N + 1):
        a = float(i) / float(N)
        guess = np.zeros(17, dtype=float)
        guess[0:3] = (1 - a) * np.asarray(p0).flatten()[:3] + a * np.asarray(pf).flatten()[:3]
        guess[3:6] = [0.0, 0.0, float(psi_ref)]
        guess[6:9] = (1 - a) * np.asarray(v0).flatten()[:3] + a * np.asarray(vf).flatten()[:3]
        guess[9:12] = 0.0
        guess[12:16] = u_act0
        guess[16] = tf0
        solver.set(i, "x", guess)
    for i in range(N):
        solver.set(i, "u", u_hover)

    t_solve0 = time.perf_counter()
    status = solver.solve()
    wall_time_s = time.perf_counter() - t_solve0

    try:
        sqp_iter = int(solver.get_stats("sqp_iter"))
    except Exception:
        sqp_iter = int(solver.get_stats("nlp_iter"))
    try:
        cost_opt = float(solver.get_cost())
    except Exception:
        cost_opt = float("nan")

    X_opt = np.zeros((16, N + 1))
    U_opt = np.zeros((4, N))
    for i in range(N + 1):
        xv = np.asarray(solver.get(i, "x"), dtype=float).flatten()
        X_opt[:, i] = xv[:16]
    for i in range(N):
        U_opt[:, i] = np.asarray(solver.get(i, "u"), dtype=float).flatten()
    tf_opt = float(np.asarray(solver.get(0, "x"), dtype=float).flatten()[16])

    tau = np.linspace(0.0, 1.0, N + 1)
    del solver
    gc.collect()

    rep = {
        "nlp_solver": "acados_sqp",
        "success": int(status) == 0,
        "return_status": f"status_{status}",
        "stats": {"sqp_iter": sqp_iter, "build_wall_s": build_wall},
        "iter_count": sqp_iter,
        "cost_history": [cost_opt],
        "cost_optimal": cost_opt,
        "dt_grid": float(tf_opt) / float(N) if N > 0 else 0.0,
        "wall_time_s": float(wall_time_s),
        "time_per_iter_s": float(wall_time_s) / max(sqp_iter, 1),
    }
    return tau, X_opt, U_opt, tf_opt, rep


def _solve_fftf_tvc_acados_noact(
    N: int,
    m: float,
    p0: np.ndarray,
    v0: np.ndarray,
    pf: np.ndarray,
    vf: np.ndarray,
    ptol: float,
    vtol: float,
    lambda1_control_smooth: float,
    vmax: float,
    T_min: float,
    T_max: float,
    theta_max_deg: float,
    udot_max: float,
    tf_bounds: Tuple[float, float],
    use_glideslope_ascent: bool,
    gamma_glide_deg: float,
    g_mag: float,
    min_tf_regularization: float,
    exact_terminal: bool,
    I_m: np.ndarray,
    r_thrust: np.ndarray,
    tvc_euler_max_deg: Tuple[float, float, float],
    tvc_w_max: float,
    tvc_tau_yaw_lim: float,
    x0f: np.ndarray,
    u_hover: np.ndarray,
    lambda_yaw: float,
    psi_ref: float,
    acados_nlp_iter: int = 150,
    acados_qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    u_cmd_prev0: Optional[np.ndarray] = None,
    export_subdir: str = "",
    verbose_solve: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """
    Acados FFTF without actuator: state ``[x_phys(12), u_cmd_prev(4), t_f(1)]`` (nx=17).
    """
    export_subdir = str(export_subdir or "").strip()
    u_prev_init = (
        np.asarray(u_cmd_prev0, dtype=float).flatten()
        if u_cmd_prev0 is not None
        else np.asarray(u_hover, dtype=float).flatten()
    )
    if u_prev_init.size != 4:
        raise ValueError("u_cmd_prev0 must have length 4")
    _setup_acados_env_min_time()
    try:
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    except ImportError as e:
        raise ImportError(
            "acados not importable for nlp_solver=acados. Install acados and extend "
            "LD_LIBRARY_PATH (see tvc_traj_opt_acados.py header), or use nlp_solver=ipopt."
        ) from e

    p0 = np.asarray(p0, dtype=float).reshape(3)
    v0 = np.asarray(v0, dtype=float).reshape(3)
    pf = np.asarray(pf, dtype=float).reshape(3)
    vf = np.asarray(vf, dtype=float).reshape(3)
    x0f = np.asarray(x0f, dtype=float).flatten()
    if x0f.size != 12:
        raise ValueError(f"no-actuator path expects x0f length 12, got {x0f.size}")

    phi_mx = float(np.radians(tvc_euler_max_deg[0]))
    th_mx = float(np.radians(tvc_euler_max_deg[1]))
    psi_mx = float(np.radians(tvc_euler_max_deg[2]))
    th_gimbal = float(np.radians(theta_max_deg))
    T_lo, T_hi = float(T_min), float(T_max)
    tw = float(tvc_tau_yaw_lim)
    wmx = float(tvc_w_max)
    vbx = float(vmax)
    tf_lo, tf_hi = float(tf_bounds[0]), float(tf_bounds[1])
    Nf = float(N)

    x_phys = ca.SX.sym("x_phys", 12)
    u_prev = ca.SX.sym("u_prev", 4)
    tf_s = ca.SX.sym("tf")
    x_full = ca.vertcat(x_phys, u_prev, tf_s)
    u = ca.SX.sym("u", 4)
    f12 = _tvc_expl_phys(
        m, I_m, r_thrust, g_mag, x_phys, u[0], u[1], u[2], u[3]
    )
    tf_safe = ca.fmax(tf_s, 1e-6)
    u_prev_dot = (u - u_prev) * (Nf / tf_safe)
    f_expl = ca.vertcat(tf_s * f12, u_prev_dot, 0)

    psi = x_phys[5]
    wx, wy, wz = x_phys[9], x_phys[10], x_phys[11]
    T_use = u[2]
    Tmag = ca.sqrt(T_use**2 + _T_NORM_OBJ_EPS)
    du = u - u_prev
    h_step = tf_safe / Nf
    la_y = float(lambda_yaw)
    psr = float(psi_ref)
    running = h_step * (Tmag + float(lambda1_control_smooth) * ca.dot(du, du))
    if la_y > 0:
        running += h_step * la_y * (psi - psr) ** 2
    term_e = 0
    if min_tf_regularization > 0:
        term_e += float(min_tf_regularization) * tf_s
    if la_y > 0:
        term_e += h_step * la_y * (psi - psr) ** 2

    model = AcadosModel()
    model.name = "tvc_fftf_sqp"
    model.x = x_full
    model.xdot = ca.SX.sym("xdot", 17)
    model.u = u
    model.f_expl_expr = f_expl
    model.cost_expr_ext_cost = running
    model.cost_expr_ext_cost_e = term_e

    vx, vy, vz = x_phys[6], x_phys[7], x_phys[8]
    phi, theta, psi_h = x_phys[3], x_phys[4], x_phys[5]
    w_sq = wx**2 + wy**2 + wz**2 - wmx**2

    h_rows = [vx, vy, vz, phi, theta, psi_h, w_sq]
    lh = np.array(
        [-vbx, -vbx, -vbx, -phi_mx, -th_mx, -psi_mx, -1e15],
        dtype=float,
    )
    uh = np.array(
        [vbx, vbx, vbx, phi_mx, th_mx, psi_mx, 0.0],
        dtype=float,
    )
    udot_lim_sq = (float(udot_max) * tf_s / Nf) ** 2
    du_rate = ca.dot(du, du) - udot_lim_sq
    h_rows.append(du_rate)
    lh = np.append(lh, -1e15)
    uh = np.append(uh, 0.0)

    if use_glideslope_ascent:
        tan_g = float(np.tan(np.radians(gamma_glide_deg)))
        tan_gsq = tan_g**2
        pk = x_phys[0:3]
        dx = pk[0] - float(p0[0])
        dy = pk[1] - float(p0[1])
        dz = pk[2] - float(p0[2])
        h_rows.append(-dz)
        h_rows.append(tan_gsq * (dx**2 + dy**2) - dz**2)
        lh = np.append(lh, [-1e15, -1e15])
        uh = np.append(uh, [0.0, 0.0])

    model.con_h_expr = ca.vertcat(*h_rows)
    ocp = AcadosOcp()
    ocp.model = model
    script_dir = Path(__file__).resolve().parent
    _sub = f"_{export_subdir}" if export_subdir else ""
    code_export_dir = str(script_dir / "c_generated_code" / f"fftf_acados_N{N}{_sub}")
    json_path = str(Path(code_export_dir) / f"{model.name}.json")
    try:
        ocp.code_gen_opts.code_export_directory = code_export_dir
    except AttributeError:
        ocp.code_export_directory = code_export_dir
    try:
        ocp.code_gen_opts.json_file = json_path
    except AttributeError:
        ocp.json_file = json_path

    ocp.solver_options.N_horizon = int(N)
    ocp.solver_options.tf = 1.0
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.lbu = np.array([-th_gimbal, -th_gimbal, T_lo, -tw], dtype=float)
    ocp.constraints.ubu = np.array([th_gimbal, th_gimbal, T_hi, tw], dtype=float)
    ocp.constraints.idxbu = np.arange(4, dtype=int)
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh

    ocp.constraints.idxbx_0 = np.arange(16, dtype=int)
    ocp.constraints.lbx_0 = np.concatenate([x0f[:12], u_prev_init])
    ocp.constraints.ubx_0 = np.concatenate([x0f[:12], u_prev_init])

    ocp.constraints.idxbx = np.array([16], dtype=int)
    ocp.constraints.lbx = np.array([tf_lo], dtype=float)
    ocp.constraints.ubx = np.array([tf_hi], dtype=float)

    xg_e = np.concatenate(
        [pf, [0.0, 0.0, float(psi_ref)], vf.ravel(), np.zeros(3)], dtype=float
    )
    if exact_terminal:
        ocp.constraints.idxbx_e = np.arange(12, dtype=int)
        ocp.constraints.lbx_e = xg_e.copy()
        ocp.constraints.ubx_e = xg_e.copy()
    else:
        ocp.constraints.idxbx_e = np.array([3, 4, 5, 9, 10, 11], dtype=int)
        ocp.constraints.lbx_e = np.array(
            [0.0, 0.0, float(psi_ref), 0.0, 0.0, 0.0], dtype=float
        )
        ocp.constraints.ubx_e = ocp.constraints.lbx_e.copy()
        ep = x_phys[0:3] - ca.vertcat(
            float(pf[0]), float(pf[1]), float(pf[2])
        )
        ev = x_phys[6:9] - ca.vertcat(
            float(vf[0]), float(vf[1]), float(vf[2])
        )

    ocp.constraints.idxbx_e = np.concatenate(
        [ocp.constraints.idxbx_e, np.array([16], dtype=int)]
    )
    ocp.constraints.lbx_e = np.concatenate([ocp.constraints.lbx_e, [tf_lo]])
    ocp.constraints.ubx_e = np.concatenate([ocp.constraints.ubx_e, [tf_hi]])

    h_e_rows = [
        vx,
        vy,
        vz,
        phi,
        theta,
        psi_h,
        w_sq,
    ]
    lh_e2 = np.array(
        [-vbx, -vbx, -vbx, -phi_mx, -th_mx, -psi_mx, -1e15],
        dtype=float,
    )
    uh_e2 = np.array(
        [vbx, vbx, vbx, phi_mx, th_mx, psi_mx, 0.0],
        dtype=float,
    )
    if use_glideslope_ascent:
        tan_gsq_e = float(np.tan(np.radians(gamma_glide_deg))) ** 2
        pk_e = x_phys[0:3]
        dx_e = pk_e[0] - float(p0[0])
        dy_e = pk_e[1] - float(p0[1])
        dz_e = pk_e[2] - float(p0[2])
        h_e_rows.append(-dz_e)
        h_e_rows.append(tan_gsq_e * (dx_e**2 + dy_e**2) - dz_e**2)
        lh_e2 = np.append(lh_e2, [-1e15, -1e15])
        uh_e2 = np.append(uh_e2, [0.0, 0.0])

    if exact_terminal:
        # Physical terminal fixed via idxbx_e; keep a trivial nl constraint (nh_e>0)
        # for older acados builds that expect terminal h to be dimensioned.
        model.con_h_expr_e = tf_s * 0
        ocp.constraints.lh_e = np.array([0.0], dtype=float)
        ocp.constraints.uh_e = np.array([0.0], dtype=float)
    else:
        model.con_h_expr_e = ca.vertcat(
            ca.dot(ep, ep) - float(ptol) ** 2,
            ca.dot(ev, ev) - float(vtol) ** 2,
            *h_e_rows,
        )
        ocp.constraints.lh_e = np.concatenate(
            [np.array([-1e15, -1e15], dtype=float), lh_e2]
        )
        ocp.constraints.uh_e = np.concatenate(
            [np.array([0.0, 0.0], dtype=float), uh_e2]
        )

    ocp.solver_options.qp_solver = acados_qp_solver
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 2 if verbose_solve else 0
    ocp.solver_options.nlp_solver_max_iter = int(acados_nlp_iter)

    t_build0 = time.perf_counter()
    try:
        try:
            solver = AcadosOcpSolver(ocp, verbose=False, build=True, generate=True)
        except TypeError:
            solver = AcadosOcpSolver(ocp, verbose=False)
    except OSError as e:
        raise RuntimeError(
            f"AcadosOcpSolver failed ({e}). Set LD_LIBRARY_PATH to acados lib "
            "or use ./scripts/run_acados.sh"
        ) from e
    build_wall = time.perf_counter() - t_build0

    tf0 = 0.5 * (tf_lo + tf_hi)
    for i in range(N + 1):
        a = float(i) / float(N)
        guess = np.zeros(17, dtype=float)
        guess[0:3] = (1 - a) * p0 + a * pf
        guess[3:6] = [0.0, 0.0, float(psi_ref)]
        guess[6:9] = (1 - a) * v0 + a * vf
        guess[9:12] = 0.0
        guess[12:16] = u_prev_init
        guess[16] = tf0
        solver.set(i, "x", guess)
    for i in range(N):
        solver.set(i, "u", u_hover)

    t_solve0 = time.perf_counter()
    status = solver.solve()
    wall_time_s = time.perf_counter() - t_solve0

    try:
        sqp_iter = int(solver.get_stats("sqp_iter"))
    except Exception:
        sqp_iter = int(solver.get_stats("nlp_iter"))
    try:
        cost_opt = float(solver.get_cost())
    except Exception:
        cost_opt = float("nan")

    X_opt = np.zeros((12, N + 1))
    U_opt = np.zeros((4, N))
    for i in range(N + 1):
        xv = np.asarray(solver.get(i, "x"), dtype=float).flatten()
        X_opt[:, i] = xv[:12]
    for i in range(N):
        U_opt[:, i] = np.asarray(solver.get(i, "u"), dtype=float).flatten()
    tf_opt = float(np.asarray(solver.get(0, "x"), dtype=float).flatten()[16])

    tau = np.linspace(0.0, 1.0, N + 1)
    del solver
    gc.collect()

    rep = {
        "nlp_solver": "acados_sqp",
        "success": int(status) == 0,
        "return_status": f"status_{status}",
        "stats": {"sqp_iter": sqp_iter, "build_wall_s": build_wall},
        "iter_count": sqp_iter,
        "cost_history": [cost_opt],
        "cost_optimal": cost_opt,
        "dt_grid": float(tf_opt) / float(N) if N > 0 else 0.0,
        "wall_time_s": float(wall_time_s),
        "time_per_iter_s": float(wall_time_s) / max(sqp_iter, 1),
    }
    return tau, X_opt, U_opt, tf_opt, rep


def _solve_fftf_tvc_acados(
    N: int,
    m: float,
    p0: np.ndarray,
    v0: np.ndarray,
    pf: np.ndarray,
    vf: np.ndarray,
    ptol: float,
    vtol: float,
    lambda1_control_smooth: float,
    vmax: float,
    T_min: float,
    T_max: float,
    theta_max_deg: float,
    udot_max: float,
    tf_bounds: Tuple[float, float],
    use_glideslope_ascent: bool,
    gamma_glide_deg: float,
    g_mag: float,
    min_tf_regularization: float,
    exact_terminal: bool,
    I_m: np.ndarray,
    r_thrust: np.ndarray,
    use_actuator_dynamics: bool,
    actuator_tau: Optional[np.ndarray],
    tvc_euler_max_deg: Tuple[float, float, float],
    tvc_w_max: float,
    tvc_tau_yaw_lim: float,
    x0_tvc: Optional[np.ndarray],
    lambda_yaw: float,
    psi_ref: float,
    acados_nlp_iter: int = 150,
    acados_qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    u_cmd_prev0: Optional[np.ndarray] = None,
    export_subdir: str = "",
    verbose_solve: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Assemble x0 and run acados FFTF (always nx=17: middle 4 states are u_cmd_prev or u_act)."""
    export_subdir = str(export_subdir or "").strip()
    u_hover = np.array([0.0, 0.0, float(m) * g_mag, 0.0], dtype=float)
    p0a = np.asarray(p0, dtype=float).reshape(3)
    v0a = np.asarray(v0, dtype=float).reshape(3)
    if x0_tvc is not None:
        x0f = np.asarray(x0_tvc, dtype=float).flatten()
        if use_actuator_dynamics:
            if x0f.size != 16:
                raise ValueError(
                    f"x0_tvc must have length 16 with actuator, got {x0f.size}"
                )
        elif x0f.size != 12:
            raise ValueError(f"x0_tvc must have length 12, got {x0f.size}")
    else:
        x0f = np.concatenate([p0a, np.zeros(3), v0a, np.zeros(3)])
        if use_actuator_dynamics:
            x0f = np.concatenate([x0f, u_hover])
    if use_actuator_dynamics:
        return _solve_fftf_tvc_acados_act(
            N,
            m,
            p0,
            v0,
            pf,
            vf,
            ptol,
            vtol,
            lambda1_control_smooth,
            vmax,
            T_min,
            T_max,
            theta_max_deg,
            udot_max,
            tf_bounds,
            use_glideslope_ascent,
            gamma_glide_deg,
            g_mag,
            min_tf_regularization,
            exact_terminal,
            I_m,
            r_thrust,
            actuator_tau,
            tvc_euler_max_deg,
            tvc_w_max,
            tvc_tau_yaw_lim,
            x0f,
            lambda_yaw,
            psi_ref,
            acados_nlp_iter,
            acados_qp_solver,
            u_hover,
            export_subdir,
            verbose_solve,
        )
    return _solve_fftf_tvc_acados_noact(
        N,
        m,
        p0,
        v0,
        pf,
        vf,
        ptol,
        vtol,
        lambda1_control_smooth,
        vmax,
        T_min,
        T_max,
        theta_max_deg,
        udot_max,
        tf_bounds,
        use_glideslope_ascent,
        gamma_glide_deg,
        g_mag,
        min_tf_regularization,
        exact_terminal,
        I_m,
        r_thrust,
        tvc_euler_max_deg,
        tvc_w_max,
        tvc_tau_yaw_lim,
        x0f[:12],
        u_hover,
        lambda_yaw,
        psi_ref,
        acados_nlp_iter,
        acados_qp_solver,
        u_cmd_prev0=u_cmd_prev0,
        export_subdir=export_subdir,
        verbose_solve=verbose_solve,
    )


def _solve_fftf_tvc(
    N: int,
    m: float,
    p0: np.ndarray,
    v0: np.ndarray,
    pf: np.ndarray,
    vf: np.ndarray,
    ptol: float,
    vtol: float,
    lambda1_control_smooth: float,
    vmax: float,
    T_min: float,
    T_max: float,
    theta_max_deg: float,
    udot_max: float,
    tf_bounds: Tuple[float, float],
    use_glideslope_ascent: bool,
    gamma_glide_deg: float,
    g_mag: float,
    min_tf_regularization: float,
    exact_terminal: bool,
    I_m: np.ndarray,
    r_thrust: np.ndarray,
    use_actuator_dynamics: bool,
    actuator_tau: Optional[np.ndarray],
    tvc_euler_max_deg: Tuple[float, float, float],
    tvc_w_max: float,
    tvc_tau_yaw_lim: float,
    x0_tvc: Optional[np.ndarray],
    lambda_yaw: float,
    psi_ref: float,
    nlp_solver: str = "acados",
    acados_nlp_iter: int = 150,
    acados_qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    u_cmd_prev0: Optional[np.ndarray] = None,
    export_subdir: str = "",
    verbose_solve: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """FFTF-style NLP with 6-DoF TVC dynamics (and optional actuator states)."""
    key = str(nlp_solver).lower().strip()
    if key == "acados":
        return _solve_fftf_tvc_acados(
            N=N,
            m=m,
            p0=p0,
            v0=v0,
            pf=pf,
            vf=vf,
            ptol=ptol,
            vtol=vtol,
            lambda1_control_smooth=lambda1_control_smooth,
            vmax=vmax,
            T_min=T_min,
            T_max=T_max,
            theta_max_deg=theta_max_deg,
            udot_max=udot_max,
            tf_bounds=tf_bounds,
            use_glideslope_ascent=use_glideslope_ascent,
            gamma_glide_deg=gamma_glide_deg,
            g_mag=g_mag,
            min_tf_regularization=min_tf_regularization,
            exact_terminal=exact_terminal,
            I_m=I_m,
            r_thrust=r_thrust,
            use_actuator_dynamics=use_actuator_dynamics,
            actuator_tau=actuator_tau,
            tvc_euler_max_deg=tvc_euler_max_deg,
            tvc_w_max=tvc_w_max,
            tvc_tau_yaw_lim=tvc_tau_yaw_lim,
            x0_tvc=x0_tvc,
            lambda_yaw=lambda_yaw,
            psi_ref=psi_ref,
            acados_nlp_iter=acados_nlp_iter,
            acados_qp_solver=acados_qp_solver,
            u_cmd_prev0=u_cmd_prev0,
            export_subdir=export_subdir,
            verbose_solve=verbose_solve,
        )
    f_fn, nx = build_tvc_rocket_dynamics_casadi(
        m,
        I_m,
        r_thrust,
        g_mag,
        use_actuator_dynamics,
        actuator_tau,
    )
    opti = ca.Opti()
    tf = opti.variable()
    X = opti.variable(nx, N + 1)
    U = opti.variable(4, N)
    u_init_p = opti.parameter(4)

    opti.subject_to(tf >= float(tf_bounds[0]))
    opti.subject_to(tf <= float(tf_bounds[1]))
    h = tf / float(N)

    p0 = np.asarray(p0, dtype=float).reshape(3)
    v0 = np.asarray(v0, dtype=float).reshape(3)
    pf = np.asarray(pf, dtype=float).reshape(3)
    vf = np.asarray(vf, dtype=float).reshape(3)
    u_hover = np.array([0.0, 0.0, float(m) * g_mag, 0.0])
    if x0_tvc is not None:
        x0f = np.asarray(x0_tvc, dtype=float).flatten()
        if x0f.size != nx:
            raise ValueError(f"x0_tvc must have length {nx}, got {x0f.size}")
    else:
        x0f = np.concatenate([p0, np.zeros(3), v0, np.zeros(3)])
        if use_actuator_dynamics:
            x0f = np.concatenate([x0f, u_hover])
    opti.subject_to(X[:, 0] == x0f)

    phi_mx = float(np.radians(tvc_euler_max_deg[0]))
    th_mx = float(np.radians(tvc_euler_max_deg[1]))
    psi_mx = float(np.radians(tvc_euler_max_deg[2]))
    th_gimbal = float(np.radians(theta_max_deg))
    T_lo, T_hi = float(T_min), float(T_max)
    tw = float(tvc_tau_yaw_lim)
    wmx = float(tvc_w_max)
    vbx = float(vmax)

    cost = 0
    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]
        x_next = xk + h * f_fn(xk, uk)
        opti.subject_to(X[:, k + 1] == x_next)
        if use_actuator_dynamics:
            T_use = xk[14]
        else:
            T_use = uk[2]
        Tmag = ca.sqrt(T_use**2 + _T_NORM_OBJ_EPS)
        if use_actuator_dynamics:
            cost += h * Tmag
        else:
            u_prev = U[:, k - 1] if k > 0 else u_init_p
            du = uk - u_prev
            cost += h * (Tmag + lambda1_control_smooth * ca.dot(du, du))
    if min_tf_regularization > 0:
        cost += min_tf_regularization * tf

    if lambda_yaw > 0:
        la_y = float(lambda_yaw)
        psr = float(psi_ref)
        for k in range(N + 1):
            dpsi = X[5, k] - psr
            cost += h * la_y * dpsi * dpsi

    err_p = X[0:3, -1] - pf
    err_v = X[6:9, -1] - vf
    if exact_terminal:
        opti.subject_to(err_p == 0)
        opti.subject_to(err_v == 0)
    else:
        opti.subject_to(ca.sumsqr(err_p) <= float(ptol) ** 2)
        opti.subject_to(ca.sumsqr(err_v) <= float(vtol) ** 2)

    # Terminal roll/pitch level; yaw fixed to reference ψ_ref; no angular velocity.
    psr_t = float(psi_ref)
    opti.subject_to(X[3, -1] == 0)
    opti.subject_to(X[4, -1] == 0)
    opti.subject_to(X[5, -1] == psr_t)
    opti.subject_to(X[9:12, -1] == 0)

    for k in range(N + 1):
        for vi in range(3):
            opti.subject_to(X[6 + vi, k] <= vbx)
            opti.subject_to(X[6 + vi, k] >= -vbx)
        opti.subject_to(X[3, k] <= phi_mx)
        opti.subject_to(X[3, k] >= -phi_mx)
        opti.subject_to(X[4, k] <= th_mx)
        opti.subject_to(X[4, k] >= -th_mx)
        opti.subject_to(X[5, k] <= psi_mx)
        opti.subject_to(X[5, k] >= -psi_mx)
        wk = X[9:12, k]
        opti.subject_to(ca.sumsqr(wk) <= wmx * wmx)
        if use_actuator_dynamics:
            opti.subject_to(X[12, k] <= th_gimbal)
            opti.subject_to(X[12, k] >= -th_gimbal)
            opti.subject_to(X[13, k] <= th_gimbal)
            opti.subject_to(X[13, k] >= -th_gimbal)
            opti.subject_to(X[14, k] >= T_lo)
            opti.subject_to(X[14, k] <= T_hi)
            opti.subject_to(X[15, k] <= tw)
            opti.subject_to(X[15, k] >= -tw)

    for k in range(N):
        opti.subject_to(U[0, k] <= th_gimbal)
        opti.subject_to(U[0, k] >= -th_gimbal)
        opti.subject_to(U[1, k] <= th_gimbal)
        opti.subject_to(U[1, k] >= -th_gimbal)
        opti.subject_to(U[2, k] >= T_lo)
        opti.subject_to(U[2, k] <= T_hi)
        opti.subject_to(U[3, k] <= tw)
        opti.subject_to(U[3, k] >= -tw)
        if not use_actuator_dynamics:
            u_prev = U[:, k - 1] if k > 0 else u_init_p
            duc = U[:, k] - u_prev
            opti.subject_to(ca.sumsqr(duc) <= (float(udot_max) * h) ** 2)

    if use_glideslope_ascent:
        tan_g = float(np.tan(np.radians(gamma_glide_deg)))
        tan_gsq = tan_g**2
        for k in range(N + 1):
            pk = X[0:3, k]
            dx = pk[0] - p0[0]
            dy = pk[1] - p0[1]
            dz = pk[2] - p0[2]
            opti.subject_to(dz >= 0)
            opti.subject_to(dz * dz >= tan_gsq * (dx * dx + dy * dy))

    opti.minimize(cost)
    solver_key = _configure_opti_nlp_solver(opti, nlp_solver)

    opti.set_value(u_init_p, u_hover)
    tf0 = 0.5 * (tf_bounds[0] + tf_bounds[1])
    opti.set_initial(tf, tf0)
    for k in range(N + 1):
        a = k / float(N)
        opti.set_initial(X[0:3, k], (1 - a) * p0 + a * pf)
        opti.set_initial(X[3, k], 0.0)
        opti.set_initial(X[4, k], 0.0)
        opti.set_initial(X[5, k], float(psi_ref))
        opti.set_initial(X[6:9, k], (1 - a) * v0 + a * vf)
        opti.set_initial(X[9:12, k], np.zeros(3))
        if use_actuator_dynamics:
            opti.set_initial(X[12:16, k], u_hover)
    opti.set_initial(U, np.tile(u_hover.reshape(4, 1), (1, N)))

    t_solve0 = time.perf_counter()
    sol = opti.solve()
    wall_time_s = time.perf_counter() - t_solve0
    tf_opt = float(sol.value(tf))
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    tau = np.linspace(0.0, 1.0, N + 1)
    rep = _solver_report_from_opti_sol(sol, N, tf_opt, wall_time_s, nlp_solver=solver_key)
    return tau, X_opt, U_opt, tf_opt, rep


def solve_fftf_guidance(
    N: int,
    m: float,
    p0: np.ndarray,
    v0: np.ndarray,
    pf: np.ndarray,
    vf: np.ndarray,
    ptol: float,
    vtol: float,
    lambda1_thrust_rate: float,
    vmax: float,
    T_min: float,
    T_max: float,
    theta_max_deg: float,
    udot_max: float,
    tf_bounds: Tuple[float, float],
    use_glideslope_ascent: bool,
    gamma_glide_deg: float,
    g_mag: float = 9.81,
    min_tf_regularization: float = 0.0,
    exact_terminal: bool = True,
    I: Optional[np.ndarray] = None,
    r_thrust: Optional[np.ndarray] = None,
    use_actuator_dynamics: bool = False,
    actuator_tau: Optional[np.ndarray] = None,
    tvc_euler_max_deg: Tuple[float, float, float] = (10.0, 10.0, 180.0),
    tvc_w_max: float = 2.0,
    tvc_tau_yaw_lim: float = 2.0,
    x0_tvc: Optional[np.ndarray] = None,
    lambda_yaw: float = 500.0,
    psi_ref_deg: float = 0.0,
    nlp_solver: str = "acados",
    acados_nlp_iter: int = 150,
    acados_qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    u_cmd_prev0: Optional[np.ndarray] = None,
    export_subdir: str = "",
    verbose_solve: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """
    Build and solve the free-final-time NLP (forward Euler in τ) for **6-DoF TVC** dynamics.

    ``lambda1_thrust_rate`` weights ``‖u_k - u_{k-1}‖²`` on **command** when **not** using actuator
    dynamics. With **actuator** dynamics, smoothing comes from the lag model; this penalty and the
    ``udot_max`` constraint are omitted.

    ``vmax`` bounds each velocity component: ``|v_x|,|v_y|,|v_z| ≤ vmax`` at all nodes.

    exact_terminal : If True (default), enforce ``p(t_f)=p_f`` and ``v(t_f)=v_f`` as equalities.
    If False, use soft balls ``‖p-p_f‖≤ptol``, ``‖v-v_f‖≤vtol``.
    In both cases, terminal ``φ=θ=0``, ``ψ=ψ_ref`` (default 0°), and ``ω=0`` are enforced.
    **Yaw shaping:** cost includes ``λ_yaw∫(ψ-ψ_ref)²dt`` with ``lambda_yaw`` (large → little yaw motion).

    nlp_solver : {'acados', 'ipopt', 'sqpmethod'}
        ``acados``: SQP + HPIPM as ``tvc_traj_opt_acados.py`` (default). ``ipopt`` / ``sqpmethod``:
        CasADi ``Opti``.

    Returns
    -------
    tau_grid : shape (N+1,) pseudo-time
    X : shape ``(12, N+1)`` or ``(16, N+1)`` if actuators (phys + u_act)
    U : shape ``(4, N)`` commanded [th_p, th_r, T, tau_yaw]
    tf : optimal final time
    info : solver report with ``nlp_solver``, ``iter_count``, ``wall_time_s``, ``time_per_iter_s``,
        ``cost_history``, ``cost_optimal``, ``dt_grid`` (= :math:`t_f/N`), ``stats`` (full CasADi dict).
    """
    I_use = (
        np.diag([0.02, 0.02, 0.01])
        if I is None
        else np.asarray(I, dtype=float).reshape(3, 3)
    )
    r_use = (
        np.array([0.0, 0.0, -0.2], dtype=float)
        if r_thrust is None
        else np.asarray(r_thrust, dtype=float).reshape(3)
    )
    return _solve_fftf_tvc(
        N=N,
        m=m,
        p0=p0,
        v0=v0,
        pf=pf,
        vf=vf,
        ptol=ptol,
        vtol=vtol,
        lambda1_control_smooth=lambda1_thrust_rate,
        vmax=vmax,
        T_min=T_min,
        T_max=T_max,
        theta_max_deg=theta_max_deg,
        udot_max=udot_max,
        tf_bounds=tf_bounds,
        use_glideslope_ascent=use_glideslope_ascent,
        gamma_glide_deg=gamma_glide_deg,
        g_mag=g_mag,
        min_tf_regularization=min_tf_regularization,
        exact_terminal=exact_terminal,
        I_m=I_use,
        r_thrust=r_use,
        use_actuator_dynamics=use_actuator_dynamics,
        actuator_tau=actuator_tau,
        tvc_euler_max_deg=tvc_euler_max_deg,
        tvc_w_max=tvc_w_max,
        tvc_tau_yaw_lim=tvc_tau_yaw_lim,
        x0_tvc=x0_tvc,
        lambda_yaw=lambda_yaw,
        psi_ref=float(np.radians(psi_ref_deg)),
        nlp_solver=nlp_solver,
        acados_nlp_iter=acados_nlp_iter,
        acados_qp_solver=acados_qp_solver,
        u_cmd_prev0=u_cmd_prev0,
        export_subdir=export_subdir,
        verbose_solve=verbose_solve,
    )


def method1_state_to_fftf_x0(
    x17: np.ndarray, m: float, g: float, use_actuator: bool
) -> np.ndarray:
    """Map GUI Method-1 state (17: p,v,q wxyz,w) to ``solve_fftf_guidance`` initial TVC vector (12 or 16)."""
    from tvc_common import quat_to_euler

    x17 = np.asarray(x17, dtype=float).flatten()
    if x17.size < 17:
        raise ValueError(f"method1 state expects 17 components, got {x17.size}")
    p, v = x17[0:3], x17[3:6]
    q, w = x17[6:10], x17[10:13]
    phi, theta, psi = quat_to_euler(q, format="wxyz")
    x12 = np.concatenate([p, [phi, theta, psi], v, w])
    u_hov = np.array([0.0, 0.0, float(m) * g, 0.0], dtype=float)
    if use_actuator:
        return np.concatenate([x12, u_hov])
    return x12


def solve_spannagl_style_waypoints(
    dt: float,
    waypoints: List[List[float]],
    m: float,
    I: np.ndarray,
    r_thrust: np.ndarray,
    weights: Dict[str, Any],
    bounds: Dict[str, Any],
    x0_method1: np.ndarray,
    max_iter: int = 150,
    callback=None,
    running_flag=None,
    iteration_callback=None,
    verbose_solve: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], Optional[np.ndarray], Dict[str, Any]]:
    """
    Spannagl et al. (arXiv:2103.04709) style **free-final-time minimum-thrust** guidance: pseudo-time
    on a fixed grid, optimize ``t_f``, running cost ∝ thrust magnitude (+ yaw shaping, optional Δu).

    Solves **one NLP per waypoint segment** (same chaining idea as Acados Methods 4–5). Requires
    ``nlp_solver=acados`` in practice for multi-segment (distinct code export per leg).

    Returns ``(xs, us, loggers, us_actual, meta)`` compatible with the Acados GUI thread.
    """
    m = float(m)
    g = float(weights.get("g", 9.81))
    use_actuator = bool(weights.get("actuator_dynamics", False))
    actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])

    T_scale = float(weights.get("min_time_T_max_scale", 1.0))
    tf_lo_def = float(weights.get("min_time_T_min", 0.15))

    v_h = float(bounds.get("state_v_horizontal_max", bounds.get("v_horizontal_max", 20.0)))
    v_z = float(bounds.get("state_v_vertical_max", bounds.get("v_vertical_max", 20.0)))
    vmax = min(v_h, v_z)

    th_pb = bounds.get("th_p", (-0.4, 0.4))
    th_rb = bounds.get("th_r", (-0.4, 0.4))
    theta_max_deg = float(
        np.degrees(max(abs(float(th_pb[1])), abs(float(th_rb[1]))))
    )

    T_b = bounds.get("T", (0.0, 30.0))
    T_min_c, T_max_c = float(T_b[0]), float(T_b[1])
    tau_b = bounds.get("tau_yaw", (-2.0, 2.0))
    tvc_tau_yaw_lim = float(max(abs(tau_b[0]), abs(tau_b[1])))

    tvc_euler_max_deg = (
        float(np.degrees(float(bounds.get("state_roll_max", np.radians(45.0))))),
        float(np.degrees(float(bounds.get("state_pitch_max", np.radians(45.0))))),
        float(np.degrees(float(bounds.get("state_yaw_max", np.radians(180.0))))),
    )
    tvc_w_max = float(bounds.get("state_w_max", 2.0))

    lambda1 = float(weights.get("w_du", 0.5))
    lambda_yaw = float(
        weights.get("spannagl_lambda_yaw", max(50.0, float(weights.get("w_yaw", 0.5)) * 400.0))
    )
    min_tf_reg = float(weights.get("spannagl_min_tf_reg", 0.0))
    exact_term = bool(weights.get("spannagl_exact_terminal", True))
    ptol = float(weights.get("spannagl_ptol", 0.05))
    vtol = float(weights.get("spannagl_vtol", 0.05))
    udot_max = float(weights.get("spannagl_udot_max", 500.0))
    use_glide = bool(weights.get("spannagl_glideslope", False))
    gamma_glide = float(weights.get("spannagl_gamma_deg", 15.0))
    nlp_key = str(weights.get("spannagl_nlp_solver", "acados")).lower().strip()

    I_m = np.asarray(I, dtype=float).reshape(3, 3)
    r_vec = np.asarray(r_thrust, dtype=float).reshape(3)

    all_xs: List[List[np.ndarray]] = []
    all_us: List[List[np.ndarray]] = []
    all_loggers: List[_CostLogger] = []
    u_blocks: List[np.ndarray] = []
    optimal_tf: List[float] = []
    boundary_acc: List[int] = []
    acc_idx = -1

    x_carry = np.asarray(x0_method1, dtype=float).flatten()
    u_prev_seg: Optional[np.ndarray] = None

    for seg_idx in range(len(waypoints) - 1):
        if running_flag is not None and not running_flag():
            break
        wp0 = waypoints[seg_idx]
        wp1 = waypoints[seg_idx + 1]
        duration = float(wp1[4]) - float(wp0[4])
        if duration <= 0:
            raise ValueError(
                f"Waypoint {seg_idx+1} time must be greater than waypoint {seg_idx} time"
            )
        tf_hi = max(duration * T_scale, tf_lo_def + 1e-3)
        tf_lo = max(tf_lo_def, 0.05)
        tf_lo = min(tf_lo, tf_hi * 0.99)

        N = max(10, int(duration / max(float(dt), 1e-6)))

        p0 = np.asarray(x_carry[0:3], dtype=float).flatten()
        v0 = np.asarray(x_carry[3:6], dtype=float).flatten()
        pf = np.asarray([wp1[0], wp1[1], wp1[2]], dtype=float)
        vf = np.zeros(3, dtype=float)
        psi_ref_deg = float(wp1[3]) if len(wp1) > 3 else 0.0

        x0_tvc = method1_state_to_fftf_x0(x_carry, m, g, use_actuator)

        tau, X, U, tf_opt, rep = solve_fftf_guidance(
            N=N,
            m=m,
            p0=p0,
            v0=v0,
            pf=pf,
            vf=vf,
            ptol=ptol,
            vtol=vtol,
            lambda1_thrust_rate=lambda1,
            vmax=vmax,
            T_min=T_min_c,
            T_max=T_max_c,
            theta_max_deg=theta_max_deg,
            udot_max=udot_max,
            tf_bounds=(tf_lo, tf_hi),
            use_glideslope_ascent=use_glide,
            gamma_glide_deg=gamma_glide,
            g_mag=g,
            min_tf_regularization=min_tf_reg,
            exact_terminal=exact_term,
            I=I_m,
            r_thrust=r_vec,
            use_actuator_dynamics=use_actuator,
            actuator_tau=actuator_tau,
            tvc_euler_max_deg=tvc_euler_max_deg,
            tvc_w_max=tvc_w_max,
            tvc_tau_yaw_lim=tvc_tau_yaw_lim,
            x0_tvc=x0_tvc,
            lambda_yaw=lambda_yaw,
            psi_ref_deg=psi_ref_deg,
            nlp_solver=nlp_key,
            acados_nlp_iter=max_iter,
            u_cmd_prev0=u_prev_seg,
            export_subdir=f"seg{seg_idx}_N{N}",
            verbose_solve=verbose_solve,
        )
        optimal_tf.append(float(tf_opt))

        seg_xs, seg_us, u_act_arr = fftf_solution_to_gui_xs_us(X, U=U)
        if callback is not None:
            callback(None, seg_idx, seg_xs, seg_us, all_xs, all_us)

        all_xs.append(seg_xs)
        all_us.append(seg_us)
        ch = rep.get("cost_history") or [rep.get("cost_optimal", 0.0)]
        all_loggers.append(_CostLogger(ch))
        if iteration_callback is not None:
            iteration_callback(
                int(rep.get("iter_count", 0)),
                float(rep.get("cost_optimal", 0.0)),
                0.0,
                seg_idx,
            )

        if u_act_arr is not None:
            u_blocks.append(np.asarray(u_act_arr, dtype=float))

        x_carry = np.asarray(seg_xs[-1], dtype=float).flatten()
        u_prev_seg = np.asarray(U[:, -1], dtype=float).flatten()

        nseg = len(seg_xs) - 1
        if seg_idx == 0:
            acc_idx = nseg
            boundary_acc.append(acc_idx)
        else:
            acc_idx += nseg
            boundary_acc.append(acc_idx)

    combined_xs: List[np.ndarray] = []
    combined_us: List[np.ndarray] = []
    has_u_act = use_actuator and len(u_blocks) > 0
    for si, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
        if si == 0:
            combined_xs.extend(seg_xs)
            combined_us.extend(seg_us)
        else:
            combined_xs.extend(seg_xs[1:])
            combined_us.extend(seg_us)

    u_actual_out: Optional[np.ndarray] = None
    if has_u_act and u_blocks:
        parts = [u_blocks[0]]
        for j in range(1, len(u_blocks)):
            parts.append(u_blocks[j][1:, :])
        u_actual_out = np.vstack(parts)

    total_T = float(sum(optimal_tf)) if optimal_tf else 0.0
    n_states = len(combined_xs)
    plot_dt = (
        total_T / max(n_states - 1, 1)
        if n_states > 1 and total_T > 1e-9
        else float(dt)
    )
    meta = {
        "spannagl_fftf": True,
        "plot_dt": plot_dt,
        "segment_boundary_indices": boundary_acc,
        "optimal_segment_times": optimal_tf,
    }
    return combined_xs, combined_us, all_loggers, u_actual_out, meta


def fftf_solution_to_gui_xs_us(
    X: np.ndarray,
    U: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
    """
    Map Acados-layout ``X`` (12 or 16 × N+1) to Method-1 ``xs``, ZOH control list ``us``, and optional
    ``us_actual`` (N+1, 4) for actuator states ``x[12:16]`` at each node (for dashed overlays in GUI).

    With **actuator dynamics** (nx=16) and ``U`` given: ``us`` = **commanded** ``U[:,k]``, ``us_actual``
    = full actual trajectory. If nx=16 but ``U`` is None, ``us`` falls back to actual at step starts only.
    """
    nx, n_nodes = X.shape
    xs_tvc: List[np.ndarray] = []
    for k in range(n_nodes):
        p = np.asarray(X[0:3, k], dtype=float).flatten()
        phi = float(X[3, k])
        theta = float(X[4, k])
        psi = float(X[5, k])
        v = np.asarray(X[6:9, k], dtype=float).flatten()
        w = np.asarray(X[9:12, k], dtype=float).flatten()
        q = euler_to_quat_wxyz(phi, theta, psi)
        xs_tvc.append(np.concatenate([p, v, q, w]))

    us_actual_arr: Optional[np.ndarray] = None
    if nx >= 16 and U is not None:
        us_actual_arr = np.asarray(X[12:16, :], dtype=float).T

    us_tvc: List[np.ndarray] = []
    for k in range(n_nodes - 1):
        if nx >= 16:
            if U is not None:
                uvec = np.asarray(U[:, k], dtype=float).flatten()
            else:
                uvec = np.asarray(X[12:16, k], dtype=float).flatten()
        else:
            if U is None:
                raise ValueError("TVC nx==12 requires U (4 x N) for control traces")
            uvec = np.asarray(U[:, k], dtype=float).flatten()
        us_tvc.append(
            np.array(
                [float(uvec[0]), float(uvec[1]), float(uvec[2]), float(uvec[3])],
                dtype=float,
            )
        )

    return xs_tvc, us_tvc, us_actual_arr


def _plot_gui_dashboard(
    X: np.ndarray,
    tf: float,
    p0: np.ndarray,
    pf: np.ndarray,
    optimization_bounds: Optional[Dict[str, Any]] = None,
    show: bool = True,
    U: Optional[np.ndarray] = None,
    solve_info: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Same full figure layout as ``tvc_traj_opt_gui`` / ``plot_gui_style_results`` (6-DoF TVC).

    ``solve_info`` : return dict from ``solve_fftf_guidance`` (cost history + timing for cost panel).
    """
    script_dir = Path(__file__).resolve().parent
    sp = str(script_dir)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    from tvc_traj_gui_plots import plot_gui_style_results

    xs, us, us_actual = fftf_solution_to_gui_xs_us(X, U=U)
    dt = float(tf) / float(len(us)) if len(us) > 0 else float(tf)
    yaw0, yawf = 0.0, 0.0
    p0a = np.asarray(p0).flatten()
    pfa = np.asarray(pf).flatten()
    waypoints = [
        [float(p0a[0]), float(p0a[1]), float(p0a[2]), yaw0, 0.0],
        [float(pfa[0]), float(pfa[1]), float(pfa[2]), yawf, float(tf)],
    ]
    suptitle = "FFTF TVC 6-DoF — " + r"$t_f$" + f" = {tf:.3f} s"
    all_loggers = None
    solve_meta = None
    if solve_info:
        ch = solve_info.get("cost_history") or []
        if ch:
            all_loggers = [_CostLogger(ch)]
        solve_meta = {
            "iter_count": solve_info.get("iter_count"),
            "wall_time_s": solve_info.get("wall_time_s"),
            "time_per_iter_s": solve_info.get("time_per_iter_s"),
            "dt_grid": solve_info.get("dt_grid"),
        }
        if solve_info.get("cost_optimal") is not None and not np.isnan(
            float(solve_info["cost_optimal"])
        ):
            co = float(solve_info["cost_optimal"])
            slv = str(solve_info.get("nlp_solver", "ipopt")).upper()
            suptitle += (
                f" | NLP obj ≈ {co:.4e} | {slv} iters {int(solve_info.get('iter_count', 0))}"
            )
    return plot_gui_style_results(
        xs,
        us,
        dt,
        waypoints=waypoints,
        optimization_bounds=optimization_bounds,
        all_loggers=all_loggers,
        suptitle=suptitle,
        show=show,
        us_actual=us_actual,
        solve_meta=solve_meta,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FFTF minimum-fuel guidance: 6-DoF TVC (same ODE as export_tvc_ode_model)"
    )
    ap.add_argument("--N", type=int, default=100, help="Euler grid points (paper uses 30)")
    ap.add_argument("--m", type=float, default=0.8, help="mass [kg] (EmboRockETH order of magnitude)")
    ap.add_argument(
        "--actuator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "first-order lag on [th_p,th_r,T,tau_yaw] (nx=16); acados uses nx=17 with u_act (no u_cmd_prev)."
        ),
    )
    ap.add_argument(
        "--actuator-tau",
        type=str,
        default="0.5,0.5,1.5,0.5",
        metavar="T1,T2,T3,T4",
        help="actuator time constants [s], four comma-separated values",
    )
    ap.add_argument("--no-plot", action="store_true", help="skip matplotlib GUI-style dashboard")
    ap.add_argument("--glideslope", action="store_true", default=False, help="enable ascent glide-slope (1e)")
    ap.add_argument(
        "--soft-terminal",
        action="store_true",
        help="use ‖p-pf‖≤ptol and ‖v-vf‖≤vtol instead of exact terminal state",
    )
    ap.add_argument(
        "--lambda-yaw",
        type=float,
        default=500.0,
        help="weight on ∫(ψ-ψ_ref)² dt to limit yaw excursion [1/(rad²·s) scale]",
    )
    ap.add_argument(
        "--psi-ref-deg",
        type=float,
        default=0.0,
        help="reference yaw ψ_ref in degrees (terminal ψ fixed here; path cost pulls ψ toward it)",
    )
    ap.add_argument(
        "--nlp-solver",
        type=str,
        default="acados",
        choices=("acados", "ipopt", "sqpmethod"),
        help="acados: SQP+HPIPM (same as tvc_traj_opt_acados); ipopt/sqpmethod: CasADi Opti",
    )
    ap.add_argument(
        "--acados-max-iter",
        type=int,
        default=150,
        help="acados SQP max iterations (only for --nlp-solver acados)",
    )
    args = ap.parse_args()

    m = args.m
    g = 9.81
    vmax = 1.0
    tvc_euler_max_deg = (8.0, 8.0, 180.0)
    T_min = 0.8 * m * g
    T_max = 1.2 * m * g
    theta_max_deg = 10.0
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.zeros(3)
    pf = np.array([5.0, 0.0, 0.0])
    vf = np.zeros(3)

    actuator_tau_arr: Optional[np.ndarray] = None
    if args.actuator_tau:
        parts = [float(p) for p in args.actuator_tau.split(",")]
        if len(parts) != 4:
            raise SystemExit("--actuator-tau requires four comma-separated numbers")
        actuator_tau_arr = np.array(parts, dtype=float)

    tau, X, U, tf, info = solve_fftf_guidance(
        N=args.N,
        m=m,
        p0=p0,
        v0=v0,
        pf=pf,
        vf=vf,
        ptol=0.15,
        vtol=0.5,
        lambda1_thrust_rate=0.05,
        vmax=vmax,
        T_min=T_min,
        T_max=T_max,
        theta_max_deg=theta_max_deg,
        udot_max=40.0,
        tf_bounds=(2.0, 25.0),
        use_glideslope_ascent=args.glideslope,
        gamma_glide_deg=10.0,
        g_mag=g,
        min_tf_regularization=1e-4,
        exact_terminal=not args.soft_terminal,
        tvc_euler_max_deg=tvc_euler_max_deg,
        use_actuator_dynamics=args.actuator,
        actuator_tau=actuator_tau_arr,
        lambda_yaw=args.lambda_yaw,
        psi_ref_deg=args.psi_ref_deg,
        nlp_solver=args.nlp_solver,
        acados_nlp_iter=args.acados_max_iter,
    )

    print(f"Optimal t_f = {tf:.4f} s")
    print(f"Final position = {X[0:3, -1]}, target = {pf}")
    print(f"Final velocity = {X[6:9, -1]}")
    n_it = int(info.get("iter_count", 0))
    wt = float(info.get("wall_time_s", 0.0))
    tpi = float(info.get("time_per_iter_s", 0.0))
    dtg = float(info.get("dt_grid", 0.0))
    co = info.get("cost_optimal")
    slv_lbl = str(info.get("nlp_solver", args.nlp_solver)).upper()
    print(
        f"{slv_lbl}: {n_it} iterations, wall time {wt:.4f} s "
        f"({tpi * 1000.0:.2f} ms/iter), return {info.get('return_status', '')}"
    )
    print(f"Grid step h = t_f/N = {dtg:.6f} s (N = {args.N})")
    if co is not None and not np.isnan(float(co)):
        print(f"NLP objective ({slv_lbl}) at solution: {float(co):.6e}")

    if not args.no_plot:
        theta_lim = np.radians(float(theta_max_deg))
        euler_lim_gui = np.radians(tvc_euler_max_deg[0])
        opt_bounds = {
            "T": (T_min, T_max),
            "th_p": (-theta_lim, theta_lim),
            "th_r": (-theta_lim, theta_lim),
            "tau_yaw": (-2.0, 2.0),
            "state_v_horizontal_max": vmax,
            "state_v_vertical_max": vmax,
            "state_roll_max": euler_lim_gui,
            "state_pitch_max": euler_lim_gui,
            "state_yaw_max": np.radians(180.0),
            "state_w_max": 2.0,
        }
        _plot_gui_dashboard(
            X,
            tf,
            p0,
            pf,
            optimization_bounds=opt_bounds,
            show=True,
            U=U,
            solve_info=info,
        )


if __name__ == "__main__":
    main()
