#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone TVC trajectory optimization (Acados / pseudo-time + segment-duration state T_seg):
optional **minimum time** or **minimum thrust integral**
(``weights['acados_objective']`` is ``min_time`` / ``min_energy``); no other Python modules in the repo.

Requires: casadi, acados_template, a built acados install, and LD_LIBRARY_PATH set before running (see end of file).

Optional: for ``main()`` to call ``tvc_traj_gui_plots.plot_gui_style_results``, put this repo's
``scripts`` on ``PYTHONPATH`` (this file auto-``sys.path.insert``s) and install ``matplotlib``.
With ``store_iterates`` enabled (e.g. ``-v``, ``--nlp-iter-metrics``), each NLP step records ``T_seg``,
discrete ``∫|T|dt``, and ``get_cost()``; unless ``--no-plot``, ``nlp_iter_T_energy`` is shown then saved as PNG.

Usage:
    cd scripts && PYTHONPATH=. python tvc_min_time_simple.py
    python tvc_min_time_simple.py --objective min_energy --no-plot
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Acados environment (same idea as tvc_traj_opt_acados)
# ---------------------------------------------------------------------------
from pathlib import Path


def _setup_acados_env():
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
                os.environ["LD_LIBRARY_PATH"] = lib_path + (os.pathsep + ld_path if ld_path else "")
            if "ACADOS_SOURCE_DIR" not in os.environ:
                os.environ["ACADOS_SOURCE_DIR"] = acados_root


_setup_acados_env()

try:
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    _ACADOS_OK = True
except ImportError:
    ca = None
    AcadosModel = None
    AcadosOcp = None
    AcadosOcpSolver = None
    _ACADOS_OK = False

# Codegen directory: same folder as this script, separate from main c_generated_code
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_EXPORT_ROOT = os.path.join(_THIS_DIR, "c_generated_code_min_simple")


# =============================================================================
# State: waypoint -> Acados 12-D
# =============================================================================
def waypoint_to_acados_state(wp, uref=None):
    x = np.zeros(12)
    x[0:3] = [float(wp[0]), float(wp[1]), float(wp[2])]
    yaw_deg = float(wp[3]) if len(wp) > 3 else 0.0
    x[5] = np.radians(yaw_deg)
    if uref is not None:
        x = np.concatenate([x, np.asarray(uref, dtype=float).flatten()[:4]])
    return x


def _acados_x_phys(x_full):
    x_full = np.asarray(x_full, dtype=float).flatten()
    if x_full.size <= 12:
        return x_full
    return x_full[:-1]


# =============================================================================
# Dynamics: pseudo-time τ∈[0,1], T_seg at end of state, dx/dτ = T_seg * f(x,u)
# =============================================================================
def _Rx_sx(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, c, -s),
        ca.horzcat(0, s, c),
    )


def _Ry_sx(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(
        ca.horzcat(c, 0, s),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-s, 0, c),
    )


def export_tvc_ode_model_pseudotime(
    m,
    I,
    r_thrust,
    g=9.81,
    model_name="tvc_rocket_pt",
    use_control_rate=False,
    use_actuator_dynamics=False,
    actuator_tau=None,
    N_shoot=10,
):
    if not _ACADOS_OK:
        raise ImportError("casadi / acados_template required")
    if actuator_tau is None:
        actuator_tau = [0.05, 0.05, 0.05, 0.05]
    if use_actuator_dynamics:
        use_control_rate = False
    Ns = float(int(max(N_shoot, 1)))

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    z = ca.SX.sym("z")
    phi = ca.SX.sym("phi")
    theta = ca.SX.sym("theta")
    psi = ca.SX.sym("psi")
    vx = ca.SX.sym("vx")
    vy = ca.SX.sym("vy")
    vz = ca.SX.sym("vz")
    wx = ca.SX.sym("wx")
    wy = ca.SX.sym("wy")
    wz = ca.SX.sym("wz")
    state_phys = ca.vertcat(x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz)

    th_p_cmd = ca.SX.sym("th_p")
    th_r_cmd = ca.SX.sym("th_r")
    T_cmd = ca.SX.sym("T")
    tau_yaw_cmd = ca.SX.sym("tau_yaw")
    control = ca.vertcat(th_p_cmd, th_r_cmd, T_cmd, tau_yaw_cmd)

    if use_actuator_dynamics:
        u_act_th_p = ca.SX.sym("u_act_th_p")
        u_act_th_r = ca.SX.sym("u_act_th_r")
        u_act_T = ca.SX.sym("u_act_T")
        u_act_tau_yaw = ca.SX.sym("u_act_tau_yaw")
        u_actual = ca.vertcat(u_act_th_p, u_act_th_r, u_act_T, u_act_tau_yaw)
        th_p, th_r, T, tau_yaw = u_act_th_p, u_act_th_r, u_act_T, u_act_tau_yaw
    else:
        th_p, th_r, T, tau_yaw = th_p_cmd, th_r_cmd, T_cmd, tau_yaw_cmd

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
    Rtvc = _Ry_sx(th_p) @ _Rx_sx(th_r)
    Fb = Rtvc @ ca.vertcat(0, 0, T)
    Fw = R @ Fb
    Fg = ca.vertcat(0, 0, -g * m)
    a_linear = (Fw + Fg) / m
    r_thrust_sx = ca.SX(r_thrust)
    tau_thrust = ca.cross(r_thrust_sx, Fb)
    tau = tau_thrust + ca.vertcat(0, 0, tau_yaw)
    I_sx = ca.SX(I)
    Iinv_sx = ca.inv(I_sx)
    w_vec = ca.vertcat(wx, wy, wz)
    a_angular = Iinv_sx @ (tau - ca.cross(w_vec, I_sx @ w_vec))
    G = ca.vertcat(
        ca.horzcat(1, sphi * sth / cth, cphi * sth / cth),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi / cth, cphi / cth),
    )
    euler_dot = G @ w_vec
    p_dot = ca.vertcat(vx, vy, vz)
    v_dot = a_linear
    f_expl_phys = ca.vertcat(p_dot, euler_dot, v_dot, a_angular)

    T_ps = ca.SX.sym("T_ps")

    if use_actuator_dynamics:
        p = ca.SX.sym("p", 4)
        inv_tau = p
        u_actual_dot = (control - u_actual) * inv_tau
        f_expl = ca.vertcat(T_ps * f_expl_phys, T_ps * u_actual_dot, 0)
        state = ca.vertcat(state_phys, u_actual, T_ps)
        nx = 17
        model = AcadosModel()
        model.p = p
    elif use_control_rate:
        u_prev_th_p = ca.SX.sym("u_prev_th_p")
        u_prev_th_r = ca.SX.sym("u_prev_th_r")
        u_prev_T = ca.SX.sym("u_prev_T")
        u_prev_tau_yaw = ca.SX.sym("u_prev_tau_yaw")
        u_prev = ca.vertcat(u_prev_th_p, u_prev_th_r, u_prev_T, u_prev_tau_yaw)
        u_prev_dot = Ns * (control - u_prev)
        f_expl = ca.vertcat(T_ps * f_expl_phys, u_prev_dot, 0)
        state = ca.vertcat(state_phys, u_prev, T_ps)
        nx = 17
        model = AcadosModel()
    else:
        f_expl = ca.vertcat(T_ps * f_expl_phys, 0)
        state = ca.vertcat(state_phys, T_ps)
        nx = 13
        model = AcadosModel()

    xdot = ca.SX.sym("xdot", nx)
    f_impl = xdot - f_expl
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = state
    model.xdot = xdot
    model.u = control
    model.name = model_name
    return model


# =============================================================================
# OCP: min_time = NONLINEAR_LS + terminal pull on T_seg; min_energy = EXTERNAL (∫|T| discrete + tracking) + terminal LS
# =============================================================================
def _default_u_sigma_from_bounds(bounds):
    b = bounds or {}
    sigma = []
    for key in ("th_p", "th_r", "T", "tau_yaw"):
        tup = b.get(key, (-0.4, 0.4) if key != "T" else (0.0, 30.0))
        lo, hi = float(tup[0]), float(tup[1])
        sigma.append(max(0.5 * abs(hi - lo), 1e-6))
    return np.array(sigma, dtype=float)


def _u_sigma_from_weights_or_bounds(weights, bounds):
    sig = weights.get("u_scale", None)
    if sig is not None:
        s = np.asarray(sig, dtype=float).ravel()
        return np.maximum(s, 1e-9)
    return _default_u_sigma_from_bounds(bounds)


def _per_channel_ls_weights(raw, sigma4):
    raw = np.asarray(raw, dtype=float).ravel()
    if raw.size == 4:
        return np.maximum(raw, 0.0)
    w0 = float(raw[0])
    s = np.asarray(sigma4, dtype=float).ravel()
    return w0 / np.maximum(s ** 2, 1e-12)


def _model_p_dim(model):
    p = getattr(model, "p", None)
    if p is None:
        return None
    sz = getattr(p, "size1", None)
    if callable(sz):
        try:
            return int(sz())
        except Exception:
            pass
    try:
        if hasattr(p, "__len__") and not isinstance(p, (str, bytes)):
            return int(len(p))
    except Exception:
        pass
    return None


def _terminal_equality_expression(model, xg_arr, weights):
    """
    Terminal equality h_e(x_N)=0. If ``terminal_constraint_full_state`` is true,
    ``x[:12] == xg`` (12-D) and per-term ``terminal_constraint*`` flags are **ignored**.
    Otherwise concatenate enabled terms (position/velocity/attitude/angular rate, 3-D each).
    Returns (expr, nh); (None, 0) when no constraints.
    """
    xg_arr = np.asarray(xg_arr, dtype=float).flatten()[:12]
    if weights.get("terminal_constraint_full_state", True):
        return model.x[:12] - ca.DM(xg_arr), 12
    h_e_list = []
    if weights.get("terminal_constraint", False):
        h_e_list.append(model.x[0:3] - ca.DM(xg_arr[0:3]))
    if weights.get("terminal_constraint_velocity", False):
        h_e_list.append(model.x[6:9] - ca.DM(xg_arr[6:9]))
    if weights.get("terminal_constraint_attitude", False):
        h_e_list.append(model.x[3:6] - ca.DM(xg_arr[3:6]))
    if weights.get("terminal_constraint_omega", False):
        h_e_list.append(model.x[9:12] - ca.DM(xg_arr[9:12]))
    if not h_e_list:
        return None, 0
    expr = h_e_list[0] if len(h_e_list) == 1 else ca.vertcat(*h_e_list)
    return expr, int(3 * len(h_e_list))


def build_acados_ocp_min_time(
    model,
    N,
    x0,
    xg,
    uref,
    weights,
    bounds,
    dt,
    T_min,
    T_max,
    T_guess,
    terminal_weights=None,
    code_export_dir=None,
    json_file=None,
    nlp_solver_max_iter=100,
    qp_solver=None,
    verbose_solve=False,
    store_iterates=None,
):
    if not _ACADOS_OK:
        raise ImportError("casadi / acados_template required")
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

    nx = int(model.x.size1())
    nu = 4
    idx_T = nx - 1
    p_dim = _model_p_dim(model)
    use_actuator_dynamics = nx == 17 and p_dim == 4
    use_control_rate = nx == 17 and not use_actuator_dynamics

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = 1.0

    w_p = weights.get("p", 1.0)
    w_v = weights.get("v", 0.2)
    w_R = weights.get("R", 0.5)
    w_roll = weights.get("roll", w_R)
    w_pitch = weights.get("pitch", w_R)
    w_yaw = weights.get("yaw", w_R)
    w_w = weights.get("w", 0.1)
    sigma_u = _u_sigma_from_weights_or_bounds(weights, bounds)
    w_u_diag = _per_channel_ls_weights(weights.get("u", 1e-3), sigma_u)
    w_du_diag = _per_channel_ls_weights(weights.get("du", 0.0), sigma_u)
    R_mat = np.diag(w_u_diag)
    Q = np.diag(
        [w_p, w_p, w_p, w_roll, w_pitch, w_yaw, w_v, w_v, w_v, w_w, w_w, w_w]
    )

    if use_actuator_dynamics:
        cost_y_expr = ca.vertcat(model.x[:16], model.u)
        yref = np.concatenate([np.asarray(xg).flatten()[:12], uref, uref])
        Q_aug = np.diag(np.concatenate([np.diag(Q), w_u_diag]))
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = cost_y_expr
        ocp.cost.yref = yref
        ocp.cost.W = np.block([[Q_aug, np.zeros((16, nu))], [np.zeros((nu, 16)), R_mat]])
    elif use_control_rate:
        u_prev = model.x[12:16]
        cost_y_expr = ca.vertcat(model.x[:16], model.u, model.u - u_prev)
        yref = np.concatenate([xg[:12], uref, uref, np.zeros(4)])
        Q_aug = np.diag(np.concatenate([np.diag(Q), w_u_diag]))
        W_du = np.diag(w_du_diag)
        W = np.block(
            [
                [Q_aug, np.zeros((16, 4)), np.zeros((16, 4))],
                [np.zeros((4, 16)), R_mat, np.zeros((4, 4))],
                [np.zeros((4, 16)), np.zeros((4, 4)), W_du],
            ]
        )
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = cost_y_expr
        ocp.cost.yref = yref
        ocp.cost.W = W
    else:
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = ca.vertcat(model.x[:12], model.u)
        ocp.cost.yref = np.concatenate([np.asarray(xg).flatten()[:12], uref])
        ocp.cost.W = np.block([[Q, np.zeros((12, nu))], [np.zeros((nu, 12)), R_mat]])

    k_term = weights.get("terminal_cost_multiplier")
    if k_term is None:
        k_term = weights.get("terminal_scale", 200.0)
    k_term = float(k_term)
    Qe = np.diag(np.diag(Q) * k_term)
    w_time = float(weights.get("min_time_weight", 1.0))
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    # Track T_seg toward 0 -> LS term w_time*T^2; with lower bound T_lo the solution may stick at T_min; set T_min with a sound physical lower bound at the caller.
    ocp.cost.yref_e = np.concatenate([np.asarray(xg).flatten()[:12], [0.0]])
    ocp.model.cost_y_expr_e = ca.vertcat(model.x[:12], model.x[idx_T])
    We_blk = np.zeros((13, 13))
    We_blk[:12, :12] = Qe
    We_blk[12, 12] = w_time
    ocp.cost.W_e = We_blk

    # Terminal equality h_e(x_N)=0: see _terminal_equality_expression; dimensions without hard constraints still use terminal LS (soft).
    xg_arr = np.asarray(xg).flatten()[:12]
    h_expr, nh = _terminal_equality_expression(model, xg_arr, weights)
    if nh > 0:
        ocp.model.con_h_expr_e = h_expr
        ocp.constraints.lh_e = np.zeros(nh)
        ocp.constraints.uh_e = np.zeros(nh)

    b = bounds or {}
    th_p = b.get("th_p", (-0.4, 0.4))
    th_r = b.get("th_r", (-0.4, 0.4))
    T_b = b.get("T", (0.0, 30.0))
    tau_yaw = b.get("tau_yaw", (-2.0, 2.0))
    ocp.constraints.lbu = np.array([th_p[0], th_r[0], T_b[0], tau_yaw[0]])
    ocp.constraints.ubu = np.array([th_p[1], th_r[1], T_b[1], tau_yaw[1]])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    v_xy_lim = b.get("state_v_horizontal_max", b.get("v_horizontal_max", 20.0))
    v_z_lim = b.get("state_v_vertical_max", b.get("v_vertical_max", 20.0))
    vx_lim = float(b.get("state_vx_max", v_xy_lim))
    vy_lim = float(b.get("state_vy_max", v_xy_lim))
    vz_lim = float(b.get("state_vz_max", v_z_lim))
    roll_max = b.get("state_roll_max", np.radians(45.0))
    pitch_max = b.get("state_pitch_max", np.radians(45.0))
    yaw_max = b.get("state_yaw_max", np.radians(180.0))
    # Angular rate: per-axis box |wx|<=wx_max, etc. (rad/s). Falls back to state_w_max (same on all axes) if state_wx_max, etc. are unset.
    _wdef = float(b.get("state_w_max", 2.0))
    wx_max = float(b.get("state_wx_max", _wdef))
    wy_max = float(b.get("state_wy_max", _wdef))
    wz_max = float(b.get("state_wz_max", _wdef))

    vx, vy, vz = model.x[6], model.x[7], model.x[8]
    phi, theta, psi = model.x[3], model.x[4], model.x[5]
    wx, wy, wz = model.x[9], model.x[10], model.x[11]
    ocp.model.con_h_expr = ca.vertcat(vx, vy, vz, phi, theta, psi, wx, wy, wz)
    ocp.constraints.lh = np.array(
        [-vx_lim, -vy_lim, -vz_lim, -roll_max, -pitch_max, -yaw_max, -wx_max, -wy_max, -wz_max]
    )
    ocp.constraints.uh = np.array(
        [vx_lim, vy_lim, vz_lim, roll_max, pitch_max, yaw_max, wx_max, wy_max, wz_max]
    )

    x0_arr = np.asarray(x0, dtype=float).flatten()
    if (use_control_rate or use_actuator_dynamics) and len(x0_arr) == 12:
        x0_arr = np.concatenate([x0_arr, np.asarray(uref).flatten()[:4]])
    T_lo = float(max(T_min, 1e-3))
    T_hi = float(max(T_lo + 1e-6, T_max))
    # Do not pin the full x vector via ocp.constraints.x0 = [x_phys, T_guess]: T_seg becomes an equality
    # fixed at T_guess≈0.55*(T_min+T_max) and min_time_weight has no effect.
    # Pin only physical states at t=0; bound T_seg with box [T_lo, T_hi] (idxbxe_0 lists physical indices only).
    nx_phys = int(idx_T)
    ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int32)
    ocp.constraints.lbx_0 = np.concatenate([x0_arr[:nx_phys], [T_lo]])
    ocp.constraints.ubx_0 = np.concatenate([x0_arr[:nx_phys], [T_hi]])
    ocp.constraints.idxbxe_0 = np.arange(nx_phys, dtype=np.int32)

    ocp.constraints.idxbx = np.array([idx_T], dtype=np.int32)
    ocp.constraints.lbx = np.array([T_lo])
    ocp.constraints.ubx = np.array([T_hi])

    ocp.constraints.idxbx_e = np.array([idx_T], dtype=np.int32)
    ocp.constraints.lbx_e = np.array([T_lo])
    ocp.constraints.ubx_e = np.array([T_hi])

    if use_actuator_dynamics:
        actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])
        actuator_tau = np.asarray(actuator_tau).flatten()
        if len(actuator_tau) < 4:
            actuator_tau = np.resize(actuator_tau, 4)
        inv_tau = 1.0 / np.maximum(np.asarray(actuator_tau, dtype=float), 1e-6)
        ocp.parameter_values = inv_tau

    ocp.solver_options.qp_solver = qp_solver or "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 2 if verbose_solve else 0
    # Replaying per-NLP-step cost needs store_iterates=True; can be enabled independently of verbose_solve
    ocp.solver_options.store_iterates = store_iterates if store_iterates is not None else verbose_solve
    ocp.solver_options.nlp_solver_max_iter = int(nlp_solver_max_iter)

    return ocp


def build_acados_ocp_min_energy(
    model,
    N,
    x0,
    xg,
    uref,
    weights,
    bounds,
    dt,
    T_min,
    T_max,
    T_guess,
    terminal_weights=None,
    code_export_dir=None,
    json_file=None,
    nlp_solver_max_iter=10,
    qp_solver=None,
    verbose_solve=False,
    store_iterates=None,
):
    """
    Same dynamics and constraints as ``build_acados_ocp_min_time``; running cost is **EXTERNAL**:

    - Tracking: quadratic form equivalent to min_time NONLINEAR_LS (reference ``xg`` at segment end for
      the whole horizon; **no** per-node yref). **Do not** enforce "pull every step to xg" as hard constraints
      (conflicts with initial state); use terminal hard equalities below for the goal.
    - ``min_energy_stage_tracking_scale`` (default ``1.0``): scales stage tracking; set ``0`` to keep only
      thrust integral + ``T_seg`` regularization (when terminal state is fixed by hard equalities).
    - Thrust integral: ``min_energy_thrust_integral_weight * (T_seg/N) * sqrt(T^2+eps)``,
      discrete approximation of ``∫ |T| dt`` (``T`` is commanded thrust; under actuator dynamics, actual thrust state).
    - **T_seg regularization (auto by default)**: ``min_energy_T_seg_regularization * T_seg^2 / N`` added to stage cost.
      With only ∫|T| and fixed terminal, the objective in ``T_seg`` is often **weakly identifiable** (or sticks to
      ``T_min``/``T_max``); SQP may stall at the initial guess. A small regularizer gives ``T_seg`` a clear
      gradient (slight preference for shorter segments). Set explicitly to ``0`` to disable.

    Terminal hard constraints: if ``terminal_constraint_full_state`` is true, ``x_N[:12]=xg`` (12 equalities) and
    per-term ``terminal_constraint`` / ``_velocity`` / ``_attitude`` / ``_omega`` are ignored; else same piecewise
    assembly as min_time.

    Terminal soft cost: weighted LS on ``x[:12]`` scaled by ``min_energy_terminal_state_ls_scale`` (default ``1.0``);
    set ``0`` when terminal state is hard-constrained; keep only ``min_energy_terminal_T_weight`` on ``T_seg``.

    Terminal: weighted LS on ``x[:12]`` (can be disabled above); weight on ``T_seg`` is ``min_energy_terminal_T_weight``
    (default ``1e-3 * min_time_weight`` if unset, same order as min_time terminal time pull; ``0`` disables).

    Optional ``min_energy_hessian_approx``: ``"EXACT"`` or ``"GAUSS_NEWTON"`` (default); with EXTERNAL cost,
    try ``EXACT`` if ``T_seg`` still does not update.
    """
    if not _ACADOS_OK:
        raise ImportError("casadi / acados_template required")
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

    nx = int(model.x.size1())
    nu = 4
    idx_T = nx - 1
    p_dim = _model_p_dim(model)
    use_actuator_dynamics = nx == 17 and p_dim == 4
    use_control_rate = nx == 17 and not use_actuator_dynamics

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = 1.0

    w_p = weights.get("p", 1.0)
    w_v = weights.get("v", 0.2)
    w_R = weights.get("R", 0.5)
    w_roll = weights.get("roll", w_R)
    w_pitch = weights.get("pitch", w_R)
    w_yaw = weights.get("yaw", w_R)
    w_w = weights.get("w", 0.1)
    sigma_u = _u_sigma_from_weights_or_bounds(weights, bounds)
    w_u_diag = _per_channel_ls_weights(weights.get("u", 1e-3), sigma_u)
    w_du_diag = _per_channel_ls_weights(weights.get("du", 0.0), sigma_u)
    R_mat = np.diag(w_u_diag)
    Q = np.diag(
        [w_p, w_p, w_p, w_roll, w_pitch, w_yaw, w_v, w_v, w_v, w_w, w_w, w_w]
    )

    Nf = float(max(int(N), 1))
    w_int = float(weights.get("min_energy_thrust_integral_weight", 0.0))
    T_eps = float(weights.get("min_energy_T_smooth_eps", 1e-10))
    if use_actuator_dynamics:
        T_for_abs = model.x[14]
    else:
        T_for_abs = model.u[2]
    T_abs = ca.sqrt(T_for_abs**2 + T_eps)
    l_thrust_int = w_int * (model.x[idx_T] / Nf) * T_abs

    _mtw = float(weights.get("min_time_weight", 1.0))
    if "min_energy_T_seg_regularization" in weights:
        w_reg_T = float(weights["min_energy_T_seg_regularization"])
    else:
        w_reg_T = 1e-4 * max(_mtw, 1e-9)
    if w_reg_T > 0.0:
        l_thrust_int = l_thrust_int + w_reg_T * (model.x[idx_T] ** 2) / Nf

    xg12_np = np.asarray(xg).flatten()[:12]
    uref_np = np.asarray(uref).flatten()[:4]
    xg_dm = ca.DM(xg12_np)
    uref_dm = ca.DM(uref_np)

    Qd = np.diag(Q).astype(float)
    l_track = 0.5 * float(Qd[0]) * (model.x[0] - xg_dm[0]) ** 2
    for i in range(1, 12):
        l_track = l_track + 0.5 * float(Qd[i]) * (model.x[i] - xg_dm[i]) ** 2

    if use_actuator_dynamics or use_control_rate:
        for j in range(4):
            l_track = l_track + 0.5 * float(w_u_diag[j]) * (model.x[12 + j] - uref_dm[j]) ** 2

    Rdiag = np.diag(R_mat).astype(float)
    for k in range(4):
        l_track = l_track + 0.5 * float(Rdiag[k]) * (model.u[k] - uref_dm[k]) ** 2

    if use_control_rate:
        du = model.u - model.x[12:16]
        for k in range(4):
            l_track = l_track + 0.5 * float(w_du_diag[k]) * du[k] ** 2

    st_scale = float(weights.get("min_energy_stage_tracking_scale", 0.0))
    model.cost_expr_ext_cost = l_thrust_int + st_scale * l_track
    ocp.cost.cost_type = "EXTERNAL"

    k_term = weights.get("terminal_cost_multiplier")
    if k_term is None:
        k_term = weights.get("terminal_scale", 200.0)
    k_term = float(k_term)
    qe_state_scale = float(weights.get("min_energy_terminal_state_ls_scale", 0.0))
    Qe = np.diag(np.diag(Q) * k_term * qe_state_scale)
    if "min_energy_terminal_T_weight" in weights:
        w_term_T = float(weights["min_energy_terminal_T_weight"])
    else:
        w_term_T = 1e-3 * max(_mtw, 1e-9)
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.yref_e = np.concatenate([np.asarray(xg).flatten()[:12], [0.0]])
    ocp.model.cost_y_expr_e = ca.vertcat(model.x[:12], model.x[idx_T])
    We_blk = np.zeros((13, 13))
    We_blk[:12, :12] = Qe
    We_blk[12, 12] = w_term_T
    ocp.cost.W_e = We_blk

    xg_arr = np.asarray(xg).flatten()[:12]
    h_expr, nh = _terminal_equality_expression(model, xg_arr, weights)
    if nh > 0:
        ocp.model.con_h_expr_e = h_expr
        ocp.constraints.lh_e = np.zeros(nh)
        ocp.constraints.uh_e = np.zeros(nh)

    b = bounds or {}
    th_p = b.get("th_p", (-0.4, 0.4))
    th_r = b.get("th_r", (-0.4, 0.4))
    T_b = b.get("T", (0.0, 30.0))
    tau_yaw = b.get("tau_yaw", (-2.0, 2.0))
    ocp.constraints.lbu = np.array([th_p[0], th_r[0], T_b[0], tau_yaw[0]])
    ocp.constraints.ubu = np.array([th_p[1], th_r[1], T_b[1], tau_yaw[1]])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    v_xy_lim = b.get("state_v_horizontal_max", b.get("v_horizontal_max", 20.0))
    v_z_lim = b.get("state_v_vertical_max", b.get("v_vertical_max", 20.0))
    vx_lim = float(b.get("state_vx_max", v_xy_lim))
    vy_lim = float(b.get("state_vy_max", v_xy_lim))
    vz_lim = float(b.get("state_vz_max", v_z_lim))
    roll_max = b.get("state_roll_max", np.radians(45.0))
    pitch_max = b.get("state_pitch_max", np.radians(45.0))
    yaw_max = b.get("state_yaw_max", np.radians(180.0))
    _wdef = float(b.get("state_w_max", 2.0))
    wx_max = float(b.get("state_wx_max", _wdef))
    wy_max = float(b.get("state_wy_max", _wdef))
    wz_max = float(b.get("state_wz_max", _wdef))

    vx, vy, vz = model.x[6], model.x[7], model.x[8]
    phi, theta, psi = model.x[3], model.x[4], model.x[5]
    wx, wy, wz = model.x[9], model.x[10], model.x[11]
    ocp.model.con_h_expr = ca.vertcat(vx, vy, vz, phi, theta, psi, wx, wy, wz)
    ocp.constraints.lh = np.array(
        [-vx_lim, -vy_lim, -vz_lim, -roll_max, -pitch_max, -yaw_max, -wx_max, -wy_max, -wz_max]
    )
    ocp.constraints.uh = np.array(
        [vx_lim, vy_lim, vz_lim, roll_max, pitch_max, yaw_max, wx_max, wy_max, wz_max]
    )

    x0_arr = np.asarray(x0, dtype=float).flatten()
    if (use_control_rate or use_actuator_dynamics) and len(x0_arr) == 12:
        x0_arr = np.concatenate([x0_arr, np.asarray(uref).flatten()[:4]])
    T_lo = float(max(T_min, 1e-3))
    T_hi = float(max(T_lo + 1e-6, T_max))
    nx_phys = int(idx_T)
    ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int32)
    ocp.constraints.lbx_0 = np.concatenate([x0_arr[:nx_phys], [T_lo]])
    ocp.constraints.ubx_0 = np.concatenate([x0_arr[:nx_phys], [T_hi]])
    ocp.constraints.idxbxe_0 = np.arange(nx_phys, dtype=np.int32)

    ocp.constraints.idxbx = np.array([idx_T], dtype=np.int32)
    ocp.constraints.lbx = np.array([T_lo])
    ocp.constraints.ubx = np.array([T_hi])

    ocp.constraints.idxbx_e = np.array([idx_T], dtype=np.int32)
    ocp.constraints.lbx_e = np.array([T_lo])
    ocp.constraints.ubx_e = np.array([T_hi])

    if use_actuator_dynamics:
        actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])
        actuator_tau = np.asarray(actuator_tau).flatten()
        if len(actuator_tau) < 4:
            actuator_tau = np.resize(actuator_tau, 4)
        inv_tau = 1.0 / np.maximum(np.asarray(actuator_tau, dtype=float), 1e-6)
        ocp.parameter_values = inv_tau

    ocp.solver_options.qp_solver = qp_solver or "PARTIAL_CONDENSING_HPIPM"
    _hess = str(weights.get("min_energy_hessian_approx", "GAUSS_NEWTON")).upper()
    ocp.solver_options.hessian_approx = _hess if _hess in ("GAUSS_NEWTON", "EXACT") else "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
    ocp.solver_options.print_level = 1 if verbose_solve else 0
    ocp.solver_options.store_iterates = store_iterates if store_iterates is not None else verbose_solve
    ocp.solver_options.nlp_solver_max_iter = int(nlp_solver_max_iter)

    return ocp


def _acados_nlp_status_hint(status: int) -> str:
    """Brief hint for ocp_nlp_solver return codes (see acados docs if versions differ)."""
    try:
        s = int(status)
    except (TypeError, ValueError):
        return ""
    if s == 0:
        return ""
    if s == 4:
        return (
            " status=4 often means QP subproblem failure (ACADOS_QP_FAILURE); "
            "SQP stops at the failing step (any iteration, not only step 1). "
            "A small sqp_iter count usually means early termination, not max_iter=1."
        )
    if s in (1, 2):
        return " Often max NLP iterations reached or not fully converged (acados version-dependent)."
    return ""


def _clear_acados_module_cache():
    to_remove = [
        k
        for k in list(sys.modules.keys())
        if (
            "c_generated_code" in k
            or "c_generated_code_min_simple" in k
            or "tvc_seg" in k
            or "tvc_min_simple" in k
        )
        and "acados_template" not in k
    ]
    for k in to_remove:
        del sys.modules[k]
    gc.collect()


def _extract_sqp_cost_history(solver, final_cost, verbose_solve):
    if not verbose_solve:
        return [float(final_cost)] if final_cost is not None else [0.0]
    costs_list = []
    try:
        nlp_iter = int(solver.get_stats("nlp_iter"))
    except Exception:
        try:
            nlp_iter = int(solver.get_stats("sqp_iter"))
        except Exception:
            nlp_iter = 0
    try:
        for k in range(nlp_iter + 1):
            it = solver.get_iterate(k)
            solver.set_iterate(it)
            costs_list.append(float(solver.get_cost()))
        try:
            solver.set_iterate(solver.get_iterate(-1))
        except Exception:
            pass
    except Exception:
        costs_list = []
    if not costs_list:
        return [float(final_cost)] if final_cost is not None else [0.0]
    if final_cost is not None and len(costs_list) >= 2:
        fc = float(final_cost)
        if costs_list[-1] > 0 and abs(costs_list[-1] - fc) / max(costs_list[-1], 1e-12) > 0.01:
            costs_list[-1] = fc
        if costs_list[0] < costs_list[-1] * 0.1 and len(costs_list) > 1:
            costs_list[0] = costs_list[1]
    return costs_list


def _replay_nlp_iterates_and_print(solver, seg_idx, segment_wall_time_s, nlp_iterate_callback=None):
    """
    After solve, replay each NLP iteration with ``get_iterate(k)`` and print cost (needs ``store_iterates=True``).

    Acados iterates inside ``solver.solve()``; Python cannot hook **during** iterations.
    For live text use ``verbose_solve=True`` (``print_level=2``) and ``python -u``.
    This function prints per-iteration NLP cost **after** the segment solve completes.
    """
    try:
        nlp_iter = int(solver.get_stats("nlp_iter"))
    except Exception:
        try:
            nlp_iter = int(solver.get_stats("sqp_iter"))
        except Exception:
            nlp_iter = 0
    print(
        f"  [Segment {seg_idx + 1}] NLP iterate replay (nlp_iter={nlp_iter}, "
        f"segment wall time {segment_wall_time_s:.4f} s)"
    )
    try:
        for k in range(nlp_iter + 1):
            it = solver.get_iterate(k)
            solver.set_iterate(it)
            c = float(solver.get_cost())
            print(f"    NLP iter {k:3d}:  cost = {c:.8e}")
            if nlp_iterate_callback is not None:
                try:
                    nlp_iterate_callback(k, c, seg_idx)
                except Exception:
                    pass
        try:
            solver.set_iterate(solver.get_iterate(-1))
        except Exception:
            pass
    except Exception as e:
        print(f"    (iterate replay failed: {e})")
    sys.stdout.flush()


def _solver_iterate_to_traj12(solver, N):
    """Read 12-D rigid-body states, controls, and T_seg at each node from the current iterate."""
    xs = [
        np.asarray(_acados_x_phys(np.array(solver.get(i, "x"), copy=True)), dtype=float).flatten()[:12]
        for i in range(N + 1)
    ]
    us = [np.asarray(solver.get(i, "u"), dtype=float).flatten() for i in range(N)]
    T_seg = float(np.asarray(solver.get(N, "x"), dtype=float).flatten()[-1])
    return xs, us, T_seg


def _acados_sqp_statistics_residuals(solver):
    """
    Extract SQP table from last ``solve()`` via ``get_stats('statistics')`` (matches terminal ``# it / res_*``).

    Acados exports shape ``(stat_n+1, n_iter)``: row 0 is iteration index ``0,1,…``; rows 1–4 are
    ``res_stat, res_eq, res_ineq, res_comp`` (same as ``print_statistics`` SQP table).
    Transposes if ndarray is ``(n_iter, stat_n+1)``; checks row 0 looks like ``arange``.
    """
    try:
        stat = np.asarray(solver.get_stats("statistics"), dtype=float)
        stat_n = int(solver.get_stats("stat_n"))
    except Exception:
        return None
    if stat.ndim != 2:
        return None
    n_field_rows = stat_n + 1
    if stat.shape[0] == n_field_rows:
        pass
    elif stat.shape[1] == n_field_rows:
        stat = stat.T
    else:
        if min(stat.shape) < 5:
            return None
        if stat.shape[0] >= 5 and stat.shape[0] <= stat.shape[1]:
            pass
        elif stat.shape[1] >= 5:
            stat = stat.T
        else:
            return None

    nr, nc = stat.shape
    if nr < 5 or nc < 1:
        return None

    def _row0_is_iter_index(s):
        r, c = s.shape
        if c < 2:
            return True
        idx = np.asarray(s[0, :], dtype=float)
        exp = np.arange(c, dtype=float)
        return float(np.max(np.abs(idx - exp))) <= 0.6

    if not _row0_is_iter_index(stat):
        stat = stat.T
        nr, nc = stat.shape
        if nr < 5 or nc < 1 or not _row0_is_iter_index(stat):
            return None

    return {
        "stats_iter": np.asarray(stat[0, :], dtype=int),
        "res_stat": np.asarray(stat[1, :], dtype=float).copy(),
        "res_eq": np.asarray(stat[2, :], dtype=float).copy(),
        "res_ineq": np.asarray(stat[3, :], dtype=float).copy(),
        "res_comp": np.asarray(stat[4, :], dtype=float).copy(),
    }


def _u_actual_nodes_from_solver_iterate(solver, N):
    """Actuator model: x[12:16] at each node is actual thrust, etc., shape (N+1,4); else None."""
    rows = []
    for i in range(N + 1):
        xa = np.asarray(solver.get(i, "x"), dtype=float).flatten()
        if xa.size < 16:
            return None
        rows.append(np.array(xa[12:16], dtype=float, copy=True))
    return np.stack(rows, axis=0)


def _collect_nlp_iterate_metrics(solver, N, use_actuator_dynamics):
    """
    Call after segment ``solve()`` with ``store_iterates`` enabled.
    Replay each NLP iterate k; record T_seg*, discrete ∫|T|dt (same as ``_segment_thrust_abs_integral``), NLP cost.
    Read res_stat / res_eq / res_ineq / res_comp from ``statistics`` **before** any set_iterate.
    """
    sqp_stats = _acados_sqp_statistics_residuals(solver)
    try:
        nlp_iter = int(solver.get_stats("nlp_iter"))
    except Exception:
        try:
            nlp_iter = int(solver.get_stats("sqp_iter"))
        except Exception:
            nlp_iter = 0
    iters = []
    ts = []
    ens = []
    costs = []
    for k in range(nlp_iter + 1):
        it = solver.get_iterate(k)
        solver.set_iterate(it)
        _, us, T_seg = _solver_iterate_to_traj12(solver, N)
        if use_actuator_dynamics:
            uan = _u_actual_nodes_from_solver_iterate(solver, N)
            e = _segment_thrust_abs_integral(us, T_seg, u_actual_nodes=uan)
        else:
            e = _segment_thrust_abs_integral(us, T_seg, None)
        try:
            c = float(solver.get_cost())
        except Exception:
            c = float("nan")
        iters.append(k)
        ts.append(float(T_seg))
        ens.append(float(e))
        costs.append(c)
    try:
        solver.set_iterate(solver.get_iterate(-1))
    except Exception:
        pass
    out = {
        "iter": np.array(iters, dtype=int),
        "T_seg_s": np.array(ts, dtype=float),
        "thrust_integral_Ns": np.array(ens, dtype=float),
        "nlp_cost": np.array(costs, dtype=float),
    }
    if sqp_stats is not None:
        out["sqp_statistics"] = sqp_stats
    return out


def _plot_nlp_iterate_T_energy_curves(
    metrics_by_segment,
    out_path=None,
    show=True,
    acados_objective="min_time",
):
    """
    Per segment **3 rows × 3 cols**: row 1 ``T_seg``, ``∫|T|dt``, NLP cost; row 2 ``res_stat``, ``res_eq``, ``res_ineq``;
    row 3 col 1 ``res_comp`` (**linear** y-axis); other cells empty. ``tight_layout`` at end (extra pad vs titles).
    ``metrics_by_segment`` entries are ``_collect_nlp_iterate_metrics`` dicts or ``None`` (skipped).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    valid = [(i, m) for i, m in enumerate(metrics_by_segment) if m is not None and m["iter"].size > 0]
    if not valid:
        return None
    n_seg = len(valid)
    sub = "min-time" if str(acados_objective).lower() == "min_time" else "min-energy"
    fig_w = 13.5
    row_h = 1.0
    fig_h = max(7.0, n_seg * 3.55 * row_h)
    fig = plt.figure(figsize=(fig_w, fig_h))
    height_ratios = []
    for _ in range(n_seg):
        height_ratios.extend([row_h, row_h, row_h])
    gs = GridSpec(
        3 * n_seg,
        3,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.55,
        wspace=0.40,
    )

    _tit_kw = {"fontsize": 9, "pad": 6}
    _res_tit_kw = {"fontsize": 8, "pad": 4}

    for s, (seg_i, m) in enumerate(valid):
        r0 = 3 * s
        k = m["iter"]
        ax0 = fig.add_subplot(gs[r0, 0])
        ax1 = fig.add_subplot(gs[r0, 1])
        ax2 = fig.add_subplot(gs[r0, 2])
        ax0.plot(k, m["T_seg_s"], "o-", ms=3, lw=1.0, color="C0")
        ax0.set_ylabel(r"$T_{\mathrm{seg}}^*$ [s]", fontsize=9)
        ax0.grid(True, alpha=0.3)
        ax0.set_title(f"S{s + 1} $T_{{\mathrm{{seg}}}}$", **_tit_kw)
        ax1.plot(k, m["thrust_integral_Ns"], "o-", ms=3, lw=1.0, color="C1")
        ax1.set_ylabel(r"$\int|T|\,\mathrm{d}t$ [N·s]", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"S{s + 1} thrust int.", **_tit_kw)
        cost = np.asarray(m.get("nlp_cost", []), dtype=float)
        if cost.size != k.size:
            cost = np.full_like(k, np.nan, dtype=float)
        ax2.plot(k, cost, "o-", ms=3, lw=1.0, color="C2")
        ax2.set_ylabel("NLP cost", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"S{s + 1} cost", **_tit_kw)
        ax0.tick_params(axis="both", labelsize=8)
        ax1.tick_params(axis="both", labelsize=8)
        ax2.tick_params(axis="both", labelsize=8)
        if s == n_seg - 1:
            ax0.set_xlabel("NLP iteration k", fontsize=9)
            ax1.set_xlabel("NLP iteration k", fontsize=9)
            ax2.set_xlabel("NLP iteration k", fontsize=9)

        st = m.get("sqp_statistics")
        if st is not None and st.get("stats_iter") is not None:
            xi = np.asarray(st["stats_iter"], dtype=int)
            pairs = [
                ("res_stat", st["res_stat"], "C3"),
                ("res_eq", st["res_eq"], "C4"),
                ("res_ineq", st["res_ineq"], "C5"),
                ("res_comp", st["res_comp"], "C6"),
            ]
            ax_r0 = None
            for j in range(3):
                name, arr, col = pairs[j]
                sub_kw = {} if ax_r0 is None else {"sharex": ax_r0}
                axrj = fig.add_subplot(gs[r0 + 1, j], **sub_kw)
                if ax_r0 is None:
                    ax_r0 = axrj
                y = np.asarray(arr, dtype=float)
                axrj.plot(xi, y, "o-", ms=3, lw=1.0, color=col)
                axrj.set_title(name, **_res_tit_kw)
                axrj.set_ylabel("res.", fontsize=8)
                axrj.grid(True, alpha=0.3)
                axrj.tick_params(axis="both", labelsize=7)
                plt.setp(axrj.get_xticklabels(), visible=(s == n_seg - 1))
                if s == n_seg - 1:
                    axrj.set_xlabel("k (# it)", fontsize=8)
            name, arr, col = pairs[3]
            axrc = fig.add_subplot(gs[r0 + 2, 0], sharex=ax_r0)
            y = np.asarray(arr, dtype=float)
            axrc.plot(xi, y, "o-", ms=3, lw=1.0, color=col)
            axrc.set_title(name, **_res_tit_kw)
            axrc.set_ylabel("res.", fontsize=8)
            axrc.grid(True, alpha=0.3)
            axrc.tick_params(axis="both", labelsize=7)
            plt.setp(axrc.get_xticklabels(), visible=(s == n_seg - 1))
            if s == n_seg - 1:
                axrc.set_xlabel("k (# it)", fontsize=8)
        else:
            ax_na = fig.add_subplot(gs[r0 + 1 : r0 + 3, :])
            ax_na.set_axis_off()
            ax_na.text(
                0.5,
                0.5,
                "SQP statistics (res_stat …) not available",
                ha="center",
                va="center",
                transform=ax_na.transAxes,
                fontsize=10,
            )

    fig.suptitle(
        f"TVC {sub}: trajectory + SQP residuals (tvc_min_time_simple)",
        fontsize=10,
        y=0.995,
    )
    try:
        fig.tight_layout(
            rect=[0.03, 0.03, 0.97, 0.90],
            pad=1.2,
            h_pad=2.0,
            w_pad=1.2,
        )
    except Exception:
        fig.subplots_adjust(
            left=0.09,
            right=0.96,
            top=0.88,
            bottom=0.08,
            hspace=0.55,
            wspace=0.42,
        )
    from tvc_traj_gui_plots import mpl_show_if_interactive, mpl_show_nonblocking_if_interactive

    # Show then save: non-blocking refresh on interactive backend, savefig, then blocking show until close (main dashboard next)
    mpl_show_nonblocking_if_interactive(show)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=120)
    mpl_show_if_interactive(show)
    plt.close(fig)
    return out_path


def plot_nlp_iterate_metrics_from_meta(meta, show=True, out_path=None):
    """Plot if ``meta['nlp_iterate_metrics_by_segment']`` has data; return save path or None if not saved."""
    mlist = meta.get("nlp_iterate_metrics_by_segment")
    if not mlist:
        return None
    if out_path is None:
        out_path = os.path.join(_THIS_DIR, "nlp_iterate_state_plots", "nlp_iter_T_energy.png")
    obj = str(meta.get("acados_objective", "min_time"))
    return _plot_nlp_iterate_T_energy_curves(mlist, out_path=out_path, show=show, acados_objective=obj)


def _plot_nlp_iterate_state_png(
    xs,
    us,
    T_seg,
    seg_idx,
    nlp_k,
    out_path,
    p_goal,
    *,
    nlp_cost=None,
    thrust_integral_Ns=None,
    objective_tag=None,
):
    """
    Plot discrete state/control as four-panel PNG (non-interactive; needs matplotlib).
    ``nlp_k`` is ``None`` for **pre-solve** initial guess; ``int`` for that NLP iterate index.
    If ``nlp_cost`` / ``thrust_integral_Ns`` are passed (same as ``_collect_nlp_iterate_metrics``), show them under the title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(xs)
    n_st = max(n - 1, 0)
    idx = np.arange(n, dtype=float)
    p = np.stack([x[0:3] for x in xs], axis=0)
    v = np.stack([x[6:9] for x in xs], axis=0)
    eul = np.stack([x[3:6] for x in xs], axis=0)
    umat = np.stack(us, axis=0) if us else np.zeros((0, 4))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.plot(idx, p[:, 0], label="x")
    ax.plot(idx, p[:, 1], label="y")
    ax.plot(idx, p[:, 2], label="z")
    if p_goal is not None and len(p_goal) >= 3:
        g = np.asarray(p_goal, dtype=float).flatten()[:3]
        ax.axhline(g[0], color="C0", ls="--", alpha=0.45, lw=1.0)
        ax.axhline(g[1], color="C1", ls="--", alpha=0.45, lw=1.0)
        ax.axhline(g[2], color="C2", ls="--", alpha=0.45, lw=1.0)
    ax.set_xlabel("node i")
    ax.set_ylabel("position [m]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"position (T_seg={T_seg:.4g} s)")

    ax = axes[0, 1]
    ax.plot(idx, v[:, 0], label="vx")
    ax.plot(idx, v[:, 1], label="vy")
    ax.plot(idx, v[:, 2], label="vz")
    ax.set_xlabel("node i")
    ax.set_ylabel("velocity [m/s]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("velocity")

    ax = axes[1, 0]
    ax.plot(idx, np.degrees(eul[:, 0]), label="roll")
    ax.plot(idx, np.degrees(eul[:, 1]), label="pitch")
    ax.plot(idx, np.degrees(eul[:, 2]), label="yaw")
    ax.set_xlabel("node i")
    ax.set_ylabel("deg")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("attitude (Euler)")

    ax = axes[1, 1]
    if umat.shape[0] == n_st and n_st > 0:
        ui = np.arange(umat.shape[0], dtype=float)
        ax.plot(ui, umat[:, 0], label="th_p")
        ax.plot(ui, umat[:, 1], label="th_r")
        ax.plot(ui, umat[:, 2], label="T")
        ax.plot(ui, umat[:, 3], label="tau_yaw")
        ax.set_xlabel("stage i")
    ax.set_ylabel("u")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("control")

    if nlp_k is None:
        # Use ASCII in figure text: default DejaVu Sans lacks CJK; non-ASCII triggers missing-glyph warnings on savefig.
        iter_label = "initial guess (before SQP, pre-solve)"
    elif isinstance(nlp_k, str):
        iter_label = nlp_k
    else:
        iter_label = f"NLP iterate {nlp_k}"
    title_lines = [f"Segment {seg_idx + 1}  {iter_label}"]
    if nlp_cost is not None or thrust_integral_Ns is not None:
        obj_pref = f"[{objective_tag}] " if objective_tag else ""
        if nlp_cost is not None and np.isfinite(nlp_cost):
            cs = f"{float(nlp_cost):.6g}"
        elif nlp_cost is not None:
            cs = "nan"
        else:
            cs = "-"
        if thrust_integral_Ns is not None:
            es = f"{float(thrust_integral_Ns):.6g}"
        else:
            es = "-"
        title_lines.append(
            f"{obj_pref}NLP cost={cs}   int|T|dt~{es} N*s   T_seg={float(T_seg):.6g} s"
        )
    fig.suptitle("\n".join(title_lines), fontsize=10, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _replay_nlp_iterates_plot_states(
    solver,
    N,
    seg_idx,
    out_dir,
    waypoints,
    nlp_plot_stride=1,
    use_actuator_dynamics=False,
    acados_objective="min_time",
):
    """
    After solve, replay iterates and write PNGs: ``get_iterate(k)`` + ``set_iterate``,
    files ``out_dir/seg{seg}_nlp{k:04d}.png``. ``k=0`` is pre-solve initial guess (no SQP update);
    no separate ``before_nlp`` figure. Each plot shows ``get_cost()``, discrete ``∫|T|dt`` (same as metrics), ``T_seg``.

    Requires ``store_iterates=True``.
    """
    os.makedirs(out_dir, exist_ok=True)
    p_goal = None
    try:
        if seg_idx + 1 < len(waypoints):
            p_goal = np.asarray(waypoints[seg_idx + 1][:3], dtype=float)
    except Exception:
        p_goal = None
    try:
        nlp_iter = int(solver.get_stats("nlp_iter"))
    except Exception:
        try:
            nlp_iter = int(solver.get_stats("sqp_iter"))
        except Exception:
            nlp_iter = 0
    stride = max(1, int(nlp_plot_stride))
    n_saved = 0
    try:
        for k in range(0, nlp_iter + 1, stride):
            it = solver.get_iterate(k)
            solver.set_iterate(it)
            xs, us, T_seg = _solver_iterate_to_traj12(solver, N)
            if use_actuator_dynamics:
                uan = _u_actual_nodes_from_solver_iterate(solver, N)
                thrust_int = _segment_thrust_abs_integral(us, T_seg, u_actual_nodes=uan)
            else:
                thrust_int = _segment_thrust_abs_integral(us, T_seg, None)
            try:
                nlp_c = float(solver.get_cost())
            except Exception:
                nlp_c = float("nan")
            fn = os.path.join(out_dir, f"seg{seg_idx:d}_nlp{k:04d}.png")
            _plot_nlp_iterate_state_png(
                xs,
                us,
                T_seg,
                seg_idx,
                k,
                fn,
                p_goal,
                nlp_cost=nlp_c,
                thrust_integral_Ns=thrust_int,
                objective_tag=str(acados_objective).lower(),
            )
            n_saved += 1
        try:
            solver.set_iterate(solver.get_iterate(-1))
        except Exception:
            pass
    except Exception as e:
        print(f"    (NLP iterate state PNG save failed: {e})", file=sys.stderr)
    print(
        f"  [Segment {seg_idx + 1}] NLP state PNGs written to {out_dir}/ ({n_saved} files, stride={stride})",
        flush=True,
    )
    return n_saved


def _min_time_velocity_check(seg_xs, T_seg, bounds):
    """
    Coarse check: whether straight-line distance dist, T_seg, and max|v| on the trajectory are consistent.
    If SQP did not converge, often max|v|*T_seg << dist (inconsistent with dynamics).
    """
    b = bounds or {}
    v_xy_lim = float(b.get("state_v_horizontal_max", b.get("v_horizontal_max", 20.0)))
    v_z_lim = float(b.get("state_v_vertical_max", b.get("v_vertical_max", 20.0)))
    vx_lim = float(b.get("state_vx_max", v_xy_lim))
    vy_lim = float(b.get("state_vy_max", v_xy_lim))
    vz_lim = float(b.get("state_vz_max", v_z_lim))
    v_box_max = float(np.sqrt(vx_lim**2 + vy_lim**2 + vz_lim**2))

    pos = np.array([np.asarray(x, dtype=float).flatten()[:3] for x in seg_xs])
    vel = np.array([np.asarray(x, dtype=float).flatten()[6:9] for x in seg_xs])
    dist = float(np.linalg.norm(pos[-1] - pos[0]))
    vnorm = np.linalg.norm(vel, axis=1)
    v_max = float(np.max(vnorm)) if vnorm.size else 0.0
    v_xy_m = float(np.max(np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2))) if vel.size else 0.0
    vz_m = float(np.max(np.abs(vel[:, 2]))) if vel.size else 0.0
    Ts = max(float(T_seg), 1e-9)
    v_req = dist / Ts
    # Uniform-speed lower bound: if max|v|*T is an order below dist, inconsistent with covering dist in time T
    path_metric = v_max * Ts
    consistent = path_metric >= 0.2 * dist if dist > 1e-6 else True
    return {
        "dist_m": dist,
        "T_seg_s": float(T_seg),
        "v_req_mean_m_s": v_req,
        "v_max_m_s": v_max,
        "v_xy_max_m_s": v_xy_m,
        "v_z_max_m_s": vz_m,
        "v_box_cap_m_s": v_box_max,
        "path_metric_vmax_times_T_m": path_metric,
        "likely_dynamics_consistent": bool(consistent),
    }


def _segment_thrust_abs_integral(seg_us, T_seg_s, u_actual_nodes=None):
    """
    Discrete approximation ``∫|T|dt``: ``∑_k (T_seg/N) |T_k|``, same form as min_energy stage term.
    * If ``u_actual_nodes`` is ``(N+1, 4)``, use column 3 actual thrust (actuator model);
      else commanded thrust from ``seg_us`` ``(N, 4)``.
    """
    Ts = max(float(T_seg_s), 1e-15)
    if u_actual_nodes is not None:
        ua = np.asarray(u_actual_nodes, dtype=float)
        n_nodes = int(ua.shape[0])
        Nu = max(n_nodes - 1, 1)
        dt = Ts / float(Nu)
        Tmag = np.abs(ua[:, 2])
        return float(dt * np.sum(Tmag[:Nu]))
    u_arr = np.asarray(seg_us, dtype=float)
    Nu = int(u_arr.shape[0])
    if Nu == 0:
        return 0.0
    dt = Ts / float(Nu)
    return float(dt * np.sum(np.abs(u_arr[:, 2])))


def _total_thrust_abs_integral(all_us, optimal_T_list, all_u_actual=None):
    """Sum of per-segment ``∫|T|dt`` approximations; ``all_u_actual`` aligned with ``all_us``, entries may be ``None``."""
    total = 0.0
    by_seg = []
    n_seg = len(all_us)
    for si in range(n_seg):
        seg_us = all_us[si]
        Topt = float(optimal_T_list[si]) if si < len(optimal_T_list) else 0.0
        ua = None
        if all_u_actual is not None and si < len(all_u_actual):
            ua = all_u_actual[si]
        v = _segment_thrust_abs_integral(seg_us, Topt, u_actual_nodes=ua)
        by_seg.append(v)
        total += v
    return total, by_seg


# =============================================================================
# solve_with_acados_waypoints (pseudo-time + free T_seg: min_time / min_energy)
# =============================================================================
# Overview:
#   1) Waypoints sorted by time; each adjacent pair is one segment; each segment gets its own OCP and solve.
#   2) Each segment is discretized on **pseudo-time** τ∈[0,1] (acados horizon N, tf=1); physical duration is
#      state component T_seg; dynamics dx/dτ = T_seg * f(x,u) with constant T_seg (decision variable).
#   3) Waypoint time difference (wp[i+1].t - wp[i].t) sets T_seg **upper bound** (with min_time_T_max_scale);
#      lower bound from min_time_T_min; objective shortens T_seg under constraints (terminal cost on T_seg).
#   4) Between segments: terminal x(N) of previous segment is initial state of next (possibly augmented).
# =============================================================================
def solve_with_acados_waypoints(
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
    print_nlp_iterate_costs=False,
    nlp_iterate_callback=None,
    plot_nlp_iterate_states=False,
    nlp_iterate_plot_dir=None,
    nlp_iterate_plot_stride=1,
    record_nlp_iterate_metrics=False,
):
    """
    Multi-segment TVC trajectory: ``weights['acados_objective']`` is ``min_time`` (terminal pull on ``T_seg``)
    or ``min_energy`` (running cost with discrete ``∫|T|dt`` + tracking); both use free ``T_seg``, pseudo-time \(\tau\in[0,1]\).

    Parameters
    ----------
    waypoints : list of [x, y, z, yaw_deg, time]
        At least 2 points; time strictly increasing. Adjacent time differences set per-segment max duration (above).
    dt : float
        Only estimates horizon length N ≈ duration/dt (**not** physical integration step; acados integrates on τ).
    weights / bounds : dict
        See `build_acados_ocp_min_time` / `build_acados_ocp_min_energy`: tracking weights, terminal scaling,
        min_time_T_min, min_time_T_min_frac, min_time_T_min_geo_scale, min_time_T_max_scale,
        min_time_weight, min_energy_thrust_integral_weight, min_energy_terminal_T_weight,
        min_energy_T_seg_regularization (default small regularizer if unset, so T_seg updates under min_energy),
        min_energy_hessian_approx (try ``EXACT``),
        ``min_energy_stage_tracking_scale`` / ``min_energy_terminal_state_ls_scale``, control/state box constraints, etc.
        ``schedule_ref`` ignored when ``acados_objective=='min_energy'`` (EXTERNAL reference is segment-end ``xg``).
        Terminal hard equalities (from segment-end ``xg`` via waypoints; typically ``v=0``, ``φ=θ=0``, ``ψ``=waypoint yaw):
        ``terminal_constraint_full_state`` true -> ``x_N[:12]=xg`` (12-D), per-term flags below ignored;
        else ``terminal_constraint`` -> ``p_N=p_g``; ``terminal_constraint_velocity`` -> ``v_N=v_g``;
        ``terminal_constraint_attitude`` -> ``(φ,θ,ψ)_N`` match target Euler;
        ``terminal_constraint_omega`` -> ``ω_N=ω_g`` (change ``xg`` or disable if target ω≠0).
        Disabled terms rely on terminal LS only (possible residual).
        ``warm_start_interpolate_horizon`` (default ``True``): if true, states at nodes ``i=1…N`` linearly interpolate
        ``x0_seg``→``xg_seg`` (actuator branch adds ``u_act`` decay); if false, repeat segment start ``x0_seg``,
        only ``T_seg`` init remains ``T_guess``. Disabling interpolation may hurt convergence.
    verbose_solve : bool
        If ``True``, OCP ``print_level=2``; Acados prints SQP to stdout during solve; use ``python -u`` for live output.
    print_nlp_iterate_costs : bool
        If ``True``, after **each segment solve**, replay with ``get_iterate(k)`` and print each NLP cost
        (needs ``store_iterates``; auto-enabled if any of verbose_solve / print_nlp_iterate_costs / plot_nlp_iterate_states).
    nlp_iterate_callback : callable or None
        If given, signature ``callback(nlp_iter_index: int, cost: float, seg_idx: int)``, called each replay step.
    plot_nlp_iterate_states : bool
        If ``True``, after ``solve()`` replay and write ``seg*_nlp{k:04d}.png``;
        ``k=0`` is pre-solve guess (same as former ``before_nlp``). Title lists
        ``get_cost()``, discrete ``∫|T|dt`` (same as ``nlp_iterate_metrics``), ``T_seg``, ``acados_objective``.
        Needs ``store_iterates``; default dir ``scripts/nlp_iterate_state_plots``, override with ``nlp_iterate_plot_dir``.
    nlp_iterate_plot_dir : str or None
        Directory for per-step state PNGs; ``None`` uses ``nlp_iterate_state_plots`` under this script.
    nlp_iterate_plot_stride : int
        Save every ``stride`` NLP steps to limit file count when iterations are large (≥1).
    record_nlp_iterate_metrics : bool
        If ``True``, force ``store_iterates`` and after each segment ``solve()`` replay iterates,
        recording ``T_seg``, discrete ``∫|T|dt``, NLP cost into ``meta['nlp_iterate_metrics_by_segment']``.
        Also recorded if any of verbose_solve / print_nlp_iterate_costs / plot_nlp_iterate_states is true.

    Returns
    -------
    xs : list[np.ndarray]
        Per node **12-D** rigid-body state [x,y,z, φ,θ,ψ, vx..vz, wx..wz] (T_seg / actuator augmentation stripped).
    us : list[np.ndarray]
        Per segment N controls, each 4-D [th_p, th_r, T, tau_yaw].
    loggers : list
        Per segment simple object with `.costs` for cost curves.
    u_actual : ndarray or None
        Actual actuator time series only under actuator model.
    meta : dict
        optimal_segment_times — optimal T_seg* per segment;
        plot_dt — equivalent sample interval from total physical time / node count (plotting);
        segment_boundary_indices — end-node indices on stitched trajectory;
        solver_status — per-segment ``solver.solve()`` return codes, 0 = success;
        velocity_checks — per-segment coarse velocity/path dict (``_min_time_velocity_check``);
        nlp_iterate_plot_dirs — PNG directories per segment if ``plot_nlp_iterate_states`` enabled.
        nlp_iterate_metrics_by_segment — per-segment dict (``iter``, ``T_seg_s``, ``thrust_integral_Ns``,
        ``nlp_cost``, optional ``sqp_statistics``: ``stats_iter``, ``res_stat``, ``res_eq``,
        ``res_ineq``, ``res_comp``) or ``None``; filled when ``store_iterates`` on (``record_nlp_iterate_metrics``).
        thrust_abs_integral_Ns — total discrete ``∫|T|dt`` [N·s] (same form as min_energy cost);
        thrust_abs_integral_by_segment_Ns — per-segment list;
        thrust_integral_uses_actual_thrust — True under actuator model (actual thrust state column).
        warm_start_interpolate_horizon — whether horizon init uses ``x0``→``xg`` interpolation (matches ``weights``).
    """
    if not _ACADOS_OK:
        raise ImportError("Install casadi and acados_template and set LD_LIBRARY_PATH.")

    obj = str(weights.get("acados_objective", "min_time")).lower()
    if obj not in ("min_time", "min_energy"):
        raise NotImplementedError(
            "weights['acados_objective'] must be 'min_time' or 'min_energy'"
        )

    # --- Physical parameters (match CasADi model) ---
    m = float(m)
    I = np.array(I, dtype=float).reshape(3, 3)
    r_thrust = np.array(r_thrust, dtype=float).reshape(3,)

    # --- Nominal segment duration from waypoint times: T_seg upper bound and horizon N ---
    durations = []
    for i in range(len(waypoints) - 1):
        d = waypoints[i + 1][4] - waypoints[i][4]
        if d <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(float(d))

    T_scale = float(weights.get("min_time_T_max_scale", 1.0))  # upper bound = duration * T_scale
    T_min_def = float(weights.get("min_time_T_min", 0.15))  # T_seg lower bound (from weights)
    warm_interp = bool(weights.get("warm_start_interpolate_horizon", True))
    # Hover reference thrust: weight balance in vertical direction, for u in yref
    uref = np.array([0.0, 0.0, m * 9.81, 0.0])

    all_xs = []  # per-segment node chain (12-D)
    all_us = []
    all_u_actual = []  # actuator model: per segment (N+1,4) actual thrust, etc.
    all_loggers = []
    optimal_T_list = []  # solved T_seg* per segment
    solver_status_list = []  # per-segment solver.solve() return (0 = success)
    velocity_checks = []  # per-segment _min_time_velocity_check
    nlp_iterate_plot_dirs = []  # per-segment NLP state PNG dir (if plot enabled)
    nlp_iterate_metrics_by_segment = []  # per-segment T_seg / ∫|T|dt / cost when store_iterates

    # Current segment start: segment 0 from waypoint 0; later segments updated to terminal state
    x0 = waypoint_to_acados_state(waypoints[0])
    total_solve_time = 0.0
    total_sqp_iters = 0

    for seg_idx in range(len(durations)):
        if running_flag is not None and not running_flag():
            break

        duration = durations[seg_idx]
        # Allowed T_seg range [T_min, T_max] for this segment (physical seconds)
        T_max = max(duration * T_scale, T_min_def + 1e-3)
        # Lower bound: min_time_T_min, fraction of nominal duration, coarse displacement/velocity estimate,
        # so T_seg is not driven to an unphysical minimum (terminal w_time*T^2 and yref_T=0 favor T→T_min).
        T_min_frac = float(weights.get("min_time_T_min_frac", 0.05))
        T_min_dur = T_min_frac * duration
        b_loc = bounds or {}
        v_xy_lim = float(b_loc.get("state_v_horizontal_max", b_loc.get("v_horizontal_max", 20.0)))
        v_z_lim = float(b_loc.get("state_v_vertical_max", b_loc.get("v_vertical_max", 20.0)))
        vx_lim = float(b_loc.get("state_vx_max", v_xy_lim))
        vy_lim = float(b_loc.get("state_vy_max", v_xy_lim))
        vz_lim = float(b_loc.get("state_vz_max", v_z_lim))
        # Box-feasible |v| upper bound (all axes saturated), for coarse uniform-flight time lower bound
        v_box_max = float(np.sqrt(max(vx_lim, 1e-6) ** 2 + max(vy_lim, 1e-6) ** 2 + max(vz_lim, 1e-6) ** 2))
        p0 = np.asarray(waypoints[seg_idx][:3], dtype=float)
        p1 = np.asarray(waypoints[seg_idx + 1][:3], dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        geo_scale = float(weights.get("min_time_T_min_geo_scale", 0.35))
        T_min_geo = (dist / max(v_box_max, 0.5)) * geo_scale
        T_min = max(T_min_def, T_min_dur, T_min_geo)
        T_min = float(min(T_min, T_max * 0.95 - 1e-6))
        end_wp = waypoints[seg_idx + 1]
        xg = waypoint_to_acados_state(end_wp)

        # Discretization: pseudo-time nodes N (finer = slower)
        N = max(50, int(duration / dt))
        # T_seg cold start: mid-lower between bounds, avoid boundary sticking
        T_guess = float(np.clip(0.5 * (T_min + T_max), T_min, T_max))

        use_actuator_dynamics = weights.get("actuator_dynamics", False)
        actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])
        use_control_rate = (not use_actuator_dynamics) and weights.get("du", 0.0) > 0

        nlp_base = int(weights.get("min_time_nlp_max_iter", max_iter))
        # From segment 2: clear loaded generated modules to avoid dimension mismatch reusing old solver
        if seg_idx > 0:
            _clear_acados_module_cache()
            nlp_max_iter = max(300, nlp_base * 3) if use_actuator_dynamics else max(200, nlp_base * 2)
            qp_solver = None
        else:
            nlp_max_iter = nlp_base
            qp_solver = None

        _mt = "_me" if obj == "min_energy" else "_mt"
        model_suffix = (
            f"{_mt}_act"
            if use_actuator_dynamics
            else (f"{_mt}_du" if use_control_rate else _mt)
        )
        code_export_dir = os.path.join(_CODE_EXPORT_ROOT, f"seg{seg_idx}{model_suffix}_N{N}")
        json_file = os.path.join(code_export_dir, f"tvc_min_simple_seg{seg_idx}{model_suffix}.json")
        model_name = f"tvc_min_simple_seg{seg_idx}{model_suffix}"

        # CasADi model: T_seg at end of state; dynamics on τ, tf=1
        model = export_tvc_ode_model_pseudotime(
            m,
            I,
            r_thrust,
            model_name=model_name,
            use_control_rate=use_control_rate,
            use_actuator_dynamics=use_actuator_dynamics,
            actuator_tau=actuator_tau,
            N_shoot=N,
        )
        _store_it = (
            verbose_solve
            or print_nlp_iterate_costs
            or plot_nlp_iterate_states
            or record_nlp_iterate_metrics
        )
        if obj == "min_energy":
            ocp = build_acados_ocp_min_energy(
                model,
                N,
                x0,
                xg,
                uref,
                weights,
                bounds,
                dt,
                T_min,
                T_max,
                T_guess,
                terminal_weights,
                code_export_dir=code_export_dir,
                json_file=json_file,
                nlp_solver_max_iter=nlp_max_iter,
                qp_solver=qp_solver,
                verbose_solve=verbose_solve,
                store_iterates=_store_it,
            )
        else:
            ocp = build_acados_ocp_min_time(
                model,
                N,
                x0,
                xg,
                uref,
                weights,
                bounds,
                dt,
                T_min,
                T_max,
                T_guess,
                terminal_weights,
                code_export_dir=code_export_dir,
                json_file=json_file,
                nlp_solver_max_iter=nlp_max_iter,
                qp_solver=qp_solver,
                verbose_solve=verbose_solve,
                store_iterates=_store_it,
            )

        # Initial x0_seg / goal xg_seg: 12-D or 16-D (control-rate / actuator augmentation)
        x0_arr = np.asarray(x0, dtype=float).flatten()
        xg_arr = np.asarray(xg, dtype=float).flatten()
        use_16 = use_control_rate or use_actuator_dynamics
        if use_16 and len(x0_arr) == 12:
            x0_seg = np.concatenate([x0_arr[:12], uref])
        elif use_16 and len(x0_arr) >= 16:
            x0_seg = x0_arr[:16]
        else:
            x0_seg = x0_arr
        if use_16 and len(xg_arr) == 12:
            xg_seg = np.concatenate([xg_arr[:12], uref])
        elif use_16 and len(xg_arr) >= 16:
            xg_seg = xg_arr[:16]
        else:
            xg_seg = xg_arr

        # Full initial vector = [physical (or augmented) state, T_seg], matches model.x dimension
        x0_full = np.concatenate([x0_seg, [T_guess]])

        try:
            try:
                solver = AcadosOcpSolver(ocp, verbose=False, check_reuse_possible=False)
            except TypeError:
                solver = AcadosOcpSolver(ocp, verbose=False)
        except OSError as e:
            if "cannot open shared object file" in str(e) or "libqpOASES" in str(e) or "libhpipm" in str(e):
                raise RuntimeError(
                    f"Acados solver failed: {e}\n"
                    "Set: export ACADOS_SOURCE_DIR=... and LD_LIBRARY_PATH=.../lib"
                ) from e
            raise

        nx_full = int(x0_full.size)
        solver.set(0, "x", x0_full)
        # Multi-segment: pin t=0 states except T_seg to previous segment terminal (avoid init drift)
        if seg_idx > 0:
            try:
                lbx0 = np.full(nx_full, -1e10)
                ubx0 = np.full(nx_full, 1e10)
                lbx0[:-1] = ubx0[:-1] = np.asarray(x0_full[:-1], dtype=float)
                solver.constraints_set(0, "lbx", lbx0)
                solver.constraints_set(0, "ubx", ubx0)
            except Exception:
                pass

        # Actuator segments: sometimes tighten SQP step for stiffness
        if seg_idx > 0 and use_actuator_dynamics:
            try:
                solver.options_set("qp_tau_min", 1e-8)
                solver.options_set("globalization", "FIXED_STEP")
                solver.options_set("globalization_fixed_step_length", 0.7)
            except Exception:
                pass

        # Actuator: set parameter p = 1/tau at each stage (matches u_act_dot in model)
        if use_actuator_dynamics:
            actuator_tau_arr = np.asarray(actuator_tau).flatten()
            if len(actuator_tau_arr) < 4:
                actuator_tau_arr = np.resize(actuator_tau_arr, 4)
            p_val = np.asarray(1.0 / np.maximum(actuator_tau_arr.astype(float), 1e-6), dtype=np.float64)
            for i in range(N):
                solver.set(i, "p", p_val)

        # Running-cost yref: only min_time + NONLINEAR_LS; min_energy is EXTERNAL, yref ignored.
        use_schedule_ref = weights.get("schedule_ref", True) and obj == "min_time"
        uref_arr_yref = np.array(uref)
        if use_schedule_ref:
            x0_12 = np.asarray(x0_seg).flatten()[:12]
            xg_12 = np.asarray(xg_seg).flatten()[:12]
            u_actual_ref = uref_arr_yref
            if seg_idx > 0 and use_actuator_dynamics:
                u_a0_ref = np.asarray(x0_seg[12:16], dtype=float)
            for i in range(N):
                alpha = float(i) / N
                x_ref = (1 - alpha) * x0_12 + alpha * xg_12
                if use_control_rate:
                    yref = np.concatenate([x_ref, uref_arr_yref, uref_arr_yref, np.zeros(4)])
                elif use_actuator_dynamics:
                    if seg_idx > 0:
                        t_frac = float(i + 1) / max(N, 1)
                        decay = 1.0 - np.exp(-4.0 * t_frac)
                        u_actual_ref = u_a0_ref + (uref_arr_yref - u_a0_ref) * decay
                    yref = np.concatenate([x_ref, u_actual_ref, uref_arr_yref])
                else:
                    yref = np.concatenate([x_ref, uref_arr_yref])
                solver.set(i, "yref", yref)

        # Full-horizon initial guess: default x0→xg linear + T_seg; if warm_start_interpolate_horizon=False, repeat x0_seg
        uref_arr = np.array(uref)
        if seg_idx > 0 and use_actuator_dynamics:
            u_a0 = np.asarray(x0_seg[12:16], dtype=float)
            for i in range(1, N + 1):
                if warm_interp:
                    alpha = float(i) / N
                    x_phys_guess = (1 - alpha) * x0_seg[:12] + alpha * xg_seg[:12]
                    t_frac = float(i) / max(N, 1)
                    decay = 1.0 - np.exp(-5.0 * t_frac)
                    u_actual_guess = u_a0 + (uref_arr - u_a0) * decay
                else:
                    x_phys_guess = np.asarray(x0_seg[:12], dtype=float).copy()
                    u_actual_guess = np.asarray(x0_seg[12:16], dtype=float).copy()
                x_guess = np.concatenate([x_phys_guess, u_actual_guess, [T_guess]])
                solver.set(i, "x", x_guess)
        else:
            # With interpolation: velocity (p1-p0)/T_guess clipped to box; without: keep velocity from x0_seg
            vx_w = float(b_loc.get("state_vx_max", v_xy_lim))
            vy_w = float(b_loc.get("state_vy_max", v_xy_lim))
            vz_w = float(b_loc.get("state_vz_max", v_z_lim))
            p0_3 = np.asarray(x0_seg[:3], dtype=float)
            p1_3 = np.asarray(xg_seg[:3], dtype=float)
            dp = p1_3 - p0_3
            v_nom = dp / max(T_guess, 1e-6)
            v_nom[0] = float(np.clip(v_nom[0], -vx_w, vx_w))
            v_nom[1] = float(np.clip(v_nom[1], -vy_w, vy_w))
            v_nom[2] = float(np.clip(v_nom[2], -vz_w, vz_w))
            for i in range(1, N + 1):
                if warm_interp:
                    alpha = float(i) / N
                    x_phys = (1 - alpha) * np.asarray(x0_seg, dtype=float) + alpha * np.asarray(
                        xg_seg, dtype=float
                    )
                    x_phys = np.array(x_phys, dtype=float, copy=True)
                    x_phys[6:9] = v_nom
                else:
                    x_phys = np.asarray(x0_seg, dtype=float).copy()
                x_guess = np.concatenate([x_phys, [T_guess]])
                solver.set(i, "x", x_guess)
        for i in range(N):
            solver.set(i, "u", uref_arr)

        if iteration_callback is not None:
            iteration_callback(0, 0.0, 0.0, seg_idx)

        # --- SQP solve ---
        t0 = time.perf_counter()
        status = solver.solve()
        solver_status_list.append(int(status))
        elapsed = time.perf_counter() - t0
        total_solve_time += elapsed

        if status != 0:
            _hint = _acados_nlp_status_hint(status)
            print(
                f"  [Warning] Segment {seg_idx + 1} ({obj}): solver.solve() returned status={status}."
                f"{_hint}"
                f" T* and trajectory may be unreliable; try larger N, looser bounds, "
                f"min_energy_hessian_approx=EXACT, or check min_time_T_min / min_time_weight / "
                f"min_energy_thrust_integral_weight.",
                file=sys.stderr,
            )

        cost_val = solver.get_cost()
        try:
            sqp_iter = solver.get_stats("sqp_iter")
        except Exception:
            sqp_iter = 1
        total_sqp_iters += int(sqp_iter)
        # Optimal T_seg is last component of terminal node state
        try:
            T_opt = float(np.asarray(solver.get(N, "x"), dtype=float).flatten()[-1])
        except Exception:
            T_opt = T_guess
        optimal_T_list.append(T_opt)

        if print_nlp_iterate_costs:
            _replay_nlp_iterates_and_print(
                solver, seg_idx, elapsed, nlp_iterate_callback=nlp_iterate_callback
            )

        if plot_nlp_iterate_states:
            base = nlp_iterate_plot_dir
            if not base:
                base = os.path.join(_THIS_DIR, "nlp_iterate_state_plots")
            seg_dir = os.path.join(base, f"run_seg{seg_idx:d}")
            _replay_nlp_iterates_plot_states(
                solver,
                N,
                seg_idx,
                seg_dir,
                waypoints,
                nlp_plot_stride=int(max(1, nlp_iterate_plot_stride)),
                use_actuator_dynamics=use_actuator_dynamics,
                acados_objective=obj,
            )
            nlp_iterate_plot_dirs.append(seg_dir)
        else:
            nlp_iterate_plot_dirs.append(None)

        if _store_it:
            try:
                nlp_iterate_metrics_by_segment.append(
                    _collect_nlp_iterate_metrics(solver, N, use_actuator_dynamics)
                )
            except Exception as e:
                print(
                    f"  [Warning] Segment {seg_idx + 1}: NLP iterate T_seg/∫|T|dt recording failed: {e}",
                    file=sys.stderr,
                )
                nlp_iterate_metrics_by_segment.append(None)
        else:
            nlp_iterate_metrics_by_segment.append(None)

        if verbose_solve:
            # Do not call print_statistics() again: print_level=2 already printed SQP table in solve();
            # print_statistics() would duplicate res_stat/res_eq… (slightly different format).
            print(
                f"  Segment {seg_idx+1} ({obj}): cost={cost_val:.6e}, T*={T_opt:.4f}s, "
                f"SQP iter={sqp_iter}, time={elapsed:.3f}s"
            )
            sys.stdout.flush()
        if iteration_callback is not None:
            iteration_callback(int(sqp_iter), float(cost_val), 0.0, seg_idx)

        # --- Extract solution: drop T_seg, keep 12-D rigid-body state for downstream use ---
        seg_u_act = None
        try:
            seg_xs = [
                _acados_x_phys(np.array(solver.get(i, "x"), copy=True))[:12]
                for i in range(N + 1)
            ]
            seg_us = [np.array(solver.get(i, "u"), copy=True) for i in range(N)]
            x0 = _acados_x_phys(np.asarray(solver.get(N, "x"), dtype=float).flatten())
            if use_actuator_dynamics:
                rows = []
                for i in range(N + 1):
                    xa = np.asarray(solver.get(i, "x"), dtype=float).flatten()
                    if xa.size < 17:
                        rows = None
                        break
                    rows.append(np.array(xa[12:16], dtype=float, copy=True))
                if rows is not None:
                    seg_u_act = np.stack(rows, axis=0)
        except Exception as e:
            if status != 0:
                seg_xs = []
                for i in range(N + 1):
                    alpha = float(i) / N
                    x_core = (1 - alpha) * x0_seg + alpha * xg_seg
                    x_ac = np.concatenate([x_core, [T_guess]])
                    seg_xs.append(_acados_x_phys(x_ac)[:12])
                seg_us = [np.array(uref_arr, copy=True) for _ in range(N)]
                x0 = np.array(xg_seg, copy=True)
                seg_u_act = None
                print(f"  [Fallback] Segment {seg_idx+1} ({obj}) using guess ({e})")
            else:
                raise

        try:
            vc = _min_time_velocity_check(seg_xs, T_opt, bounds)
            velocity_checks.append(vc)
            if (not vc["likely_dynamics_consistent"]) or (status != 0):
                print(
                    f"  [Velocity check] Segment {seg_idx + 1}: displacement ~ {vc['dist_m']:.3f} m, T*={vc['T_seg_s']:.4f} s -> "
                    f"uniform speed needs ~{vc['v_req_mean_m_s']:.2f} m/s; trajectory max|v|={vc['v_max_m_s']:.3f} m/s, "
                    f"max|v|·T*={vc['path_metric_vmax_times_T_m']:.3f} m (velocity box |v|<={vc['v_box_cap_m_s']:.2f} m/s)."
                    f"{' Do not trust T* if severely inconsistent (often SQP not converged).' if not vc['likely_dynamics_consistent'] else ''}",
                    file=sys.stderr,
                )
        except Exception:
            velocity_checks.append(None)

        costs_list = _extract_sqp_cost_history(
            solver, cost_val, verbose_solve or print_nlp_iterate_costs
        )

        class SimpleLogger:
            def __init__(self, costs):
                self.costs = (
                    costs if isinstance(costs, (list, tuple)) else [costs] if costs is not None else [0.0]
                )

        all_loggers.append(SimpleLogger(costs_list))

        if callback is not None:
            callback(None, seg_idx, seg_xs, seg_us, all_xs, all_us)

        all_xs.append(seg_xs)
        all_us.append(seg_us)
        all_u_actual.append(seg_u_act)

        del solver
        gc.collect()

    thrust_total, thrust_by_seg = _total_thrust_abs_integral(
        all_us, optimal_T_list, all_u_actual
    )
    uses_actual = bool(all_u_actual) and all(a is not None for a in all_u_actual)

    if verbose_solve and optimal_T_list:
        print(
            f"  [{obj}] Total: SQP iter={total_sqp_iters}, wall={total_solve_time:.3f}s, "
            f"T* [s]={[f'{t:.3f}' for t in optimal_T_list]}, "
            f"∫|T|dt≈{thrust_total:.5f} N·s"
        )
        sys.stdout.flush()

    # --- Stitch segments: drop duplicate start node between adjacent segments ---
    combined_xs = []
    combined_us = []
    u_blocks = []
    has_u_actual = bool(all_u_actual) and all(a is not None for a in all_u_actual)
    boundary_acc = []
    acc_idx = -1
    for si, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
        nseg = len(seg_xs) - 1
        if si == 0:
            combined_xs.extend(seg_xs)
            combined_us.extend(seg_us)
            acc_idx = nseg
            boundary_acc.append(acc_idx)
            if has_u_actual:
                u_blocks.append(np.asarray(all_u_actual[si], dtype=float))
        else:
            combined_xs.extend(seg_xs[1:])
            combined_us.extend(seg_us)
            acc_idx += nseg
            boundary_acc.append(acc_idx)
            if has_u_actual:
                u_blocks.append(np.asarray(all_u_actual[si][1:], dtype=float))
    u_actual_out = np.vstack(u_blocks) if has_u_actual and u_blocks else None

    # Equivalent time step from total physical duration (for plotting)
    total_T = float(sum(optimal_T_list)) if optimal_T_list else 0.0
    n_states = len(combined_xs)
    plot_dt = total_T / max(n_states - 1, 1) if n_states > 1 and total_T > 1e-9 else float(dt)

    meta = {
        "min_time": True,
        "acados_objective": obj,
        "warm_start_interpolate_horizon": warm_interp,
        "plot_dt": plot_dt,
        "segment_boundary_indices": boundary_acc,
        "optimal_segment_times": optimal_T_list,
        "solver_status": solver_status_list,
        "velocity_checks": velocity_checks,
        "nlp_iterate_plot_dirs": nlp_iterate_plot_dirs,
        "nlp_iterate_metrics_by_segment": nlp_iterate_metrics_by_segment,
        "thrust_abs_integral_Ns": thrust_total,
        "thrust_abs_integral_by_segment_Ns": thrust_by_seg,
        "thrust_integral_uses_actual_thrust": uses_actual,
    }
    return combined_xs, combined_us, all_loggers, u_actual_out, meta


def _acados12_states_to_method1(xs):
    """12-D Acados state -> GUI Method-1 (17-D: p,v,q wxyz,w)."""
    from tvc_common import euler_to_quat_wxyz

    out = []
    for x in xs:
        x = np.asarray(x, dtype=float).flatten()
        xm = np.zeros(17)
        xm[0:3] = x[0:3]
        xm[3:6] = x[6:9]
        xm[6:10] = euler_to_quat_wxyz(x[3], x[4], x[5])
        xm[10:13] = x[9:12]
        out.append(xm)
    return out


def plot_min_time_run(xs, us, loggers, u_actual, meta, waypoints, bounds, dt_nominal):
    """
    Call ``plot_gui_style_results`` consistent with the GUI.
    Multi-segment: build physical time axis from ``meta['optimal_segment_times']`` and
    ``segment_boundary_indices``; pass segment-end indices for 3D coloring; do not infer boundaries from nominal waypoint times / global ``plot_dt``.
    """
    from tvc_common import physical_time_grid_per_shooting_segment
    from tvc_traj_gui_plots import plot_gui_style_results

    xs_m1 = _acados12_states_to_method1(xs)
    plot_dt = float(meta.get("plot_dt", dt_nominal))
    ots = meta.get("optimal_segment_times") or []
    sbo = meta.get("segment_boundary_indices") or []
    time_states = None
    if (
        ots
        and sbo
        and len(ots) == len(sbo)
        and len(xs_m1) == int(sbo[-1]) + 1
    ):
        time_states = physical_time_grid_per_shooting_segment(ots, sbo)
    obj = str(meta.get("acados_objective", "min_time"))
    sub = "min-time" if obj == "min_time" else "min-energy (∫|T|)"
    return plot_gui_style_results(
        xs_m1,
        us,
        plot_dt,
        waypoints=waypoints,
        optimization_bounds=bounds,
        all_loggers=loggers,
        suptitle=f"TVC {sub} (tvc_min_time_simple / Acados)",
        show=True,
        us_actual=u_actual,
        solve_meta={"plot_dt": plot_dt, "dt_grid": plot_dt},
        segment_boundaries_override=list(sbo) if sbo else None,
        time_states=time_states,
    )


def main(): 
    parser = argparse.ArgumentParser(
        description="Standalone TVC example: pseudo-time + free T_seg, min time or min thrust integral"
    )
    parser.add_argument(
        "--objective",
        choices=("min_time", "min_energy"),
        default="min_energy",
        help="min_time=terminal pull on T_seg; min_energy=running cost with ∫|T|dt approx (still optimizes T_seg)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not show plots after solve")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Acados print_level=2: print SQP to terminal during solve (use python -u recommended)",
    )
    parser.add_argument(
        "--print-nlp-iters",
        action="store_true",
        default=False,
        help="After each segment solve, replay and print each NLP iterate cost (needs store_iterates; independent of -v)",
    )
    parser.add_argument(
        "--plot-nlp-states",
        action="store_true",
        default=True,
        help="Four-panel PNG of state/control along horizon per NLP iterate (needs store_iterates; see --nlp-plot-dir)",
    )
    parser.add_argument(
        "--nlp-plot-dir",
        type=str,
        default=None,
        help="Root dir for state PNGs; default nlp_iterate_state_plots/ under this script; subdirs run_seg0, run_seg1…",
    )
    parser.add_argument(
        "--nlp-plot-stride",
        type=int,
        default=1,
        help="Save one PNG every this many NLP steps to limit file count (default: every step)",
    )
    parser.add_argument(
        "--nlp-iter-metrics",
        action="store_true",
        default=False,
        help="Enable store_iterates without -v / --print-nlp-iters / --plot-nlp-states; "
        "record T_seg, ∫|T|dt, NLP cost per step; save curves after solve (unless --no-plot)",
    )
    parser.add_argument(
        "--no-warm-start-interpolate",
        action="store_true",
        default=True,
        help="Horizon init without x0→xg interpolation: nodes i>0 repeat segment start (T_seg init still T_guess); default interpolates",
    )
    args = parser.parse_args()

    if not _ACADOS_OK:
        print("casadi / acados_template not found. Install and set LD_LIBRARY_PATH.")
        sys.exit(1)

    dt = 0.05
    waypoints = [
        # [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0, 0.0, 10.0],
        # [4.0, 0.0, 0.0, 0.0, 20.0],
    ]
    m = 0.6
    I = np.diag([0.02, 0.02, 0.01])
    r_thrust = np.array([0.0, 0.0, -0.2])

    weights = {
        "acados_objective": args.objective,
        "p": 1.0,
        "v": 1.0,
        "R": 1.0,
        "yaw": 1.0,
        "w": 1.0,
        "u": 1.0,
        "schedule_ref": True,
        "terminal_cost_multiplier": 200.0,
        # Terminal hard equalities (goal from waypoint xg; too strict may be infeasible—disable per term to test)
        "terminal_constraint": True,
        "terminal_constraint_velocity": True,
        "terminal_constraint_attitude": True,
        "terminal_constraint_omega": True,  # enable when terminal ω=0 is required
        # T_seg lower bound: max with min_time_T_min_frac*duration and geometric estimate; do not set too small alone.
        "min_time_T_min": 2.3,
        "min_time_T_min_frac": 0.05,
        "min_time_T_min_geo_scale": 0.25,
        "min_time_T_max_scale": 1.0,
        "min_time_weight": 1.0,
        "min_time_nlp_max_iter": 10,
        # min_energy: thrust integral weight; small T_seg regularization auto if key omitted (see build docstring)
        "min_energy_thrust_integral_weight": 1.0,
    }
    if args.no_warm_start_interpolate:
        weights["warm_start_interpolate_horizon"] = False
    bounds = {
        "th_p": (-0.1, 0.1),
        "th_r": (-0.1, 0.1),
        "T": (0.0, 15.0),
        "tau_yaw": (-0.1, 0.1),
        "state_v_horizontal_max": 1.0,
        "state_v_vertical_max": 1.5,
        "state_roll_max": np.radians(10.0),
        "state_pitch_max": np.radians(10.0),
        "state_yaw_max": np.radians(10.0),
        # Angular rate box [|wx|,|wy|,|wz|] per axis (rad/s). state_w_max sets all three; or state_wx_max, etc. separately.
        "state_w_max": 0.4,
    }

    xs, us, loggers, u_actual, meta = solve_with_acados_waypoints(
        dt,
        waypoints,
        m,
        I,
        r_thrust,
        weights,
        bounds,
        verbose_solve=args.verbose,
        print_nlp_iterate_costs=args.print_nlp_iters,
        plot_nlp_iterate_states=args.plot_nlp_states,
        nlp_iterate_plot_dir=args.nlp_plot_dir,
        nlp_iterate_plot_stride=args.nlp_plot_stride,
        record_nlp_iterate_metrics=args.nlp_iter_metrics,
    )

    if meta.get("warm_start_interpolate_horizon") is False:
        print("Horizon init: x0→xg interpolation disabled (nodes i>0 repeat segment start state)")

    times = meta.get("optimal_segment_times") or []
    if times:
        print(f"Optimal segment duration T* = {float(times[0]):.4f} s")
    ti = meta.get("thrust_abs_integral_Ns")
    if ti is not None:
        src = "actual thrust state" if meta.get("thrust_integral_uses_actual_thrust") else "commanded thrust u[:,2]"
        print(
            f"Energy metric ∫|T|dt ≈ {float(ti):.5f} N·s ({src}; discrete ∑(T_seg/N)|T_k|, min_time/min_energy)"
        )
        tib = meta.get("thrust_abs_integral_by_segment_Ns") or []
        if tib:
            print(f"  per segment [N·s]: {[float(x) for x in tib]}")
    p_end = np.asarray(xs[-1][:3], dtype=float)
    p_tgt = np.asarray(waypoints[-1][:3], dtype=float)
    print(f"Terminal position: {p_end}")
    print(f"Target position: {p_tgt}")
    print(f"Position error norm: {float(np.linalg.norm(p_end - p_tgt)):.4e} m")
    st = meta.get("solver_status") or []
    if st and any(s != 0 for s in st):
        print(f"Solver status (per segment): {st} — non-zero: do not trust T* or trajectory.")
    vcs = meta.get("velocity_checks") or []
    if vcs and vcs[0]:
        vc0 = vcs[0]
        print(
            f"Velocity check: max|v|={vc0['v_max_m_s']:.3f} m/s, "
            f"displacement/T*≈{vc0['v_req_mean_m_s']:.2f} m/s, "
            f"max|v|·T*={vc0['path_metric_vmax_times_T_m']:.3f} m vs displacement {vc0['dist_m']:.3f} m"
        )
    plot_dirs = meta.get("nlp_iterate_plot_dirs") or []
    if any(d for d in plot_dirs if d):
        print(f"NLP iterate state PNG directories: {[d for d in plot_dirs if d]}")

    if not args.no_plot:
        try:
            p_met = plot_nlp_iterate_metrics_from_meta(meta, show=True)
            if p_met:
                print(f"NLP iterate T_seg / ∫|T|dt / cost curves: {p_met}")
        except Exception as e:
            print(f"NLP iterate curve plot skipped: {e}", file=sys.stderr)

    if not args.no_plot:
        try:
            plot_min_time_run(xs, us, loggers, u_actual, meta, waypoints, bounds, dt)
        except ImportError as e:
            print(f"Plot skipped (missing dependency): {e}")


if __name__ == "__main__":
    main()
