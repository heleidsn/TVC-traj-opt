# -*- coding: utf-8 -*-
"""
Acados OCP cost: nonlinear LS (Method 4/5), EXTERNAL free-tf (Method 7),
and unified multi-segment LS helpers.

Dynamics: ``tvc_traj_opt_acados_dynamics.py``. Waypoint maps: ``tvc_traj_opt_acados_state.py``.
"""
import numpy as np

try:
    import casadi as ca
    from acados_template import AcadosOcp
except ImportError:
    ca = None
    AcadosOcp = None

from tvc_traj_opt_acados_state import waypoint_to_acados_state


def _require_acados_ocp():
    if ca is None or AcadosOcp is None:
        raise ImportError(
            "casadi and acados_template (AcadosOcp) are required to build TVC OCP costs."
        )


def _default_u_sigma_from_bounds(bounds):
    """
    Characteristic scales σ for [th_p, th_r, T, tau_yaw] (rad, rad, N, N·m).
    Uses half the control box width so a deviation of order ~one-sided range is normalized similarly.
    """
    b = bounds or {}
    th_p = b.get("th_p", (-0.4, 0.4))
    th_r = b.get("th_r", (-0.4, 0.4))
    T_b = b.get("T", (0.0, 30.0))
    tau_yaw = b.get("tau_yaw", (-2.0, 2.0))
    sigma = []
    for tup in (th_p, th_r, T_b, tau_yaw):
        lo, hi = float(tup[0]), float(tup[1])
        sigma.append(max(0.5 * abs(hi - lo), 1e-6))
    return np.array(sigma, dtype=float)


def _u_sigma_from_weights_or_bounds(weights, bounds):
    sig = weights.get("u_scale", None)
    if sig is not None:
        s = np.asarray(sig, dtype=float).ravel()
        if s.size != 4:
            raise ValueError("weights['u_scale'] must have length 4 [th_p, th_r, T, tau_yaw]")
        return np.maximum(s, 1e-9)
    return _default_u_sigma_from_bounds(bounds)


def _per_channel_ls_weights(raw, sigma4):
    """
    Least-squares diagonal weights for one control block (u, u_prev, u_act, or du).
    If ``raw`` is length 4, use as W_ii directly; if scalar w, use w/σ_i² so penalties are comparable across units.
    """
    raw = np.asarray(raw, dtype=float).ravel()
    if raw.size == 4:
        return np.maximum(raw, 0.0)
    if raw.size != 1:
        raise ValueError("weight must be a scalar or a length-4 vector [th_p, th_r, T, tau_yaw]")
    w0 = float(raw[0])
    s = np.asarray(sigma4, dtype=float).ravel()
    if s.size != 4:
        raise ValueError("sigma must have length 4")
    return w0 / np.maximum(s ** 2, 1e-12)


def _model_p_dim(model):
    """
    Dimension of ``model.p`` for Acados. Some ``acados_template`` builds expose ``p`` as a CasADi
    symbol (use ``size1()``); others use an empty ``list``, which must not be passed to ``size1()``.
    """
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


def _unified_hover_reference_state12():
    """Nominal hover regulation target: p=0, level attitude, v=w=0 (paired with small running W on non-waypoint stages)."""
    return np.zeros(12)


def _unified_waypoint_stage_W(Q_terminal_12, weights, use_actuator_dynamics, use_control_rate, bounds=None):
    """Waypoint shooting stages: same structure as running LS, but Q on physical state uses terminal-scale diagonal."""
    sigma_u = _u_sigma_from_weights_or_bounds(weights, bounds)
    w_u_diag = _per_channel_ls_weights(weights.get("u", 1e-3), sigma_u)
    R_mat = np.diag(w_u_diag)
    if use_actuator_dynamics:
        Q_aug = np.diag(np.concatenate([np.diag(Q_terminal_12), w_u_diag]))
        return np.block([[Q_aug, np.zeros((16, 4))], [np.zeros((4, 16)), R_mat]])
    if use_control_rate:
        Q_aug = np.diag(np.concatenate([np.diag(Q_terminal_12), w_u_diag]))
        w_du_diag = _per_channel_ls_weights(weights.get("du", 0.0), sigma_u)
        W_du = np.diag(w_du_diag)
        return np.block([
            [Q_aug, np.zeros((16, 4)), np.zeros((16, 4))],
            [np.zeros((4, 16)), R_mat, np.zeros((4, 4))],
            [np.zeros((4, 16)), np.zeros((4, 4)), W_du],
        ])
    return np.block([[Q_terminal_12, np.zeros((12, 4))], [np.zeros((4, 12)), R_mat]])


def _unified_waypoint_running_W_from_terminal(ocp, weights, use_actuator_dynamics, use_control_rate, bounds=None):
    """
    Running cost W at waypoint shooting stages: **same 12×12 physical block as terminal** ``ocp.cost.W_e``,
    plus u / du blocks matching ``build_acados_ocp`` running structure. Single source of truth as terminal.
    """
    W_e = np.asarray(ocp.cost.W_e, dtype=float)
    q12 = np.asarray(np.diag(W_e), dtype=float).ravel()
    Q12 = np.diag(q12)
    return _unified_waypoint_stage_W(Q12, weights, use_actuator_dynamics, use_control_rate, bounds=bounds)


def _unified_yref_waypoint(wp, uref, use_actuator_dynamics, use_control_rate):
    """LS reference y_ref at a waypoint stage: full waypoint state + hover u (and aug. channels if needed)."""
    uref = np.asarray(uref, dtype=float).flatten()[:4]
    x12 = waypoint_to_acados_state(wp)[:12]
    if use_control_rate:
        x16 = waypoint_to_acados_state(wp, uref=uref)
        return np.concatenate([x16, uref, uref, np.zeros(4)])
    if use_actuator_dynamics:
        x16 = waypoint_to_acados_state(wp, uref=uref)
        return np.concatenate([x16, uref])
    return np.concatenate([x12, uref])


def _sync_acados_terminal_ls_from_ocp(solver, ocp, N=None):
    """
    Ensure stage-N (Mayer) nonlinear-LS matches ocp: some solver/template combos need an
    explicit cost_set/set after per-stage cost tinkering, or the terminal term is missing
    from the NLP (then the optimizer favors a feasible hover with tiny path cost only).
    """
    if N is None:
        N = int(solver.N)
    try:
        W_e = np.asfortranarray(np.array(ocp.cost.W_e, dtype=np.float64, copy=True))
        yref_e = np.asarray(ocp.cost.yref_e, dtype=np.float64).ravel()
        try:
            solver.cost_set(N, "W", W_e, api="new")
        except TypeError:
            solver.cost_set(N, "W", W_e)
        solver.set(N, "yref", yref_e)
    except Exception:
        pass


def _unified_yref_regulation(uref, use_actuator_dynamics, use_control_rate):
    """LS reference on non-waypoint stages: regulation toward nominal hover x and u = uref (no path interpolation)."""
    uref = np.asarray(uref, dtype=float).flatten()[:4]
    x12 = _unified_hover_reference_state12()
    if use_control_rate:
        x16 = np.concatenate([x12, uref])
        return np.concatenate([x16, uref, uref, np.zeros(4)])
    if use_actuator_dynamics:
        x16 = np.concatenate([x12, uref])
        return np.concatenate([x16, uref])
    return np.concatenate([x12, uref])


def _unified_W_reg_no_state_tracking(W_reg, use_actuator_dynamics, use_control_rate):
    """
    Same running cost layout as W_reg, but **zero weight on all state components** in the LS vector
    (only ``u`` and, if applicable, ``u-u_prev`` stay weighted). Non-waypoint ``yref`` often sets
    x -> hover at origin; keeping Q_x would spuriously attract the trajectory toward the world origin.
    """
    W = np.array(W_reg, dtype=float, copy=True)
    if use_control_rate:
        nx_in_y = 16  # cost_y = [x(16), u(4), du(4)]
    elif use_actuator_dynamics:
        nx_in_y = 16
    else:
        nx_in_y = 12
    W[:nx_in_y, :nx_in_y] = 0.0
    return W


def _unified_stage_segment_bounds(N_per_seg):
    """Return list of (i0, i1_exclusive) for each segment in shooting indices 0..sum(N_per_seg)-1."""
    bounds = []
    i0 = 0
    for nj in N_per_seg:
        bounds.append((i0, i0 + nj))
        i0 += nj
    return bounds


def _unified_segment_idx_for_stage(i, N_per_seg):
    """Shooting stage index ``i`` in ``0 .. sum(N_per_seg)-1`` → segment index ``seg`` (0 .. n_seg-1)."""
    acc = 0
    for seg, nj in enumerate(N_per_seg):
        if i < acc + nj:
            return seg
        acc += nj
    raise RuntimeError(f"unified: stage {i} out of range for N_per_seg={N_per_seg}")


def _unified_yref_along_segment(i, waypoints, N_per_seg, uref, use_actuator_dynamics, use_control_rate):
    """
    LS reference for a **non-waypoint** running stage: linearly interpolate spatial ref between the
    segment's start and end waypoints (position + yaw from waypoint list). This avoids pulling the
    vehicle back to the world origin on interior stages (see _unified_yref_regulation / hover only).
    """
    uref = np.asarray(uref, dtype=float).flatten()[:4]
    seg_b = _unified_stage_segment_bounds(N_per_seg)
    for seg, (a, b) in enumerate(seg_b):
        if a <= i < b:
            nj = b - a
            alpha = float(i - a) / float(max(nj, 1))
            wp0 = waypoints[seg]
            wp1 = waypoints[seg + 1]
            px = (1.0 - alpha) * float(wp0[0]) + alpha * float(wp1[0])
            py = (1.0 - alpha) * float(wp0[1]) + alpha * float(wp1[1])
            pz = (1.0 - alpha) * float(wp0[2]) + alpha * float(wp1[2])
            yaw0 = float(wp0[3]) if len(wp0) > 3 else 0.0
            yaw1 = float(wp1[3]) if len(wp1) > 3 else 0.0
            yaw = (1.0 - alpha) * yaw0 + alpha * yaw1
            fake_wp = [px, py, pz, yaw, 0.0]
            return _unified_yref_waypoint(fake_wp, uref, use_actuator_dynamics, use_control_rate)
    # Fallback (should not happen for i in 0..N_total-1)
    return _unified_yref_regulation(uref, use_actuator_dynamics, use_control_rate)


def _state_tracking_q_diag(weights):
    """12-state tracking diagonal for [p(3), euler(3), v(3), w(3)]."""
    w_p = weights.get("p", 1.0)
    w_v = weights.get("v", 0.2)
    w_R = weights.get("R", 0.5)
    w_roll = weights.get("roll", w_R)
    w_pitch = weights.get("pitch", w_R)
    w_yaw = weights.get("yaw", w_R)
    w_w = weights.get("w", 0.1)
    return np.array(
        [w_p, w_p, w_p, w_roll, w_pitch, w_yaw, w_v, w_v, w_v, w_w, w_w, w_w],
        dtype=float,
    )


def _build_nonlinear_ls_running_cost(model, xg, uref, Q, w_u_diag, w_du_diag, use_actuator_dynamics, use_control_rate):
    """Build NONLINEAR_LS running cost tuple: (cost_y_expr, yref, W)."""
    nu = 4
    xg12 = np.asarray(xg, dtype=float).flatten()[:12]
    uref4 = np.asarray(uref, dtype=float).flatten()[:4]
    R_mat = np.diag(w_u_diag)
    if use_actuator_dynamics:
        cost_y_expr = ca.vertcat(model.x[:16], model.u)
        yref = np.concatenate([xg12, uref4, uref4])
        Q_aug = np.diag(np.concatenate([np.diag(Q), w_u_diag]))
        W = np.block([[Q_aug, np.zeros((16, nu))], [np.zeros((nu, 16)), R_mat]])
        return cost_y_expr, yref, W
    if use_control_rate:
        u_prev = model.x[12:16]
        cost_y_expr = ca.vertcat(model.x[:16], model.u, model.u - u_prev)
        yref = np.concatenate([xg12, uref4, uref4, np.zeros(4)])
        Q_aug = np.diag(np.concatenate([np.diag(Q), w_u_diag]))
        W_du = np.diag(w_du_diag)
        W = np.block(
            [
                [Q_aug, np.zeros((16, 4)), np.zeros((16, 4))],
                [np.zeros((4, 16)), R_mat, np.zeros((4, 4))],
                [np.zeros((4, 16)), np.zeros((4, 4)), W_du],
            ]
        )
        return cost_y_expr, yref, W
    cost_y_expr = ca.vertcat(model.x[:12], model.u)
    yref = np.concatenate([xg12, uref4])
    W = np.block([[Q, np.zeros((12, nu))], [np.zeros((nu, 12)), R_mat]])
    return cost_y_expr, yref, W


def _apply_nonlinear_ls_running_cost(ocp, model, xg, uref, Q, w_u_diag, w_du_diag, use_actuator_dynamics, use_control_rate):
    """Apply NONLINEAR_LS running cost on ocp/model."""
    cost_y_expr, yref, W = _build_nonlinear_ls_running_cost(
        model=model,
        xg=xg,
        uref=uref,
        Q=Q,
        w_u_diag=w_u_diag,
        w_du_diag=w_du_diag,
        use_actuator_dynamics=use_actuator_dynamics,
        use_control_rate=use_control_rate,
    )
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = cost_y_expr
    ocp.cost.yref = yref
    ocp.cost.W = W


def _terminal_scale(weights):
    """Terminal scaling coefficient (new key first, then legacy key)."""
    k_term = weights.get("terminal_cost_multiplier")
    if k_term is None:
        k_term = weights.get("terminal_scale", 200.0)
    return float(k_term)


def _apply_nonlinear_ls_terminal_cost_method4(ocp, model, xg, Q, weights):
    """Method 4 terminal NONLINEAR_LS on x[:12]."""
    Qe = np.diag(np.diag(Q) * _terminal_scale(weights))
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.yref_e = np.asarray(xg, dtype=float).flatten()[:12]
    ocp.model.cost_y_expr_e = model.x[:12]
    ocp.cost.W_e = Qe


def _apply_nonlinear_ls_terminal_cost_method5(ocp, model, xg, Q, weights, idx_T):
    """Method 5 terminal NONLINEAR_LS on [x[:12], T_seg]."""
    Qe = np.diag(np.diag(Q) * _terminal_scale(weights))
    w_time = float(weights.get("min_time_weight", 1.0))
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.yref_e = np.concatenate([np.asarray(xg, dtype=float).flatten()[:12], [0.0]])
    ocp.model.cost_y_expr_e = ca.vertcat(model.x[:12], model.x[idx_T])
    We_blk = np.zeros((13, 13))
    We_blk[:12, :12] = Qe
    We_blk[12, 12] = w_time
    ocp.cost.W_e = We_blk


def _build_external_free_tf_cost_expr(model, xg, weights, N, idx_T, use_actuator_dynamics):
    """Build Method 7 EXTERNAL running/terminal expressions."""
    xg12 = np.asarray(xg, dtype=float).flatten()[:12]
    x_phys = model.x[:12]
    tf_s = model.x[idx_T]
    th_p_c, th_r_c, T_cmd, tau_y_c = model.u[0], model.u[1], model.u[2], model.u[3]
    T_m = model.x[14] if use_actuator_dynamics else T_cmd

    w_T = float(weights.get("free_tf_w_T", 1.0))
    w_tvc = float(weights.get("free_tf_w_tvc", 1e-2))
    w_ty = float(weights.get("free_tf_w_tau_yaw", 1e-2))
    w_tf_term = float(weights.get("free_tf_w_terminal_time", 10.0))
    include_x_term = bool(weights.get("free_tf_include_state_terminal", True))

    Nf = float(max(int(N), 1))
    l_run = w_T * T_m**2 + w_tvc * (th_p_c**2 + th_r_c**2) + w_ty * tau_y_c**2
    running = (tf_s / Nf) * l_run

    Qe_diag = _state_tracking_q_diag(weights) * _terminal_scale(weights)
    term_e = w_tf_term * tf_s
    if include_x_term:
        for i in range(12):
            term_e = term_e + float(Qe_diag[i]) * (x_phys[i] - float(xg12[i])) ** 2
    return running, term_e


def _apply_terminal_position_constraint(ocp, model, xg, weights):
    """Optional terminal equality p_N = p_g."""
    if not bool(weights.get("terminal_constraint", False)):
        return
    xg_arr = np.asarray(xg, dtype=float).flatten()[:12]
    ocp.model.con_h_expr_e = model.x[0:3] - ca.DM(xg_arr[0:3])
    ocp.constraints.lh_e = np.zeros(3)
    ocp.constraints.uh_e = np.zeros(3)


def _apply_control_bounds(ocp, bounds):
    """Apply control box constraints for [th_p, th_r, T, tau_yaw]."""
    b = bounds or {}
    th_p = b.get("th_p", (-0.4, 0.4))
    th_r = b.get("th_r", (-0.4, 0.4))
    T_b = b.get("T", (0.0, 30.0))
    tau_yaw = b.get("tau_yaw", (-2.0, 2.0))
    ocp.constraints.lbu = np.array([th_p[0], th_r[0], T_b[0], tau_yaw[0]])
    ocp.constraints.ubu = np.array([th_p[1], th_r[1], T_b[1], tau_yaw[1]])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])


def _apply_state_path_bounds(ocp, model, bounds):
    """Apply path bounds on [vx, vy, vz, phi, theta, psi, wx, wy, wz]."""
    b = bounds or {}
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


def _augment_x0_with_uref_if_needed(x0, uref, use_actuator_dynamics, use_control_rate):
    """If model state includes [u_prev] or [u_actual], append uref to a 12D x0."""
    x0_arr = np.asarray(x0, dtype=float).flatten()
    if (use_control_rate or use_actuator_dynamics) and len(x0_arr) == 12:
        x0_arr = np.concatenate([x0_arr, np.asarray(uref, dtype=float).flatten()[:4]])
    return x0_arr


def _apply_free_tf_state_time_bounds(ocp, nx, idx_T, x0_arr, T_min, T_max):
    """State bounds for free-tf models: x0 fixed on physical states, T state boxed on all stages."""
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


def _apply_actuator_parameter_values(ocp, weights):
    """Set model parameter values to actuator inverse time constants."""
    actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])
    actuator_tau = np.asarray(actuator_tau).flatten()
    if len(actuator_tau) < 4:
        actuator_tau = np.resize(actuator_tau, 4)
    inv_tau = 1.0 / np.maximum(np.asarray(actuator_tau, dtype=float), 1e-6)
    ocp.parameter_values = inv_tau


# method 4: fixed-time nonlinear least squares
def build_acados_ocp(model, N, Tf, x0, xg, uref, weights, bounds, dt, terminal_weights=None,
                     code_export_dir=None, json_file=None, nlp_solver_max_iter=100, qp_solver=None,
                     verbose_solve=False):
    """Build Acados OCP for one segment. ``terminal_weights`` is kept for API compatibility and ignored."""
    _require_acados_ocp()
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
    
    nx = int(model.x.size1())  # 12 or 16 (with u_prev or u_actual)
    nu = 4
    p_dim = _model_p_dim(model)
    use_actuator_dynamics = nx == 16 and p_dim == 4
    use_control_rate = nx == 16 and p_dim == 1
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    
    # Cost: nonlinear least-squares
    sigma_u = _u_sigma_from_weights_or_bounds(weights, bounds)
    w_u_diag = _per_channel_ls_weights(weights.get("u", 1e-3), sigma_u)
    w_du_diag = _per_channel_ls_weights(weights.get("du", 0.0), sigma_u)
    Q = np.diag(_state_tracking_q_diag(weights))
    _apply_nonlinear_ls_running_cost(
        ocp=ocp,
        model=model,
        xg=xg,
        uref=uref,
        Q=Q,
        w_u_diag=w_u_diag,
        w_du_diag=w_du_diag,
        use_actuator_dynamics=use_actuator_dynamics,
        use_control_rate=use_control_rate,
    )
    
    # Terminal LS on x[:12]: same diagonal structure as running Q, scaled by one coefficient.
    # Prefer terminal_cost_multiplier; fall back to legacy terminal_scale (old GUI key).
    _apply_nonlinear_ls_terminal_cost_method4(ocp, model, xg, Q, weights)
    
    _apply_terminal_position_constraint(ocp, model, xg, weights)
    _apply_control_bounds(ocp, bounds)
    _apply_state_path_bounds(ocp, model, bounds)
    
    # x0: augment with u_prev/u_actual=uref when using control rate or actuator
    x0_arr = _augment_x0_with_uref_if_needed(x0, uref, use_actuator_dynamics, use_control_rate)
    ocp.constraints.x0 = x0_arr
    
    # When model has param p: control rate uses 1/dt; actuator uses 1/tau per channel
    if use_control_rate:
        ocp.parameter_values = np.array([N / Tf], dtype=float)
    elif use_actuator_dynamics:
        _apply_actuator_parameter_values(ocp, weights)
    
    ocp.solver_options.qp_solver = qp_solver or 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.print_level = 2 if verbose_solve else 0  # 2: print cost during solve
    ocp.solver_options.store_iterates = verbose_solve  # for cost convergence plot
    ocp.solver_options.nlp_solver_max_iter = int(nlp_solver_max_iter)
    
    return ocp

# method 5: minimum time with pseudo-time state T_seg
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
):
    """
    Minimum-time (free duration) OCP on pseudo-time horizon τ∈[0,1]: ``solver_options.tf=1``,
    physical segment length is state ``T_seg`` (last component of ``model.x``).
    """
    _require_acados_ocp()
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

    sigma_u = _u_sigma_from_weights_or_bounds(weights, bounds)
    w_u_diag = _per_channel_ls_weights(weights.get("u", 1e-3), sigma_u)
    w_du_diag = _per_channel_ls_weights(weights.get("du", 0.0), sigma_u)
    Q = np.diag(_state_tracking_q_diag(weights))
    _apply_nonlinear_ls_running_cost(
        ocp=ocp,
        model=model,
        xg=xg,
        uref=uref,
        Q=Q,
        w_u_diag=w_u_diag,
        w_du_diag=w_du_diag,
        use_actuator_dynamics=use_actuator_dynamics,
        use_control_rate=use_control_rate,
    )
    _apply_nonlinear_ls_terminal_cost_method5(ocp, model, xg, Q, weights, idx_T)

    _apply_terminal_position_constraint(ocp, model, xg, weights)
    _apply_control_bounds(ocp, bounds)
    _apply_state_path_bounds(ocp, model, bounds)

    x0_arr = _augment_x0_with_uref_if_needed(x0, uref, use_actuator_dynamics, use_control_rate)
    # Do not pin full x0 via ocp.constraints.x0 — that fixes T_seg to the initial guess.
    _apply_free_tf_state_time_bounds(ocp, nx, idx_T, x0_arr, T_min, T_max)

    if use_actuator_dynamics:
        _apply_actuator_parameter_values(ocp, weights)

    ocp.solver_options.qp_solver = qp_solver or "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 2 if verbose_solve else 0
    ocp.solver_options.store_iterates = verbose_solve
    ocp.solver_options.nlp_solver_max_iter = int(nlp_solver_max_iter)

    return ocp

# method 7: free-tf with external cost
def build_acados_ocp_free_tf_external(
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
):
    """
    Free final time on pseudo-time τ∈[0,1] with physical duration ``tf`` as the last state (same
    dynamics as ``export_tvc_ode_model_pseudotime``). **EXTERNAL** cost (cf. time-scaling):

    - Running (per stage): ``(tf/N) * ( w_T*T^2 + w_tvc*(th_p^2+th_r^2) + w_ty*tau_yaw^2 )``
      with ``T`` the commanded thrust unless actuators are used (then actual thrust state).
    - Terminal: ``w_terminal_tf * tf`` plus optional quadratic on ``x[:12]`` vs goal (Mayer-style).

    Uses ``hessian_approx = EXACT`` (recommended for general EXTERNAL costs).
    """
    _require_acados_ocp()
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

    running, term_e = _build_external_free_tf_cost_expr(
        model=model,
        xg=xg,
        weights=weights,
        N=N,
        idx_T=idx_T,
        use_actuator_dynamics=use_actuator_dynamics,
    )

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    model.cost_expr_ext_cost = running
    model.cost_expr_ext_cost_e = term_e

    _apply_terminal_position_constraint(ocp, model, xg, weights)
    _apply_control_bounds(ocp, bounds)
    _apply_state_path_bounds(ocp, model, bounds)
    x0_arr = _augment_x0_with_uref_if_needed(x0, uref, use_actuator_dynamics, use_control_rate)
    _apply_free_tf_state_time_bounds(ocp, nx, idx_T, x0_arr, T_min, T_max)

    if use_actuator_dynamics:
        _apply_actuator_parameter_values(ocp, weights)

    ocp.solver_options.qp_solver = qp_solver or "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 2 if verbose_solve else 0
    ocp.solver_options.store_iterates = verbose_solve
    ocp.solver_options.nlp_solver_max_iter = int(nlp_solver_max_iter)

    return ocp
