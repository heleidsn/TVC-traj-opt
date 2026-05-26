# -*- coding: utf-8 -*-
"""
TVC rigid-body ODE as ``AcadosModel`` (CasADi SX).

- ``export_tvc_ode_model``: physical-time dynamics, 12 / 16 states.
- ``export_tvc_ode_model_pseudotime``: normalized-time formulation with segment duration state.

OCP costs: ``tvc_traj_opt_acados_cost.py``. State / waypoint maps: ``tvc_traj_opt_acados_state.py``.
"""

try:
    import casadi as ca
    from acados_template import AcadosModel
except ImportError:
    ca = None
    AcadosModel = None


def _require_casadi_acados_model():
    if ca is None or AcadosModel is None:
        raise ImportError(
            "casadi and acados_template (AcadosModel) are required for TVC dynamics export."
        )


def _Rx_sx(a):
    """CasADi SX rotation matrix around x"""
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, c, -s),
        ca.horzcat(0, s, c),
    )


def _Ry_sx(a):
    """CasADi SX rotation matrix around y"""
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(
        ca.horzcat(c, 0, s),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-s, 0, c),
    )


def export_tvc_ode_model(
    m,
    I,
    r_thrust,
    g=9.81,
    model_name="tvc_rocket",
    use_control_rate=False,
    use_actuator_dynamics=False,
    actuator_tau=None,
):
    """
    Export TVC rocket ODE model for Acados.
    State: [x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz] (12 dim)
           or with use_control_rate: + [u_prev_th_p, u_prev_th_r, u_prev_T, u_prev_tau_yaw] (16 dim)
           or with use_actuator_dynamics: + [u_act_th_p, u_act_th_r, u_act_T, u_act_tau_yaw] (16 dim)
    Control: [th_p, th_r, T, tau_yaw] (4 dim) - commanded; plant uses u_actual when actuator
    use_control_rate: if True, augment state with u_prev for control rate penalty
    use_actuator_dynamics: if True, first-order actuator u_act_dot = (u_cmd - u_act)/tau per channel
    actuator_tau: [tau_pitch, tau_roll, tau_T, tau_yaw] in seconds; default [0.05]*4
    """
    _require_casadi_acados_model()
    if actuator_tau is None:
        actuator_tau = [0.05, 0.05, 0.05, 0.05]
    # Actuator dynamics and control rate are mutually exclusive (both extend state to 16)
    if use_actuator_dynamics:
        use_control_rate = False
    # State
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

    # Control (u_cmd)
    th_p_cmd = ca.SX.sym("th_p")
    th_r_cmd = ca.SX.sym("th_r")
    T_cmd = ca.SX.sym("T")
    tau_yaw_cmd = ca.SX.sym("tau_yaw")
    control = ca.vertcat(th_p_cmd, th_r_cmd, T_cmd, tau_yaw_cmd)

    # Plant input: u_actual when actuator, else u_cmd
    if use_actuator_dynamics:
        u_act_th_p = ca.SX.sym("u_act_th_p")
        u_act_th_r = ca.SX.sym("u_act_th_r")
        u_act_T = ca.SX.sym("u_act_T")
        u_act_tau_yaw = ca.SX.sym("u_act_tau_yaw")
        u_actual = ca.vertcat(u_act_th_p, u_act_th_r, u_act_T, u_act_tau_yaw)
        th_p, th_r, T, tau_yaw = u_act_th_p, u_act_th_r, u_act_T, u_act_tau_yaw
    else:
        th_p, th_r, T, tau_yaw = th_p_cmd, th_r_cmd, T_cmd, tau_yaw_cmd

    # Rotation matrices (ZYX order: R = Rz(psi) @ Ry(theta) @ Rx(phi))
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

    # TVC rotation (pitch then roll)
    Rtvc = _Ry_sx(th_p) @ _Rx_sx(th_r)
    Fb = Rtvc @ ca.vertcat(0, 0, T)
    Fw = R @ Fb
    Fg = ca.vertcat(0, 0, -g * m)
    a_linear = (Fw + Fg) / m

    # Torque
    r_thrust_sx = ca.SX(r_thrust)
    tau_thrust = ca.cross(r_thrust_sx, Fb)
    tau = tau_thrust + ca.vertcat(0, 0, tau_yaw)

    # Angular acceleration
    I_sx = ca.SX(I)
    Iinv_sx = ca.inv(I_sx)
    w_vec = ca.vertcat(wx, wy, wz)
    a_angular = Iinv_sx @ (tau - ca.cross(w_vec, I_sx @ w_vec))

    # Euler rate: eulerdot = G * w (body angular velocity)
    # ZYX convention
    G = ca.vertcat(
        ca.horzcat(1, sphi * sth / cth, cphi * sth / cth),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi / cth, cphi / cth),
    )
    euler_dot = G @ w_vec

    # xdot (physical)
    p_dot = ca.vertcat(vx, vy, vz)
    v_dot = a_linear
    f_expl_phys = ca.vertcat(p_dot, euler_dot, v_dot, a_angular)

    if use_actuator_dynamics:
        # First-order actuator: u_actual_dot = (u_cmd - u_actual) / tau per channel
        state = ca.vertcat(state_phys, u_actual)
        p = ca.SX.sym("p", 4)  # [1/tau_pitch, 1/tau_roll, 1/tau_T, 1/tau_yaw]
        inv_tau = p
        u_actual_dot = (control - u_actual) * inv_tau
        f_expl = ca.vertcat(f_expl_phys, u_actual_dot)
        nx = 16
        model = AcadosModel()
        model.p = p
    elif use_control_rate:
        # Augment state with u_prev for control rate penalty
        u_prev_th_p = ca.SX.sym("u_prev_th_p")
        u_prev_th_r = ca.SX.sym("u_prev_th_r")
        u_prev_T = ca.SX.sym("u_prev_T")
        u_prev_tau_yaw = ca.SX.sym("u_prev_tau_yaw")
        u_prev = ca.vertcat(u_prev_th_p, u_prev_th_r, u_prev_T, u_prev_tau_yaw)
        state = ca.vertcat(state_phys, u_prev)
        # u_prev_dot = (u - u_prev) / dt, p[0] = 1/dt; over one step u_prev -> u
        p = ca.SX.sym("p", 1)
        inv_dt = p[0]  # set to N/Tf per segment
        u_prev_dot = (control - u_prev) * inv_dt
        f_expl = ca.vertcat(f_expl_phys, u_prev_dot)
        nx = 16
        model = AcadosModel()
        model.p = p
    else:
        state = state_phys
        f_expl = f_expl_phys
        nx = 12
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
    """
    Same plant as ``export_tvc_ode_model`` but in **pseudo-time** τ ∈ [0,1] with an extra state ``T_seg``
    (physical segment duration, constant: dT/dτ = 0). Dynamics dx/dτ = T_seg * f(x,u) so that
    integrating τ from 0→1 equals one segment of length ``T_seg`` in physical time.

    ``N_shoot`` is required when ``use_control_rate`` (baked into u_prev dynamics in τ-space).
    """
    _require_casadi_acados_model()
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
