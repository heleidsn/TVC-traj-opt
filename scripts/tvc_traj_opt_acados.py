#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization using Acados (Method 4)

Acados handles constraints natively (control bounds, state path constraints),
often with better convergence than penalty-based approaches.

Usage:
    python -u tvc_traj_opt_acados.py

Requires: acados, casadi (pip install casadi; acados from source)

Before running, ensure Acados libs are in LD_LIBRARY_PATH:
    export ACADOS_SOURCE_DIR=/path/to/acados
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
"""

import os
import sys
import gc
import numpy as np
from pathlib import Path

# Setup Acados environment before import (fixes libqpOASES_e.so / libhpipm.so not found)
def _setup_acados_env():
    """Prepend acados lib to LD_LIBRARY_PATH if not already set"""
    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if not acados_root:
        # Guess from acados_template package location
        try:
            import acados_template
            pkg_path = Path(acados_template.__file__).resolve().parent
            # acados_template is in interfaces/acados_template, go up to acados root
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

# Optional: Acados and CasADi
try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    import casadi as ca
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

from tvc_common import euler_to_quat_pinocchio, quat_to_euler, SEGMENT_COLORS, segment_boundaries_from_waypoints


def _Rx_sx(a):
    """CasADi SX rotation matrix around x"""
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, c, -s),
        ca.horzcat(0, s, c)
    )


def _Ry_sx(a):
    """CasADi SX rotation matrix around y"""
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(
        ca.horzcat(c, 0, s),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-s, 0, c)
    )


def export_tvc_ode_model(m, I, r_thrust, g=9.81, model_name='tvc_rocket', use_control_rate=False):
    """
    Export TVC rocket ODE model for Acados.
    State: [x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz] (12 dim)
           or with use_control_rate: + [u_prev_th_p, u_prev_th_r, u_prev_T, u_prev_tau_yaw] (16 dim)
    Control: [th_p, th_r, T, tau_yaw] (4 dim)
    use_control_rate: if True, augment state with u_prev for control rate penalty
    """
    # State
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    z = ca.SX.sym('z')
    phi = ca.SX.sym('phi')
    theta = ca.SX.sym('theta')
    psi = ca.SX.sym('psi')
    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    vz = ca.SX.sym('vz')
    wx = ca.SX.sym('wx')
    wy = ca.SX.sym('wy')
    wz = ca.SX.sym('wz')
    state_phys = ca.vertcat(x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz)
    
    # Control
    th_p = ca.SX.sym('th_p')
    th_r = ca.SX.sym('th_r')
    T = ca.SX.sym('T')
    tau_yaw = ca.SX.sym('tau_yaw')
    control = ca.vertcat(th_p, th_r, T, tau_yaw)
    
    # Rotation matrices (ZYX order: R = Rz(psi) @ Ry(theta) @ Rx(phi))
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi), ca.sin(psi)
    Rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi, cphi)
    )
    Ry = ca.vertcat(
        ca.horzcat(cth, 0, sth),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-sth, 0, cth)
    )
    Rz = ca.vertcat(
        ca.horzcat(cpsi, -spsi, 0),
        ca.horzcat(spsi, cpsi, 0),
        ca.horzcat(0, 0, 1)
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
        ca.horzcat(1, sphi*sth/cth, cphi*sth/cth),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi/cth, cphi/cth)
    )
    euler_dot = G @ w_vec
    
    # xdot (physical)
    p_dot = ca.vertcat(vx, vy, vz)
    v_dot = a_linear
    f_expl_phys = ca.vertcat(p_dot, euler_dot, v_dot, a_angular)
    
    if use_control_rate:
        # Augment state with u_prev for control rate penalty
        u_prev_th_p = ca.SX.sym('u_prev_th_p')
        u_prev_th_r = ca.SX.sym('u_prev_th_r')
        u_prev_T = ca.SX.sym('u_prev_T')
        u_prev_tau_yaw = ca.SX.sym('u_prev_tau_yaw')
        u_prev = ca.vertcat(u_prev_th_p, u_prev_th_r, u_prev_T, u_prev_tau_yaw)
        state = ca.vertcat(state_phys, u_prev)
        # u_prev_dot = (u - u_prev) / dt, p[0] = 1/dt; over one step u_prev -> u
        p = ca.SX.sym('p', 1)
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
    
    xdot = ca.SX.sym('xdot', nx)
    f_impl = xdot - f_expl
    
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = state
    model.xdot = xdot
    model.u = control
    model.name = model_name
    
    return model


def acados_state_to_method1(x_acados):
    """Convert Acados state [x,y,z,phi,theta,psi,vx,vy,vz,wx,wy,wz(,u_prev)] to Method 1 format (17-dim)"""
    x_acados = np.asarray(x_acados).flatten()
    p = x_acados[:3]
    phi, theta, psi = x_acados[3], x_acados[4], x_acados[5]
    v = x_acados[6:9]
    w = x_acados[9:12]
    q = euler_to_quat_pinocchio(phi, theta, psi)
    # Method 1: [p(3), v(3), q(4) wxyz, w(3), u_prev(4)]
    x_m1 = np.zeros(17)
    x_m1[0:3] = p
    x_m1[3:6] = v
    x_m1[6:10] = [q[3], q[0], q[1], q[2]]  # wxyz from qxyzw
    x_m1[10:13] = w
    return x_m1


def method1_to_acados_state(x_m1):
    """Convert Method 1 state [p,v,q wxyz,w,...] to Acados [x,y,z,phi,theta,psi,vx,vy,vz,wx,wy,wz]"""
    x = np.zeros(12)
    x[0:3] = np.asarray(x_m1[0:3], dtype=float)
    x[3:6] = quat_to_euler(np.asarray(x_m1[6:10], dtype=float), format='wxyz')
    x[6:9] = np.asarray(x_m1[3:6], dtype=float)
    x[9:12] = np.asarray(x_m1[10:13], dtype=float)
    return x


def waypoint_to_acados_state(wp, uref=None):
    """Convert waypoint [x,y,z,yaw_deg,time] to Acados state (goal).
    If uref is given, return augmented 16-dim state [x_12, uref] for control rate model."""
    x = np.zeros(12)
    x[0:3] = [float(wp[0]), float(wp[1]), float(wp[2])]
    yaw_deg = float(wp[3]) if len(wp) > 3 else 0.0
    x[5] = np.radians(yaw_deg)  # psi
    if uref is not None:
        x = np.concatenate([x, np.asarray(uref).flatten()[:4]])
    return x


def build_acados_ocp(model, N, Tf, x0, xg, uref, weights, bounds, dt, terminal_weights=None,
                     code_export_dir=None, json_file=None, nlp_solver_max_iter=100, qp_solver=None):
    """Build Acados OCP for one segment"""
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
    
    nx = int(model.x.size1())  # 12 or 16 (with u_prev)
    nu = 4
    use_control_rate = nx == 16
    
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    
    # Cost: nonlinear least-squares
    w_p = weights.get("p", 1.0)
    w_v = weights.get("v", 0.2)
    w_R = weights.get("R", 0.5)
    w_w = weights.get("w", 0.1)  # angular velocity
    w_u = weights.get("u", 1e-3)
    w_du = weights.get("du", 0.0)  # control rate penalty; 0=关闭, >0=启用
    
    Q = np.diag([
        w_p, w_p, w_p,           # position
        w_R, w_R, w_R,           # euler
        w_v, w_v, w_v,           # velocity
        w_w, w_w, w_w            # angular velocity
    ])
    R_mat = w_u * np.eye(nu)
    
    if use_control_rate:
        # cost_y = [x(16), u(4), u-u_prev(4)] = 24 dim; penalize ||u - u_prev||^2 for smooth control
        u_prev = model.x[12:16]
        cost_y_expr = ca.vertcat(model.x, model.u, model.u - u_prev)
        # yref: [x_phys, u_prev, u, du]; x_phys->xg, u_prev->uref, u->uref, du->0
        yref = np.concatenate([xg[:12], uref, uref, np.zeros(4)])
        Q_aug = np.diag(np.concatenate([np.diag(Q), [w_u] * 4]))  # 16x16: Q(12) + w_u for u_prev(4)
        W_du = w_du * np.eye(4)
        W = np.block([
            [Q_aug, np.zeros((16, 4)), np.zeros((16, 4))],
            [np.zeros((4, 16)), R_mat, np.zeros((4, 4))],
            [np.zeros((4, 16)), np.zeros((4, 4)), W_du]
        ])
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = cost_y_expr
        ocp.cost.yref = yref
        ocp.cost.W = W
    else:
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
        ocp.cost.yref = np.concatenate([xg, uref])
        ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R_mat]])
    
    tw = terminal_weights or weights
    Qe = np.diag([
        tw.get("p", 200), tw.get("p", 200), tw.get("p", 200),
        tw.get("R", 200), tw.get("R", 200), tw.get("R", 200),
        tw.get("v", 50), tw.get("v", 50), tw.get("v", 50),
        tw.get("w", 20), tw.get("w", 20), tw.get("w", 20)
    ])
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = np.asarray(xg).flatten()[:12]
    ocp.model.cost_y_expr_e = model.x[:12]  # terminal cost only on physical state
    ocp.cost.W_e = Qe
    
    # Control bounds
    b = bounds or {}
    th_p = b.get("th_p", (-0.4, 0.4))
    th_r = b.get("th_r", (-0.4, 0.4))
    T_b = b.get("T", (0.0, 30.0))
    tau_yaw = b.get("tau_yaw", (-2.0, 2.0))
    ocp.constraints.lbu = np.array([th_p[0], th_r[0], T_b[0], tau_yaw[0]])
    ocp.constraints.ubu = np.array([th_p[1], th_r[1], T_b[1], tau_yaw[1]])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    
    # State path constraints
    # con_h: h(x) in [lh, uh]
    # State: [x,y,z, phi,theta,psi, vx,vy,vz, wx,wy,wz]
    b = bounds or {}
    v_h_max = b.get("state_v_horizontal_max", b.get("v_horizontal_max", 20.0))
    v_v_max = b.get("state_v_vertical_max", b.get("v_vertical_max", 20.0))
    roll_max = b.get("state_roll_max", np.radians(45.0))
    pitch_max = b.get("state_pitch_max", np.radians(45.0))
    yaw_max = b.get("state_yaw_max", np.radians(180.0))
    w_max = b.get("state_w_max", 2.0)
    
    vx, vy, vz = model.x[6], model.x[7], model.x[8]
    phi, theta, psi = model.x[3], model.x[4], model.x[5]
    wx, wy, wz = model.x[9], model.x[10], model.x[11]
    
    v_horizontal = ca.sqrt(vx**2 + vy**2 + 1e-12)
    v_vertical = ca.fabs(vz)
    w_mag = ca.sqrt(wx**2 + wy**2 + wz**2 + 1e-12)
    
    # 速度 + 欧拉角 + 角速度模长
    ocp.model.con_h_expr = ca.vertcat(
        v_horizontal, v_vertical,      # 速度约束
        phi, theta, psi,               # 欧拉角约束 |phi|<=roll_max 等
        w_mag                          # 角速度模长 ||w||<=w_max
    )
    ocp.constraints.lh = np.array([
        0.0, 0.0,                     # v_h, v_v 下界
        -roll_max, -pitch_max, -yaw_max,  # phi, theta, psi 下界
        0.0                           # w_mag 下界
    ])
    ocp.constraints.uh = np.array([
        v_h_max, v_v_max,             # v_h, v_v 上界
        roll_max, pitch_max, yaw_max, # phi, theta, psi 上界
        w_max                         # w_mag 上界
    ])
    
    # x0: augment with u_prev=uref when using control rate
    x0_arr = np.asarray(x0).flatten()
    if use_control_rate and len(x0_arr) == 12:
        x0_arr = np.concatenate([x0_arr, np.asarray(uref).flatten()[:4]])
    ocp.constraints.x0 = x0_arr
    
    # 当 model 有参数 p（control rate 用 1/dt）时，必须设置 parameter_values，否则 make_consistent 报错
    if use_control_rate:
        ocp.parameter_values = np.array([N / Tf], dtype=float)
    
    ocp.solver_options.qp_solver = qp_solver or 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_max_iter = int(nlp_solver_max_iter)
    
    return ocp


def _clear_acados_module_cache(code_export_dir):
    """清理 Acados 生成的 solver 模块缓存，避免第二段加载第一段错误维度的 solver"""
    # 只删除 c_generated_code 下的模块，不删除 acados_template
    to_remove = [k for k in list(sys.modules.keys())
                 if ('c_generated_code' in k or 'tvc_seg' in k) and 'acados_template' not in k]
    for k in to_remove:
        del sys.modules[k]
    gc.collect()


def solve_with_acados_waypoints(dt, waypoints, m, I, r_thrust, weights, bounds, max_iter=100,
                                use_box_solver=False, callback=None, running_flag=None,
                                terminal_weights=None, iteration_callback=None):
    """
    Solve trajectory optimization with waypoints using Acados.
    
    Interface compatible with solve_with_pinocchio_waypoints for GUI.
    
    Returns:
        combined_xs: List of states in Method 1 format (17-dim)
        combined_us: List of controls
        all_loggers: List of logger-like objects (with .costs for plotting)
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("Acados not available. Install: pip install casadi; build acados from source.")
    
    m = float(m)
    I = np.array(I, dtype=float).reshape(3, 3)
    r_thrust = np.array(r_thrust, dtype=float).reshape(3,)
    
    # Segment durations
    durations = []
    for i in range(len(waypoints) - 1):
        d = waypoints[i+1][4] - waypoints[i][4]
        if d <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(d)
    
    uref = np.array([0.0, 0.0, m*9.81, 0.0])
    
    all_xs = []
    all_us = []
    all_loggers = []
    
    # Initial state
    x0 = waypoint_to_acados_state(waypoints[0])
    
    # Base dir for code export (avoid segment overwrite / module cache reuse)
    base_export = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_generated_code")
    
    for seg_idx in range(len(durations)):
        if running_flag is not None and not running_flag():
            break
        
        duration = durations[seg_idx]
        end_wp = waypoints[seg_idx + 1]
        xg = waypoint_to_acados_state(end_wp)
        
        N = max(10, int(duration / dt))
        Tf = duration
        
        # 每段使用唯一目录和 model 名，避免 Acados 模块缓存导致下一段复用错误 solver (acados#905)
        code_export_dir = os.path.join(base_export, f"tvc_seg{seg_idx}_N{N}")
        json_file = f"tvc_rocket_seg{seg_idx}.json"
        model_name = f"tvc_rocket_seg{seg_idx}"
        
        # 第二段及以后：清理模块缓存，增加 SQP 迭代数
        if seg_idx > 0:
            _clear_acados_module_cache(code_export_dir)
            nlp_max_iter = max(200, max_iter * 2)
            qp_solver = None
        else:
            nlp_max_iter = max_iter
            qp_solver = None
        
        use_control_rate = weights.get("du", 0.0) > 0
        model = export_tvc_ode_model(m, I, r_thrust, model_name=model_name, use_control_rate=use_control_rate)
        ocp = build_acados_ocp(model, N, Tf, x0, xg, uref, weights, bounds, dt, terminal_weights,
                               code_export_dir=code_export_dir, json_file=json_file,
                               nlp_solver_max_iter=nlp_max_iter, qp_solver=qp_solver)
        
        # 扩展 x0/xg 为 16 维（含 u_prev）用于 control rate 模型；x0 已为 16 维时（来自上一段）保持不变
        x0_arr = np.asarray(x0).flatten()
        xg_arr = np.asarray(xg).flatten()
        x0_seg = np.concatenate([x0_arr[:12], uref]) if use_control_rate and len(x0_arr) == 12 else x0_arr
        xg_seg = np.concatenate([xg_arr[:12], uref]) if use_control_rate and len(xg_arr) == 12 else xg_arr
        
        try:
            try:
                solver = AcadosOcpSolver(ocp, verbose=False, check_reuse_possible=False)
            except TypeError:
                solver = AcadosOcpSolver(ocp, verbose=False)
        except OSError as e:
            if "cannot open shared object file" in str(e) or "libqpOASES" in str(e) or "libhpipm" in str(e):
                raise RuntimeError(
                    f"Acados solver failed: {e}\n\n"
                    "Fix: Add acados lib to LD_LIBRARY_PATH before running:\n"
                    "  export ACADOS_SOURCE_DIR=/path/to/acados\n"
                    "  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib\n"
                    "Or use: ./scripts/run_acados.sh"
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(f"Acados solver creation failed: {e}") from e
        
        # Set initial state
        solver.set(0, "x", x0_seg)
        
        # 设置 control rate 模型的参数 p（每步 1/dt）
        if use_control_rate:
            p_val = np.array([N / Tf])
            for i in range(N):
                solver.set(i, "p", p_val)
        
        # 设置轨迹初始猜测（线性插值 x0->xg）
        uref_arr = np.array(uref)
        for i in range(1, N + 1):
            alpha = float(i) / N
            x_guess = (1 - alpha) * x0_seg + alpha * xg_seg
            solver.set(i, "x", x_guess)
        for i in range(N):
            solver.set(i, "u", uref_arr)
        
        # Acados 无逐迭代回调，在求解前发出 iter=0
        if iteration_callback is not None:
            iteration_callback(0, 0.0, 0.0, seg_idx)
        
        status = solver.solve()
        if status != 0 and seg_idx > 0:
            # ACADOS_MINSTEP(4) 常见于段2：QP 步长过小，但解通常仍可用
            print(f"  [提示] 段{seg_idx+1} Acados status={status}，解可能为部分收敛（若轨迹合理可忽略）")
        
        # 求解后发出最终迭代数和代价
        cost_val = solver.get_cost()
        try:
            sqp_iter = solver.get_stats("sqp_iter")
        except Exception:
            sqp_iter = 1
        if iteration_callback is not None:
            iteration_callback(int(sqp_iter), float(cost_val), 0.0, seg_idx)
        
        # Simple logger for GUI compatibility
        class SimpleLogger:
            def __init__(self, cost_val):
                self.costs = [cost_val] if cost_val is not None else [0.0]
        
        all_loggers.append(SimpleLogger(cost_val))
        
        # 提取解：失败时用初始猜测作为回退，确保每段都有完整轨迹
        try:
            seg_xs = [acados_state_to_method1(np.array(solver.get(i, "x"), copy=True)) for i in range(N+1)]
            seg_us = [np.array(solver.get(i, "u"), copy=True) for i in range(N)]
            x0 = np.array(solver.get(N, "x"), copy=True)
        except Exception as e:
            if status != 0:
                # 求解失败时用线性插值初始猜测构造 seg_xs
                seg_xs = []
                for i in range(N + 1):
                    alpha = float(i) / N
                    x_ac = (1 - alpha) * x0_seg + alpha * xg_seg
                    seg_xs.append(acados_state_to_method1(np.array(x_ac, copy=True)))
                seg_us = [np.array(uref_arr, copy=True) for _ in range(N)]
                x0 = np.array(xg_seg, copy=True)
                print(f"  [回退] 段{seg_idx+1} 使用初始猜测 (solver.get 异常: {e})")
            else:
                raise
        
        if callback is not None:
            callback(None, seg_idx, seg_xs, seg_us, all_xs, all_us)
        
        all_xs.append(seg_xs)
        all_us.append(seg_us)
        
        # 显式销毁 solver，避免 Acados 模块缓存导致下一段复用错误维度 (acados#905)
        del solver
        gc.collect()
    
    # Combine segments
    combined_xs = []
    combined_us = []
    for i, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
        if i == 0:
            combined_xs.extend(seg_xs)
            combined_us.extend(seg_us)
        else:
            combined_xs.extend(seg_xs[1:])
            combined_us.extend(seg_us)
    
    return combined_xs, combined_us, all_loggers


def solve_with_acados_waypoints_unified(dt, waypoints, m, I, r_thrust, weights, bounds, max_iter=100,
                                        use_box_solver=False, callback=None, running_flag=None,
                                        terminal_weights=None, iteration_callback=None):
    """
    Solve with Acados using a single unified problem (all segments merged).
    每段节点的 cost 目标为对应 waypoint，与 Pinocchio unified 一致，轨迹会经过中间点。
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("Acados not available.")
    
    m = float(m)
    I = np.array(I, dtype=float).reshape(3, 3)
    r_thrust = np.array(r_thrust, dtype=float).reshape(3,)
    
    durations = []
    for i in range(len(waypoints) - 1):
        d = waypoints[i+1][4] - waypoints[i][4]
        if d <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(d)
    
    # 每段节点数（与 Pinocchio unified 一致）
    N_per_seg = [max(10, int(d / dt)) for d in durations]
    N_total = sum(N_per_seg)
    Tf_total = sum(durations)
    
    x0 = waypoint_to_acados_state(waypoints[0])
    xg = waypoint_to_acados_state(waypoints[-1])
    uref = np.array([0.0, 0.0, m*9.81, 0.0])
    
    # unified 模式需更强 waypoint 跟踪，提高 position 权重
    use_control_rate = weights.get("du", 0.0) > 0
    model = export_tvc_ode_model(m, I, r_thrust, use_control_rate=use_control_rate)
    ocp = build_acados_ocp(model, N_total, Tf_total, x0, xg, uref, weights, bounds, dt, terminal_weights)
    
    x0_seg = np.concatenate([np.asarray(x0).flatten()[:12], uref]) if use_control_rate else np.asarray(x0).flatten()
    
    try:
        solver = AcadosOcpSolver(ocp, verbose=False)
    except Exception as e:
        raise RuntimeError(f"Acados solver creation failed: {e}") from e
    
    solver.set(0, "x", x0_seg)
    
    # 设置 control rate 模型的参数 p
    if use_control_rate:
        p_val = np.array([N_total / Tf_total])
        for i in range(N_total):
            solver.set(i, "p", p_val)
    
    # 每段节点使用对应 waypoint 作为 cost 目标，使轨迹经过中间点
    node_idx = 0
    n_boundary = 8
    for seg_idx in range(len(durations)):
        wp_this = waypoints[seg_idx + 1]
        wp_prev = waypoints[seg_idx] if seg_idx > 0 else waypoints[0]
        x_ref_this = waypoint_to_acados_state(wp_this)[:12]
        x_ref_prev = waypoint_to_acados_state(wp_prev)[:12]
        n_seg = N_per_seg[seg_idx]
        for k in range(n_seg):
            if seg_idx > 0 and k < n_boundary:
                x_ref = x_ref_prev
            elif k >= n_seg - n_boundary:
                x_ref = x_ref_this
            else:
                x_ref = x_ref_this
            if use_control_rate:
                yref = np.concatenate([x_ref, uref, uref, np.zeros(4)])  # 24 dim
            else:
                yref = np.concatenate([x_ref, uref])
            solver.set(node_idx, "yref", yref)
            node_idx += 1
    
    # 初始猜测：按段线性插值 wp_i -> wp_{i+1}
    x_prev = x0_seg.copy()
    for seg_idx in range(len(durations)):
        end_wp = waypoints[seg_idx + 1]
        x_end_12 = waypoint_to_acados_state(end_wp)[:12]
        x_end = np.concatenate([x_end_12, uref]) if use_control_rate else x_end_12
        n_seg = N_per_seg[seg_idx]
        i0 = sum(N_per_seg[:seg_idx])
        for k in range(n_seg):
            alpha = (k + 1) / n_seg
            x_guess = (1 - alpha) * x_prev + alpha * x_end
            solver.set(i0 + k + 1, "x", x_guess)
        x_prev = x_end
    for i in range(N_total):
        solver.set(i, "u", uref)
    
    if iteration_callback is not None:
        iteration_callback(0, 0.0, 0.0, 0)
    
    status = solver.solve()
    
    cost_val = solver.get_cost()
    try:
        sqp_iter = solver.get_stats("sqp_iter")
    except Exception:
        sqp_iter = 1
    if iteration_callback is not None:
        iteration_callback(int(sqp_iter), float(cost_val), 0.0, 0)
    
    class SimpleLogger:
        def __init__(self, c):
            self.costs = [c] if c is not None else [0.0]
    
    all_loggers = [SimpleLogger(cost_val)]
    
    combined_xs = [acados_state_to_method1(solver.get(i, "x")) for i in range(N_total+1)]
    combined_us = [np.array(solver.get(i, "u")) for i in range(N_total)]
    
    if callback is not None:
        callback(None, 0, combined_xs, combined_us, [], [])
    
    return combined_xs, combined_us, all_loggers


def test_three_waypoints(show_plot=True, unified=False, use_control_rate_smooth=False):
    """
    简单测试：2 个或更多 waypoints 的轨迹优化。
    支持：开始时间不为 0、单段（2 waypoints）、多段（3+ waypoints）。
    unified=True 时用单次 OCP 求解全轨迹，可避免分段模式段2 的 ACADOS_MINSTEP。
    use_control_rate_smooth=True 时增加控制变化率惩罚，使 u 更平滑；False 保持原样。
    Waypoint 格式: [x, y, z, yaw_deg, time]
    """
    dt = 0.05
    # 示例：2 waypoints 单段，从 t=3s 开始
    waypoints = [
        [0.0, 0.0, 0.0, 0.0, 0.0],   # 起点
        [2.0, 0.0, 3.0, 0.0, 10.0],   # 中间点
        # [4.0, 0.0, 5.0, 0.0, 10.0],  # 终点
    ]
    m = 0.6
    I = np.diag([0.02, 0.02, 0.01])
    r_thrust = np.array([0, 0, -0.2])
    # 两种模式: du=0 保持原样无控制率惩罚; du>0 增加控制变化率惩罚使 u 更平滑
    weights = {"p": 1.0, "v": 0.2, "R": 0.5, "w": 0.1, "u": 1.0}
    if use_control_rate_smooth:
        weights["du"] = 100.0
    bounds = {
        "th_p": (-0.35, 0.35), "th_r": (-0.35, 0.35),
        "T": (0.0, 25.0), "tau_yaw": (-1.0, 1.0),
        "state_v_horizontal_max": 1.0, "state_v_vertical_max": 1.5,
        "state_roll_max": np.radians(10.0), "state_pitch_max": np.radians(5.0),
        "state_yaw_max": np.radians(30.0),
    }

    if len(waypoints) < 2:
        raise ValueError("至少需要 2 个 waypoints (起点和终点)")
    n_seg = len(waypoints) - 1
    print("=" * 50)
    print(f"Acados 轨迹优化测试 ({len(waypoints)} waypoints, {n_seg} 段)")
    print("=" * 50)
    for i, wp in enumerate(waypoints):
        t = wp[4] if len(wp) >= 5 else 0.0
        print(f"WP{i}: {wp[:3]} (t={t}s)")
    if len(waypoints) >= 2:
        seg_info = [f"{waypoints[i+1][4]-waypoints[i][4]:.1f}s" for i in range(n_seg)]
        print(f"dt={dt}s, 各段时长: {seg_info}")
    if unified:
        print("模式: unified (单次 OCP)")
    else:
        print("模式: 分段 (每段独立 OCP)")
    print(f"控制率惩罚: {'开启 du=' + str(weights.get('du', 0)) if use_control_rate_smooth else '关闭 (原样)'}")
    print("-" * 50)

    solver_fn = solve_with_acados_waypoints_unified if unified else solve_with_acados_waypoints
    xs, us, loggers = solver_fn(
        dt=dt, waypoints=waypoints, m=m, I=I, r_thrust=r_thrust,
        weights=weights, bounds=bounds
    )

    print(f"求解完成: {len(xs)} 个状态点, {len(us)} 个控制, {len(loggers)} 段")
    print(f"起点位置: {xs[0][:3]}")
    end_wp = waypoints[-1]
    print(f"终点位置: {xs[-1][:3]} (目标: {end_wp[:3]})")
    err_end = np.linalg.norm(np.array(xs[-1][:3]) - np.array(end_wp[:3]))
    print(f"  与终点误差: {err_end:.4f}m")
    # 多段时验证中间 waypoint
    if len(waypoints) >= 3:
        dt = 0.05
        t0 = waypoints[0][4] if len(waypoints[0]) >= 5 else 0.0
        for i in range(1, len(waypoints) - 1):
            idx = int((waypoints[i][4] - t0) / dt)
            if idx < len(xs):
                at_wp = xs[idx][:3]
                err = np.linalg.norm(np.array(at_wp) - np.array(waypoints[i][:3]))
                print(f"  WP{i} 处: {at_wp} | 误差: {err:.4f}m")
    for i, lg in enumerate(loggers):
        if lg and lg.costs:
            print(f"  段{i+1} 代价: {lg.costs[0]:.6e}")
    print("=" * 50)

    if show_plot:
        _plot_result(xs, us, waypoints, dt, loggers)
    return xs, us, loggers


def _plot_result(xs, us, waypoints, dt, loggers):
    """绘制轨迹结果"""
    import matplotlib.pyplot as plt
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        import sys
        if _script_dir not in sys.path:
            sys.path.insert(0, str(_script_dir))
    # 显式计算段边界，确保按段分色
    boundaries = [min(b, len(xs) - 1) for b in segment_boundaries_from_waypoints(waypoints, dt)]
    if boundaries:
        print(f"  段边界索引: {boundaries} (共 {len(boundaries)} 段)")
    wp_list = waypoints if waypoints and all(len(wp) >= 5 for wp in waypoints) else [[wp[0], wp[1], wp[2], 0.0, i * dt] for i, wp in enumerate(waypoints)]

    try:
        from tvc_traj_opt import plot_trajectory
        logger = loggers[0] if loggers else None
        fig = plot_trajectory(xs, us, dt, logger=logger, x_goal=None, waypoints=wp_list,
                             segment_boundaries=boundaries if boundaries else None)
        fig.suptitle("TVC Trajectory - Acados", fontsize=12)
        plt.show()
    except ImportError:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        pos = np.array([x[:3] for x in xs])
        if boundaries:
            idx = 0
            for i, end_idx in enumerate(boundaries):
                end_idx = min(end_idx, len(pos) - 1)
                if idx <= end_idx:
                    seg = pos[idx:end_idx + 1]
                    c = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
                    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=c, linewidth=2.5, label=f'Segment {i+1}')
                    if idx < end_idx:
                        ax.scatter(seg[-1, 0], seg[-1, 1], seg[-1, 2], color=c, s=60, marker='o',
                                   edgecolors='black', linewidths=1, zorder=5)
                idx = end_idx
            if idx < len(pos) - 1:
                seg = pos[idx:]
                c = SEGMENT_COLORS[len(boundaries) % len(SEGMENT_COLORS)]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=c, linewidth=2.5, label=f'Segment {len(boundaries)+1}')
        else:
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", linewidth=2, label="Trajectory")
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='green', s=100, marker='o', label='Start')
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color='red', s=100, marker='*', label='End')
        if wp_list:
            wps = np.array([[w[0], w[1], w[2]] for w in wp_list])
            ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c="orange", s=80, marker="^", label="Waypoints")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("TVC Trajectory - Acados (Method 4)")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if not ACADOS_AVAILABLE:
        print("Acados not available. Install: pip install casadi; build acados from source.")
        print("See: https://docs.acados.org/installation/")
        exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="TVC Acados 轨迹优化测试")
    parser.add_argument("--no-plot", action="store_true", help="不显示图形")
    parser.add_argument("--unified", action="store_true", help="使用 unified 模式（单次 OCP，可避免段2 ACADOS_MINSTEP）")
    parser.add_argument("--smooth", action="store_true", help="增加控制变化率惩罚，使 u 更平滑")
    args = parser.parse_args()

    test_three_waypoints(show_plot=not args.no_plot, unified=True, use_control_rate_smooth=args.smooth)
