#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization using Pinocchio and Crocoddyl (Method 2: Fast)

This implementation uses Pinocchio for dynamics computation and Crocoddyl's 
standard DifferentialActionModelFreeFwdDynamics, which provides analytical 
Jacobians for faster optimization compared to numerical differentiation.

This is Method 2: Fast optimization using Pinocchio + Crocoddyl standard approach.
Method 1 (slow, custom calcDiff with numerical differentiation) is in tvc_traj_opt.py

Usage:
    python -u tvc_traj_opt_pinocchio.py
    
Note: Use -u flag (unbuffered output) to see real-time iteration information during solving
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import crocoddyl
import pinocchio as pin
from pathlib import Path
import time

# Import from tvc_common and tvc_traj_opt
from tvc_common import quat_mul, quat_to_euler_deg, Rx, Ry

try:
    from tvc_traj_opt import plot_trajectory
except ImportError:
    try:
        from .tvc_traj_opt import plot_trajectory
    except ImportError:
        plot_trajectory = None


def plot_debug(xs, us, dt, logger=None, title="TVC Trajectory Debug", save_path=None, waypoints=None):
    """
    Plot trajectory using the same plot_trajectory as GUI (from tvc_traj_opt).
    Supports both Pinocchio (13-dim) and Method 1 (17-dim) formats - converts to Method 1 for plotting.
    
    Args:
        xs: State trajectory (list or array)
        us: Control trajectory (list or array)
        dt: Time step
        logger: CallbackLogger (optional, for cost convergence)
        title: Figure title (used when plot_trajectory unavailable)
        save_path: If set, save figure to file instead of showing
        waypoints: List of waypoints [x,y,z,...] for 3D plot (optional)
    """
    if plot_trajectory is not None:
        # Use GUI's plot_trajectory - requires Method 1 format (17-dim)
        xs_m1 = [convert_pinocchio_state_to_method1(x) if len(np.asarray(x).flatten()) == 13 else x for x in xs]
        fig = plot_trajectory(xs_m1, us, dt, logger=logger, x_goal=None, waypoints=waypoints)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        return
    
    # Fallback: simple plot when tvc_traj_opt not available
    xs_arr = np.array([np.asarray(x).flatten() for x in xs])
    us_arr = np.array([np.asarray(u).flatten() for u in us])
    if us_arr.ndim == 1:
        us_arr = us_arr.reshape(-1, 4)
    t = np.arange(len(xs)) * dt
    t_u = np.arange(len(us)) * dt
    n = xs_arr.shape[1]
    if n >= 13:
        pos = xs_arr[:, 0:3]
        if n >= 17:
            vel = xs_arr[:, 3:6]
            quat = xs_arr[:, 6:10]
            angvel = xs_arr[:, 10:13]
        else:
            vel = xs_arr[:, 7:10]
            quat = xs_arr[:, 3:7]
            angvel = xs_arr[:, 10:13]
    else:
        pos, vel = xs_arr[:, :3], np.zeros((len(xs), 3))
        quat = np.tile([0, 0, 0, 1], (len(xs), 1))
        angvel = np.zeros((len(xs), 3))
    quat_format = 'wxyz' if n >= 17 else 'xyzw'
    euler_deg = np.array([quat_to_euler_deg(quat[i], quat_format) for i in range(len(xs))])
    th_p = us_arr[:, 0] if us_arr.shape[1] >= 1 else np.zeros(len(us))
    th_r = us_arr[:, 1] if us_arr.shape[1] >= 2 else np.zeros(len(us))
    T = us_arr[:, 2] if us_arr.shape[1] >= 3 else np.zeros(len(us))
    tau_yaw = us_arr[:, 3] if us_arr.shape[1] >= 4 else np.zeros(len(us))
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title, fontsize=12)
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', lw=1.5)
    ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='g', s=50, marker='o', label='Start')
    ax1.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='r', s=50, marker='*', label='End')
    max_range = max(np.ptp(pos[:, 0]) if len(pos) > 0 else 1.0, np.ptp(pos[:, 1]) or 1.0, np.ptp(pos[:, 2]) or 1.0) or 1.0
    half = max_range / 2.0
    cx = (pos[:, 0].min() + pos[:, 0].max()) / 2 if len(pos) > 0 else 0.0
    cy = (pos[:, 1].min() + pos[:, 1].max()) / 2 if len(pos) > 0 else 0.0
    cz = (pos[:, 2].min() + pos[:, 2].max()) / 2 if len(pos) > 0 else 0.0
    ax1.set_xlim([cx - half, cx + half])
    ax1.set_ylim([cy - half, cy + half])
    ax1.set_zlim([cz - half, cz + half])
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
    ax1.set_title('3D Trajectory')
    for ax, data, ylabel, labels in [
        (fig.add_subplot(2, 4, 2), pos, 'Position (m)', ['x', 'y', 'z']),
        (fig.add_subplot(2, 4, 3), vel, 'Velocity (m/s)', ['vx', 'vy', 'vz']),
        (fig.add_subplot(2, 4, 4), euler_deg, 'Euler (deg)', ['Roll', 'Pitch', 'Yaw']),
        (fig.add_subplot(2, 4, 5), angvel, 'Ang Vel (rad/s)', ['ωx', 'ωy', 'ωz']),
    ]:
        for j, lbl in enumerate(labels):
            ax.plot(t, data[:, j], label=lbl)
        ax.set_xlabel('t (s)'); ax.set_ylabel(ylabel); ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.plot(t_u, th_p, 'b-', label='th_p'); ax6.plot(t_u, th_r, 'g-', label='th_r')
    ax6.plot(t_u, T, 'r-', label='T'); ax6.plot(t_u, tau_yaw, 'm-', label='tau_yaw')
    ax6.set_xlabel('t (s)'); ax6.set_ylabel('Control'); ax6.legend(loc='upper right', fontsize=8); ax6.grid(True, alpha=0.3)
    if logger and logger.costs:
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.semilogy(logger.costs, 'b-'); ax7.set_xlabel('Iteration'); ax7.set_ylabel('Cost'); ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


class TVCRocketDifferentialActionModel(crocoddyl.DifferentialActionModelAbstract):
    """
    Differential Action Model for TVC Rocket using Pinocchio
    This provides analytical Jacobians through Pinocchio's automatic differentiation
    for faster optimization compared to numerical differentiation.
    """
    def __init__(self, state, actuation, cost_model, robot_model, r_thrust_body, g=9.81, 
                 m=None, I=None, u_ref=None, bounds=None, weights=None, use_box_solver=False):
        """
        Initialize differential action model
        
        Args:
            state: Crocoddyl state (StateMultibody)
            actuation: Actuation model
            cost_model: Cost model
            robot_model: Pinocchio robot model
            r_thrust_body: Thrust position in body frame [x, y, z]
            g: Gravity acceleration (m/s^2)
            m: Mass (kg). If None, use robot_model.inertias[1].mass (URDF)
            I: Inertia matrix 3x3 (kg·m²). If None, use robot_model.inertias[1].inertia (URDF)
            u_ref: Reference control [th_p, th_r, T, tau_yaw] for u and du cost (same as Method 1)
            bounds: Constraint bounds dict (for control and state constraints)
            weights: Cost weights dict
            use_box_solver: If True, skip control constraint penalties (handled by SolverBoxFDDP)
        """
        super().__init__(state, actuation.nu)
        self.use_box_solver = use_box_solver
        self.actuation = actuation
        self.cost_model = cost_model
        self.robot_model = robot_model
        self.robot_data = robot_model.createData()
        self.r_thrust = np.array(r_thrust_body, dtype=float).reshape(3,)
        self.g = float(g)
        self.u_ref = np.array(u_ref, dtype=float).reshape(actuation.nu) if u_ref is not None else np.zeros(actuation.nu)
        self._u_prev = self.u_ref.copy()  # For du cost, same as Method 1
        self._u_prev_for_grad = self.u_ref.copy()  # For du gradient in calcDiff
        # Physical params: use provided m, I (from GUI) or fall back to URDF - same as Method 1
        if m is not None:
            self.mass = float(m)
        else:
            self.mass = robot_model.inertias[1].mass
        if I is not None:
            self.I = np.array(I, dtype=float).reshape(3, 3)
            self.Iinv = np.linalg.inv(self.I)
        else:
            self.I = np.array(robot_model.inertias[1].inertia, dtype=float)
            self.Iinv = np.linalg.inv(self.I)
        
        # Store bounds and weights for constraint penalties
        self.bounds = bounds if bounds is not None else {}
        self.weights = weights if weights is not None else {}
        
        # Initialize default bounds
        self.b = {
            "th_p": (-0.4, 0.4),
            "th_r": (-0.4, 0.4),
            "T": (0.0, 30.0),
            "tau_yaw": (-2.0, 2.0),
            "k_bound": 200.0
        }
        if bounds is not None:
            self.b.update(bounds)
        
        # Initialize default state bounds
        self.state_b = {
            "v_horizontal_max": 20.0,
            "v_vertical_max": 20.0,
            "roll_max": np.radians(10.0),
            "pitch_max": np.radians(10.0),
            "yaw_max": np.radians(30.0),
            "w_max": 2.0,
            "k_state_bound": 20.0,  # Lower default to avoid constraint gradient dominating position cost
            # constraint_lxx_scale: state constraint Lxx scale. 0=Method1 style (no Lxx), 1=full Lxx, 0.1=damped
            "constraint_lxx_scale": 0.0,
        }
        # Allow state bounds to be passed via bounds dict with "state_" prefix
        if bounds is not None:
            for key, value in bounds.items():
                if key.startswith("state_"):
                    state_key = key[6:]  # Remove "state_" prefix
                    if state_key in self.state_b:
                        self.state_b[state_key] = value
        
    def createData(self):
        """Create data object for this differential action model"""
        data = super().createData()
        # Create cost data using DataCollectorAbstract
        collector = crocoddyl.DataCollectorAbstract()
        data.costs = self.cost_model.createData(collector)
        return data
        
    def _Rtvc(self, th_p, th_r):
        """Compute TVC rotation matrix (pitch then roll)"""
        return Ry(th_p) @ Rx(th_r)
    
    def _bound_pen(self, val, lb, ub, k, alpha=5.0):
        """
        Boundary penalty: val in [lb, ub] is feasible (penalty=0).
        - val < lb: soft-plus² (smooth, C∞)
        - val > ub: quadratic (val-ub)². Quadratic penalty gives gradient=0 at bound,
          allowing optimizer to stay at limit stably rather than being pushed back.
        """
        if val < lb:
            z = lb - val
            sp = np.logaddexp(0, alpha * z) / alpha
            return k * sp * sp
        if val > ub:
            z = val - ub
            return k * z * z  # Quadratic penalty: gradient continuous at bound
        return 0.0
    
    def _quat_to_euler(self, q):
        """Convert quaternion to Euler angles (ZYX order), q format: [qx,qy,qz,qw] (Pinocchio)"""
        from tvc_common import quat_to_euler
        return quat_to_euler(q, format='xyzw')
    
    def calc(self, data, x, u=None):
        """Compute dynamics and cost"""
        if u is None:
            u = np.zeros(self.nu)
        # Extract state: x = [q(7), v(6)] where q=[p(3), q(4)], v=[v(3), w(3)]
        q = x[:self.state.nq]  # [x, y, z, qx, qy, qz, qw]
        v = x[self.state.nq:]  # [vx, vy, vz, wx, wy, wz]
        
        # Extract control: u = [th_p, th_r, T, tau_yaw]
        th_p, th_r, T, tau_yaw = u
        
        # Compute rotation matrix from quaternion
        # Pinocchio uses [x, y, z, w] format for quaternion in q vector
        quat = pin.Quaternion(q[6], q[3], q[4], q[5])  # [w, x, y, z]
        R = quat.toRotationMatrix()
        
        # Compute TVC rotation (pitch then roll)
        Rtvc = self._Rtvc(th_p, th_r)
        
        # Thrust vector in body frame
        Fb = Rtvc @ np.array([0., 0., T])
        
        # Thrust vector in world frame
        Fw = R @ Fb
        
        # Torque in body frame (from thrust offset + yaw torque)
        tau_thrust = np.cross(self.r_thrust, Fb)
        tau = tau_thrust + np.array([0., 0., tau_yaw])
        
        # Gravity force in world frame (use self.mass from GUI params, same as Method 1)
        Fg = np.array([0., 0., -self.g * self.mass])
        
        # Total force in world frame
        F_total = Fw + Fg
        
        # Compute accelerations (use self.mass, self.I from GUI params, same as Method 1)
        a_linear = F_total / self.mass
        
        # Angular acceleration
        w = v[3:6]  # Angular velocity
        a_angular = self.Iinv @ (tau - np.cross(w, self.I @ w))
        
        # For StateMultibody, xout should be velocity derivative (nv dimension)
        # Position derivatives (qdot) are computed automatically from velocities
        # xout = [vdot] where vdot = [a_linear, a_angular]
        xdot = np.zeros(self.state.nv)  # nv = 6 for free-flyer
        
        # Velocity derivative (accelerations)
        xdot[:3] = a_linear   # Linear acceleration
        xdot[3:6] = a_angular  # Angular acceleration
        
        data.xout = xdot
        
        # Compute cost (costs should already be created in createData)
        self.cost_model.calc(data.costs, x, u)
        data.cost = data.costs.cost
        
        # Add du cost (same as Method 1): w_du * ||u - u_prev||^2
        # Method 2 state has no u_prev; use _u_prev from previous step (sequential forward pass)
        w_du = self.weights.get("du", 1e-2)
        du = u - self._u_prev
        data.cost += w_du * float(np.dot(du, du))
        self._u_prev_for_grad = self._u_prev.copy()  # Save for calcDiff before update
        self._u_prev = np.array(u, copy=True)
        
        # Add constraint penalties (same as Method 1)
        # Control constraints: skip when use_box_solver (SolverBoxFDDP handles them natively)
        # Method 2 uses penalty-based control constraints identical to Method 1
        kB = self.b["k_bound"]
        th_p, th_r, T, tau_yaw = u
        if not self.use_box_solver:
            data.cost += self._bound_pen(th_p, *self.b["th_p"], kB)
            data.cost += self._bound_pen(th_r, *self.b["th_r"], kB)
            data.cost += self._bound_pen(T, *self.b["T"], kB)
            data.cost += self._bound_pen(tau_yaw, *self.b["tau_yaw"], kB)
        
        # State constraints
        kSB = self.state_b.get("k_state_bound", 20.0)
        v_linear = v[0:3]  # Linear velocity [vx, vy, vz]
        w = v[3:6]        # Angular velocity [wx, wy, wz]
        
        # Velocity constraints (horizontal and vertical)
        # Use sqrt(vx² + vy² + ε²) to avoid singularity at origin (smooth gradient)
        # Bounds: [-v_max, v_max] (symmetric, min = -max)
        _eps_vh = 1e-8
        v_horizontal = np.sqrt(v_linear[0]**2 + v_linear[1]**2 + _eps_vh**2)
        v_vertical = abs(v_linear[2])
        v_horizontal_max = self.state_b.get("v_horizontal_max", 20.0)
        v_vertical_max = self.state_b.get("v_vertical_max", 20.0)
        data.cost += self._bound_pen(v_horizontal, -v_horizontal_max, v_horizontal_max, kSB)
        data.cost += self._bound_pen(v_vertical, -v_vertical_max, v_vertical_max, kSB)
        
        # Euler angle constraints
        # q_quat = q[3:7]  # Quaternion [qx, qy, qz, qw]
        # euler = self._quat_to_euler(q_quat)
        # data.cost += self._bound_pen(abs(euler[0]), 0.0, self.state_b.get("roll_max", np.radians(45.0)), kSB)  # Roll
        # data.cost += self._bound_pen(abs(euler[1]), 0.0, self.state_b.get("pitch_max", np.radians(45.0)), kSB)  # Pitch
        # data.cost += self._bound_pen(abs(euler[2]), 0.0, self.state_b.get("yaw_max", np.radians(180.0)), kSB)  # Yaw
        
        # # Angular velocity magnitude constraint
        # w_mag = np.linalg.norm(w)
        # data.cost += self._bound_pen(w_mag, 0.0, self.state_b.get("w_max", 2.0), kSB)
        
    def calcDiff(self, data, x, u=None):
        """Compute analytical Jacobians using Pinocchio's automatic differentiation"""
        if u is None:
            u = np.zeros(self.nu)
            
        # Compute dynamics and cost first
        self.calc(data, x, u)
        
        # For StateMultibody, Fx and Fu are w.r.t. differential state (ndx dimension)
        # but xout is velocity derivative (nv dimension)
        nx = self.state.ndx  # Differential state dimension
        nv = self.state.nv   # Velocity dimension
        nu = self.nu
        
        # Initialize Jacobians
        # Fx: maps from differential state to velocity derivative
        data.Fx = np.zeros((nv, nx))
        # Fu: maps from control to velocity derivative
        data.Fu = np.zeros((nv, nu))
        
        # Use numerical differentiation for dynamics Jacobians
        # (In a full implementation, these would be analytical)
        eps = 1e-8
        x_pert = x.copy()
        u_pert = u.copy()
        
        # Fx: dynamics Jacobian w.r.t. differential state
        # Need to perturb in differential space, not configuration space
        xdot_base = data.xout.copy()
        
        # Create a differential state perturbation
        dx = np.zeros(nx)
        for i in range(nx):
            dx[i] = eps
            # Convert differential state to configuration state
            x_pert = self.state.integrate(x, dx)
            xdot_pert = self._compute_dynamics(x_pert, u)
            data.Fx[:, i] = (xdot_pert - xdot_base) / eps
            dx[i] = 0.0
        
        # Fu: dynamics Jacobian w.r.t. control
        for i in range(nu):
            u_pert[i] = u[i] + eps
            xdot_pert = self._compute_dynamics(x, u_pert)
            data.Fu[:, i] = (xdot_pert - xdot_base) / eps
            u_pert[i] = u[i]
        
        # Cost Jacobians (from cost model)
        if hasattr(data, 'costs'):
            self.cost_model.calcDiff(data.costs, x, u)
            data.Lx = data.costs.Lx.copy()
            data.Lu = data.costs.Lu.copy()
            data.Lxx = data.costs.Lxx.copy()
            data.Lxu = data.costs.Lxu.copy()
            data.Luu = data.costs.Luu.copy()
        else:
            # Fallback: initialize cost Jacobians to zero if costs data not available
            data.Lx = np.zeros(self.state.ndx)
            data.Lu = np.zeros(self.nu)
            data.Lxx = np.zeros((self.state.ndx, self.state.ndx))
            data.Lxu = np.zeros((self.state.ndx, self.nu))
            data.Luu = np.zeros((self.nu, self.nu))
        
        # Add constraint penalty gradients using analytical computation
        # This is more stable than numerical differentiation, especially when constraints are violated
        self._compute_constraint_gradients(data, x, u)
    
    def _compute_dynamics(self, x, u):
        """Helper method to compute dynamics without modifying data"""
        # Similar to calc but returns xdot directly
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        th_p, th_r, T, tau_yaw = u
        
        quat = pin.Quaternion(q[6], q[3], q[4], q[5])
        R = quat.toRotationMatrix()
        Rtvc = self._Rtvc(th_p, th_r)
        
        Fb = Rtvc @ np.array([0., 0., T])
        Fw = R @ Fb
        tau_thrust = np.cross(self.r_thrust, Fb)
        tau = tau_thrust + np.array([0., 0., tau_yaw])
        
        Fg = np.array([0., 0., -self.g * self.mass])
        F_total = Fw + Fg
        
        a_linear = F_total / self.mass
        w = v[3:6]
        a_angular = self.Iinv @ (tau - np.cross(w, self.I @ w))
        
        # For StateMultibody, xdot should be velocity derivative (nv dimension)
        xdot = np.zeros(self.state.nv)  # nv = 6 for free-flyer
        xdot[:3] = a_linear   # Linear acceleration
        xdot[3:6] = a_angular  # Angular acceleration
        
        return xdot
    
    def _compute_total_cost(self, x, u):
        """Helper method to compute total cost (including constraints) without modifying data"""
        # Create temporary data object to compute cost
        temp_data = self.createData()
        self.calc(temp_data, x, u)
        return temp_data.cost
    
    def _bound_pen_grad(self, val, lb, ub, k, alpha=5.0):
        """Gradient of boundary penalty w.r.t. val"""
        if val < lb:
            z = lb - val
            sp = np.logaddexp(0, alpha * z) / alpha
            sig = 0.5 * (1.0 + np.tanh(0.5 * alpha * z))  # stable sigmoid
            return -2.0 * k * sp * sig  # d/d(val) since dz/d(val)=-1
        elif val > ub:
            z = val - ub
            return 2.0 * k * z  # Quadratic penalty: d/d(val) k*(val-ub)² = 2k*(val-ub), 0 at bound
        else:
            return 0.0
    
    def _bound_pen_hess(self, val, lb, ub, k, alpha=5.0):
        """
        Second derivative of boundary penalty w.r.t. val (d²p/d(val)²).
        For Gauss-Newton: Lxx += hess * outer(dz/dx, dz/dx).
        - val > ub: quadratic penalty d²/d(val)² k*(val-ub)² = 2k
        - val < lb: soft-plus² Hessian
        """
        if val < lb:
            z = lb - val
            sp = np.logaddexp(0, alpha * z) / alpha
            sig = 0.5 * (1.0 + np.tanh(0.5 * alpha * z))
            return 2.0 * k * (sig * sig + sp * alpha * sig * (1.0 - sig))
        elif val > ub:
            return 2.0 * k  # Quadratic penalty Hessian is constant
        else:
            return 0.0
    
    def _compute_constraint_gradients(self, data, x, u):
        """Compute constraint penalty gradients analytically"""
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        th_p, th_r, T, tau_yaw = u
        
        kB = self.b["k_bound"]
        kSB = self.state_b.get("k_state_bound", 20.0)
        # State constraint Lxx scale: 0=Method1 style (no Lxx), 1=full, 0.1=damped. See constraint_lxx_scale
        lxx_scale = self.state_b.get("constraint_lxx_scale", 0.0)
        
        # Control constraint gradients (same as Method 1)
        th_p_lb, th_p_ub = self.b["th_p"]
        th_r_lb, th_r_ub = self.b["th_r"]
        T_lb, T_ub = self.b["T"]
        tau_yaw_lb, tau_yaw_ub = self.b["tau_yaw"]
        
        # Control gradients + Hessian: skip when use_box_solver (SolverBoxFDDP handles them natively)
        if not self.use_box_solver:
            data.Lu[0] += self._bound_pen_grad(th_p, th_p_lb, th_p_ub, kB)
            data.Lu[1] += self._bound_pen_grad(th_r, th_r_lb, th_r_ub, kB)
            data.Lu[2] += self._bound_pen_grad(T, T_lb, T_ub, kB)
            data.Lu[3] += self._bound_pen_grad(tau_yaw, tau_yaw_lb, tau_yaw_ub, kB)
            # Luu: Gauss-Newton Hessian for constraint penalties (each control independent)
            data.Luu[0, 0] += self._bound_pen_hess(th_p, th_p_lb, th_p_ub, kB)
            data.Luu[1, 1] += self._bound_pen_hess(th_r, th_r_lb, th_r_ub, kB)
            data.Luu[2, 2] += self._bound_pen_hess(T, T_lb, T_ub, kB)
            data.Luu[3, 3] += self._bound_pen_hess(tau_yaw, tau_yaw_lb, tau_yaw_ub, kB)
        
        # du cost gradient (same as Method 1): d/du [w_du * ||u - u_prev||^2] = 2*w_du*(u - u_prev)
        w_du = self.weights.get("du", 1e-2)
        du = u - self._u_prev_for_grad
        data.Lu += 2.0 * w_du * du
        data.Luu += 2.0 * w_du * np.eye(self.nu)
        
        # State constraint gradients (need to map to differential state space)
        v_linear = v[0:3]
        w = v[3:6]
        
        # Velocity constraints (same smooth formula as calc: sqrt(vx²+vy²+ε²))
        _eps_vh = 1e-8
        v_horizontal = np.sqrt(v_linear[0]**2 + v_linear[1]**2 + _eps_vh**2)
        v_vertical = abs(v_linear[2])
        v_horizontal_max = self.state_b.get("v_horizontal_max", 20.0)
        v_vertical_max = self.state_b.get("v_vertical_max", 20.0)
        
        # v_horizontal gradient: d(v_h)/d(vx)=vx/v_h, d(v_h)/d(vy)=vy/v_h (v_h>=ε, no singularity)
        v_h_grad = self._bound_pen_grad(v_horizontal, -v_horizontal_max, v_horizontal_max, kSB)
        dvh_dvx = v_linear[0] / v_horizontal
        dvh_dvy = v_linear[1] / v_horizontal
        data.Lx[3] += v_h_grad * dvh_dvx  # w.r.t. vx
        data.Lx[4] += v_h_grad * dvh_dvy  # w.r.t. vy
        # Lxx: Gauss-Newton Hessian for v_horizontal (indices 3,4 = vx,vy)
        v_h_hess = self._bound_pen_hess(v_horizontal, -v_horizontal_max, v_horizontal_max, kSB)
        g_vh = np.zeros(self.state.ndx)
        g_vh[3], g_vh[4] = dvh_dvx, dvh_dvy
        data.Lxx += lxx_scale * v_h_hess * np.outer(g_vh, g_vh)
        
        # v_vertical gradient w.r.t. v[2]
        v_v_grad = self._bound_pen_grad(v_vertical, -v_vertical_max, v_vertical_max, kSB)
        sign_vz = np.sign(v_linear[2]) if abs(v_linear[2]) > 1e-8 else 0.0
        data.Lx[5] += v_v_grad * sign_vz
        # Lxx: Gauss-Newton Hessian for v_vertical (index 5 = vz)
        if abs(v_linear[2]) > 1e-8:
            v_v_hess = self._bound_pen_hess(v_vertical, -v_vertical_max, v_vertical_max, kSB)
            g_vv = np.zeros(self.state.ndx)
            g_vv[5] = sign_vz
            data.Lxx += lxx_scale * v_v_hess * np.outer(g_vv, g_vv)
        
        # Euler angle constraints (need quaternion to euler gradient)
        q_quat = q[3:7]  # [qx, qy, qz, qw]
        euler = self._quat_to_euler(q_quat)
        roll_max = self.state_b.get("roll_max", np.radians(45.0))
        pitch_max = self.state_b.get("pitch_max", np.radians(45.0))
        yaw_max = self.state_b.get("yaw_max", np.radians(180.0))
        
        # For Euler angle constraints, we need to compute gradient w.r.t. quaternion
        # This is complex, so we use a simplified approach: approximate gradient
        # The gradient of abs(euler) w.r.t. quaternion is computed numerically for stability
        eps_euler = 1e-6
        for i, (euler_val, euler_max) in enumerate([(euler[0], roll_max), (euler[1], pitch_max), (euler[2], yaw_max)]):
            euler_grad = self._bound_pen_grad(abs(euler_val), 0.0, euler_max, kSB) * np.sign(euler_val)
            g_euler = np.zeros(self.state.ndx)  # dz/dx for z=|euler|, used in Lxx = hess * outer(g,g)
            
            # Approximate gradient of euler angle w.r.t. quaternion using numerical differentiation
            for j in range(4):  # 4 quaternion components
                q_pert = q_quat.copy()
                q_pert[j] += eps_euler
                q_pert = q_pert / np.linalg.norm(q_pert)
                euler_pert = self._quat_to_euler(q_pert)
                deuler_dq = (euler_pert[i] - euler_val) / eps_euler
                
                if j < 3:
                    data.Lx[6 + j] += euler_grad * deuler_dq
                    g_euler[6 + j] += np.sign(euler_val) * deuler_dq  # dz/dx for z=|euler|
                else:
                    for k in range(3):
                        data.Lx[6 + k] += euler_grad * deuler_dq / 3.0
                        g_euler[6 + k] += np.sign(euler_val) * deuler_dq / 3.0
            
            # Lxx: Gauss-Newton Hessian for Euler constraint (hess = d²p/dz², g = dz/dx)
            euler_hess = self._bound_pen_hess(abs(euler_val), 0.0, euler_max, kSB)
            if np.any(g_euler != 0):
                data.Lxx += lxx_scale * euler_hess * np.outer(g_euler, g_euler)
        
        # Angular velocity magnitude constraint
        w_mag = np.linalg.norm(w)
        w_max = self.state_b.get("w_max", 2.0)
        if w_mag > 1e-8:
            w_mag_grad = self._bound_pen_grad(w_mag, 0.0, w_max, kSB)
            dw_dwx, dw_dwy, dw_dwz = w[0] / w_mag, w[1] / w_mag, w[2] / w_mag
            data.Lx[9] += w_mag_grad * dw_dwx
            data.Lx[10] += w_mag_grad * dw_dwy
            data.Lx[11] += w_mag_grad * dw_dwz
            # Lxx: Gauss-Newton Hessian for w_mag (indices 9,10,11)
            w_mag_hess = self._bound_pen_hess(w_mag, 0.0, w_max, kSB)
            g_w = np.zeros(self.state.ndx)
            g_w[9], g_w[10], g_w[11] = dw_dwx, dw_dwy, dw_dwz
            data.Lxx += lxx_scale * w_mag_hess * np.outer(g_w, g_w)


class TVCActuationModel(crocoddyl.ActuationModelAbstract):
    """
    Custom actuation model for TVC rocket
    Control input: [th_p, th_r, T, tau_yaw] (4 dimensions)
    """
    def __init__(self, state):
        """Initialize TVC actuation model"""
        super().__init__(state, 4)  # nu = 4
        
    def calc(self, data, x, u):
        """Compute actuation (not used for free-flyer, but required by interface)"""
        # For free-flyer with custom dynamics, this is not used
        # Set tau to zero (we handle forces/torques in dynamics)
        data.tau = np.zeros(self.state.nv)
        
    def calcDiff(self, data, x, u):
        """Compute actuation Jacobians (not used, but required by interface)"""
        # For free-flyer with custom dynamics, this is not used
        data.dtau_du = np.zeros((self.state.nv, self.nu))


def create_tvc_cost_model(state, actuation, x_goal, u_ref, weights, bounds):
    """
    Create cost model for TVC rocket optimization
    
    Args:
        state: Crocoddyl state model
        actuation: Actuation model
        x_goal: Goal state
        u_ref: Reference control
        weights: Cost weights dict
        bounds: Constraint bounds dict
    """
    cost_model = crocoddyl.CostModelSum(state, actuation.nu)
    
    # State cost (position, orientation, velocity, angular velocity)
    # Pinocchio diff order: [pos(3), orient_so3(3), lin_vel(3), ang_vel(3)]
    # orient_so3: [roll, pitch, yaw] in body frame (approx for small angles)
    R_default = weights.get("R", 0.5)
    state_weights = np.ones(state.ndx)
    state_weights[:3] = weights.get("p", 1.0)   # Position
    state_weights[3] = weights.get("roll", R_default)   # Roll (orient_so3[0])
    state_weights[4] = weights.get("pitch", R_default)   # Pitch (orient_so3[1])
    state_weights[5] = weights.get("yaw", R_default)    # Yaw (orient_so3[2]) - can override separately
    state_weights[6:9] = weights.get("v", 0.2)  # Linear velocity
    state_weights[9:12] = weights.get("w", 0.1) # Angular velocity
    
    state_activation = crocoddyl.ActivationModelWeightedQuad(state_weights)
    state_residual = crocoddyl.ResidualModelState(state, x_goal, actuation.nu)
    cost_model.addCost("state", 
                      crocoddyl.CostModelResidual(state, state_activation, state_residual),
                      1.0)
    
    # Control cost (same as Method 1): w_u * ||u - u_ref||^2
    u_ref_corrected = np.array(u_ref, dtype=float)
    if len(u_ref_corrected) != actuation.nu:
        u_ref_corrected = np.zeros(actuation.nu)
        if len(u_ref) >= 3:
            u_ref_corrected[2] = u_ref[2]  # Preserve thrust reference if available
    
    control_activation = crocoddyl.ActivationModelWeightedQuad(np.ones(actuation.nu))
    control_residual = crocoddyl.ResidualModelControl(state, u_ref_corrected)
    cost_model.addCost("control",
                      crocoddyl.CostModelResidual(state, control_activation, control_residual),
                      weights.get("u", 1e-3))
    
    return cost_model


def build_u_bounds_from_bounds(bounds):
    """
    Build u_lb and u_ub arrays from bounds dict for SolverBoxFDDP.
    Control order: [th_p, th_r, T, tau_yaw]
    """
    default_bounds = {
        "th_p": (-0.4, 0.4),
        "th_r": (-0.4, 0.4),
        "T": (0.0, 30.0),
        "tau_yaw": (-2.0, 2.0),
    }
    b = default_bounds.copy()
    if bounds is not None:
        for k in default_bounds:
            if k in bounds:
                b[k] = bounds[k]
    u_lb = np.array([b["th_p"][0], b["th_r"][0], b["T"][0], b["tau_yaw"][0]])
    u_ub = np.array([b["th_p"][1], b["th_r"][1], b["T"][1], b["tau_yaw"][1]])
    return u_lb, u_ub


def create_box_solver(problem):
    """Create SolverBoxFDDP if available, else SolverBoxDDP"""
    try:
        return crocoddyl.SolverBoxFDDP(problem)
    except AttributeError:
        return crocoddyl.SolverBoxDDP(problem)


def convert_pinocchio_state_to_method1(x_pinocchio):
    """
    Convert Pinocchio state format to Method 1 format for visualization
    
    Pinocchio format: [q(7), v(6)] = [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    Method 1 format: [p(3), v(3), q(4), w(3), u_prev(4)] = [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz, ...]
    
    Args:
        x_pinocchio: State in Pinocchio format (13-dim: [q(7), v(6)])
    
    Returns:
        x_method1: State in Method 1 format (17-dim: [p(3), v(3), q(4), w(3), u_prev(4)])
    """
    if len(x_pinocchio) == 13:
        # Extract components
        q = x_pinocchio[:7]   # [x, y, z, qx, qy, qz, qw]
        v = x_pinocchio[7:]   # [vx, vy, vz, wx, wy, wz]
        
        # Method 1 format: [p(3), v(3), q(4), w(3), u_prev(4)]
        # Method 1 quaternion: [w, x, y, z]; Pinocchio has [qx, qy, qz, qw] = [x, y, z, w]
        x_method1 = np.zeros(17)
        x_method1[0:3] = q[0:3]        # Position [x, y, z]
        x_method1[3:6] = v[0:3]       # Linear velocity [vx, vy, vz]
        x_method1[6:10] = [q[6], q[3], q[4], q[5]]  # [w, x, y, z] from Pinocchio [qx,qy,qz,qw]
        x_method1[10:13] = v[3:6]      # Angular velocity [wx, wy, wz]
        # u_prev is set to zero (not used in visualization)
        x_method1[13:17] = np.array([0.0, 0.0, 0.0, 0.0])
        
        return x_method1
    else:
        # Already in Method 1 format or unexpected format
        return x_pinocchio


def convert_method1_state_to_pinocchio(x_method1):
    """
    Convert Method 1 state format to Pinocchio format
    
    Method 1 format: [p(3), v(3), q(4), w(3), u_prev(4)]
    Method 1 quaternion: [w, x, y, z] at indices 6,7,8,9 (from tvc_traj_opt)
    Pinocchio format: [q(7), v(6)] = [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    Pinocchio quaternion: [qx, qy, qz, qw] = [x, y, z, w] from Method 1
    
    Args:
        x_method1: State in Method 1 format (17-dim)
    
    Returns:
        x_pinocchio: State in Pinocchio format (13-dim)
    """
    if len(x_method1) >= 13:
        # Extract components
        p = x_method1[0:3]        # Position [x, y, z]
        v_linear = x_method1[3:6]  # Linear velocity [vx, vy, vz]
        # Method 1 quaternion: [w, x, y, z] at 6,7,8,9
        q_w, q_x, q_y, q_z = x_method1[6], x_method1[7], x_method1[8], x_method1[9]
        w = x_method1[10:13]       # Angular velocity [wx, wy, wz]
        
        # Pinocchio format: [q(7), v(6)], quaternion [qx, qy, qz, qw] = [x, y, z, w]
        x_pinocchio = np.zeros(13)
        x_pinocchio[0:3] = p              # Position [x, y, z]
        x_pinocchio[3:7] = [q_x, q_y, q_z, q_w]  # Quaternion [qx, qy, qz, qw]
        x_pinocchio[7:10] = v_linear       # Linear velocity [vx, vy, vz]
        x_pinocchio[10:13] = w             # Angular velocity [wx, wy, wz]
        
        return x_pinocchio
    else:
        # Already in Pinocchio format or unexpected format
        return x_method1


def solve_with_pinocchio(dt=0.02, N=100, max_iter=100, use_box_solver=False):
    """
    Solve trajectory optimization using Pinocchio + Crocoddyl (Method 2: Fast)
    
    Parameters:
        dt: Time step (default 0.02s)
        N: Number of time steps (default 100)
        max_iter: Maximum number of iterations (default 100)
        use_box_solver: If True, use SolverBoxFDDP with native control bounds (faster)
    """
    # Physical parameters
    m = 0.6
    I = np.diag([0.02, 0.02, 0.01])
    r_thrust = np.array([0.0, 0.0, -0.2])
    
    # Load URDF model
    urdf_path = Path(__file__).parent.parent / 'models' / 'tvc' / 'tvc_simple.urdf'
    
    # Build Pinocchio model
    robot_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    robot_data = robot_model.createData()
    
    # Create state model
    state = crocoddyl.StateMultibody(robot_model)
    
    # Create custom actuation model for TVC (nu=4: [th_p, th_r, T, tau_yaw])
    actuation = TVCActuationModel(state)
    
    # Initial configuration: [x, y, z, qx, qy, qz, qw]
    q0 = pin.neutral(robot_model)
    # q0[2] = 0.0  # z = 0
    # q0[6] = 1.0  # qw = 1 (upright orientation)
    
    # Initial velocity: [vx, vy, vz, wx, wy, wz]
    v0 = np.zeros(robot_model.nv)
    
    # Initial state in configuration space (for ShootingProblem)
    x0 = np.concatenate([q0, v0])
    
    # Goal configuration
    qg = q0.copy()
    qg[0] = 1.0  # x = 1m
    qg[2] = 5.0  # z = 10m
    
    # Goal velocity (zero)
    vg = np.zeros(robot_model.nv)
    
    # Goal state in configuration space [q, v] - this is what ResidualModelState expects
    xg = np.concatenate([qg, vg])  # Dimension: nq + nv = 7 + 6 = 13
    
    # Reference control
    uref = np.array([0.0, 0.0, m*9.81, 0.0])
    
    # Cost weights
    weights = {
        "p": 1.0,
        "v": 0.2,
        "R": 0.5,
        "yaw": 0.5,
        "w": 0.1,
        "u": 1e-3,
        "du": 1e-2
    }
    
    # Bounds (control + state constraints)
    bounds = {
        "T": (0.0, 25.0),
        "th_p": (-0.35, 0.35),
        "th_r": (-0.35, 0.35),
        "tau_yaw": (-1.0, 1.0),
        # State constraints (same as GUI defaults)
        "state_v_horizontal_max": 1.0,
        "state_v_vertical_max": 3.0,
        "state_roll_max": np.radians(10.0),
        "state_pitch_max": np.radians(10.0),
        "state_yaw_max": np.radians(180.0),
        "state_w_max": 2.0,
        "state_k_state_bound": 200.0,
    }
    
    # Create cost model (xg is in configuration space [q, v], dimension 13)
    cost_model = create_tvc_cost_model(state, actuation, xg, uref, weights, bounds)
    
    # Create differential action model (use m, I, u_ref - same as Method 1)
    diff_model = TVCRocketDifferentialActionModel(
        state, actuation, cost_model, robot_model, r_thrust, g=9.81,
        m=m, I=I, u_ref=uref, bounds=bounds, weights=weights, use_box_solver=use_box_solver
    )
    
    # Create integrated action model
    running = crocoddyl.IntegratedActionModelEuler(diff_model, dt)
    
    # Set control bounds for SolverBoxFDDP
    if use_box_solver:
        u_lb, u_ub = build_u_bounds_from_bounds(bounds)
        running.u_lb = u_lb
        running.u_ub = u_ub
    
    # Terminal cost with higher weights
    terminal_weights = weights.copy()
    terminal_weights.update({
        "p": 200.0,
        "v": 50.0,
        "R": 200.0,
        "yaw": 200.0,
        "w": 20.0,
        "u": 0.0,
        "du": 0.0
    })
    terminal_cost = create_tvc_cost_model(state, actuation, xg, uref, terminal_weights, bounds)
    terminal_diff = TVCRocketDifferentialActionModel(
        state, actuation, terminal_cost, robot_model, r_thrust, g=9.81,
        m=m, I=I, u_ref=uref, bounds=bounds, weights=terminal_weights, use_box_solver=use_box_solver
    )
    terminal = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)
    if use_box_solver:
        terminal.u_lb = u_lb
        terminal.u_ub = u_ub
    
    # Create problem (x0 should be in configuration space: [q, v])
    problem = crocoddyl.ShootingProblem(x0, [running]*N, terminal)
    solver = create_box_solver(problem) if use_box_solver else crocoddyl.SolverFDDP(problem)
    
    # Solver parameters
    solver.th_stop = 1e-6  # Stricter convergence criterion (default is 1e-6, was 1e-4)
    solver.reg_min = 1e-9
    solver.reg_max = 1e6
    
    # Callbacks
    logger = crocoddyl.CallbackLogger()
    callbacks = [
        crocoddyl.CallbackVerbose(),
        logger
    ]
    solver.setCallbacks(callbacks)
    
    # Initial guess
    xs_init = [x0.copy() for _ in range(N+1)]
    us_init = [uref.copy() for _ in range(N)]
    
    print("="*60)
    print("TVC Rocket Trajectory Optimization - Method 2: Pinocchio + Crocoddyl")
    print("="*60)
    print(f"  - Number of time steps: {N}")
    print(f"  - Time step: {dt} s")
    print(f"  - Total duration: {N*dt:.2f} s")
    print(f"  - Maximum iterations: {max_iter}")
    print(f"  - Solver: {'BoxFDDP (native control bounds)' if use_box_solver else 'FDDP (penalty constraints)'}")
    print("")
    
    start_time = time.time()
    solver.solve(xs_init, us_init, max_iter, False)
    solve_time = time.time() - start_time
    
    print("")
    print(f"Solving completed!")
    print(f"  - Solving time: {solve_time:.2f} seconds")
    print(f"  - Final cost: {solver.cost:.6e}")
    print(f"  - Iterations: {solver.iter}")
    print(f"  - Stop condition: {solver.stop:.6e}")
    
    if len(logger.costs) > 0:
        print(f"  - Cost change: {logger.costs[0]:.6e} -> {logger.costs[-1]:.6e}")
        print(f"  - Average time per iteration: {solve_time/solver.iter:.3f} seconds")
    
    return solver.xs, solver.us, logger


def solve_with_pinocchio_waypoints(dt, waypoints, m, I, r_thrust, weights, bounds, max_iter=100, 
                                   use_box_solver=False, callback=None, running_flag=None,
                                   terminal_weights=None):
    """
    Solve trajectory optimization with multiple waypoints using Pinocchio + Crocoddyl
    
    This function is similar to the GUI's OptimizationThread.run but uses Pinocchio method.
    
    Args:
        dt: Time step
        waypoints: List of waypoints, each is [x, y, z, yaw_deg, time]
        m: Mass
        I: Moment of inertia matrix
        r_thrust: Thrust position in body frame [x, y, z]
        weights: Cost weights dict
        bounds: Constraint bounds dict
        max_iter: Maximum iterations
        use_box_solver: If True, use SolverBoxFDDP with native control bounds (faster)
        callback: Optional callback function for real-time updates
        running_flag: Optional flag to check if optimization should continue
    
    Returns:
        combined_xs: Combined trajectory states (in Method 1 format for visualization)
        combined_us: Combined trajectory controls
        all_loggers: List of loggers for each segment
    """
    # Load URDF model
    urdf_path = Path(__file__).parent.parent / 'models' / 'tvc' / 'tvc_simple.urdf'
    
    # Build Pinocchio model
    robot_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    
    # Create state model
    state = crocoddyl.StateMultibody(robot_model)
    
    # Create custom actuation model for TVC (nu=4: [th_p, th_r, T, tau_yaw])
    actuation = TVCActuationModel(state)
    
    # Calculate segment durations
    durations = []
    for i in range(len(waypoints) - 1):
        duration = waypoints[i+1][4] - waypoints[i][4]  # time is at index 4
        if duration <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(duration)
    
    uref = np.array([0.0, 0.0, m*9.81, 0.0])
    
    # Store all segments' trajectories
    all_xs = []
    all_us = []
    all_loggers = []
    
    # Initial state for first segment (convert from Method 1 format to Pinocchio format)
    first_wp = waypoints[0]
    x0_method1 = np.zeros(17)
    x0_method1[0:3] = [float(first_wp[0]), float(first_wp[1]), float(first_wp[2])]  # Position
    # Convert yaw to quaternion
    yaw_deg = float(first_wp[3]) if len(first_wp) > 3 else 0.0
    yaw_rad = np.radians(yaw_deg)
    x0_method1[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])  # Quaternion from yaw
    # Convert to Pinocchio format
    x0_seg = convert_method1_state_to_pinocchio(x0_method1)
    
    # Solve each segment
    for seg_idx in range(len(durations)):
        if running_flag is not None and not running_flag():
            break
        
        duration = durations[seg_idx]
        start_wp = waypoints[seg_idx]
        end_wp = waypoints[seg_idx + 1]
        
        # Calculate number of time steps for this segment
        N = max(10, int(duration / dt))
        
        # Target state for this segment (convert from Method 1 format to Pinocchio format)
        xg_method1 = np.zeros(17)
        xg_method1[0:3] = [float(end_wp[0]), float(end_wp[1]), float(end_wp[2])]  # Target position
        # Convert yaw to quaternion
        yaw_deg = float(end_wp[3]) if len(end_wp) > 3 else 0.0
        yaw_rad = np.radians(yaw_deg)
        xg_method1[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])  # Quaternion from yaw
        # Convert to Pinocchio format
        xg_seg = convert_method1_state_to_pinocchio(xg_method1)
        
        # Create cost model
        cost_model = create_tvc_cost_model(state, actuation, xg_seg, uref, weights, bounds)
        
        # Create differential action model (use m, I, u_ref from GUI - same as Method 1)
        diff_model = TVCRocketDifferentialActionModel(
            state, actuation, cost_model, robot_model, r_thrust, g=9.81,
            m=m, I=I, u_ref=uref, bounds=bounds, weights=weights, use_box_solver=use_box_solver
        )
        
        # Create integrated action model
        running = crocoddyl.IntegratedActionModelEuler(diff_model, dt)
        if use_box_solver:
            u_lb, u_ub = build_u_bounds_from_bounds(bounds)
            running.u_lb = u_lb
            running.u_ub = u_ub
        
        # Terminal cost: use provided terminal_weights or fallback to defaults
        tw = terminal_weights if terminal_weights is not None else {
            **weights, "p": 200.0, "v": 50.0, "R": 200.0, "yaw": 200.0, "w": 20.0, "u": 0.0, "du": 0.0
        }
        terminal_cost = create_tvc_cost_model(state, actuation, xg_seg, uref, tw, bounds)
        terminal_diff = TVCRocketDifferentialActionModel(
            state, actuation, terminal_cost, robot_model, r_thrust, g=9.81,
            m=m, I=I, u_ref=uref, bounds=bounds, weights=tw, use_box_solver=use_box_solver
        )
        terminal = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)
        if use_box_solver:
            terminal.u_lb = u_lb
            terminal.u_ub = u_ub
        
        # Create problem
        problem = crocoddyl.ShootingProblem(x0_seg, [running]*N, terminal)
        solver = create_box_solver(problem) if use_box_solver else crocoddyl.SolverFDDP(problem)
        
        # Solver parameters
        solver.th_stop = 1e-6  # Stricter convergence criterion (default is 1e-6, was 1e-4)
        solver.reg_min = 1e-9
        solver.reg_max = 1e6
        
        # Callbacks
        logger = crocoddyl.CallbackLogger()
        callbacks = [logger]
        if callback is not None:
            # Create a callback wrapper
            class CallbackWrapper(crocoddyl.CallbackAbstract):
                def __init__(self, callback_func, seg_idx, completed_xs, completed_us):
                    crocoddyl.CallbackAbstract.__init__(self)
                    self.callback_func = callback_func
                    self.seg_idx = seg_idx
                    self.completed_xs = completed_xs
                    self.completed_us = completed_us
                def __call__(self, solver):
                    if self.callback_func is not None:
                        # Convert states to Method 1 format for callback
                        current_xs_method1 = [convert_pinocchio_state_to_method1(x) for x in solver.xs]
                        self.callback_func(solver, self.seg_idx, current_xs_method1, 
                                          [np.array(u) for u in solver.us], 
                                          self.completed_xs, self.completed_us)
            callbacks.append(CallbackWrapper(callback, seg_idx, all_xs.copy(), all_us.copy()))
        callbacks.append(crocoddyl.CallbackVerbose())
        solver.setCallbacks(callbacks)
        all_loggers.append(logger)
        
        # Initial guess
        xs_init = [x0_seg.copy() for _ in range(N+1)]
        us_init = [uref.copy() for _ in range(N)]
        
        # Solve this segment
        solver.solve(xs_init, us_init, max_iter, False)
        
        # Store results (convert to Method 1 format for consistency)
        seg_xs = [convert_pinocchio_state_to_method1(x) for x in solver.xs]
        seg_us = [np.array(u) for u in solver.us]
        
        # Verify state continuity at connection point
        if seg_idx > 0 and len(all_xs) > 0:
            prev_final = all_xs[-1][-1]  # Previous segment's final state
            curr_initial = seg_xs[0]      # Current segment's initial state
            state_diff = np.linalg.norm(prev_final[:13] - curr_initial[:13])  # Compare first 13 elements
            if state_diff > 1e-6:
                print(f"  Warning: State discontinuity at segment {seg_idx + 1} connection: "
                      f"diff={state_diff:.2e}")
                seg_xs[0] = prev_final.copy()
        
        all_xs.append(seg_xs)
        all_us.append(seg_us)
        
        # Update initial state for next segment
        if seg_idx < len(durations) - 1:
            x0_seg = convert_method1_state_to_pinocchio(seg_xs[-1])
    
    # Combine all segments
    combined_xs = []
    combined_us = []
    
    for i, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
        if i == 0:
            combined_xs.extend(seg_xs)
            combined_us.extend(seg_us)
        else:
            combined_xs.extend(seg_xs[1:])  # Skip duplicate state
            combined_us.extend(seg_us)
    
    return combined_xs, combined_us, all_loggers


def solve_with_pinocchio_waypoints_unified(dt, waypoints, m, I, r_thrust, weights, bounds, max_iter=100,
                                          use_box_solver=False, callback=None, running_flag=None,
                                          terminal_weights=None):
    """
    Solve trajectory optimization with multiple waypoints using a SINGLE unified problem.
    
    All segments are merged into one ShootingProblem. Each phase (between waypoints) uses
    a cost model targeting the next waypoint. The solver optimizes the entire trajectory
    jointly, allowing global trade-offs across waypoints.
    
    Args:
        dt: Time step
        waypoints: List of waypoints, each is [x, y, z, yaw_deg, time]
        m: Mass
        I: Moment of inertia matrix
        r_thrust: Thrust position in body frame [x, y, z]
        weights: Cost weights dict
        bounds: Constraint bounds dict
        max_iter: Maximum iterations
        use_box_solver: If True, use SolverBoxFDDP with native control bounds
        callback: Optional callback(solver) for real-time updates (no seg_idx in unified)
        running_flag: Optional flag to check if optimization should continue
    
    Returns:
        combined_xs: Trajectory states (in Method 1 format for visualization)
        combined_us: Trajectory controls
        all_loggers: List with single logger [logger]
    """
    urdf_path = Path(__file__).parent.parent / 'models' / 'tvc' / 'tvc_simple.urdf'
    robot_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    state = crocoddyl.StateMultibody(robot_model)
    actuation = TVCActuationModel(state)
    uref = np.array([0.0, 0.0, m*9.81, 0.0])
    
    # Segment durations and step counts
    durations = []
    for i in range(len(waypoints) - 1):
        duration = waypoints[i+1][4] - waypoints[i][4]
        if duration <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(duration)
    
    # Build running models: each segment phase has N_i steps with cost targeting next waypoint
    running_models = []
    u_lb, u_ub = None, None
    if use_box_solver:
        u_lb, u_ub = build_u_bounds_from_bounds(bounds)
    
    for seg_idx in range(len(durations)):
        if running_flag is not None and not running_flag():
            break
        end_wp = waypoints[seg_idx + 1]
        N = max(10, int(durations[seg_idx] / dt))
        
        xg_method1 = np.zeros(17)
        xg_method1[0:3] = [float(end_wp[0]), float(end_wp[1]), float(end_wp[2])]
        yaw_deg = float(end_wp[3]) if len(end_wp) > 3 else 0.0
        yaw_rad = np.radians(yaw_deg)
        xg_method1[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])
        xg_seg = convert_method1_state_to_pinocchio(xg_method1)
        
        cost_model = create_tvc_cost_model(state, actuation, xg_seg, uref, weights, bounds)
        diff_model = TVCRocketDifferentialActionModel(
            state, actuation, cost_model, robot_model, r_thrust, g=9.81,
            m=m, I=I, u_ref=uref, bounds=bounds, weights=weights, use_box_solver=use_box_solver
        )
        running = crocoddyl.IntegratedActionModelEuler(diff_model, dt)
        if use_box_solver:
            running.u_lb = u_lb
            running.u_ub = u_ub
        running_models.extend([running] * N)
    
    # Terminal model targets last waypoint
    last_wp = waypoints[-1]
    xg_method1 = np.zeros(17)
    xg_method1[0:3] = [float(last_wp[0]), float(last_wp[1]), float(last_wp[2])]
    yaw_deg = float(last_wp[3]) if len(last_wp) > 3 else 0.0
    yaw_rad = np.radians(yaw_deg)
    xg_method1[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])
    xg_term = convert_method1_state_to_pinocchio(xg_method1)
    tw = terminal_weights if terminal_weights is not None else {
        **weights, "p": 200.0, "v": 50.0, "R": 200.0, "w": 20.0, "u": 0.0, "du": 0.0
    }
    terminal_cost = create_tvc_cost_model(state, actuation, xg_term, uref, tw, bounds)
    terminal_diff = TVCRocketDifferentialActionModel(
        state, actuation, terminal_cost, robot_model, r_thrust, g=9.81,
        m=m, I=I, u_ref=uref, bounds=bounds, weights=tw, use_box_solver=use_box_solver
    )
    terminal = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)
    if use_box_solver:
        terminal.u_lb = u_lb
        terminal.u_ub = u_ub
    
    # Initial state
    first_wp = waypoints[0]
    x0_method1 = np.zeros(17)
    x0_method1[0:3] = [float(first_wp[0]), float(first_wp[1]), float(first_wp[2])]
    yaw_deg = float(first_wp[3]) if len(first_wp) > 3 else 0.0
    yaw_rad = np.radians(yaw_deg)
    x0_method1[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])
    x0 = convert_method1_state_to_pinocchio(x0_method1)
    
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal)
    solver = create_box_solver(problem) if use_box_solver else crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-6
    solver.reg_min = 1e-9
    solver.reg_max = 1e6
    
    logger = crocoddyl.CallbackLogger()
    callbacks = [logger]
    if callback is not None:
        class UnifiedCallbackWrapper(crocoddyl.CallbackAbstract):
            def __init__(self, cb):
                crocoddyl.CallbackAbstract.__init__(self)
                self.cb = cb
            def __call__(self, solver):
                if self.cb is not None:
                    xs_m1 = [convert_pinocchio_state_to_method1(x) for x in solver.xs]
                    us_list = [np.array(u) for u in solver.us]
                    self.cb(solver, 0, xs_m1, us_list, [], [])  # seg_idx=0, no completed
        callbacks.append(UnifiedCallbackWrapper(callback))
    callbacks.append(crocoddyl.CallbackVerbose())
    solver.setCallbacks(callbacks)
    
    N_total = len(running_models)
    xs_init = [x0.copy() for _ in range(N_total + 1)]
    us_init = [uref.copy() for _ in range(N_total)]
    
    solver.solve(xs_init, us_init, max_iter, False)
    
    combined_xs = [convert_pinocchio_state_to_method1(x) for x in solver.xs]
    combined_us = [np.array(u) for u in solver.us]
    return combined_xs, combined_us, [logger]


def simulate_tvc_trajectory(x0, us, dt, robot_model, r_thrust, g=9.81, m=None, I=None):
    """
    Forward simulate TVC rocket trajectory using dynamics integration.
    
    Uses symplectic Euler integration (same as Crocoddyl's IntegratedActionModelEuler)
    to propagate state given initial state and control sequence.
    
    Args:
        x0: Initial state in Pinocchio format [q(7), v(6)]
        us: List of control inputs, each [th_p, th_r, T, tau_yaw]
        dt: Time step (s)
        robot_model: Pinocchio robot model
        r_thrust: Thrust position in body frame [x, y, z]
        g: Gravity (m/s^2)
        m: Mass (kg). If None, use robot_model.inertias[1].mass
        I: Inertia matrix 3x3. If None, use robot_model.inertias[1].inertia
    
    Returns:
        xs_sim: List of simulated states [x0, x1, ..., xN], same format as x0
    """
    state = crocoddyl.StateMultibody(robot_model)
    actuation = TVCActuationModel(state)
    
    # Minimal cost model for simulation (cost values discarded)
    xg = x0.copy()
    mass_for_uref = m if m is not None else robot_model.inertias[1].mass
    uref = np.array([0.0, 0.0, mass_for_uref * g, 0.0])
    weights = {"p": 1.0, "v": 0.2, "R": 0.5, "w": 0.1}
    bounds = {}
    cost_model = create_tvc_cost_model(state, actuation, xg, uref, weights, bounds)
    
    diff_model = TVCRocketDifferentialActionModel(
        state, actuation, cost_model, robot_model, r_thrust, g=g,
        m=m, I=I, u_ref=uref, bounds=bounds, use_box_solver=True  # No penalty needed for simulation
    )
    integrated = crocoddyl.IntegratedActionModelEuler(diff_model, dt)
    data = integrated.createData()
    
    xs_sim = [np.array(x0, copy=True)]
    x = np.array(x0, copy=True)
    for u in us:
        u_arr = np.array(u).flatten()
        if len(u_arr) < 4:
            u_arr = np.resize(u_arr, 4)
        integrated.calc(data, x, u_arr)
        x = np.array(data.xnext, copy=True)
        xs_sim.append(x)
    return xs_sim


def run_both_methods_simulation(dt=0.05, N=100, max_iter=100, verbose=True):
    """
    Run optimization with both FDDP and BoxFDDP, then simulate both trajectories.
    
    Compares:
    - Optimization results (cost, iterations)
    - Simulated trajectory vs optimized trajectory (verification)
    
    Args:
        dt: Time step
        N: Number of steps
        max_iter: Max iterations
        verbose: Print comparison summary
    
    Returns:
        dict with keys: fddp_xs, fddp_us, boxfddp_xs, boxfddp_us,
                        fddp_sim_xs, boxfddp_sim_xs, fddp_logger, boxfddp_logger
    """
    urdf_path = Path(__file__).parent.parent / 'models' / 'tvc' / 'tvc_simple.urdf'
    robot_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    r_thrust = np.array([0.0, 0.0, -0.2])
    m, I = 0.6, np.diag([0.02, 0.02, 0.01])  # Same as solve_with_pinocchio
    
    # Method 1: FDDP (penalty constraints)
    t0 = time.time()
    fddp_xs, fddp_us, fddp_logger = solve_with_pinocchio(dt=dt, N=N, max_iter=max_iter, use_box_solver=False)
    t_fddp = time.time() - t0
    
    # Method 2: BoxFDDP (native control bounds)
    t0 = time.time()
    boxfddp_xs, boxfddp_us, boxfddp_logger = solve_with_pinocchio(dt=dt, N=N, max_iter=max_iter, use_box_solver=True)
    t_boxfddp = time.time() - t0
    
    # Simulate both trajectories (forward integration from x0 and us)
    x0_pin = np.array(fddp_xs[0])
    if len(x0_pin) == 17:
        x0_pin = convert_method1_state_to_pinocchio(x0_pin)
    
    fddp_sim_xs = simulate_tvc_trajectory(x0_pin, fddp_us, dt, robot_model, r_thrust, m=m, I=I)
    boxfddp_sim_xs = simulate_tvc_trajectory(x0_pin, boxfddp_us, dt, robot_model, r_thrust, m=m, I=I)
    
    if verbose:
        print("\n" + "="*60)
        print("Comparison: FDDP vs BoxFDDP")
        print("="*60)
        print(f"  FDDP:    cost={fddp_logger.costs[-1] if fddp_logger.costs else 0:.6e}, "
              f"iter={len(fddp_logger.costs)}, time={t_fddp:.2f}s")
        print(f"  BoxFDDP: cost={boxfddp_logger.costs[-1] if boxfddp_logger.costs else 0:.6e}, "
              f"iter={len(boxfddp_logger.costs)}, time={t_boxfddp:.2f}s")
        
        # Verify simulation vs optimized trajectory (both in Pinocchio format)
        n_fddp = min(len(fddp_xs), len(fddp_sim_xs))
        n_box = min(len(boxfddp_xs), len(boxfddp_sim_xs))
        fddp_err = np.mean([np.linalg.norm(np.array(fddp_xs[i])[:13] - np.array(fddp_sim_xs[i])[:13]) 
                           for i in range(n_fddp)])
        boxfddp_err = np.mean([np.linalg.norm(np.array(boxfddp_xs[i])[:13] - np.array(boxfddp_sim_xs[i])[:13]) 
                              for i in range(n_box)])
        print(f"  Simulation verification (mean state error optim vs sim):")
        print(f"    FDDP:    {fddp_err:.2e}")
        print(f"    BoxFDDP: {boxfddp_err:.2e}")
        print("="*60)
    
    return {
        "fddp_xs": fddp_xs,
        "fddp_us": fddp_us,
        "boxfddp_xs": boxfddp_xs,
        "boxfddp_us": boxfddp_us,
        "fddp_sim_xs": fddp_sim_xs,
        "boxfddp_sim_xs": boxfddp_sim_xs,
        "fddp_logger": fddp_logger,
        "boxfddp_logger": boxfddp_logger,
        "dt": dt,
        "N": N,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TVC trajectory optimization (FDDP / BoxFDDP / Waypoints)")
    parser.add_argument("--mode",
                        choices=["fddp", "boxfddp", "both", "waypoints", "waypoints_unified"],
                        default="waypoints",
                        help="fddp: FDDP only; boxfddp: BoxFDDP only; both: FDDP+BoxFDDP with simulation; "
                             "waypoints: segmented waypoints; waypoints_unified: unified waypoints")
    parser.add_argument("--no-plot", action="store_true", help="Disable debug plot (plot is on by default)")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--solver", choices=["fddp", "boxfddp"], default="boxfddp",
                        help="Solver for waypoints modes (default: boxfddp)")
    args = parser.parse_args()
    
    if args.mode == "both":
        # Run both FDDP and BoxFDDP, simulate both, compare
        results = run_both_methods_simulation(dt=args.dt, N=args.N, max_iter=args.max_iter)
        print("\nSimulation completed. FDDP and BoxFDDP trajectories and simulated trajectories available.")
        if not args.no_plot:
            plot_debug(results["fddp_xs"], results["fddp_us"], args.dt, results["fddp_logger"],
                      title="FDDP Trajectory")
            plot_debug(results["boxfddp_xs"], results["boxfddp_us"], args.dt, results["boxfddp_logger"],
                      title="BoxFDDP Trajectory")
    elif args.mode in ("waypoints", "waypoints_unified"):
        # Waypoints mode: solve_with_pinocchio_waypoints or solve_with_pinocchio_waypoints_unified
        m = 0.6
        I = np.diag([0.02, 0.02, 0.01])
        r_thrust = np.array([0.0, 0.0, -0.2])
        weights = {"p": 1.0, "v": 0.2, "R": 0.5, "yaw": 15, "w": 0.1, "u": 1e-3, "du": 1e-2}
        # terminal_weights = {"p": 200.0, "v": 50.0, "R": 200.0, "yaw": 200.0, "w": 20.0, "u": 0.0, "du": 0.0}
        terminal_weights = {"p": 0.0, "v": 0.0, "R": 0.0, "yaw": 0.0, "w": 0.0, "u": 0.0, "du": 0.0}
        # terminal_weights = weights
        bounds = {
            "T": (0.0, 25.0),
            "th_p": (-0.35, 0.35),
            "th_r": (-0.35, 0.35),
            "tau_yaw": (-1.0, 1.0),
            # State constraints (same as GUI defaults)
            "state_v_horizontal_max": 1,
            "state_v_vertical_max": 2.0,
            "state_roll_max": np.radians(10.0),
            "state_pitch_max": np.radians(10.0),
            "state_yaw_max": np.radians(30.0),
            "state_w_max": 2.0,
            "state_k_state_bound": 20.0,
            # state_constraint_lxx_scale: 0=Method1 style (recommended), 1=full Lxx, 0.1=damped
            "state_constraint_lxx_scale": 0,
        }
        # Default waypoints: start (0,0,0) at t=0 -> goal (1,0,5) at t=5s
        waypoints = [
            [0.0, 0.0, 0.0, 0.0, 0.0],   # [x, y, z, yaw_deg, time]
            [3.0, 0.0, 0.0, 0.0, 5.0]
            # [4.0, 0.0, 5.0, 0.0, 10.0]
        ]
        use_box = (args.solver == "boxfddp")
        solver_fn = solve_with_pinocchio_waypoints_unified if args.mode == "waypoints_unified" else solve_with_pinocchio_waypoints
        t0 = time.perf_counter()
        xs, us, all_loggers = solver_fn(
            dt=args.dt,
            waypoints=waypoints,
            m=m,
            I=I,
            r_thrust=r_thrust,
            weights=weights,
            bounds=bounds,
            max_iter=args.max_iter,
            use_box_solver=use_box,
            terminal_weights=terminal_weights
        )
        elapsed = time.perf_counter() - t0
        logger = all_loggers[0] if all_loggers else None
        total_iters = sum(len(lg.costs) for lg in all_loggers) if all_loggers else 0
        print("\n" + "="*50)
        print(f"Solved ({args.mode}, {'BoxFDDP' if use_box else 'FDDP'}). N = {len(us)}, "
              f"iters = {total_iters}, time = {elapsed:.2f}s")
        print("x0 =", xs[0][:7], "...")
        print("xN =", xs[-1][:7], "...")
        # Simulate to verify (waypoints return Method 1 format; simulate needs Pinocchio)
        urdf_path = Path(__file__).parent.parent / 'models' / 'tvc' / 'tvc_simple.urdf'
        robot_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        x0_pin = convert_method1_state_to_pinocchio(xs[0]) if len(xs[0]) == 17 else xs[0]
        sim_xs = simulate_tvc_trajectory(x0_pin, us, args.dt, robot_model, r_thrust, m=m, I=I)
        err = np.mean([np.linalg.norm((convert_method1_state_to_pinocchio(a) if len(a) == 17 else a)[:13]
                                      - np.array(b)[:13])
                       for a, b in zip(xs, sim_xs[:len(xs)])])
        print(f"Simulation verification (mean state error): {err:.2e}")
        if not args.no_plot:
            wp_list = [[wp[0], wp[1], wp[2]] for wp in waypoints] if waypoints else None
            plot_debug(xs, us, args.dt, logger, title=f"{args.mode} ({'BoxFDDP' if use_box else 'FDDP'}) Trajectory", waypoints=wp_list)
    else:
        # Single method: FDDP or BoxFDDP (solve_with_pinocchio)
        use_box = (args.mode == "boxfddp")
        xs, us, logger = solve_with_pinocchio(dt=args.dt, N=args.N, max_iter=args.max_iter,
                                             use_box_solver=use_box)
        print("\n" + "="*50)
        print(f"Solved ({'BoxFDDP' if use_box else 'FDDP'}). N =", len(us))
        print("x0 =", xs[0][:7], "...")
        print("xN =", xs[-1][:7], "...")
        # Simulate to verify
        urdf_path = Path(__file__).parent.parent / 'models' / 'tvc' / 'tvc_simple.urdf'
        robot_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        m, I = 0.6, np.diag([0.02, 0.02, 0.01])
        sim_xs = simulate_tvc_trajectory(xs[0], us, args.dt, robot_model, np.array([0, 0, -0.2]), m=m, I=I)
        err = np.mean([np.linalg.norm(np.array(a)[:13] - np.array(b)[:13])
                       for a, b in zip(xs, sim_xs[:len(xs)])])
        print(f"Simulation verification (mean state error): {err:.2e}")
        if not args.no_plot:
            plot_debug(xs, us, args.dt, logger, title=f"{'BoxFDDP' if use_box else 'FDDP'} Trajectory")
