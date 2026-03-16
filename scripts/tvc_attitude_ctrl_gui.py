#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Attitude Control Simulation GUI

Simulates P+PID attitude control with first-order actuator dynamics.
- Outer loop: P on angle error → desired angular velocity
- Inner loop: PID on angular velocity error → actuator command (PWM)
- Actuator: first-order lag between command and actual angle

Usage:
    python tvc_attitude_ctrl_gui.py

At t=2s, a step command is applied to roll angle. Plots show controller response.
"""

import sys
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.path.dirname(script_dir), 'config')
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'tvc_params.json')


def get_default_params():
    """Default parameter values for TVC attitude control."""
    return {
        'dt': 0.01, 'T_end': 10.0, 'step_time': 2.0,
        'step_angle_deg': 10.0, 'theta_ref_period': 5.0,
        'step_omega': 0.5, 'step_u': 0.2,
        'cmd_mode': 'angle',
        'Kp_angle': 2.0, 'Kp': 1.0, 'Ki': 0.5, 'Kd': 0.05, 'int_max': 10.0,
        'tau_act': 0.053, 'K_act': 1.0, 'theta_gimbal_max_deg': 10.0,
        'I_roll': 0.02, 'F_thrust': 6.0, 'L': 0.2, 'm': 0.6,
    }
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np
import matplotlib

try:
    import PyQt5
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QPushButton, QLabel, QGroupBox,
                                 QGridLayout, QDoubleSpinBox, QMessageBox, QComboBox, QTabWidget,
                                 QFileDialog)
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    QT_AVAILABLE = True
except ImportError:
    try:
        import PySide2
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                       QHBoxLayout, QPushButton, QLabel, QGroupBox,
                                       QGridLayout, QDoubleSpinBox, QMessageBox, QComboBox, QTabWidget,
                                       QFileDialog)
        from PySide2.QtCore import Qt
        from PySide2.QtGui import QFont
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        print("Error: PyQt5 or PySide2 required. pip install PyQt5")
        sys.exit(1)

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


def _run_omega_pid(e_omega, int_e_omega, e_omega_prev, dt, Kp, Ki, Kd, int_max):
    """PID on angular velocity error. Returns (u, int_e_omega_new)."""
    int_e = int_e_omega + e_omega * dt
    int_e = np.clip(int_e, -int_max, int_max)
    de_dt = (e_omega - e_omega_prev) / dt if dt > 0 else 0.0
    u = Kp * e_omega + Ki * int_e + Kd * de_dt
    return u, int_e


def _get_tvc_physics(params):
    """Extract TVC rigid body params: I_roll (kg·m²), F_thrust (N), L (m), m (kg), g (m/s²)."""
    I_roll = max(params.get('I_roll', 0.02), 1e-6)
    F_thrust = max(params.get('F_thrust', 6.0), 0.1)
    L = max(params.get('L', 0.2), 0.01)
    m = max(params.get('m', 0.6), 0.01)
    g = params.get('g', 9.81)
    return I_roll, F_thrust, L, m, g


def run_omega_control(params):
    """
    Angular velocity control only. ω_des step → PID → u → actuator → θ_gimbal → τ → I·α → ω.
    TVC rigid body: τ = F_thrust·L·θ_gimbal, I·dω/dt = τ.
    """
    dt = params['dt']
    T_end = params['T_end']
    step_time = params['step_time']
    step_omega = params.get('step_omega', 0.5)
    Kp, Ki, Kd = params['Kp'], params['Ki'], params['Kd']
    tau_act = max(params['tau_act'], 1e-6)
    K_act = params['K_act']
    int_max = params.get('int_max', 10.0)
    theta_gimbal_max = np.radians(params.get('theta_gimbal_max_deg', 10.0))
    I_roll, F_thrust, L, m, g = _get_tvc_physics(params)
    
    n = int(T_end / dt) + 1
    t = np.linspace(0, T_end, n)
    theta_ref = np.zeros(n)
    omega_des = np.zeros(n)
    omega_des[t >= step_time] = step_omega
    
    theta_gimbal = np.zeros(n)
    theta_act = np.zeros(n)
    omega_act = np.zeros(n)
    pos = np.zeros((n, 3))   # x, y, z
    vel = np.zeros((n, 3))   # vx, vy, vz
    u_cmd = np.zeros(n)
    
    int_e_omega = 0.0
    e_omega_prev = 0.0
    
    for i in range(1, n):
        e_omega = omega_des[i-1] - omega_act[i-1]
        u, int_e_omega = _run_omega_pid(e_omega, int_e_omega, e_omega_prev, dt, Kp, Ki, Kd, int_max)
        e_omega_prev = e_omega
        u = np.clip(u, -theta_gimbal_max, theta_gimbal_max)
        
        # Actuator: τ_act*dθ_gimbal/dt + θ_gimbal = K_act*u
        dtheta_gimbal_dt = (K_act * u - theta_gimbal[i-1]) / tau_act
        theta_gimbal[i] = theta_gimbal[i-1] + dtheta_gimbal_dt * dt
        
        # TVC torque: τ = F_thrust * L * θ_gimbal (small angle)
        tau = F_thrust * L * theta_gimbal[i]
        # Rigid body rotation: I·dω/dt = τ
        domega_dt = tau / I_roll
        omega_act[i] = omega_act[i-1] + domega_dt * dt
        theta_act[i] = theta_act[i-1] + omega_act[i] * dt
        
        # Translational: F_world = R_roll(θ) @ F_body, F_body = [0, -T*sin(θ_gimbal), T*cos(θ_gimbal)]
        # Small angle: F_body ≈ [0, -T*θ_gimbal, T], F_world ≈ [0, -T*(θ_act+θ_gimbal), T]
        th = theta_act[i]
        th_g = theta_gimbal[i]
        F_body = np.array([0, -F_thrust * np.sin(th_g), F_thrust * np.cos(th_g)])
        R_roll = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        F_world = R_roll @ F_body + np.array([0, 0, -m * g])
        a_world = F_world / m
        vel[i] = vel[i-1] + a_world * dt
        pos[i] = pos[i-1] + vel[i] * dt
        u_cmd[i-1] = u
    
    u_cmd[-1] = u_cmd[-2]
    return t, theta_ref, theta_act, omega_des, omega_act, u_cmd, theta_gimbal, pos, vel


def run_angle_control(params):
    """
    Angle control: θ_ref step → P(angle) → ω_des → PID(omega) → u → actuator → θ_gimbal → τ → I·α → ω → θ.
    TVC rigid body: τ = F_thrust·L·θ_gimbal, I·dω/dt = τ, dθ/dt = ω.
    """
    dt = params['dt']
    T_end = params['T_end']
    step_time = params['step_time']
    step_angle_deg = params['step_angle_deg']
    Kp_angle = params['Kp_angle']
    Kp, Ki, Kd = params['Kp'], params['Ki'], params['Kd']
    tau_act = max(params['tau_act'], 1e-6)
    K_act = params['K_act']
    int_max = params.get('int_max', 10.0)
    theta_gimbal_max = np.radians(params.get('theta_gimbal_max_deg', 10.0))
    I_roll, F_thrust, L, m, g = _get_tvc_physics(params)
    
    n = int(T_end / dt) + 1
    t = np.linspace(0, T_end, n)
    theta_ref = np.zeros(n)
    T_period = params.get('theta_ref_period', 5.0)
    t_rel = t - step_time
    mask_pos = (t >= step_time) & (t_rel < T_period)
    mask_zero = (t >= step_time + T_period) & (t_rel < 2 * T_period)
    mask_neg = t >= step_time + 2 * T_period
    theta_ref[mask_pos] = np.radians(step_angle_deg)
    theta_ref[mask_zero] = 0.0
    theta_ref[mask_neg] = np.radians(-step_angle_deg)
    
    omega_des = np.zeros(n)
    theta_gimbal = np.zeros(n)
    theta_act = np.zeros(n)
    omega_act = np.zeros(n)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    u_cmd = np.zeros(n)
    
    int_e_omega = 0.0
    e_omega_prev = 0.0
    
    for i in range(1, n):
        e_theta = theta_ref[i-1] - theta_act[i-1]
        omega_des[i-1] = Kp_angle * e_theta
        
        e_omega = omega_des[i-1] - omega_act[i-1]
        u, int_e_omega = _run_omega_pid(e_omega, int_e_omega, e_omega_prev, dt, Kp, Ki, Kd, int_max)
        e_omega_prev = e_omega
        u = np.clip(u, -theta_gimbal_max, theta_gimbal_max)
        
        # Actuator: τ_act*dθ_gimbal/dt + θ_gimbal = K_act*u
        dtheta_gimbal_dt = (K_act * u - theta_gimbal[i-1]) / tau_act
        theta_gimbal[i] = theta_gimbal[i-1] + dtheta_gimbal_dt * dt
        
        # TVC torque: τ = F_thrust * L * θ_gimbal
        tau = F_thrust * L * theta_gimbal[i]
        # Rigid body: I·dω/dt = τ
        domega_dt = tau / I_roll
        omega_act[i] = omega_act[i-1] + domega_dt * dt
        theta_act[i] = theta_act[i-1] + omega_act[i] * dt
        
        # Translational
        th, th_g = theta_act[i], theta_gimbal[i]
        F_body = np.array([0, -F_thrust * np.sin(th_g), F_thrust * np.cos(th_g)])
        R_roll = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        F_world = R_roll @ F_body + np.array([0, 0, -m * g])
        a_world = F_world / m
        vel[i] = vel[i-1] + a_world * dt
        pos[i] = pos[i-1] + vel[i] * dt
        u_cmd[i-1] = u
    
    omega_des[-1] = omega_des[-2]
    u_cmd[-1] = u_cmd[-2]
    return t, theta_ref, theta_act, omega_des, omega_act, u_cmd, theta_gimbal, pos, vel


def run_actuator_step(params):
    """Actuator step test: u step → actuator → θ_gimbal → τ → I·α → ω → θ. Bypass controllers."""
    dt = params['dt']
    T_end = params['T_end']
    step_time = params['step_time']
    step_u = params.get('step_u', 0.2)
    tau_act = max(params['tau_act'], 1e-6)
    K_act = params['K_act']
    theta_gimbal_max = np.radians(params.get('theta_gimbal_max_deg', 10.0))
    I_roll, F_thrust, L, m, g = _get_tvc_physics(params)
    
    n = int(T_end / dt) + 1
    t = np.linspace(0, T_end, n)
    theta_ref = np.zeros(n)
    omega_des = np.zeros(n)
    theta_gimbal = np.zeros(n)
    theta_act = np.zeros(n)
    omega_act = np.zeros(n)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    u_cmd = np.zeros(n)
    u_cmd[t >= step_time] = step_u
    
    for i in range(1, n):
        u = np.clip(u_cmd[i-1], -theta_gimbal_max, theta_gimbal_max)
        u_cmd[i-1] = u
        # Actuator
        dtheta_gimbal_dt = (K_act * u - theta_gimbal[i-1]) / tau_act
        theta_gimbal[i] = theta_gimbal[i-1] + dtheta_gimbal_dt * dt
        # TVC + rigid body
        tau = F_thrust * L * theta_gimbal[i]
        domega_dt = tau / I_roll
        omega_act[i] = omega_act[i-1] + domega_dt * dt
        theta_act[i] = theta_act[i-1] + omega_act[i] * dt
        # Translational
        th, th_g = theta_act[i], theta_gimbal[i]
        F_body = np.array([0, -F_thrust * np.sin(th_g), F_thrust * np.cos(th_g)])
        R_roll = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        F_world = R_roll @ F_body + np.array([0, 0, -m * g])
        a_world = F_world / m
        vel[i] = vel[i-1] + a_world * dt
        pos[i] = pos[i-1] + vel[i] * dt
    
    return t, theta_ref, theta_act, omega_des, omega_act, u_cmd, theta_gimbal, pos, vel


def run_simulation(params):
    """
    Run TVC attitude control simulation.
    Command mode: 'angle' | 'omega' | 'actuator'
    Returns: t, theta_ref, theta_act, omega_des, omega_act, u_cmd, theta_gimbal, pos, vel
    """
    cmd_mode = params.get('cmd_mode', 'angle')
    if cmd_mode == 'omega':
        return run_omega_control(params)
    if cmd_mode == 'actuator':
        return run_actuator_step(params)
    return run_angle_control(params)


def actuator_theory(t, step_time, step_u, K_act, tau_act):
    """Theoretical first-order response: θ = K_act*u*(1 - exp(-(t-t0)/τ)) for t>=t0"""
    theta = np.zeros_like(t)
    mask = t >= step_time
    theta[mask] = K_act * step_u * (1 - np.exp(-(t[mask] - step_time) / max(tau_act, 1e-6)))
    return theta


class MainWindow(QMainWindow):
    """TVC Attitude Control Simulation"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('TVC Attitude Control Simulation (Roll)')
        self.resize(1200, 700)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: parameters
        left = self.create_param_panel()
        main_layout.addWidget(left, 1)
        
        # Load params from default config on startup
        self.load_params_from_default()
        
        # Right: plots
        right = self.create_plot_panel()
        main_layout.addWidget(right, 2)
        
    def create_param_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        title = QLabel('Parameters')
        title.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(title)
        
        # Simulation
        sim_group = QGroupBox('Simulation')
        sim_layout = QGridLayout()
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 0.1)
        self.dt_spin.setValue(0.01)
        self.dt_spin.setDecimals(3)
        self.dt_spin.setSingleStep(0.005)
        self.dt_spin.setMaximumHeight(28)
        self.dt_spin.setToolTip('Integration time step (s)')
        self.T_end_spin = QDoubleSpinBox()
        self.T_end_spin.setRange(1, 60)
        self.T_end_spin.setValue(10.0)
        self.T_end_spin.setDecimals(1)
        self.T_end_spin.setMaximumHeight(28)
        self.step_time_spin = QDoubleSpinBox()
        self.step_time_spin.setRange(0, 60)
        self.step_time_spin.setValue(2.0)
        self.step_time_spin.setDecimals(1)
        self.step_time_spin.setMaximumHeight(28)
        self.step_time_spin.setToolTip('Time when step command is applied (s)')
        self.cmd_mode_combo = QComboBox()
        self.cmd_mode_combo.addItem('Angle step (θ_ref → P → ω_des)', 'angle')
        self.cmd_mode_combo.addItem('Angular velocity step (ω_des direct)', 'omega')
        self.cmd_mode_combo.addItem('Actuator step (u direct, test τ)', 'actuator')
        self.cmd_mode_combo.setMaximumHeight(28)
        self.cmd_mode_combo.currentIndexChanged.connect(self.on_cmd_mode_changed)
        self.step_angle_spin = QDoubleSpinBox()
        self.step_angle_spin.setRange(-90, 90)
        self.step_angle_spin.setValue(10.0)
        self.step_angle_spin.setDecimals(1)
        self.step_angle_spin.setMaximumHeight(28)
        self.step_angle_spin.setToolTip('Step command: roll angle (deg), used in angle mode')
        self.theta_ref_period_spin = QDoubleSpinBox()
        self.theta_ref_period_spin.setRange(0.5, 60)
        self.theta_ref_period_spin.setValue(5.0)
        self.theta_ref_period_spin.setDecimals(1)
        self.theta_ref_period_spin.setMaximumHeight(28)
        self.theta_ref_period_spin.setToolTip('Angle mode: θ_ref period (s). Sequence: +θ → 0 → -θ, each T_period')
        self.step_omega_spin = QDoubleSpinBox()
        self.step_omega_spin.setRange(-5, 5)
        self.step_omega_spin.setValue(0.5)
        self.step_omega_spin.setDecimals(3)
        self.step_omega_spin.setMaximumHeight(28)
        self.step_omega_spin.setToolTip('Step command: angular velocity (rad/s), used in omega mode')
        self.step_u_spin = QDoubleSpinBox()
        self.step_u_spin.setRange(-2, 2)
        self.step_u_spin.setValue(0.2)
        self.step_u_spin.setDecimals(3)
        self.step_u_spin.setMaximumHeight(28)
        self.step_u_spin.setToolTip('Step command: actuator input u (rad), used in actuator mode. τ*dθ/dt+θ=K_act*u')
        sim_layout.addWidget(QLabel('dt (s):'), 0, 0)
        sim_layout.addWidget(self.dt_spin, 0, 1)
        sim_layout.addWidget(QLabel('T_end (s):'), 1, 0)
        sim_layout.addWidget(self.T_end_spin, 1, 1)
        sim_layout.addWidget(QLabel('Step time (s):'), 2, 0)
        sim_layout.addWidget(self.step_time_spin, 2, 1)
        sim_layout.addWidget(QLabel('Command:'), 3, 0)
        sim_layout.addWidget(self.cmd_mode_combo, 3, 1)
        sim_layout.addWidget(QLabel('Step angle (deg):'), 4, 0)
        sim_layout.addWidget(self.step_angle_spin, 4, 1)
        sim_layout.addWidget(QLabel('θ_ref period (s):'), 5, 0)
        sim_layout.addWidget(self.theta_ref_period_spin, 5, 1)
        sim_layout.addWidget(QLabel('Step ω (rad/s):'), 6, 0)
        sim_layout.addWidget(self.step_omega_spin, 6, 1)
        sim_layout.addWidget(QLabel('Step u (rad):'), 7, 0)
        sim_layout.addWidget(self.step_u_spin, 7, 1)
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # Outer P (angle)
        outer_group = QGroupBox('Outer Loop (P on angle)')
        outer_layout = QGridLayout()
        self.Kp_angle_spin = QDoubleSpinBox()
        self.Kp_angle_spin.setRange(0, 100)
        self.Kp_angle_spin.setValue(2.0)
        self.Kp_angle_spin.setDecimals(2)
        self.Kp_angle_spin.setMaximumHeight(28)
        self.Kp_angle_spin.setToolTip('Kp_angle: ω_des = Kp_angle * (θ_ref - θ_act)')
        outer_layout.addWidget(QLabel('Kp_angle:'), 0, 0)
        outer_layout.addWidget(self.Kp_angle_spin, 0, 1)
        outer_group.setLayout(outer_layout)
        layout.addWidget(outer_group)
        
        # Inner PID (angular velocity)
        inner_group = QGroupBox('Inner Loop (PID on angular velocity)')
        inner_layout = QGridLayout()
        self.Kp_spin = QDoubleSpinBox()
        self.Kp_spin.setRange(0, 100)
        self.Kp_spin.setValue(1.0)
        self.Kp_spin.setDecimals(3)
        self.Kp_spin.setMaximumHeight(28)
        self.Kp_spin.setToolTip('Proportional gain: u += Kp * (ω_des - ω_act)')
        self.Ki_spin = QDoubleSpinBox()
        self.Ki_spin.setRange(0, 100)
        self.Ki_spin.setValue(0.5)
        self.Ki_spin.setDecimals(3)
        self.Ki_spin.setMaximumHeight(28)
        self.Ki_spin.setToolTip('Integral gain: u += Ki * ∫(ω_des - ω_act)dt')
        self.Kd_spin = QDoubleSpinBox()
        self.Kd_spin.setRange(0, 10)
        self.Kd_spin.setValue(0.05)
        self.Kd_spin.setDecimals(3)
        self.Kd_spin.setMaximumHeight(28)
        self.Kd_spin.setToolTip('Derivative gain: u += Kd * d(ω_des - ω_act)/dt')
        self.int_max_spin = QDoubleSpinBox()
        self.int_max_spin.setRange(0.1, 100)
        self.int_max_spin.setValue(10.0)
        self.int_max_spin.setDecimals(1)
        self.int_max_spin.setMaximumHeight(28)
        self.int_max_spin.setToolTip('Integrator anti-windup limit')
        inner_layout.addWidget(QLabel('Kp:'), 0, 0)
        inner_layout.addWidget(self.Kp_spin, 0, 1)
        inner_layout.addWidget(QLabel('Ki:'), 1, 0)
        inner_layout.addWidget(self.Ki_spin, 1, 1)
        inner_layout.addWidget(QLabel('Kd:'), 2, 0)
        inner_layout.addWidget(self.Kd_spin, 2, 1)
        inner_layout.addWidget(QLabel('Int limit:'), 3, 0)
        inner_layout.addWidget(self.int_max_spin, 3, 1)
        inner_group.setLayout(inner_layout)
        layout.addWidget(inner_group)
        
        # TVC rigid body
        tvc_group = QGroupBox('TVC Rigid Body')
        tvc_layout = QGridLayout()
        self.I_roll_spin = QDoubleSpinBox()
        self.I_roll_spin.setRange(0.001, 1.0)
        self.I_roll_spin.setValue(0.02)
        self.I_roll_spin.setDecimals(3)
        self.I_roll_spin.setMaximumHeight(28)
        self.I_roll_spin.setToolTip('Roll axis moment of inertia (kg·m²)')
        self.F_thrust_spin = QDoubleSpinBox()
        self.F_thrust_spin.setRange(0.1, 100)
        self.F_thrust_spin.setValue(6.0)
        self.F_thrust_spin.setDecimals(1)
        self.F_thrust_spin.setMaximumHeight(28)
        self.F_thrust_spin.setToolTip('Thrust magnitude (N). τ = F·L·θ_gimbal')
        self.L_spin = QDoubleSpinBox()
        self.L_spin.setRange(0.01, 2.0)
        self.L_spin.setValue(0.2)
        self.L_spin.setDecimals(3)
        self.L_spin.setMaximumHeight(28)
        self.L_spin.setToolTip('Moment arm: thrust point to CG (m). |r_thrust|')
        tvc_layout.addWidget(QLabel('I_roll (kg·m²):'), 0, 0)
        tvc_layout.addWidget(self.I_roll_spin, 0, 1)
        tvc_layout.addWidget(QLabel('F_thrust (N):'), 1, 0)
        tvc_layout.addWidget(self.F_thrust_spin, 1, 1)
        tvc_layout.addWidget(QLabel('L (m):'), 2, 0)
        tvc_layout.addWidget(self.L_spin, 2, 1)
        self.m_spin = QDoubleSpinBox()
        self.m_spin.setRange(0.01, 100)
        self.m_spin.setValue(0.6)
        self.m_spin.setDecimals(2)
        self.m_spin.setMaximumHeight(28)
        self.m_spin.setToolTip('Rocket mass (kg) for translational dynamics')
        tvc_layout.addWidget(QLabel('m (kg):'), 3, 0)
        tvc_layout.addWidget(self.m_spin, 3, 1)
        tvc_group.setLayout(tvc_layout)
        layout.addWidget(tvc_group)
        
        # Actuator (first-order)
        act_group = QGroupBox('Actuator (first-order)')
        act_layout = QGridLayout()
        self.tau_act_spin = QDoubleSpinBox()
        self.tau_act_spin.setRange(0.01, 2.0)
        self.tau_act_spin.setValue(0.053)
        self.tau_act_spin.setDecimals(3)
        self.tau_act_spin.setSingleStep(0.01)
        self.tau_act_spin.setMaximumHeight(28)
        self.tau_act_spin.setToolTip('Time constant τ (s). ~0.053 for 3Hz bandwidth')
        self.K_act_spin = QDoubleSpinBox()
        self.K_act_spin.setRange(0.1, 10)
        self.K_act_spin.setValue(1.0)
        self.K_act_spin.setDecimals(2)
        self.K_act_spin.setMaximumHeight(28)
        self.K_act_spin.setToolTip('Actuator gain: τ*dθ/dt + θ = K_act*u')
        self.theta_gimbal_max_spin = QDoubleSpinBox()
        self.theta_gimbal_max_spin.setRange(1, 90)
        self.theta_gimbal_max_spin.setValue(10.0)
        self.theta_gimbal_max_spin.setDecimals(1)
        self.theta_gimbal_max_spin.setMaximumHeight(28)
        self.theta_gimbal_max_spin.setToolTip('u_cmd limit ±θ_max (deg), clamps control before actuator')
        act_layout.addWidget(QLabel('τ (s):'), 0, 0)
        act_layout.addWidget(self.tau_act_spin, 0, 1)
        act_layout.addWidget(QLabel('K_act:'), 1, 0)
        act_layout.addWidget(self.K_act_spin, 1, 1)
        act_layout.addWidget(QLabel('θ_gimbal max (deg):'), 2, 0)
        act_layout.addWidget(self.theta_gimbal_max_spin, 2, 1)
        act_group.setLayout(act_layout)
        layout.addWidget(act_group)
        
        # Run button
        self.run_btn = QPushButton('Run Simulation')
        self.run_btn.setStyleSheet('background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;')
        self.run_btn.clicked.connect(self.run_sim)
        layout.addWidget(self.run_btn)
        
        # Param file buttons
        self.save_params_btn = QPushButton('Save')
        self.save_params_btn.setStyleSheet('background-color: #2196F3; color: white; padding: 8px;')
        self.save_params_btn.setToolTip('Overwrite default config file')
        self.save_params_btn.clicked.connect(self.save_params_to_default)
        layout.addWidget(self.save_params_btn)
        
        self.save_to_btn = QPushButton('Save to...')
        self.save_to_btn.setStyleSheet('padding: 8px;')
        self.save_to_btn.clicked.connect(self.save_params_to_file)
        layout.addWidget(self.save_to_btn)
        
        self.load_params_btn = QPushButton('Load from...')
        self.load_params_btn.setStyleSheet('padding: 8px;')
        self.load_params_btn.clicked.connect(self.load_params_from_file)
        layout.addWidget(self.load_params_btn)
        
        layout.addStretch()
        self.on_cmd_mode_changed()  # initial visibility
        return panel
        
    def apply_params(self, params):
        """Apply params dict to GUI widgets."""
        defaults = get_default_params()
        merged = {**defaults, **{k: v for k, v in params.items() if k in defaults}}
        self.dt_spin.setValue(merged.get('dt', 0.01))
        self.T_end_spin.setValue(merged.get('T_end', 10.0))
        self.step_time_spin.setValue(merged.get('step_time', 2.0))
        self.step_angle_spin.setValue(merged.get('step_angle_deg', 10.0))
        self.theta_ref_period_spin.setValue(merged.get('theta_ref_period', 5.0))
        self.step_omega_spin.setValue(merged.get('step_omega', 0.5))
        self.step_u_spin.setValue(merged.get('step_u', 0.2))
        mode = merged.get('cmd_mode', 'angle')
        idx = {'angle': 0, 'omega': 1, 'actuator': 2}.get(mode, 0)
        self.cmd_mode_combo.setCurrentIndex(idx)
        self.Kp_angle_spin.setValue(merged.get('Kp_angle', 2.0))
        self.Kp_spin.setValue(merged.get('Kp', 1.0))
        self.Ki_spin.setValue(merged.get('Ki', 0.5))
        self.Kd_spin.setValue(merged.get('Kd', 0.05))
        self.int_max_spin.setValue(merged.get('int_max', 10.0))
        self.tau_act_spin.setValue(merged.get('tau_act', 0.053))
        self.K_act_spin.setValue(merged.get('K_act', 1.0))
        self.theta_gimbal_max_spin.setValue(merged.get('theta_gimbal_max_deg', 10.0))
        self.I_roll_spin.setValue(merged.get('I_roll', 0.02))
        self.F_thrust_spin.setValue(merged.get('F_thrust', 6.0))
        self.L_spin.setValue(merged.get('L', 0.2))
        self.m_spin.setValue(merged.get('m', 0.6))
        self.on_cmd_mode_changed()
    
    def save_params_to_default(self):
        """Save current params to default config file (overwrite)."""
        os.makedirs(CONFIG_DIR, exist_ok=True)
        params = self.get_params()
        try:
            with open(DEFAULT_CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, 'Saved', f'Saved to default config:\n{DEFAULT_CONFIG_PATH}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save: {e}')
    
    def save_params_to_file(self):
        """Save params to user-selected file."""
        os.makedirs(CONFIG_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Parameters to File', DEFAULT_CONFIG_PATH,
            'JSON (*.json);;All Files (*)'
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.get_params(), f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, 'Saved', f'Saved to:\n{path}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save: {e}')
    
    def load_params_from_default(self):
        """Load params from default config file."""
        if os.path.isfile(DEFAULT_CONFIG_PATH):
            try:
                with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                self.apply_params(params)
            except Exception as e:
                QMessageBox.warning(self, 'Load Warning', f'Failed to load default config: {e}\nUsing defaults.')
        else:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(get_default_params(), f, indent=2, ensure_ascii=False)
            self.apply_params(get_default_params())
    
    def load_params_from_file(self):
        """Load params from user-selected file."""
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Parameters', DEFAULT_CONFIG_PATH,
            'JSON (*.json);;All Files (*)'
        )
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                self.apply_params(params)
                QMessageBox.information(self, 'Loaded', f'Loaded from:\n{path}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load: {e}')
        
    def on_cmd_mode_changed(self):
        """Enable step params based on command mode; use larger τ in actuator mode for visible dynamics"""
        idx = self.cmd_mode_combo.currentIndex()
        self.step_angle_spin.setEnabled(idx == 0)
        self.theta_ref_period_spin.setEnabled(idx == 0)
        self.step_omega_spin.setEnabled(idx == 1)
        self.step_u_spin.setEnabled(idx == 2)
        if idx == 2 and self.tau_act_spin.value() < 0.2:
            self.tau_act_spin.setValue(0.3)  # larger τ to see first-order dynamics
        
    def create_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.plot_tabs = QTabWidget()
        
        # Tab 1: Attitude & Control (angle, omega, u)
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.fig_att = Figure(figsize=(12, 8))
        self.canvas_att = FigureCanvas(self.fig_att)
        gs1 = GridSpec(3, 1, figure=self.fig_att, hspace=0.4)
        self.fig_att.suptitle('Attitude & Control', fontsize=14, fontweight='bold', y=0.98)
        self.ax_angle = self.fig_att.add_subplot(gs1[0])
        self.ax_angle.set_xlabel('Time (s)', fontsize=10)
        self.ax_angle.set_ylabel('Roll angle (deg)', fontsize=10)
        self.ax_angle.set_title('Attitude: Reference vs Actual', fontsize=11, fontweight='bold')
        self.ax_angle.grid(True, alpha=0.3)
        self.ax_omega = self.fig_att.add_subplot(gs1[1])
        self.ax_omega.set_xlabel('Time (s)', fontsize=10)
        self.ax_omega.set_ylabel('Angular velocity (deg/s)', fontsize=10)
        self.ax_omega.set_title('Angular Velocity: Desired vs Actual', fontsize=11, fontweight='bold')
        self.ax_omega.grid(True, alpha=0.3)
        self.ax_u = self.fig_att.add_subplot(gs1[2])
        self.ax_u.set_xlabel('Time (s)', fontsize=10)
        self.ax_u.set_ylabel('Control u (PWM/angle cmd)', fontsize=10)
        self.ax_u.set_title('Controller Output (Actuator Command)', fontsize=11, fontweight='bold')
        self.ax_u.grid(True, alpha=0.3)
        tab1_layout.addWidget(self.canvas_att)
        self.plot_tabs.addTab(tab1, 'Attitude & Control')
        
        # Tab 2: Position & Velocity
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        self.fig_traj = Figure(figsize=(12, 8))
        self.canvas_traj = FigureCanvas(self.fig_traj)
        gs2 = GridSpec(2, 1, figure=self.fig_traj, hspace=0.4)
        self.fig_traj.suptitle('TVC Body Position & Velocity', fontsize=14, fontweight='bold', y=0.98)
        self.ax_pos = self.fig_traj.add_subplot(gs2[0])
        self.ax_pos.set_xlabel('Time (s)', fontsize=10)
        self.ax_pos.set_ylabel('Position (m)', fontsize=10)
        self.ax_pos.set_title('TVC Body Position (x, y, z)', fontsize=11, fontweight='bold')
        self.ax_pos.grid(True, alpha=0.3)
        self.ax_vel = self.fig_traj.add_subplot(gs2[1])
        self.ax_vel.set_xlabel('Time (s)', fontsize=10)
        self.ax_vel.set_ylabel('Velocity (m/s)', fontsize=10)
        self.ax_vel.set_title('TVC Body Velocity (vx, vy, vz)', fontsize=11, fontweight='bold')
        self.ax_vel.grid(True, alpha=0.3)
        tab2_layout.addWidget(self.canvas_traj)
        self.plot_tabs.addTab(tab2, 'Position & Velocity')
        
        layout.addWidget(self.plot_tabs)
        return panel
        
    def get_params(self):
        return {
            'dt': self.dt_spin.value(),
            'T_end': self.T_end_spin.value(),
            'step_time': self.step_time_spin.value(),
            'step_angle_deg': self.step_angle_spin.value(),
            'theta_ref_period': self.theta_ref_period_spin.value(),
            'step_omega': self.step_omega_spin.value(),
            'step_u': self.step_u_spin.value(),
            'cmd_mode': ['angle', 'omega', 'actuator'][self.cmd_mode_combo.currentIndex()],
            'Kp_angle': self.Kp_angle_spin.value(),
            'Kp': self.Kp_spin.value(),
            'Ki': self.Ki_spin.value(),
            'Kd': self.Kd_spin.value(),
            'int_max': self.int_max_spin.value(),
            'tau_act': self.tau_act_spin.value(),
            'K_act': self.K_act_spin.value(),
            'theta_gimbal_max_deg': self.theta_gimbal_max_spin.value(),
            'I_roll': self.I_roll_spin.value(),
            'F_thrust': self.F_thrust_spin.value(),
            'L': self.L_spin.value(),
            'm': self.m_spin.value(),
        }
        
    def run_sim(self):
        try:
            params = self.get_params()
            t, theta_ref, theta_act, omega_des, omega_act, u_cmd, theta_gimbal, pos, vel = run_simulation(params)
            
            # Plot
            self.ax_angle.clear()
            self.ax_angle.plot(t, np.degrees(theta_ref), 'b--', label='θ_ref', linewidth=2)
            self.ax_angle.plot(t, np.degrees(theta_act), 'r-', label='θ_act', linewidth=1.5)
            if params['cmd_mode'] == 'actuator':
                # θ_gimbal theory (actuator first-order); θ_act comes from rigid body
                theta_gimbal_theory = actuator_theory(t, params['step_time'], params['step_u'],
                                                      params['K_act'], params['tau_act'])
                self.ax_angle.plot(t, np.degrees(theta_gimbal_theory), 'g:', label='θ_gimbal theory', linewidth=1.5)
            self.ax_angle.set_xlabel('Time (s)', fontsize=10)
            self.ax_angle.set_ylabel('Roll angle (deg)', fontsize=10)
            self.ax_angle.set_title('Angle: Reference vs Actual', fontsize=11, fontweight='bold')
            # self.ax_angle.set_ylim(-50, 50)
            self.ax_angle.legend(loc='lower right', fontsize=9)
            self.ax_angle.grid(True, alpha=0.3)
            self.ax_angle.tick_params(axis='both', labelsize=9)
            
            self.ax_omega.clear()
            self.ax_omega.plot(t, np.degrees(omega_des), 'b--', label='ω_des', linewidth=2)
            self.ax_omega.plot(t, np.degrees(omega_act), 'r-', label='ω_act', linewidth=1.5)
            self.ax_omega.set_xlabel('Time (s)', fontsize=10)
            self.ax_omega.set_ylabel('Angular velocity (deg/s)', fontsize=10)
            self.ax_omega.set_title('Angular Velocity: Desired vs Actual', fontsize=11, fontweight='bold')
            # self.ax_omega.set_ylim(-100, 100)  # ±100 deg/s (plot in deg/s)
            self.ax_omega.legend(loc='upper right', fontsize=9)
            self.ax_omega.grid(True, alpha=0.3)
            self.ax_omega.tick_params(axis='both', labelsize=9)
            
            # TVC body position and velocity
            self.ax_pos.clear()
            self.ax_pos.plot(t, pos[:, 0], 'b-', label='x', linewidth=1.5)
            self.ax_pos.plot(t, pos[:, 1], 'g-', label='y', linewidth=1.5)
            self.ax_pos.plot(t, pos[:, 2], 'r-', label='z', linewidth=1.5)
            self.ax_pos.set_xlabel('Time (s)', fontsize=10)
            self.ax_pos.set_ylabel('Position (m)', fontsize=10)
            self.ax_pos.set_title('TVC Body Position (x, y, z)', fontsize=11, fontweight='bold')
            self.ax_pos.legend(loc='upper right', fontsize=9)
            self.ax_pos.grid(True, alpha=0.3)
            self.ax_pos.tick_params(axis='both', labelsize=9)
            
            self.ax_vel.clear()
            self.ax_vel.plot(t, vel[:, 0], 'b-', label='vx', linewidth=1.5)
            self.ax_vel.plot(t, vel[:, 1], 'g-', label='vy', linewidth=1.5)
            self.ax_vel.plot(t, vel[:, 2], 'r-', label='vz', linewidth=1.5)
            self.ax_vel.set_xlabel('Time (s)', fontsize=10)
            self.ax_vel.set_ylabel('Velocity (m/s)', fontsize=10)
            self.ax_vel.set_title('TVC Body Velocity (vx, vy, vz)', fontsize=11, fontweight='bold')
            self.ax_vel.legend(loc='upper right', fontsize=9)
            self.ax_vel.grid(True, alpha=0.3)
            self.ax_vel.tick_params(axis='both', labelsize=9)
            
            self.ax_u.clear()
            cmd_mode = params['cmd_mode']
            # u_cmd: control command; theta_gimbal: actuator output (TVC gimbal angle)
            self.ax_u.plot(t, np.degrees(u_cmd), 'g-', label='u (command)', linewidth=1.5)
            if theta_gimbal is not None:
                self.ax_u.plot(t, np.degrees(theta_gimbal), 'm-', label='θ_gimbal (actual)', linewidth=1.5)
            self.ax_u.set_xlabel('Time (s)', fontsize=10)
            self.ax_u.set_ylabel('Control (deg)', fontsize=10)
            # self.ax_u.set_ylim(-50, 50)
            self.ax_u.set_title('Controller Output: Command vs Actual Response', fontsize=11, fontweight='bold')
            self.ax_u.legend(loc='upper right', fontsize=9)
            self.ax_u.grid(True, alpha=0.3)
            self.ax_u.tick_params(axis='both', labelsize=9)
            
            self.fig_att.tight_layout(rect=[0, 0, 1, 0.96])
            self.canvas_att.draw()
            self.fig_traj.tight_layout(rect=[0, 0, 1, 0.96])
            self.canvas_traj.draw()
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
