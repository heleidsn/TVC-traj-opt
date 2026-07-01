# -*- coding: utf-8 -*-
"""Numerical step-response simulation for isolated PX4 cascade tuning levels."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np

from .actuator_dynamics import ActuatorDynamics, actuator_config_from_params
from .linear_model import lqr_to_control_opt, state12_to_state13, state13_to_state12
from .nonlinear_plant import plant_from_phy_gui, tracker_phy_from_gui
from .simulator import numerical_sim_timing
from .px4_cascade import (
    PX4CascadeTracker,
    TUNE_LEVEL_ATTITUDE,
    TUNE_LEVEL_POSITION,
    TUNE_LEVEL_RATE,
    TUNE_LEVEL_VELOCITY,
    TUNE_LEVELS,
)

_DEG = np.pi / 180.0

TUNE_LEVEL_LABELS = {
    TUNE_LEVEL_RATE: 'Rate (rate PID only)',
    TUNE_LEVEL_ATTITUDE: 'Attitude (att P → rate PID)',
    TUNE_LEVEL_VELOCITY: 'Velocity (vel PID → att P → rate PID)',
    TUNE_LEVEL_POSITION: 'Position (pos P → vel PID → att P → rate PID)',
}

DEFAULT_PX4_TUNE_CONFIG: Dict[str, Any] = {
    'level': TUNE_LEVEL_RATE,
    'duration_s': 5.0,
    'setpoints': {
        TUNE_LEVEL_RATE: {'p_deg_s': 20.0, 'q_deg_s': 0.0, 'r_deg_s': 0.0},
        TUNE_LEVEL_ATTITUDE: {'roll_deg': 5.0, 'pitch_deg': 0.0, 'yaw_deg': 0.0},
        TUNE_LEVEL_VELOCITY: {'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'yaw_deg': 0.0},
        TUNE_LEVEL_POSITION: {'x': 0.5, 'y': 0.0, 'z': 0.2, 'yaw_deg': 0.0},
    },
}


def default_px4_tune_config() -> Dict[str, Any]:
    return deepcopy(DEFAULT_PX4_TUNE_CONFIG)


def tune_reference_state(level: str, setpoints: Dict[str, Any] | None = None) -> np.ndarray:
    """Build a 12-state reference vector for plotting at the active tune level."""
    x_ref = np.zeros(12, dtype=float)
    sp = dict(setpoints or {})
    if level == TUNE_LEVEL_POSITION:
        x_ref[0:3] = [
            float(sp.get('x', 0.0)),
            float(sp.get('y', 0.0)),
            float(sp.get('z', 0.0)),
        ]
        yaw = float(sp.get('yaw_deg', 0.0)) * _DEG
        x_ref[8] = 0.5 * yaw
    elif level == TUNE_LEVEL_VELOCITY:
        x_ref[3:6] = [
            float(sp.get('vx', 0.0)),
            float(sp.get('vy', 0.0)),
            float(sp.get('vz', 0.0)),
        ]
        yaw = float(sp.get('yaw_deg', 0.0)) * _DEG
        x_ref[8] = 0.5 * yaw
    elif level == TUNE_LEVEL_ATTITUDE:
        roll = float(sp.get('roll_deg', 0.0)) * _DEG
        pitch = float(sp.get('pitch_deg', 0.0)) * _DEG
        yaw = float(sp.get('yaw_deg', 0.0)) * _DEG
        x_ref[6:9] = [0.5 * roll, 0.5 * pitch, 0.5 * yaw]
    elif level == TUNE_LEVEL_RATE:
        x_ref[9:12] = [
            float(sp.get('p_deg_s', 0.0)) * _DEG,
            float(sp.get('q_deg_s', 0.0)) * _DEG,
            float(sp.get('r_deg_s', 0.0)) * _DEG,
        ]
    return x_ref


def run_px4_cascade_tune_sim(
    params: Dict[str, Any],
    phy_gui: Dict[str, Any],
    tune_config: Dict[str, Any],
    x0: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Step-response simulation with one cascade level isolated (nonlinear dynamics model).

    Returns the same keys as run_tracking_simulation plus tune_level / tune_setpoints.
    """
    level = tune_config.get('level', TUNE_LEVEL_RATE)
    if level not in TUNE_LEVELS:
        raise ValueError(f'Unknown tune level: {level}')

    duration = float(tune_config.get('duration_s', 5.0))
    sp_map = tune_config.get('setpoints') or {}
    setpoints = dict(sp_map.get(level, sp_map))

    phy = tracker_phy_from_gui(phy_gui)
    dyn_model = plant_from_phy_gui(phy_gui)
    mass, g = phy['MASS'], phy['G']
    tracker = PX4CascadeTracker(phy, params)
    tracker.reset()

    sim_dt, control_dt, steps_per_control = numerical_sim_timing(params)
    t_end = max(duration, sim_dt)
    actuator = ActuatorDynamics(actuator_config_from_params(params))
    if actuator.any_enabled():
        actuator.reset(np.array([0.0, 0.0, mass * g, 0.0], dtype=float))

    if x0 is None:
        x12_0 = np.zeros(12, dtype=float)
    else:
        x12_0 = np.asarray(x0, dtype=float).reshape(12)
    x13 = state12_to_state13(x12_0)

    t_hist = []
    x12_hist = []
    u_opt_hist = []
    u_cmd_hist = []
    sp_pos_hist = []
    sp_vel_hist = []
    sp_att_hist = []
    sp_rate_hist = []

    u_cmd_hold = None
    cascade_hold = None
    substep = 0

    t = 0.0
    while True:
        x12 = state13_to_state12(x13)

        if substep == 0:
            u_lqr, sig = tracker.compute_tune(x12, control_dt, level, setpoints, use_ff=False)
            u_cmd_hold = lqr_to_control_opt(u_lqr, mass, g)
            cascade_hold = sig

        u_plant = actuator.step(u_cmd_hold, sim_dt) if actuator.any_enabled() else u_cmd_hold

        t_hist.append(t)
        x12_hist.append(x12.copy())
        u_opt_hist.append(u_plant.copy())
        u_cmd_hist.append(u_cmd_hold.copy())
        if cascade_hold is not None:
            sp_pos_hist.append(cascade_hold['pos'].copy())
            sp_vel_hist.append(cascade_hold['vel'].copy())
            sp_att_hist.append(cascade_hold['att_rad'].copy())
            sp_rate_hist.append(cascade_hold['rate_rad_s'].copy())

        if t >= t_end - 1e-12:
            break
        x13 = dyn_model.step(x13, u_plant, sim_dt)
        t += sim_dt
        substep = (substep + 1) % steps_per_control

    t_arr = np.asarray(t_hist, dtype=float)
    x_sim = np.asarray(x12_hist, dtype=float)
    x_ref_arr = np.array([tune_reference_state(level, setpoints) for _ in t_arr])
    pos_err = x_sim[:, 0:3] - x_ref_arr[:, 0:3]
    err_norm = np.linalg.norm(pos_err, axis=1)

    result = {
        't': t_arr,
        'x_sim': x_sim,
        'x_ref': x_ref_arr,
        'pos_err': pos_err,
        'max_pos_err_m': float(np.max(err_norm)),
        'rmse_pos_m': float(np.sqrt(np.mean(err_norm ** 2))),
        'controller_id': 'px4_cascade',
        'u_hist': np.asarray(u_opt_hist, dtype=float),
        'use_feedforward': False,
        'tune_level': level,
        'tune_setpoints': setpoints,
        'tune_duration_s': duration,
        'is_cascade_tune': True,
        'dynamics_model': 'nonlinear',
        'actuator_dynamics_enabled': actuator.any_enabled(),
        'r_thrust_body': np.array([
            float(phy_gui.get('r_thrust_x', 0.0)),
            float(phy_gui.get('r_thrust_y', 0.0)),
            float(phy_gui.get('r_thrust_z', -0.2)),
        ], dtype=float),
        'cascade_sp': {
            'pos': np.asarray(sp_pos_hist, dtype=float),
            'vel': np.asarray(sp_vel_hist, dtype=float),
            'att_rad': np.asarray(sp_att_hist, dtype=float),
            'rate_rad_s': np.asarray(sp_rate_hist, dtype=float),
        },
        'u_cmd_hist': np.asarray(u_cmd_hist, dtype=float),
    }
    if actuator.any_enabled():
        result['actuator_config'] = actuator_config_from_params(params)
    return result
