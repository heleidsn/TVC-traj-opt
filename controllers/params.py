# -*- coding: utf-8 -*-
"""Controller parameter schemas and defaults for the Tracking GUI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .px4_params import default_px4_gui_params, migrate_px4_params, px4_param_specs

CONTROLLER_PX4 = 'px4_cascade'
CONTROLLER_LQR = 'lqr'
CONTROLLER_MPC = 'mpc'

CONTROLLER_IDS = (CONTROLLER_PX4, CONTROLLER_LQR, CONTROLLER_MPC)
CONTROLLER_LABELS = {
    CONTROLLER_PX4: 'PX4 cascade (pos→vel→att→rate)',
    CONTROLLER_LQR: 'LQR (full-state + feedforward)',
    CONTROLLER_MPC: 'MPC (linear receding horizon)',
}

SIM_NUMERICAL = 'numerical'
SIM_SITL = 'px4_sitl'

# Tracking options (per controller; timing is in numerical_sim block)
_COMMON_SIM_SPECS = [
    {'key': 'use_feedforward', 'label': 'Use planned control FF', 'default': 1.0, 'min': 0.0, 'max': 1.0,
     'decimals': 1, 'checkbox': True},
    {'key': 'clip_pos_error', 'label': 'Max pos error [m]', 'default': 0.5, 'min': 0.05, 'max': 5.0, 'decimals': 2},
    {'key': 'clip_vel_error', 'label': 'Max vel error [m/s]', 'default': 0.5, 'min': 0.05, 'max': 5.0, 'decimals': 2},
]

# Actuator dynamics for numerical tracking sim (configured in Tracking tab, not per-controller).
_ACTUATOR_TRACKING_KEYS = (
    'act_dyn_enable',
    'thrust_quant_enable', 'thrust_resolution_N',
    'tau_gimbal', 'tau_thrust', 'tau_yaw_torque',
)

_LQR_SPECS = []
for k, lbl, default in [
    ('Q_x', 'Q pos x', 1.0), ('Q_y', 'Q pos y', 1.0), ('Q_z', 'Q pos z', 1.0),
    ('Q_vx', 'Q vel x', 1.0), ('Q_vy', 'Q vel y', 1.0), ('Q_vz', 'Q vel z', 1.0),
    ('Q_qx', 'Q quat x', 1.0), ('Q_qy', 'Q quat y', 1.0), ('Q_qz', 'Q quat z', 0.1),
    ('Q_p', 'Q rate p', 1.0), ('Q_q', 'Q rate q', 1.0), ('Q_r', 'Q rate r', 0.01),
    ('R_qx', 'R gimbal x', 10.0), ('R_qy', 'R gimbal y', 10.0), ('R_T', 'R thrust', 1.0),
    ('R_r', 'R yaw torque', 10.0),
]:
    _LQR_SPECS.append({
        'key': k, 'label': lbl, 'default': default,
        'min': 0.001, 'max': 1000.0, 'decimals': 3,
    })

_MPC_SPECS = [
    {'key': 'horizon', 'label': 'Horizon steps', 'default': 20, 'min': 5, 'max': 80, 'decimals': 0, 'integer': True},
    {'key': 'mpc_dt', 'label': 'MPC dt [s]', 'default': 0.05, 'min': 0.01, 'max': 0.2, 'decimals': 3},
] + _LQR_SPECS  # reuse Q/R weights

_PARAM_REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    CONTROLLER_LQR: _LQR_SPECS + _COMMON_SIM_SPECS,
    CONTROLLER_MPC: _MPC_SPECS + _COMMON_SIM_SPECS,
}


def param_specs_for(controller_id: str, px4_params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Return flat GUI spinbox specs for the given controller."""
    from .param_groups import flatten_param_groups, param_groups_for
    return flatten_param_groups(param_groups_for(controller_id, px4_params=px4_params))


def default_params_for(controller_id: str) -> Dict[str, float]:
    """Default parameter dict for one controller."""
    if controller_id == CONTROLLER_PX4:
        return default_px4_gui_params(share_rp=True)
    out = {}
    for spec in param_specs_for(controller_id):
        key = spec['key']
        if spec.get('checkbox'):
            out[key] = bool(spec['default'])
        elif spec.get('integer'):
            out[key] = int(spec['default'])
        else:
            out[key] = float(spec['default'])
    return out


def default_numerical_sim_config() -> Dict[str, float]:
    """Plant integration step and controller update period for numerical tracking."""
    return {
        'sim_dt': 0.005,
        'control_dt': 0.02,
    }


def migrate_numerical_sim_config(cfg: Dict[str, Any] | None) -> Dict[str, float]:
    """Load numerical_sim block; lift legacy sim_dt from per-controller params."""
    if cfg and cfg.get('numerical_sim'):
        ns = dict(cfg['numerical_sim'])
        return {
            'sim_dt': float(ns.get('sim_dt', 0.005)),
            'control_dt': float(ns.get('control_dt', 0.02)),
        }
    params_map = (cfg or {}).get('params') or {}
    for raw in params_map.values():
        if isinstance(raw, dict) and 'sim_dt' in raw:
            old_dt = float(raw['sim_dt'])
            plant_dt = min(old_dt, 0.01)
            return {'sim_dt': plant_dt, 'control_dt': old_dt}
    return default_numerical_sim_config()


def all_default_tracking_config() -> Dict[str, Any]:
    """Full tracking section for JSON persistence."""
    from .actuator_dynamics import default_actuator_tracking_config
    return {
        'controller': CONTROLLER_PX4,
        'sim_mode': SIM_NUMERICAL,
        'numerical_sim': default_numerical_sim_config(),
        'actuator': default_actuator_tracking_config(),
        'params': {
            cid: default_params_for(cid) for cid in CONTROLLER_IDS
        },
    }
