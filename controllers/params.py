# -*- coding: utf-8 -*-
"""Controller parameter schemas and defaults for the Tracking GUI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .px4_params import default_px4_gui_params, migrate_px4_params, px4_param_specs

CONTROLLER_PX4 = 'px4_cascade'
CONTROLLER_LQR = 'lqr'
CONTROLLER_MPC = 'mpc'
CONTROLLER_ACADOS_NMPC = 'acados_nmpc'
CONTROLLER_FLATNESS = 'flatness_ff'

CONTROLLER_IDS = (
    CONTROLLER_PX4, CONTROLLER_LQR, CONTROLLER_MPC,
    CONTROLLER_ACADOS_NMPC, CONTROLLER_FLATNESS,
)
CONTROLLER_LABELS = {
    CONTROLLER_PX4: 'PX4 cascade (pos→vel→att→rate)',
    CONTROLLER_LQR: 'LQR (full-state + feedforward)',
    CONTROLLER_MPC: 'MPC (linear horizon, u_ref + Δu)',
    CONTROLLER_ACADOS_NMPC: 'NMPC (acados nonlinear, direct u)',
    CONTROLLER_FLATNESS: 'Flatness cascade (ξ loop + planned FF)',
}

SIM_NUMERICAL = 'numerical'
SIM_SITL = 'px4_sitl'

# Fixed closed-loop tracking behaviour (no longer exposed in the Tracking GUI).
TRACKING_USE_FEEDFORWARD = True
TRACKING_CLIP_POS_ERROR_M = 0.5
TRACKING_CLIP_VEL_ERROR_M_S = 0.5
TRACKING_REF_MASS_KG = 0.6
TRACKING_GIMBAL_RATE_LIMIT_DEG_S = 45.0
# After planned trajectory time, hold the final waypoint for this duration [s].
TRACKING_TERMINAL_HOLD_DURATION_S = 3.0
TRACKING_MANEUVER_GIMBAL_R_EXPONENT = 1.5
# Terminal hover: stronger gimbal authority + larger error window (MPC still used).
TRACKING_TERMINAL_HOLD_GIMBAL_R_EXPONENT = 1.0
TRACKING_TERMINAL_CLIP_POS_ERROR_M = 1.0
TRACKING_TERMINAL_CLIP_VEL_ERROR_M_S = 1.0


def scale_tracking_gimbal_r(
    params: Dict[str, Any],
    mass_kg: float,
    ref_mass: float = TRACKING_REF_MASS_KG,
    exponent: float = TRACKING_MANEUVER_GIMBAL_R_EXPONENT,
) -> Dict[str, Any]:
    """
    Increase R_qx/R_qy for heavier platforms so gimbal feedback does not saturate.

    GUI defaults are tuned for the ~0.6 kg proxy; the 20 kg real vehicle needs
    much softer attitude corrections to avoid ±gimbal limit chatter during
    aggressive tracking.
    """
    scaled = dict(params)
    if mass_kg <= ref_mass * 1.05:
        return scaled
    gain = (float(mass_kg) / ref_mass) ** float(exponent)
    for key in ('R_qx', 'R_qy'):
        if key in scaled:
            scaled[key] = float(scaled[key]) * gain
    return scaled

_LEGACY_TRACKING_OPTION_KEYS = (
    'use_feedforward',
    'clip_pos_error',
    'clip_vel_error',
)

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

_NMPC_SPECS = [
    {'key': 'horizon', 'label': 'Horizon steps', 'default': 15, 'min': 5, 'max': 60, 'decimals': 0, 'integer': True},
    {'key': 'nmpc_dt', 'label': 'NMPC dt [s]', 'default': 0.05, 'min': 0.01, 'max': 0.2, 'decimals': 3},
    {'key': 'nmpc_du_weight', 'label': 'Control-rate weight du', 'default': 0.05, 'min': 0.0, 'max': 10.0, 'decimals': 4},
    {'key': 'nmpc_terminal_scale', 'label': 'Terminal cost scale', 'default': 10.0, 'min': 1.0, 'max': 500.0, 'decimals': 1},
    {'key': 'nmpc_nlp_max_iter', 'label': 'SQP max iter', 'default': 10, 'min': 1, 'max': 50, 'decimals': 0, 'integer': True},
] + _LQR_SPECS

_PARAM_REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    CONTROLLER_LQR: _LQR_SPECS,
    CONTROLLER_MPC: _MPC_SPECS,
    CONTROLLER_ACADOS_NMPC: _NMPC_SPECS,
}


def param_specs_for(controller_id: str, px4_params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Return flat GUI spinbox specs for the given controller."""
    from .param_groups import flatten_param_groups, param_groups_for
    return flatten_param_groups(param_groups_for(controller_id, px4_params=px4_params))


def default_params_for(controller_id: str) -> Dict[str, float]:
    """Default parameter dict for one controller."""
    if controller_id in (CONTROLLER_PX4, CONTROLLER_FLATNESS):
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


def default_controller_params_map() -> Dict[str, Dict[str, Any]]:
    """Default params for every controller (one platform slot)."""
    return {cid: default_params_for(cid) for cid in CONTROLLER_IDS}


_TRACKING_PLATFORM_IDS = ('proxy', 'real')


def default_params_by_platform() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Independent controller-gain sets for each rocket platform."""
    return {pid: default_controller_params_map() for pid in _TRACKING_PLATFORM_IDS}


def _normalize_tracking_platform_id(platform_id: str | None) -> str:
    pid = str(platform_id or 'proxy').strip().lower()
    if pid in ('real', 'flight'):
        return 'real'
    return 'proxy'


def _merge_controller_params(controller_id: str, base: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    merged.update(dict(raw or {}))
    if controller_id in (CONTROLLER_PX4, CONTROLLER_FLATNESS):
        merged = migrate_px4_params(merged)
    return strip_legacy_tracking_options(merged)


def migrate_params_by_platform(cfg: Dict[str, Any] | None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build per-platform controller params from a tracking config dict.

    New format: ``params_by_platform.{proxy|real}.{controller_id}``.
    Legacy flat ``params`` is treated as the proxy set (historically tuned there);
    real keeps independent defaults so both platforms no longer share gains.
    """
    out = default_params_by_platform()
    raw_by = (cfg or {}).get('params_by_platform')
    if isinstance(raw_by, dict) and raw_by:
        for pid in _TRACKING_PLATFORM_IDS:
            src = raw_by.get(pid)
            if pid == 'real' and not isinstance(src, dict):
                src = raw_by.get('flight')
            if not isinstance(src, dict):
                continue
            for cid in CONTROLLER_IDS:
                if cid in src and isinstance(src[cid], dict):
                    out[pid][cid] = _merge_controller_params(cid, out[pid][cid], src[cid])
        return out

    legacy = (cfg or {}).get('params') or {}
    if isinstance(legacy, dict):
        for cid in CONTROLLER_IDS:
            if cid in legacy and isinstance(legacy[cid], dict):
                out['proxy'][cid] = _merge_controller_params(
                    cid, out['proxy'][cid], legacy[cid],
                )
    return out


def params_map_for_platform(
    cfg: Dict[str, Any] | None,
    platform_id: str | None,
) -> Dict[str, Dict[str, Any]]:
    """Return the controller-params map for one platform (ensures defaults exist)."""
    by_plat = (cfg or {}).get('params_by_platform')
    if not isinstance(by_plat, dict) or not by_plat:
        by_plat = migrate_params_by_platform(cfg)
    pid = _normalize_tracking_platform_id(platform_id)
    if pid not in by_plat or not isinstance(by_plat[pid], dict):
        by_plat[pid] = default_controller_params_map()
    for cid in CONTROLLER_IDS:
        if cid not in by_plat[pid] or not isinstance(by_plat[pid][cid], dict):
            by_plat[pid][cid] = default_params_for(cid)
    return by_plat[pid]


def default_numerical_sim_config() -> Dict[str, float]:
    """Plant step, controller period, and simulation horizon for numerical tracking."""
    return {
        'sim_dt': 0.005,
        'control_dt': 0.02,
        'terminal_hold_duration_s': TRACKING_TERMINAL_HOLD_DURATION_S,
        'total_duration_s': 0.0,
    }


def strip_legacy_tracking_options(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove retired Tracking-options keys from a saved controller param dict."""
    cleaned = dict(params or {})
    for key in _LEGACY_TRACKING_OPTION_KEYS:
        cleaned.pop(key, None)
    return cleaned

def migrate_numerical_sim_config(cfg: Dict[str, Any] | None) -> Dict[str, float]:
    """Load numerical_sim block; lift legacy sim_dt from per-controller params."""
    defaults = default_numerical_sim_config()
    if cfg and cfg.get('numerical_sim'):
        ns = dict(cfg['numerical_sim'])
        return {
            'sim_dt': float(ns.get('sim_dt', defaults['sim_dt'])),
            'control_dt': float(ns.get('control_dt', defaults['control_dt'])),
            'terminal_hold_duration_s': float(
                ns.get('terminal_hold_duration_s', defaults['terminal_hold_duration_s']),
            ),
            'total_duration_s': float(ns.get('total_duration_s', defaults['total_duration_s'])),
        }
    params_map = (cfg or {}).get('params') or {}
    for raw in params_map.values():
        if isinstance(raw, dict) and 'sim_dt' in raw:
            old_dt = float(raw['sim_dt'])
            plant_dt = min(old_dt, 0.01)
            out = dict(defaults)
            out['sim_dt'] = plant_dt
            out['control_dt'] = old_dt
            return out
    return defaults


def all_default_tracking_config() -> Dict[str, Any]:
    """Full tracking section for JSON persistence."""
    from .actuator_dynamics import default_actuator_tracking_config
    by_platform = default_params_by_platform()
    return {
        'controller': CONTROLLER_PX4,
        'sim_mode': SIM_NUMERICAL,
        'numerical_sim': default_numerical_sim_config(),
        'actuator': default_actuator_tracking_config(),
        # Independent gain sets per rocket platform (proxy vs real).
        'params_by_platform': by_platform,
        # Back-compat active view: same object as params_by_platform['proxy'].
        'params': by_platform['proxy'],
    }
