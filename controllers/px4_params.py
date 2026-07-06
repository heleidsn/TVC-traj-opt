# -*- coding: utf-8 -*-
"""PX4 cascade parameter schema, migration, and normalization for simulation."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List

_DEG = math.pi / 180.0

# Legacy key mapping → new keys (values copied if new key absent).
_LEGACY_ALIASES = {
    'Kp_pos': 'Kp_pos_xy',
    'Ki_pos': 'Ki_pos_xy',
    'Kd_pos': 'Kd_pos_xy',
    'Kp_vel': 'Kp_vel_xy',
    'Ki_vel': 'Ki_vel_xy',
    'Kd_vel': 'Kd_vel_xy',
    'Kp_att': 'Kp_att_rp_deg',
    'Ki_att': 'Ki_att_rp_deg',
    'Kd_att': 'Kd_att_rp_deg',
    'Kp_rate': 'Kp_rate_rp',
    'Ki_rate': 'Ki_rate_rp',
    'Kd_rate': 'Kd_rate_rp',
    'tilt_max': 'tilt_max_deg',
    'gimbal_max': 'gimbal_max_deg',
}


def _spin(key, label, default, min_v, max_v, decimals=3, integer=False, checkbox=False, full_width=False):
    return {
        'key': key, 'label': label, 'default': default,
        'min': min_v, 'max': max_v, 'decimals': decimals,
        'integer': integer, 'checkbox': checkbox, 'full_width': full_width,
    }


def px4_param_groups(share_rp: bool = True) -> List[Dict[str, Any]]:
    """Grouped GUI parameter sections for PX4 cascade (P + PID per loop pair)."""
    groups = [
        {
            'title': 'General',
            'specs': [
                _spin('share_rp_gains', 'Roll/pitch share same gains', 1.0, 0, 1, 1,
                      checkbox=True, full_width=True),
                _spin('use_acc_feedforward', 'Velocity loop: planned acc FF', 0.0, 0, 1, 1,
                      checkbox=True, full_width=True),
            ],
        },
        {
            'title': 'Position P + velocity PID (XY)',
            'specs': [
                _spin('Kp_pos_xy', 'Pos — Kp', 0.95, 0.0, 10.0),
                _spin('Kp_vel_xy', 'Vel — Kp', 1.8, 0.0, 20.0),
                _spin('Ki_vel_xy', 'Vel — Ki', 0.4, 0.0, 10.0),
                _spin('Kd_vel_xy', 'Vel — Kd', 0.02, 0.0, 5.0),
            ],
        },
        {
            'title': 'Position P + velocity PID (Z)',
            'specs': [
                _spin('Kp_pos_z', 'Pos — Kp', 1.0, 0.0, 10.0),
                _spin('Kp_vel_z', 'Vel — Kp', 1.5, 0.0, 20.0),
                _spin('Ki_vel_z', 'Vel — Ki', 0.2, 0.0, 10.0),
                _spin('Kd_vel_z', 'Vel — Kd', 0.02, 0.0, 5.0),
            ],
        },
    ]
    if share_rp:
        groups.append({
            'title': 'Attitude P + rate PID (roll / pitch)',
            'specs': [
                _spin('Kp_att_rp_deg', 'Att — Kp [1/deg]', 6.5, 0.0, 50.0, 2),
                _spin('Kp_rate_rp', 'Rate — Kp', 18.0, 0.0, 100.0, 2),
                _spin('Ki_rate_rp', 'Rate — Ki', 0.2, 0.0, 50.0, 2),
                _spin('Kd_rate_rp', 'Rate — Kd', 0.01, 0.0, 20.0, 2),
            ],
        })
    else:
        rp_specs = []
        for axis in ('roll', 'pitch'):
            cap = axis.capitalize()
            rp_specs.extend([
                _spin(f'Kp_att_{axis}_deg', f'{cap} att — Kp [1/deg]', 6.5, 0.0, 50.0, 2),
                _spin(f'Kp_rate_{axis}', f'{cap} rate — Kp', 18.0, 0.0, 100.0, 2),
                _spin(f'Ki_rate_{axis}', f'{cap} rate — Ki', 0.2, 0.0, 50.0, 2),
                _spin(f'Kd_rate_{axis}', f'{cap} rate — Kd', 0.01, 0.0, 20.0, 2),
            ])
        groups.append({'title': 'Attitude P + rate PID (roll / pitch)', 'specs': rp_specs})
    groups.extend([
        {
            'title': 'Attitude P + rate PID (yaw)',
            'specs': [
                _spin('Kp_att_yaw_deg', 'Att — Kp [1/deg]', 4.0, 0.0, 50.0, 2),
                _spin('Kp_rate_yaw', 'Rate — Kp', 12.0, 0.0, 100.0, 2),
                _spin('Ki_rate_yaw', 'Rate — Ki', 0.1, 0.0, 50.0, 2),
                _spin('Kd_rate_yaw', 'Rate — Kd', 0.0, 0.0, 20.0, 2),
            ],
        },
    ])
    return groups


def px4_param_specs(share_rp: bool = True) -> List[Dict[str, Any]]:
    """Flat list of all PX4 parameter specs (for migration defaults)."""
    specs = []
    for group in px4_param_groups(share_rp):
        specs.extend(group['specs'])
    return specs


def migrate_px4_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade saved JSON / legacy keys to the current PX4 parameter set."""
    p = dict(raw or {})
    for old, new in _LEGACY_ALIASES.items():
        if old in p and new not in p:
            val = p[old]
            if old in ('tilt_max', 'gimbal_max') and val < 3.0:
                p[new] = float(val) * (180.0 / math.pi)
            else:
                p[new] = val
    if 'share_rp_gains' not in p:
        p['share_rp_gains'] = True
    share = bool(p.get('share_rp_gains', True))
    if share:
        if 'Kp_att_rp_deg' in p:
            for axis in ('roll', 'pitch'):
                p.setdefault(f'Kp_att_{axis}_deg', p['Kp_att_rp_deg'])
        for prefix in ('Kp_rate', 'Ki_rate', 'Kd_rate'):
            rp_key = f'{prefix}_rp'
            if rp_key in p:
                for axis in ('roll', 'pitch'):
                    p.setdefault(f'{prefix}_{axis}', p[rp_key])
    defaults = {s['key']: s['default'] for s in px4_param_specs(share_rp=share)}
    for k, v in defaults.items():
        p.setdefault(k, v)
    return p


def _tilt_max_rad_from_params(p: Dict[str, Any]) -> float:
    """Max tilt [rad] from Parameters → Constraints (pitch/roll), or legacy saved keys."""
    if 'pitch_max' in p or 'roll_max' in p:
        pitch_deg = float(p.get('pitch_max', 10.0))
        roll_deg = float(p.get('roll_max', pitch_deg))
        return min(pitch_deg, roll_deg) * _DEG
    if 'tilt_max_deg' in p:
        return float(p['tilt_max_deg']) * _DEG
    if 'tilt_max' in p and float(p['tilt_max']) < 3.0:
        return float(p['tilt_max'])
    return 10.0 * _DEG


def gimbal_pitch_roll_max_rad(p: Dict[str, Any]) -> tuple[float, float]:
    """TVC pitch / roll limits [rad] from Parameters → Constraints or legacy keys."""
    if 'th_p_max' in p or 'th_r_max' in p:
        th_p_deg = float(p.get('th_p_max', 10.0))
        th_r_deg = float(p.get('th_r_max', th_p_deg))
        return th_p_deg * _DEG, th_r_deg * _DEG
    if 'gimbal_max_deg' in p:
        g = float(p['gimbal_max_deg']) * _DEG
        return g, g
    if 'gimbal_max' in p and float(p['gimbal_max']) < 3.0:
        g = float(p['gimbal_max'])
        return g, g
    g = 10.0 * _DEG
    return g, g


def _gimbal_max_rad_from_params(p: Dict[str, Any]) -> float:
    """Legacy scalar gimbal limit [rad] (min of pitch and roll)."""
    th_p, th_r = gimbal_pitch_roll_max_rad(p)
    return min(th_p, th_r)


def clip_lqr_gimbal_cmd(
    u4,
    th_p_max_rad: float,
    th_r_max_rad: float,
):
    """
  Clip LQR inputs [qx, qy, T_delta, r] to TVC gimbal limits.

  Small-angle map: qx ≈ th_r/2, qy ≈ th_p/2.
  """
    import numpy as np

    u = np.asarray(u4, dtype=float).reshape(4).copy()
    u[0] = np.clip(u[0], -th_r_max_rad / 2.0, th_r_max_rad / 2.0)
    u[1] = np.clip(u[1], -th_p_max_rad / 2.0, th_p_max_rad / 2.0)
    return u


def clip_control_opt(
    u4,
    th_p_max_rad: float,
    th_r_max_rad: float,
    T_min: float | None = None,
    T_max: float | None = None,
    tau_yaw_max: float | None = None,
):
    """Clip planner controls [th_p, th_r, T, tau_yaw] to constraint limits."""
    import numpy as np

    u = np.asarray(u4, dtype=float).reshape(4).copy()
    u[0] = np.clip(u[0], -th_p_max_rad, th_p_max_rad)
    u[1] = np.clip(u[1], -th_r_max_rad, th_r_max_rad)
    if T_min is not None and T_max is not None:
        u[2] = np.clip(u[2], float(T_min), float(T_max))
    if tau_yaw_max is not None:
        tau = float(tau_yaw_max)
        u[3] = np.clip(u[3], -tau, tau)
    return u


def normalize_px4_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert GUI-stored PX4 parameters to internal units for PX4CascadeTracker.

    Angle limits: deg → rad. Attitude P gains are per-degree.
    """
    p = migrate_px4_params(raw)
    share = bool(p.get('share_rp_gains', True))

    def _pair(prefix, deg_suffix=False):
        if share and deg_suffix:
            base = prefix + '_rp_deg'
            return (
                float(p.get(base, p.get(prefix + '_roll_deg', 0.0))),
                float(p.get(base, p.get(prefix + '_pitch_deg', 0.0))),
            )
        if share and not deg_suffix:
            base = prefix + '_rp'
            return (
                float(p.get(base, p.get(prefix + '_roll', 0.0))),
                float(p.get(base, p.get(prefix + '_pitch', 0.0))),
            )
        if deg_suffix:
            return (
                float(p.get(prefix + '_roll_deg', 0.0)),
                float(p.get(prefix + '_pitch_deg', 0.0)),
            )
        return (
            float(p.get(prefix + '_roll', 0.0)),
            float(p.get(prefix + '_pitch', 0.0)),
        )

    out = {
        'share_rp_gains': share,
        'use_acc_feedforward': bool(p.get('use_acc_feedforward', False)),
        'Kp_pos_xy': float(p['Kp_pos_xy']),
        'Kp_pos_z': float(p['Kp_pos_z']),
        'Kp_vel_xy': float(p['Kp_vel_xy']), 'Ki_vel_xy': float(p['Ki_vel_xy']),
        'Kd_vel_xy': float(p['Kd_vel_xy']),
        'Kp_vel_z': float(p['Kp_vel_z']), 'Ki_vel_z': float(p['Ki_vel_z']),
        'Kd_vel_z': float(p['Kd_vel_z']),
        'Kp_att_yaw_deg': float(p['Kp_att_yaw_deg']),
        'Kp_rate_yaw': float(p['Kp_rate_yaw']),
        'Ki_rate_yaw': float(p['Ki_rate_yaw']),
        'Kd_rate_yaw': float(p['Kd_rate_yaw']),
        'tilt_max': _tilt_max_rad_from_params(p),
        'gimbal_max': _gimbal_max_rad_from_params(p),
    }
    th_p_lim, th_r_lim = gimbal_pitch_roll_max_rad(p)
    out['gimbal_max_pitch'] = th_p_lim
    out['gimbal_max_roll'] = th_r_lim
    kpr, kpp = _pair('Kp_att', deg_suffix=True)
    out.update({
        'Kp_att_roll_deg': kpr, 'Kp_att_pitch_deg': kpp,
    })
    krr, krp = _pair('Kp_rate', deg_suffix=False)
    kirr, kirp = _pair('Ki_rate', deg_suffix=False)
    kdrr, kdrp = _pair('Kd_rate', deg_suffix=False)
    out.update({
        'Kp_rate_roll': krr, 'Kp_rate_pitch': krp,
        'Ki_rate_roll': kirr, 'Ki_rate_pitch': kirp,
        'Kd_rate_roll': kdrr, 'Kd_rate_pitch': kdrp,
    })
    return out


def default_px4_gui_params(share_rp: bool = True) -> Dict[str, Any]:
    specs = px4_param_specs(share_rp=share_rp)
    out = {}
    for spec in specs:
        key = spec['key']
        if spec.get('checkbox'):
            out[key] = bool(spec['default'])
        elif spec.get('integer'):
            out[key] = int(spec['default'])
        else:
            out[key] = float(spec['default'])
    return out
