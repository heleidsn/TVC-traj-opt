# -*- coding: utf-8 -*-
"""Actuator effects for numerical closed-loop simulation: lag and thrust quantization."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

CHANNEL_TH_P = 0
CHANNEL_TH_R = 1
CHANNEL_T = 2
CHANNEL_TAU_YAW = 3

TAU_MIN = 1e-3
TAU_MAX = 2.0
BW_MIN_HZ = 1.0 / (2.0 * math.pi * TAU_MAX)
BW_MAX_HZ = 1.0 / (2.0 * math.pi * TAU_MIN)

THRUST_RES_MIN = 0.01
THRUST_RES_MAX = 50.0


def tau_to_bandwidth_hz(tau: float) -> float:
    """-3 dB bandwidth [Hz] for first-order lag u_dot = (u_cmd - u) / tau."""
    return 1.0 / (2.0 * math.pi * max(float(tau), TAU_MIN))


def bandwidth_hz_to_tau(fc_hz: float) -> float:
    """Convert -3 dB bandwidth [Hz] to time constant [s]."""
    return 1.0 / (2.0 * math.pi * max(float(fc_hz), BW_MIN_HZ))


def quantize_thrust(value: float, resolution: float) -> float:
    """Round thrust [N] to nearest discrete step (e.g. 0.5 N or 10 N)."""
    res = float(resolution)
    val = float(value)
    if res <= 0.0 or not np.isfinite(val):
        return val
    return round(val / res) * res


def default_actuator_tracking_config(platform_id: str | None = None) -> Dict[str, Any]:
    res = default_thrust_resolution_for_platform(platform_id)
    return {
        'enabled': False,
        'thrust_quant_enabled': False,
        'thrust_resolution_N': res,
        'tau_gimbal': 0.05,
        'tau_thrust': 0.05,
        'tau_yaw_torque': 0.05,
    }


def default_thrust_resolution_for_platform(platform_id: str | None = None) -> float:
    """Typical thrust DAC / engine quantization step per platform."""
    pid = str(platform_id or 'proxy').strip().lower()
    if pid in ('real', 'flight'):
        return 10.0
    return 0.5


def actuator_config_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build actuator settings dict from tracking parameter map."""
    p = dict(params or {})
    if 'act_dyn_enable' in p:
        lag_on = bool(p.get('act_dyn_enable', False))
    else:
        lag_on = any(bool(p.get(k, False)) for k in (
            'act_dyn_gimbal', 'act_dyn_thrust', 'act_dyn_yaw',
        ))
    thrust_quant = bool(p.get('thrust_quant_enable', p.get('thrust_quant_enabled', False)))
    return {
        'act_dyn_enable': lag_on,
        'act_dyn_gimbal': lag_on,
        'act_dyn_thrust': lag_on,
        'act_dyn_yaw': lag_on,
        'thrust_quant_enable': thrust_quant,
        'thrust_quant_enabled': thrust_quant,
        'thrust_resolution_N': float(p.get('thrust_resolution_N', 0.5)),
        'tau_gimbal': float(p.get('tau_gimbal', 0.05)),
        'tau_thrust': float(p.get('tau_thrust', 0.05)),
        'tau_yaw_torque': float(p.get('tau_yaw_torque', 0.05)),
    }


class ActuatorDynamics:
    """
    Optional actuator effects on u = [th_p, th_r, T, tau_yaw]:

    - First-order lag per channel when lag enabled.
    - Thrust quantization to a fixed resolution [N] when enabled.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        cfg = actuator_config_from_params(config or {})
        lag_on = bool(cfg.get('act_dyn_enable', False))
        self.lag_enable = np.array([lag_on, lag_on, lag_on, lag_on], dtype=bool)
        self.thrust_quant_enable = bool(cfg.get('thrust_quant_enable', False))
        self.thrust_resolution = max(float(cfg.get('thrust_resolution_N', 0.0)), 0.0)
        self.tau = np.array([
            float(cfg.get('tau_gimbal', 0.05)),
            float(cfg.get('tau_gimbal', 0.05)),
            float(cfg.get('tau_thrust', 0.05)),
            float(cfg.get('tau_yaw_torque', 0.05)),
        ], dtype=float)
        self.u_act = np.zeros(4, dtype=float)

    def any_enabled(self) -> bool:
        return bool(np.any(self.lag_enable) or (
            self.thrust_quant_enable and self.thrust_resolution > 0.0
        ))

    def reset(self, u0=None):
        if u0 is None:
            self.u_act = np.zeros(4, dtype=float)
        else:
            self.u_act = np.asarray(u0, dtype=float).reshape(4).copy()
        if self.thrust_quant_enable and self.thrust_resolution > 0.0:
            self.u_act[CHANNEL_T] = quantize_thrust(self.u_act[CHANNEL_T], self.thrust_resolution)

    def step(self, u_cmd, dt):
        """Apply lag and/or thrust quantization; return plant input u_act."""
        u_cmd = np.asarray(u_cmd, dtype=float).reshape(4)
        dt = max(float(dt), 0.0)
        for i in range(4):
            if self.lag_enable[i]:
                tau = max(float(self.tau[i]), 1e-6)
                alpha = min(dt / tau, 1.0)
                self.u_act[i] += (u_cmd[i] - self.u_act[i]) * alpha
            else:
                self.u_act[i] = float(u_cmd[i])
        if self.thrust_quant_enable and self.thrust_resolution > 0.0:
            self.u_act[CHANNEL_T] = quantize_thrust(
                self.u_act[CHANNEL_T], self.thrust_resolution,
            )
        return self.u_act.copy()
