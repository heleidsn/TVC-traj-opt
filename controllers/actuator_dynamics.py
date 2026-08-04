# -*- coding: utf-8 -*-
"""Actuator effects for numerical closed-loop simulation: lag, mismatch, thrust quantization."""

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

SCALE_MIN = 0.1
SCALE_MAX = 3.0


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
        # Plant-side calibration mismatch: u_plant = scale * u_lag + bias
        'mismatch_enable': False,
        'scale_gimbal': 1.0,
        'bias_gimbal': 0.0,
        'scale_thrust': 1.0,
        'bias_thrust': 0.0,
        'scale_yaw_torque': 1.0,
        'bias_yaw_torque': 0.0,
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
    mismatch_on = bool(p.get('mismatch_enable', p.get('act_mismatch_enable', False)))
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
        'mismatch_enable': mismatch_on,
        'scale_gimbal': float(p.get('scale_gimbal', 1.0)),
        'bias_gimbal': float(p.get('bias_gimbal', 0.0)),
        'scale_thrust': float(p.get('scale_thrust', 1.0)),
        'bias_thrust': float(p.get('bias_thrust', 0.0)),
        'scale_yaw_torque': float(p.get('scale_yaw_torque', 1.0)),
        'bias_yaw_torque': float(p.get('bias_yaw_torque', 0.0)),
    }


class ActuatorDynamics:
    """
    Optional actuator effects on u = [th_p, th_r, T, tau_yaw]:

    Pipeline: lag → scale/bias mismatch → thrust quantization.

    Internal lag state ``u_lag`` tracks the command; plant input ``u_act`` is
    derived each step so scale/bias does not feed back into the lag.
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
        self.mismatch_enable = bool(cfg.get('mismatch_enable', False))
        sg = float(cfg.get('scale_gimbal', 1.0))
        bg = float(cfg.get('bias_gimbal', 0.0))
        self.scale = np.array([
            sg,
            sg,
            float(cfg.get('scale_thrust', 1.0)),
            float(cfg.get('scale_yaw_torque', 1.0)),
        ], dtype=float)
        self.bias = np.array([
            bg,
            bg,
            float(cfg.get('bias_thrust', 0.0)),
            float(cfg.get('bias_yaw_torque', 0.0)),
        ], dtype=float)
        self.u_lag = np.zeros(4, dtype=float)
        self.u_act = np.zeros(4, dtype=float)

    def any_enabled(self) -> bool:
        mismatch_active = self.mismatch_enable and (
            np.any(np.abs(self.scale - 1.0) > 1e-12) or np.any(np.abs(self.bias) > 1e-12)
        )
        return bool(
            np.any(self.lag_enable)
            or (self.thrust_quant_enable and self.thrust_resolution > 0.0)
            or mismatch_active
            or self.mismatch_enable
        )

    def _to_plant(self, u_lag: np.ndarray) -> np.ndarray:
        u = np.asarray(u_lag, dtype=float).reshape(4).copy()
        if self.mismatch_enable:
            u = self.scale * u + self.bias
        if self.thrust_quant_enable and self.thrust_resolution > 0.0:
            u[CHANNEL_T] = quantize_thrust(u[CHANNEL_T], self.thrust_resolution)
        return u

    def reset(self, u0=None):
        if u0 is None:
            self.u_lag = np.zeros(4, dtype=float)
        else:
            self.u_lag = np.asarray(u0, dtype=float).reshape(4).copy()
        self.u_act = self._to_plant(self.u_lag)

    def step(self, u_cmd, dt):
        """Apply lag, then optional scale/bias, then optional thrust quantization."""
        u_cmd = np.asarray(u_cmd, dtype=float).reshape(4)
        dt = max(float(dt), 0.0)
        for i in range(4):
            if self.lag_enable[i]:
                tau = max(float(self.tau[i]), 1e-6)
                alpha = min(dt / tau, 1.0)
                self.u_lag[i] += (u_cmd[i] - self.u_lag[i]) * alpha
            else:
                self.u_lag[i] = float(u_cmd[i])
        self.u_act = self._to_plant(self.u_lag)
        return self.u_act.copy()
