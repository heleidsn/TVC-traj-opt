# -*- coding: utf-8 -*-
"""Differential flatness utilities for the TVC rocket (center-of-oscillation output)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .linear_model import physics_from_gui, state12_to_state13


@dataclass
class FlatnessParams:
    """Physical parameters for flat-output reconstruction."""

    mass: float
    g: float
    ixx: float
    iyy: float
    izz: float
    lever_arm: float

    @classmethod
    def from_gui(cls, mass, ixx, iyy, izz, r_thrust_z, g=9.81):
        phy = physics_from_gui(mass, ixx, iyy, izz, r_thrust_z, g=g)
        return cls(
            mass=phy['MASS'],
            g=phy['G'],
            ixx=phy['I_XX'],
            iyy=phy['I_YY'],
            izz=phy['I_ZZ'],
            lever_arm=phy['DIST_COM_2_THRUST'],
        )

    @classmethod
    def from_phy_dict(cls, phy: Dict[str, float]):
        return cls(
            mass=float(phy['MASS']),
            g=float(phy['G']),
            ixx=float(phy['I_XX']),
            iyy=float(phy['I_YY']),
            izz=float(phy['I_ZZ']),
            lever_arm=float(phy['DIST_COM_2_THRUST']),
        )

    @property
    def omega_z_pitch(self) -> float:
        return float(np.sqrt(self.lever_arm * self.mass * self.g / self.iyy))

    @property
    def omega_z_roll(self) -> float:
        return float(np.sqrt(self.lever_arm * self.mass * self.g / self.ixx))

    @property
    def flat_offset_pitch(self) -> float:
        return self.iyy / (self.mass * self.lever_arm)

    @property
    def flat_offset_roll(self) -> float:
        return self.ixx / (self.mass * self.lever_arm)


def min_snap_1d(t, t0, t1, x0, x1):
    """
  9th-order rest-to-rest polynomial on [t0, t1] with zero 1st..4th derivatives at ends.

  Returns (value, d1, d2, d3, d4) at scalar or array time t.
  """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    if scalar:
        t = t.reshape(1)
    T = max(float(t1 - t0), 1e-6)
    s = np.clip((t - t0) / T, 0.0, 1.0)
    c = np.array([0, 0, 0, 0, 0, 126.0, -420.0, 540.0, -315.0, 70.0], dtype=float)
    dx = float(x1 - x0)

    def deriv(order, sval):
        val = np.zeros_like(sval, dtype=float)
        for k in range(len(c)):
            if k < order:
                continue
            fall = 1.0
            for j in range(order):
                fall *= (k - j)
            val += fall * c[k] * sval ** (k - order)
        return val

    p0 = deriv(0, s)
    p1 = deriv(1, s)
    p2 = deriv(2, s)
    p3 = deriv(3, s)
    p4 = deriv(4, s)
    xi = x0 + dx * p0
    dxi = dx * p1 / T
    ddxi = dx * p2 / T ** 2
    dddxi = dx * p3 / T ** 3
    ddddxi = dx * p4 / T ** 4
    if scalar:
        return float(xi[0]), float(dxi[0]), float(ddxi[0]), float(dddxi[0]), float(ddddxi[0])
    return xi, dxi, ddxi, dddxi, ddddxi


def lateral_flat_reconstruct(
    fp: FlatnessParams,
    xi, dxi, ddxi, dddxi, ddddxi,
    *,
    channel: str = 'pitch',
) -> Dict[str, float]:
    """
    Map flat lateral output and derivatives to COM state and gimbal angle.

    channel='pitch' affects world x (gimbal pitch / qy); 'roll' affects world y.
    """
    g = fp.g
    if channel == 'pitch':
        wz2 = fp.omega_z_pitch ** 2
        pos = xi - ddxi / wz2
        vel = dxi - dddxi / wz2
        q_comp = -ddxi / (2.0 * g)
        rate_comp = -dddxi / g
        delta = ddddxi / (g * wz2)
        return dict(pos=pos, vel=vel, q_comp=q_comp, rate_comp=rate_comp, delta=delta)
    wz2 = fp.omega_z_roll ** 2
    pos = xi - ddxi / wz2
    vel = dxi - dddxi / wz2
    q_comp = ddxi / (2.0 * g)
    rate_comp = dddxi / g
    delta = -ddddxi / (g * wz2)
    return dict(pos=pos, vel=vel, q_comp=q_comp, rate_comp=rate_comp, delta=delta)


def altitude_flat_reconstruct(fp: FlatnessParams, z, dz, ddz) -> Tuple[float, float, float]:
    """Minimum-phase altitude channel: thrust from vertical acceleration."""
    thrust = fp.mass * (fp.g + ddz)
    return float(z), float(dz), float(thrust)


def yaw_flat_reconstruct(fp: FlatnessParams, psi, dpsi, ddpsi) -> Tuple[float, float, float]:
    """Yaw is minimum-phase: body rate and torque from yaw angle profile."""
    return float(psi), float(dpsi), float(fp.izz * ddpsi)


def flat_state_control_at(
    fp: FlatnessParams,
    xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x,
    xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y,
    z, dz, ddz,
    psi, dpsi, ddpsi,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble 12-state and planner control [th_p, th_r, T, tau_yaw] from flat outputs.
    """
    fx = lateral_flat_reconstruct(
        fp, xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x, channel='pitch',
    )
    fy = lateral_flat_reconstruct(
        fp, xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y, channel='roll',
    )
    z_pos, z_vel, thrust = altitude_flat_reconstruct(fp, z, dz, ddz)
    yaw, yaw_rate, tau_yaw = yaw_flat_reconstruct(fp, psi, dpsi, ddpsi)

    qx = float(fy['q_comp'])
    qy = float(fx['q_comp'])
    qz = float(np.sin(yaw / 2.0))
    qw = float(np.sqrt(max(1.0 - qx * qx - qy * qy - qz * qz, 0.0)))

    x12 = np.array([
        fx['pos'], fy['pos'], z_pos,
        fx['vel'], fy['vel'], z_vel,
        qx, qy, qz,
        float(fy['rate_comp']), float(fx['rate_comp']), yaw_rate,
    ], dtype=float)

    th_p = float(fx['delta'])
    th_r = float(fy['delta'])
    u_opt = np.array([th_p, th_r, thrust, tau_yaw], dtype=float)
    return x12, u_opt


def state13_from_flat(
    fp: FlatnessParams,
    xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x,
    xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y,
    z, dz, ddz,
    psi, dpsi, ddpsi,
) -> Tuple[np.ndarray, np.ndarray]:
    """13-state planner format + control."""
    x12, u_opt = flat_state_control_at(
        fp,
        xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x,
        xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y,
        z, dz, ddz,
        psi, dpsi, ddpsi,
    )
    x13 = state12_to_state13(x12)
    return x13, u_opt


def method1_from_flat(
    fp: FlatnessParams,
    xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x,
    xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y,
    z, dz, ddz,
    psi, dpsi, ddpsi,
) -> Tuple[np.ndarray, np.ndarray]:
    """17-dim Method-1 state for GUI plotting + control."""
    x13, u_opt = state13_from_flat(
        fp,
        xi_x, dxi_x, ddxi_x, dddxi_x, ddddxi_x,
        xi_y, dxi_y, ddxi_y, dddxi_y, ddddxi_y,
        z, dz, ddz,
        psi, dpsi, ddpsi,
    )
    x17 = np.zeros(17, dtype=float)
    x17[0:13] = x13
    x17[13:17] = u_opt
    return x17, u_opt


def estimate_flat_outputs_from_state12(
    fp: FlatnessParams,
    x12: np.ndarray,
) -> Tuple[float, float]:
    """Inverse map: COM position + attitude -> flat lateral outputs (small-angle)."""
    x12 = np.asarray(x12, dtype=float).reshape(12)
    pitch = 2.0 * x12[7]
    roll = 2.0 * x12[6]
    xi_x = x12[0] - fp.flat_offset_pitch * pitch
    xi_y = x12[1] - fp.flat_offset_roll * roll
    return float(xi_x), float(xi_y)


def estimate_flat_rates_from_state12(
    fp: FlatnessParams,
    x12: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Inverse map: COM velocity + body rates -> flat-output rates (small-angle)."""
    x12 = np.asarray(x12, dtype=float).reshape(12)
    p, q_rate, r = float(x12[9]), float(x12[10]), float(x12[11])
    dxi_x = float(x12[3]) - fp.flat_offset_pitch * 2.0 * q_rate
    dxi_y = float(x12[4]) - fp.flat_offset_roll * 2.0 * p
    dz = float(x12[5])
    dpsi = r
    return dxi_x, dxi_y, dz, dpsi


def estimate_flat_state_from_state12(
    fp: FlatnessParams,
    x12: np.ndarray,
) -> Dict[str, float]:
    """Measured flat outputs (ξ_x, ξ_y, z, ψ) and their 1st derivatives."""
    x12 = np.asarray(x12, dtype=float).reshape(12)
    xi_x, xi_y = estimate_flat_outputs_from_state12(fp, x12)
    dxi_x, dxi_y, dz, dpsi = estimate_flat_rates_from_state12(fp, x12)
    psi = float(2.0 * np.arcsin(np.clip(x12[8], -1.0, 1.0)))
    return {
        'xi_x': xi_x, 'xi_y': xi_y, 'z': float(x12[2]), 'psi': psi,
        'dxi_x': dxi_x, 'dxi_y': dxi_y, 'dz': dz, 'dpsi': dpsi,
    }


def clip_controls(
    u_opt: np.ndarray,
    th_p_max_deg: float,
    th_r_max_deg: float,
    t_min: float,
    t_max: float,
    tau_max: float,
) -> np.ndarray:
    u = np.asarray(u_opt, dtype=float).reshape(4).copy()
    th_max = np.radians(max(th_p_max_deg, th_r_max_deg))
    u[0] = np.clip(u[0], -th_max, th_max)
    u[1] = np.clip(u[1], -th_max, th_max)
    u[2] = np.clip(u[2], float(t_min), float(t_max))
    u[3] = np.clip(u[3], -float(tau_max), float(tau_max))
    return u
