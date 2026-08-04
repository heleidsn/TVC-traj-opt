# -*- coding: utf-8 -*-
"""
Continuous SISO open-loop Bode / margins for PX4 cascade inner loops.

Hover linearization + cascade rate/attitude loops, with optional first-order
actuator lag.  Intended for comparing phase margin with vs without actuator
dynamics after controller gains are set in the GUI.

Assumptions (shown in the plot tab notes):
* continuous time (ignores control_dt sampling / ZOH)
* hover small-angle linear plant (same ω_z² as linear_model / PX4 allocation)
* unity-feedback SISO; no saturation / thrust quantization
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from .actuator_dynamics import actuator_config_from_params
from .linear_model import physics_from_gui
from .px4_params import normalize_px4_params

LOOP_RATE = 'rate'
LOOP_ATTITUDE = 'attitude'
LOOP_VELOCITY = 'velocity'
LOOP_POSITION = 'position'
LOOP_IDS = (LOOP_RATE, LOOP_ATTITUDE, LOOP_VELOCITY, LOOP_POSITION)

AXIS_PITCH = 'pitch'
AXIS_ROLL = 'roll'
AXIS_YAW = 'yaw'
AXIS_IDS = (AXIS_PITCH, AXIS_ROLL, AXIS_YAW)


def _finite_or_nan(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(v):
        return float('nan')
    return v


def compute_margins_from_bode(
    w: np.ndarray,
    mag_db: np.ndarray,
    phase_deg: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate gain/phase margins from Bode samples.

    Phase margin: at gain crossover (|L|≈0 dB), PM = 180° + ∠L.
    Gain margin: at phase crossover (∠L≈−180°), GM = −|L|_dB.
    """
    w = np.asarray(w, dtype=float)
    mag_db = np.asarray(mag_db, dtype=float)
    phase_deg = np.asarray(phase_deg, dtype=float)
    out = {
        'pm_deg': float('nan'),
        'gm_db': float('nan'),
        'w_gc_rad_s': float('nan'),
        'w_pc_rad_s': float('nan'),
    }
    if w.size < 3:
        return out

    # Unwrap phase for consistent −180 crossings, then wrap for PM display.
    phase_unwrapped = np.unwrap(np.deg2rad(phase_deg))
    phase_deg_u = np.rad2deg(phase_unwrapped)

    # Gain crossover: mag_db crosses 0 from above (typical) or either side.
    mag_sign = np.sign(mag_db)
    for i in range(len(w) - 1):
        if mag_sign[i] == 0:
            w_gc = w[i]
            ph = phase_deg_u[i]
            out['w_gc_rad_s'] = float(w_gc)
            out['pm_deg'] = float(180.0 + ph)
            break
        if mag_sign[i] * mag_sign[i + 1] < 0:
            # Linear interpolate in log-ω / dB.
            t = mag_db[i] / (mag_db[i] - mag_db[i + 1])
            log_w = np.log(w[i]) + t * (np.log(w[i + 1]) - np.log(w[i]))
            w_gc = float(np.exp(log_w))
            ph = phase_deg_u[i] + t * (phase_deg_u[i + 1] - phase_deg_u[i])
            out['w_gc_rad_s'] = w_gc
            out['pm_deg'] = float(180.0 + ph)
            break

    # Phase crossover: phase crosses −180° (or −180 − 360k).
    target = -180.0
    # Shift so we look for crossing of nearest odd multiple of −180 near data.
    for i in range(len(w) - 1):
        p0, p1 = phase_deg_u[i], phase_deg_u[i + 1]
        # Cross any …, -540, -180, +180, … odd multiples of 180 below 0
        for k in range(-4, 3):
            thr = target + 360.0 * k
            if (p0 - thr) * (p1 - thr) <= 0 and abs(p1 - p0) > 1e-12:
                t = (thr - p0) / (p1 - p0)
                log_w = np.log(w[i]) + t * (np.log(w[i + 1]) - np.log(w[i]))
                w_pc = float(np.exp(log_w))
                mag = mag_db[i] + t * (mag_db[i + 1] - mag_db[i])
                out['w_pc_rad_s'] = w_pc
                out['gm_db'] = float(-mag)
                return out
    return out


def _rate_gains(gains: Dict[str, Any], axis: str) -> Tuple[float, float, float]:
    g = gains
    if axis == AXIS_ROLL:
        return (
            float(g['Kp_rate_roll']),
            float(g['Ki_rate_roll']),
            float(g['Kd_rate_roll']),
        )
    if axis == AXIS_YAW:
        return (
            float(g['Kp_rate_yaw']),
            float(g['Ki_rate_yaw']),
            float(g['Kd_rate_yaw']),
        )
    return (
        float(g['Kp_rate_pitch']),
        float(g['Ki_rate_pitch']),
        float(g['Kd_rate_pitch']),
    )


def _att_gain_si(gains: Dict[str, Any], axis: str) -> float:
    """
    Attitude P from rad error → rad/s setpoint.

    Controller uses: rate_sp = Kp_att_deg * degrees(err) * π/180 = Kp_att_deg * err_rad.
    """
    g = gains
    if axis == AXIS_ROLL:
        return float(g['Kp_att_roll_deg'])
    if axis == AXIS_YAW:
        return float(g['Kp_att_yaw_deg'])
    return float(g['Kp_att_pitch_deg'])


def _vel_gains(gains: Dict[str, Any], axis: str) -> Tuple[float, float, float]:
    """Velocity PID. Pitch/roll → XY; yaw axis uses Z (vertical thrust channel)."""
    g = gains
    if axis == AXIS_YAW:
        return (
            float(g['Kp_vel_z']),
            float(g['Ki_vel_z']),
            float(g['Kd_vel_z']),
        )
    return (
        float(g['Kp_vel_xy']),
        float(g['Ki_vel_xy']),
        float(g['Kd_vel_xy']),
    )


def _pos_gain(gains: Dict[str, Any], axis: str) -> float:
    g = gains
    if axis == AXIS_YAW:
        return float(g['Kp_pos_z'])
    return float(g['Kp_pos_xy'])


def _actuator_tau_for_axis(act_cfg: Dict[str, Any], axis: str, loop: str = LOOP_RATE) -> float:
    """
    Gimbal τ for pitch/roll attitude chain; yaw-torque τ for yaw rate/att;
    thrust τ for vertical (yaw-axis) velocity/position.
    """
    if loop in (LOOP_VELOCITY, LOOP_POSITION) and axis == AXIS_YAW:
        return float(act_cfg.get('tau_thrust', 0.0) or 0.0)
    if axis == AXIS_YAW:
        return float(act_cfg.get('tau_yaw_torque', 0.0) or 0.0)
    return float(act_cfg.get('tau_gimbal', 0.0) or 0.0)


def _pid_tf(kp: float, ki: float, kd: float) -> signal.TransferFunction:
    """C(s) = Kp + Ki/s + Kd s = (Kd s^2 + Kp s + Ki) / s."""
    return _make_tf([float(kd), float(kp), float(ki)], [1.0, 0.0])


def _integrator_tf() -> signal.TransferFunction:
    return _make_tf([1.0], [1.0, 0.0])


def _lag_tf(tau: float) -> signal.TransferFunction:
    tau = float(tau)
    if tau <= 0.0:
        return _make_tf([1.0], [1.0])
    return _make_tf([1.0], [tau, 1.0])


def _tf_num_den(sys: signal.TransferFunction) -> Tuple[np.ndarray, np.ndarray]:
    num = np.atleast_1d(np.asarray(sys.num, dtype=float)).ravel()
    den = np.atleast_1d(np.asarray(sys.den, dtype=float)).ravel()
    return num, den


def _pad_poly(a: np.ndarray, n: int) -> np.ndarray:
    a = np.atleast_1d(np.asarray(a, dtype=float)).ravel()
    if len(a) >= n:
        return a.copy()
    out = np.zeros(n, dtype=float)
    out[-len(a):] = a
    return out


def _trim_tf(num: np.ndarray, den: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop leading zeros and scale so den leading coeff is 1."""
    num = np.atleast_1d(np.asarray(num, dtype=float)).ravel()
    den = np.atleast_1d(np.asarray(den, dtype=float)).ravel()
    # Strip leading zeros from both
    while len(num) > 1 and abs(num[0]) < 1e-14:
        num = num[1:]
    while len(den) > 1 and abs(den[0]) < 1e-14:
        den = den[1:]
    if abs(den[0]) > 1e-14:
        scale = den[0]
        num = num / scale
        den = den / scale
    return num, den


def _make_tf(num, den) -> signal.TransferFunction:
    num, den = _trim_tf(num, den)
    return signal.TransferFunction(num, den)


def tf_series(sys_a: signal.TransferFunction, sys_b: signal.TransferFunction) -> signal.TransferFunction:
    """Series connection G_a(s) * G_b(s)."""
    na, da = _tf_num_den(sys_a)
    nb, db = _tf_num_den(sys_b)
    return _make_tf(np.convolve(na, nb), np.convolve(da, db))


def tf_unity_feedback(sys_ol: signal.TransferFunction) -> signal.TransferFunction:
    """Closed-loop T = L / (1 + L) for unity negative feedback."""
    num, den = _tf_num_den(sys_ol)
    n = max(len(num), len(den))
    num_p = _pad_poly(num, n)
    den_p = _pad_poly(den, n)
    return _make_tf(num_p, den_p + num_p)


def rate_open_loop_tf(
    kp: float, ki: float, kd: float, tau: float = 0.0,
) -> signal.TransferFunction:
    """
    Rate-loop open-loop L(s) after plant/allocation cancellation:

        L(s) = (Kd s^2 + Kp s + Ki) / s^2            (no actuator)
        L(s) = (Kd s^2 + Kp s + Ki) / (s^2 (τ s+1))  (with lag)
    """
    num = [float(kd), float(kp), float(ki)]
    if tau is None or float(tau) <= 0.0:
        den = [1.0, 0.0, 0.0]
    else:
        tau = float(tau)
        den = [tau, 1.0, 0.0, 0.0]
    return _make_tf(num, den)


def attitude_open_loop_tf(
    kp_att: float,
    kp_rate: float,
    ki_rate: float,
    kd_rate: float,
    tau: float = 0.0,
) -> signal.TransferFunction:
    """
    Attitude open-loop with nested rate closed loop:

        L_rate = rate_open_loop_tf(...)
        T_rate = L_rate / (1 + L_rate)
        L_att  = (Kp_att / s) * T_rate
    """
    L_rate = rate_open_loop_tf(kp_rate, ki_rate, kd_rate, tau=tau)
    T_rate = tf_unity_feedback(L_rate)
    C_att = signal.TransferFunction([float(kp_att)], [1.0, 0.0])
    return tf_series(C_att, T_rate)


def attitude_closed_loop_tf(
    kp_att: float,
    kp_rate: float,
    ki_rate: float,
    kd_rate: float,
    tau: float = 0.0,
) -> signal.TransferFunction:
    return tf_unity_feedback(
        attitude_open_loop_tf(kp_att, kp_rate, ki_rate, kd_rate, tau=tau)
    )


def velocity_open_loop_tf(
    kp_vel: float,
    ki_vel: float,
    kd_vel: float,
    kp_att: float,
    kp_rate: float,
    ki_rate: float,
    kd_rate: float,
    tau: float = 0.0,
    horizontal: bool = True,
) -> signal.TransferFunction:
    """
    Velocity open-loop.

    Horizontal (pitch→vx / roll→vy):
        acc_sp --(1/g)--> tilt_sp --T_att--> tilt --(g)--> a --> 1/s --> v
        ⇒ L_vel = C_vel * T_att / s   (g cancels; actuator is inside T_att)

    Vertical (z / thrust):
        L_vel = C_vel * 1/(τs+1) / s
    """
    C_vel = _pid_tf(kp_vel, ki_vel, kd_vel)
    if horizontal:
        T_att = attitude_closed_loop_tf(kp_att, kp_rate, ki_rate, kd_rate, tau=tau)
        return tf_series(tf_series(C_vel, T_att), _integrator_tf())
    plant = tf_series(_lag_tf(tau), _integrator_tf())
    return tf_series(C_vel, plant)


def position_open_loop_tf(
    kp_pos: float,
    kp_vel: float,
    ki_vel: float,
    kd_vel: float,
    kp_att: float,
    kp_rate: float,
    ki_rate: float,
    kd_rate: float,
    tau: float = 0.0,
    horizontal: bool = True,
) -> signal.TransferFunction:
    """
    Position open-loop with nested velocity closed loop:

        L_pos = Kp_pos * T_vel / s
    """
    L_vel = velocity_open_loop_tf(
        kp_vel, ki_vel, kd_vel, kp_att, kp_rate, ki_rate, kd_rate,
        tau=tau, horizontal=horizontal,
    )
    T_vel = tf_unity_feedback(L_vel)
    C_pos = _make_tf([float(kp_pos)], [1.0])
    return tf_series(tf_series(C_pos, T_vel), _integrator_tf())


def bode_response(
    sys: signal.TransferFunction,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if w is None:
        w = np.logspace(-2, 3, 700)
    w_out, mag_db, phase_deg = signal.bode(sys, w=w)
    return np.asarray(w_out), np.asarray(mag_db), np.asarray(phase_deg)


def analyze_loop(
    phy_gui: Dict[str, Any],
    controller_params: Dict[str, Any],
    loop: str = LOOP_RATE,
    axis: str = AXIS_PITCH,
    w: Optional[np.ndarray] = None,
    force_actuator: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Compute Bode + margins with and without actuator lag for one cascade loop.

    Axis mapping for velocity/position:
      * pitch → horizontal X (via pitch attitude)
      * roll  → horizontal Y (via roll attitude)
      * yaw   → vertical Z (thrust channel; no tilt cascade)
    """
    phy = physics_from_gui(
        phy_gui['mass'], phy_gui['Ixx'], phy_gui['Iyy'], phy_gui['Izz'],
        phy_gui['r_thrust_z'], g=float(phy_gui.get('g', 9.81)),
    )
    gains = normalize_px4_params(controller_params)
    act_cfg = actuator_config_from_params(controller_params)
    enabled = bool(act_cfg.get('act_dyn_enable', False))
    if force_actuator is not None:
        enabled = bool(force_actuator)

    loop = str(loop or LOOP_RATE).lower()
    if loop not in LOOP_IDS:
        loop = LOOP_RATE
    axis = str(axis or AXIS_PITCH).lower()
    if axis not in AXIS_IDS:
        axis = AXIS_PITCH

    tau_gui = _actuator_tau_for_axis(act_cfg, axis, loop=loop)
    tau_plot = float(tau_gui) if (enabled and tau_gui > 0.0) else 0.0

    kp_r, ki_r, kd_r = _rate_gains(gains, axis)
    kp_att = _att_gain_si(gains, axis)
    kp_v, ki_v, kd_v = _vel_gains(gains, axis)
    kp_pos = _pos_gain(gains, axis)
    horizontal = axis != AXIS_YAW

    def _make(tau: float):
        if loop == LOOP_POSITION:
            return position_open_loop_tf(
                kp_pos, kp_v, ki_v, kd_v, kp_att, kp_r, ki_r, kd_r,
                tau=tau, horizontal=horizontal,
            )
        if loop == LOOP_VELOCITY:
            return velocity_open_loop_tf(
                kp_v, ki_v, kd_v, kp_att, kp_r, ki_r, kd_r,
                tau=tau, horizontal=horizontal,
            )
        if loop == LOOP_ATTITUDE:
            return attitude_open_loop_tf(kp_att, kp_r, ki_r, kd_r, tau=tau)
        return rate_open_loop_tf(kp_r, ki_r, kd_r, tau=tau)

    if w is None:
        # Outer loops need lower frequencies.
        w_lo = -2.0 if loop in (LOOP_VELOCITY, LOOP_POSITION) else -1.0
        w = np.logspace(w_lo, 3, 700)

    sys_wo = _make(0.0)
    w0, mag0, ph0 = bode_response(sys_wo, w)
    m0 = compute_margins_from_bode(w0, mag0, ph0)

    sys_w = _make(tau_plot) if tau_plot > 0.0 else sys_wo
    w1, mag1, ph1 = bode_response(sys_w, w)
    m1 = compute_margins_from_bode(w1, mag1, ph1)

    wz2 = phy['DIST_COM_2_THRUST'] * phy['MASS'] * phy['G'] / (
        phy['I_YY'] if axis == AXIS_PITCH else (
            phy['I_XX'] if axis == AXIS_ROLL else phy['I_ZZ']
        )
    )

    channel = {
        AXIS_PITCH: 'X (via pitch)',
        AXIS_ROLL: 'Y (via roll)',
        AXIS_YAW: 'Z (thrust)' if loop in (LOOP_VELOCITY, LOOP_POSITION) else 'yaw',
    }.get(axis, axis)

    notes = (
        'Continuous cascade SISO; ignores sampling/saturation/quantization. '
    )
    if loop == LOOP_RATE:
        notes += 'Rate open-loop cancels ω_z² (depends on PID + τ).'
    elif loop == LOOP_ATTITUDE:
        notes += 'Attitude nests rate closed loop.'
    elif loop == LOOP_VELOCITY:
        if horizontal:
            notes += f'Velocity nests attitude closed loop; channel {channel}.'
        else:
            notes += 'Vertical velocity: C_vel·1/(τ_thrust s+1)/s.'
    else:
        if horizontal:
            notes += f'Position nests velocity closed loop; channel {channel}.'
        else:
            notes += 'Vertical position nests vertical velocity closed loop.'

    return {
        'loop': loop,
        'axis': axis,
        'channel': channel,
        'omega_z2': float(wz2) if axis != AXIS_YAW else float('nan'),
        'gains': {
            'Kp_rate': kp_r, 'Ki_rate': ki_r, 'Kd_rate': kd_r,
            'Kp_att': kp_att,
            'Kp_vel': kp_v, 'Ki_vel': ki_v, 'Kd_vel': kd_v,
            'Kp_pos': kp_pos,
        },
        'actuator_enabled': enabled,
        'tau_s': float(tau_gui),
        'tau_used_with_s': float(tau_plot),
        'w': w0,
        'without': {
            'mag_db': mag0, 'phase_deg': ph0, **m0,
        },
        'with_actuator': {
            'mag_db': mag1, 'phase_deg': ph1, **m1,
        },
        'notes': notes,
    }


def analyze_all_loops(
    phy_gui: Dict[str, Any],
    controller_params: Dict[str, Any],
    axis: str = AXIS_PITCH,
    w: Optional[np.ndarray] = None,
    force_actuator: Optional[bool] = None,
) -> Dict[str, Any]:
    """Analyze Rate / Attitude / Velocity / Position for one axis."""
    if w is None:
        w = np.logspace(-2.0, 3.0, 700)
    by_loop: Dict[str, Any] = {}
    for loop in LOOP_IDS:
        by_loop[loop] = analyze_loop(
            phy_gui, controller_params,
            loop=loop, axis=axis, w=w, force_actuator=force_actuator,
        )
    first = by_loop[LOOP_RATE]
    axis = first['axis']
    if axis == AXIS_YAW:
        channel = 'yaw (rate/att) + Z thrust (vel/pos)'
    else:
        channel = first.get('channel', '')
    return {
        'axis': axis,
        'channel': channel,
        'gains': first.get('gains') or {},
        'actuator_enabled': first.get('actuator_enabled', False),
        'by_loop': by_loop,
        'notes': (
            'Solid = no lag · Dashed = with actuator. '
            'Shared frequency grid; outer loops use lower ω_gc.'
        ),
    }


def format_margins_summary(result: Dict[str, Any]) -> str:
    """Short monospace text for the margins info panel (single loop)."""
    if not result:
        return 'No analysis yet.'
    if isinstance(result.get('by_loop'), dict):
        return format_margins_summary_all(result)
    wo = result['without']
    wa = result['with_actuator']
    g = result.get('gains') or {}
    tau = result.get('tau_s', 0.0)
    en = result.get('actuator_enabled', False)
    lines = [
        f"Loop : {result['loop']} / {result['axis']}  [{result.get('channel', '')}]",
        f"Rate : Kp={g.get('Kp_rate', 0):.3g}  Ki={g.get('Ki_rate', 0):.3g}  "
        f"Kd={g.get('Kd_rate', 0):.3g}",
        f"Att  : Kp={g.get('Kp_att', 0):.3g}",
        f"Vel  : Kp={g.get('Kp_vel', 0):.3g}  Ki={g.get('Ki_vel', 0):.3g}  "
        f"Kd={g.get('Kd_vel', 0):.3g}",
        f"Pos  : Kp={g.get('Kp_pos', 0):.3g}",
        f"τ_act: {tau:.4f} s  |  lag {'ON' if en else 'OFF'}  |  "
        f"dashed uses τ={result.get('tau_used_with_s', 0):.4f} s",
        '',
        '── Without actuator ──',
        f"PM   : {_fmt_pm(wo.get('pm_deg'))}",
        f"ω_gc : {_fmt_w(wo.get('w_gc_rad_s'))}",
        f"GM   : {_fmt_gm(wo.get('gm_db'))}",
        '',
        '── With actuator lag ──',
        f"PM   : {_fmt_pm(wa.get('pm_deg'))}",
        f"ω_gc : {_fmt_w(wa.get('w_gc_rad_s'))}",
        f"GM   : {_fmt_gm(wa.get('gm_db'))}",
        '',
        'ΔPM  : '
        + (
            f"{wa['pm_deg'] - wo['pm_deg']:+.1f} deg"
            if np.isfinite(wa.get('pm_deg', np.nan)) and np.isfinite(wo.get('pm_deg', np.nan))
            else 'n/a'
        ),
        '',
        result.get('notes', ''),
    ]
    return '\n'.join(lines)


def format_margins_summary_all(
    bundle: Dict[str, Any],
    visible_loops: Optional[Dict[str, bool]] = None,
) -> str:
    """Compact summary for cascade loops on one axis (optionally filtered)."""
    if not bundle or not isinstance(bundle.get('by_loop'), dict):
        return 'No analysis yet.'
    visible = dict(visible_loops or {})
    for loop in LOOP_IDS:
        visible.setdefault(loop, True)
    g = bundle.get('gains') or {}
    en = bundle.get('actuator_enabled', False)
    lines = [
        f"Axis : {bundle.get('axis', '')}  [{bundle.get('channel', '')}]",
        f"Rate : Kp={g.get('Kp_rate', 0):.3g}  Ki={g.get('Ki_rate', 0):.3g}  "
        f"Kd={g.get('Kd_rate', 0):.3g}",
        f"Att  : Kp={g.get('Kp_att', 0):.3g}",
        f"Vel  : Kp={g.get('Kp_vel', 0):.3g}  Ki={g.get('Ki_vel', 0):.3g}  "
        f"Kd={g.get('Kd_vel', 0):.3g}",
        f"Pos  : Kp={g.get('Kp_pos', 0):.3g}",
        f"Act  : lag {'ON' if en else 'OFF'}",
        '',
        'Loop       τ[s]   PM₀     PM_τ    ΔPM',
        '─────────────────────────────────────',
    ]
    labels = {
        LOOP_RATE: 'Rate',
        LOOP_ATTITUDE: 'Att',
        LOOP_VELOCITY: 'Vel',
        LOOP_POSITION: 'Pos',
    }
    any_shown = False
    for loop in LOOP_IDS:
        if not visible.get(loop, True):
            continue
        any_shown = True
        r = bundle['by_loop'].get(loop) or {}
        wo = r.get('without') or {}
        wa = r.get('with_actuator') or {}
        pm0 = wo.get('pm_deg', float('nan'))
        pm1 = wa.get('pm_deg', float('nan'))
        dpm = (
            f"{pm1 - pm0:+.1f}"
            if np.isfinite(pm0) and np.isfinite(pm1) else 'n/a'
        )
        lines.append(
            f"{labels.get(loop, loop):<10}"
            f"{float(r.get('tau_used_with_s', 0.0)):.3f}  "
            f"{_fmt_pm(pm0):<8} "
            f"{_fmt_pm(pm1):<8} "
            f"{dpm}"
        )
        lines.append(
            f"  ω_gc₀={_fmt_w(wo.get('w_gc_rad_s'))}"
        )
        lines.append(
            f"  ω_gcτ={_fmt_w(wa.get('w_gc_rad_s'))}"
        )
    if not any_shown:
        lines.append('(no loops selected)')
    lines.extend(['', bundle.get('notes', '')])
    return '\n'.join(lines)


def _fmt_pm(v) -> str:
    v = _finite_or_nan(v)
    return 'n/a' if not np.isfinite(v) else f'{v:.1f} deg'


def _fmt_gm(v) -> str:
    v = _finite_or_nan(v)
    return 'n/a' if not np.isfinite(v) else f'{v:.1f} dB'


def _fmt_w(v) -> str:
    v = _finite_or_nan(v)
    if not np.isfinite(v) or v <= 0:
        return 'n/a'
    return f'{v:.2f} rad/s ({v / (2 * np.pi):.2f} Hz)'


def format_loop_margin_tag(loop_result: Optional[Dict[str, Any]]) -> str:
    """
    Compact one-line tag for States panel titles:
    'PM 58→55°  f_gc 0.19→0.18 Hz'
    """
    if not loop_result:
        return ''
    wo = loop_result.get('without') or {}
    wa = loop_result.get('with_actuator') or {}
    pm0 = _finite_or_nan(wo.get('pm_deg'))
    pm1 = _finite_or_nan(wa.get('pm_deg'))
    w0 = _finite_or_nan(wo.get('w_gc_rad_s'))
    w1 = _finite_or_nan(wa.get('w_gc_rad_s'))

    if np.isfinite(pm0) and np.isfinite(pm1):
        pm_txt = f'PM {pm0:.0f}→{pm1:.0f}°'
    elif np.isfinite(pm0):
        pm_txt = f'PM {pm0:.0f}°'
    else:
        pm_txt = 'PM n/a'

    def _f(w):
        return w / (2.0 * np.pi) if np.isfinite(w) and w > 0 else float('nan')

    f0, f1 = _f(w0), _f(w1)
    if np.isfinite(f0) and np.isfinite(f1):
        f_txt = f'f_gc {f0:.2f}→{f1:.2f} Hz'
    elif np.isfinite(f0):
        f_txt = f'f_gc {f0:.2f} Hz'
    else:
        f_txt = 'f_gc n/a'
    return f'{pm_txt}  {f_txt}'
