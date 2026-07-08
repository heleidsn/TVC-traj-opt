# -*- coding: utf-8 -*-
"""Closed-loop numerical tracking simulation (nonlinear TVC dynamics model)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .actuator_dynamics import ActuatorDynamics, actuator_config_from_params
from .linear_model import lqr_to_control_opt, state12_to_state13, state13_to_state12
from .lqr_tracker import LQRTracker
from .mpc_tracker import MPCTracker
from .nonlinear_plant import plant_from_phy_gui, tracker_phy_from_gui
from .params import (
    CONTROLLER_ACADOS_NMPC,
    CONTROLLER_FLATNESS,
    CONTROLLER_LQR,
    CONTROLLER_MPC,
    CONTROLLER_PX4,
    TRACKING_CLIP_POS_ERROR_M,
    TRACKING_CLIP_VEL_ERROR_M_S,
    TRACKING_GIMBAL_RATE_LIMIT_DEG_S,
    TRACKING_MANEUVER_GIMBAL_R_EXPONENT,
    TRACKING_TERMINAL_CLIP_POS_ERROR_M,
    TRACKING_TERMINAL_CLIP_VEL_ERROR_M_S,
    TRACKING_TERMINAL_HOLD_DURATION_S,
    TRACKING_TERMINAL_HOLD_GIMBAL_R_EXPONENT,
    TRACKING_USE_FEEDFORWARD,
    scale_tracking_gimbal_r,
)
from .px4_cascade import PX4CascadeTracker
from .px4_params import (
    clip_control_opt,
    clip_lqr_gimbal_cmd,
    gimbal_pitch_roll_max_rad,
)
from .trajectory_ref import TrajectoryReference


def numerical_sim_end_time(ref, params: Dict[str, Any]) -> float:
    """Simulation horizon [s] from trajectory start (t=0 in the loop)."""
    plan_span = max(float(ref.plan_end_time() - ref.t[0]), 0.0)
    hold = float(
        params.get('terminal_hold_duration_s', TRACKING_TERMINAL_HOLD_DURATION_S),
    )
    total_override = float(params.get('total_duration_s', 0.0))
    if total_override > 0.0:
        return max(total_override, 0.1)
    return max(plan_span + max(hold, 0.0), 0.1)


def numerical_sim_timing(params: Dict[str, Any]) -> tuple[float, float, int]:
    """
    Parse plant step (sim_dt) and controller period (control_dt).

    Returns (sim_dt, control_dt, steps_per_control) with control_dt = ratio * sim_dt.
    """
    sim_dt = max(float(params.get('sim_dt', 0.005)), 1e-4)
    control_dt = max(float(params.get('control_dt', 0.02)), sim_dt)
    ratio = max(1, int(round(control_dt / sim_dt)))
    control_dt = ratio * sim_dt
    return sim_dt, control_dt, ratio


def _make_tracker(
    controller_id, phy, params, phy_gui=None, flat_outputs=None, flatness_physics=None,
):
    if controller_id == CONTROLLER_PX4:
        return PX4CascadeTracker(phy, params)
    if controller_id == CONTROLLER_LQR:
        return LQRTracker(phy, params)
    if controller_id == CONTROLLER_FLATNESS:
        from .flatness_tracker import FlatnessCascadeTracker
        mass = float(phy_gui.get('mass', phy['MASS'])) if phy_gui else float(phy['MASS'])
        g = float(phy_gui.get('g', phy['G'])) if phy_gui else float(phy['G'])
        return FlatnessCascadeTracker(
            phy, params, flat_outputs=flat_outputs, flatness_physics=flatness_physics,
            mass=mass, g=g,
        )
    if controller_id == CONTROLLER_MPC:
        return MPCTracker(phy, params)
    if controller_id == CONTROLLER_ACADOS_NMPC:
        from .acados_nmpc_tracker import AcadosNmpcTracker, acados_nmpc_available
        if not acados_nmpc_available():
            raise ImportError(
                "Acados NMPC requires CasADi and a built acados installation."
            )
        if phy_gui is None:
            raise ValueError("phy_gui is required for Acados NMPC tracker")
        return AcadosNmpcTracker(phy_gui, params)
    raise ValueError(f'Unknown controller: {controller_id}')


def _control_limits_from_params(params):
    th_p_max, th_r_max = gimbal_pitch_roll_max_rad(params)
    T_min = float(params['T_min']) if 'T_min' in params else None
    T_max = float(params['T_max']) if 'T_max' in params else None
    tau_max = float(params['tau_yaw_max']) if 'tau_yaw_max' in params else None
    return th_p_max, th_r_max, T_min, T_max, tau_max


def _clip_plant_control(u_opt, params):
    th_p_max, th_r_max, T_min, T_max, tau_max = _control_limits_from_params(params)
    return clip_control_opt(u_opt, th_p_max, th_r_max, T_min, T_max, tau_max)


def _init_actuator(actuator, mass, g, u_ff=None):
    u0 = np.array([0.0, 0.0, mass * g, 0.0], dtype=float)
    if u_ff is not None:
        u0 = np.asarray(u_ff, dtype=float).reshape(4).copy()
    if u0[2] <= 0:
        u0[2] = mass * g
    actuator.reset(u0)


def run_tracking_simulation(
    xs,
    us,
    time_states,
    controller_id,
    params,
    phy_gui,
    x0=None,
    flat_outputs=None,
    flatness_physics=None,
) -> Dict[str, Any]:
    """
    Simulate closed-loop tracking of a planned trajectory.

    Dynamics model: nonlinear TVC rigid body (same as trajectory optimization).
    Controllers still use the linearized 12-state formulation.

    Returns dict with keys: t, x_sim (N,12), x_ref (N,12), pos_err (N,3),
    max_pos_err_m, rmse_pos_m, controller_id, u_hist (N,4) plant inputs,
    u_cmd_hist when actuator dynamics enabled.
    """
    phy = tracker_phy_from_gui(phy_gui)
    dyn_model = plant_from_phy_gui(phy_gui)
    mass, g = phy['MASS'], phy['G']

    ref = TrajectoryReference(
        xs, us, time_states=time_states,
        mass=mass, g=g,
    )

    sim_dt, control_dt, steps_per_control = numerical_sim_timing(params)
    mass = float(phy_gui.get('mass', phy['MASS']))
    ctrl_params = params
    r_gimbal_scale = 1.0
    if controller_id in (CONTROLLER_LQR, CONTROLLER_MPC):
        ctrl_params = scale_tracking_gimbal_r(
            params, mass, exponent=TRACKING_MANEUVER_GIMBAL_R_EXPONENT,
        )
        r_gimbal_scale = float(ctrl_params.get('R_qx', 10.0)) / max(
            float(params.get('R_qx', 10.0)), 1e-9,
        )

    tracker = _make_tracker(
        controller_id, phy, ctrl_params, phy_gui=phy_gui,
        flat_outputs=flat_outputs, flatness_physics=flatness_physics,
    )
    terminal_ctrl_params = None
    if controller_id in (CONTROLLER_LQR, CONTROLLER_MPC):
        terminal_ctrl_params = scale_tracking_gimbal_r(
            params, mass, exponent=TRACKING_TERMINAL_HOLD_GIMBAL_R_EXPONENT,
        )
    if controller_id == CONTROLLER_MPC:
        mpc_dt = float(ctrl_params.get('mpc_dt', control_dt))
        if abs(mpc_dt - control_dt) > 1e-9 and hasattr(tracker, 'update_params'):
            mpc_params = dict(ctrl_params)
            mpc_params['mpc_dt'] = control_dt
            tracker.update_params(mpc_params)
    if controller_id == CONTROLLER_ACADOS_NMPC:
        nmpc_params = dict(params)
        nmpc_params['control_dt'] = control_dt
        nmpc_params['nmpc_dt'] = control_dt
        if hasattr(tracker, 'update_params'):
            tracker.update_params(nmpc_params)

    px4_tracking = controller_id == CONTROLLER_PX4
    # PX4 cascade is tuned as pure feedback; other controllers may use planned u_ff.
    use_ff = TRACKING_USE_FEEDFORWARD and not px4_tracking
    clip_pos = TRACKING_CLIP_POS_ERROR_M
    clip_vel = TRACKING_CLIP_VEL_ERROR_M_S

    actuator = ActuatorDynamics(actuator_config_from_params(params))

    if x0 is None:
        x13 = np.asarray(ref.xs[0, :13], dtype=float).copy()
    else:
        x13 = state12_to_state13(np.asarray(x0, dtype=float).reshape(12))

    if px4_tracking:
        t_end = max(ref.duration(), 0.1)
    else:
        t_end = numerical_sim_end_time(ref, params)
    terminal_hold_s = float(
        params.get('terminal_hold_duration_s', TRACKING_TERMINAL_HOLD_DURATION_S),
    )
    plan_end_t = ref.plan_end_time()

    if hasattr(tracker, 'reset'):
        tracker.reset()

    u_ff0 = ref.control_lqr_at(0.0) if use_ff else None
    if actuator.any_enabled():
        _init_actuator(actuator, mass, g, u_ff=u_ff0)

    t_hist = []
    x12_hist = []
    u_opt_hist = []
    u_cmd_hist = []
    sp_pos_hist = []
    sp_vel_hist = []
    sp_att_hist = []
    sp_rate_hist = []
    record_cascade = controller_id in (CONTROLLER_PX4, CONTROLLER_FLATNESS)

    u_cmd_hold = None
    prev_u_cmd_hold = None
    cascade_hold = None
    substep = 0
    th_p_lim, th_r_lim = gimbal_pitch_roll_max_rad(params)
    max_gimbal_slew = np.radians(TRACKING_GIMBAL_RATE_LIMIT_DEG_S) * control_dt
    terminal_gains_active = None
    was_in_terminal = False

    t = 0.0
    while True:
        x12 = state13_to_state12(x13)
        t_clamped = min(t, plan_end_t)
        in_terminal = ref.in_terminal_hold(t)
        terminal_gains = ref.use_terminal_gains(t)

        if substep == 0:
            entering_hover_gains = (
                terminal_gains and terminal_gains_active is not True
            )
            entering_terminal_mode = in_terminal and not was_in_terminal

            if (
                controller_id in (CONTROLLER_LQR, CONTROLLER_MPC)
                and terminal_ctrl_params is not None
                and terminal_gains != terminal_gains_active
            ):
                tuned = terminal_ctrl_params if terminal_gains else ctrl_params
                if hasattr(tracker, 'update_gains'):
                    tracker.update_gains(tuned)
                elif hasattr(tracker, 'update_params'):
                    mpc_params = dict(tuned)
                    mpc_params['mpc_dt'] = control_dt
                    tracker.update_params(mpc_params)
                terminal_gains_active = terminal_gains

            if entering_hover_gains or entering_terminal_mode:
                prev_u_cmd_hold = None
            was_in_terminal = in_terminal

            clip_p = (
                TRACKING_TERMINAL_CLIP_POS_ERROR_M
                if terminal_gains else clip_pos
            )
            clip_v = (
                TRACKING_TERMINAL_CLIP_VEL_ERROR_M_S
                if terminal_gains else clip_vel
            )

            if px4_tracking:
                x_ref = ref.state12_at(t_clamped)
                acc_ref = ref.accel_at(t_clamped)
                u_ff = None
            else:
                x_ref = ref.tracking_state12_at(t)
                acc_ref = np.zeros(3, dtype=float) if in_terminal else ref.accel_at(t_clamped)
                if use_ff:
                    if in_terminal:
                        u_ff = ref.terminal_hold_control_lqr()
                    else:
                        u_ff = ref.control_lqr_at(t_clamped)
                else:
                    u_ff = None

            if controller_id == CONTROLLER_PX4:
                u_lqr, sig = tracker.compute(
                    x12, x_ref, acc_ref, control_dt, u_ff=None, use_ff=False,
                )
                cascade_hold = sig
                u_ref_lqr = np.zeros(4)
            elif controller_id == CONTROLLER_MPC:
                horizon = int(params.get('horizon', 20))
                mpc_dt = float(getattr(tracker, 'mpc_dt', params.get('mpc_dt', control_dt)))
                ref_h = ref.horizon_window(t, horizon, mpc_dt)
                u_ref_h = ref.control_lqr_horizon(t, horizon, mpc_dt)
                u_lqr = tracker.compute(
                    x12, ref_h, u_ref_horizon=u_ref_h,
                    clip_pos=clip_p, clip_vel=clip_v,
                )
                u_ref_lqr = u_ref_h[0]
            elif controller_id == CONTROLLER_ACADOS_NMPC:
                horizon = int(params.get('horizon', 15))
                u_cmd_hold = _clip_plant_control(
                    tracker.compute(x13, ref, t, horizon=horizon, dt=control_dt),
                    params,
                )
            elif controller_id == CONTROLLER_FLATNESS:
                u_lqr, sig = tracker.compute(
                    x12, ref, acc_ref, control_dt,
                    u_ff=u_ff, use_ff=use_ff, t_query=t_clamped,
                )
                cascade_hold = sig
                u_ref_lqr = u_ff if (use_ff and u_ff is not None) else np.zeros(4)
            else:
                u_lqr = tracker.compute(
                    x12, x_ref, u_ff=u_ff, use_ff=use_ff,
                    clip_pos=clip_p, clip_vel=clip_v,
                )
                u_ref_lqr = u_ff if (use_ff and u_ff is not None) else np.zeros(4)

            if controller_id == CONTROLLER_ACADOS_NMPC:
                pass
            elif px4_tracking:
                u_cmd_hold = lqr_to_control_opt(u_lqr, mass, g)
            else:
                u_lqr = clip_lqr_gimbal_cmd(u_lqr, th_p_lim, th_r_lim)
                u_cmd_hold = _clip_plant_control(
                    lqr_to_control_opt(u_lqr, mass, g), params,
                )
            if prev_u_cmd_hold is not None and controller_id not in (CONTROLLER_PX4, CONTROLLER_FLATNESS):
                for axis in (0, 1):
                    du = u_cmd_hold[axis] - prev_u_cmd_hold[axis]
                    u_cmd_hold[axis] = prev_u_cmd_hold[axis] + np.clip(
                        du, -max_gimbal_slew, max_gimbal_slew,
                    )
            prev_u_cmd_hold = u_cmd_hold.copy()

        if actuator.any_enabled():
            u_plant = actuator.step(u_cmd_hold, sim_dt)
        else:
            u_plant = u_cmd_hold

        t_hist.append(t)
        x12_hist.append(x12.copy())
        u_opt_hist.append(u_plant.copy())
        if record_cascade and cascade_hold is not None:
            sp_pos_hist.append(cascade_hold['pos'].copy())
            sp_vel_hist.append(cascade_hold['vel'].copy())
            sp_att_hist.append(cascade_hold['att_rad'].copy())
            sp_rate_hist.append(cascade_hold['rate_rad_s'].copy())
        if record_cascade or actuator.any_enabled():
            u_cmd_hist.append(u_cmd_hold.copy())

        if t >= t_end - 1e-12:
            break
        x13 = dyn_model.step(x13, u_plant, sim_dt)
        t += sim_dt
        substep = (substep + 1) % steps_per_control

    t_arr = np.asarray(t_hist, dtype=float)
    x_sim = np.asarray(x12_hist, dtype=float)
    if px4_tracking:
        x_ref_arr = np.array([ref.state12_at(min(ti, ref.t[-1])) for ti in t_arr])
    else:
        x_ref_arr = np.array([ref.tracking_state12_at(ti) for ti in t_arr])
    pos_err = x_sim[:, 0:3] - x_ref_arr[:, 0:3]
    err_norm = np.linalg.norm(pos_err, axis=1)

    result = {
        't': t_arr,
        'x_sim': x_sim,
        'x_ref': x_ref_arr,
        'pos_err': pos_err,
        'max_pos_err_m': float(np.max(err_norm)),
        'rmse_pos_m': float(np.sqrt(np.mean(err_norm ** 2))),
        'controller_id': controller_id,
        'u_hist': np.asarray(u_opt_hist, dtype=float),
        'use_feedforward': use_ff,
        'use_acc_feedforward': (
            bool(getattr(tracker, 'gains', {}).get('use_acc_feedforward', False))
            if controller_id == CONTROLLER_PX4 else None
        ),
        'dynamics_model': 'nonlinear',
        'actuator_dynamics_enabled': actuator.any_enabled(),
        'sim_dt': sim_dt,
        'control_dt': control_dt,
        'plan_duration_s': ref.duration(),
        'terminal_hold_duration_s': terminal_hold_s,
        'sim_total_duration_s': float(t_arr[-1]) if len(t_arr) else t_end,
        'total_duration_s': float(params.get('total_duration_s', 0.0)),
        'r_gimbal_scale': (
            r_gimbal_scale
            if controller_id in (CONTROLLER_LQR, CONTROLLER_MPC)
            else 1.0
        ),
        'platform_id': str(phy_gui.get('platform_id', 'proxy')),
        'r_thrust_body': np.array([
            float(phy_gui.get('r_thrust_x', 0.0)),
            float(phy_gui.get('r_thrust_y', 0.0)),
            float(phy_gui.get('r_thrust_z', -0.2)),
        ], dtype=float),
    }
    if actuator.any_enabled():
        result['u_cmd_hist'] = np.asarray(u_cmd_hist, dtype=float)
        result['actuator_config'] = actuator_config_from_params(params)
    elif record_cascade and u_cmd_hist:
        result['u_cmd_hist'] = np.asarray(u_cmd_hist, dtype=float)
    if record_cascade:
        result['cascade_sp'] = {
            'pos': np.asarray(sp_pos_hist, dtype=float),
            'vel': np.asarray(sp_vel_hist, dtype=float),
            'att_rad': np.asarray(sp_att_hist, dtype=float),
            'rate_rad_s': np.asarray(sp_rate_hist, dtype=float),
        }
    return result
