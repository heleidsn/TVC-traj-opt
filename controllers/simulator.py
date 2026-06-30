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
from .params import CONTROLLER_LQR, CONTROLLER_MPC, CONTROLLER_PX4
from .px4_cascade import PX4CascadeTracker
from .trajectory_ref import TrajectoryReference


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


def _make_tracker(controller_id, phy, params):
    if controller_id == CONTROLLER_PX4:
        return PX4CascadeTracker(phy, params)
    if controller_id == CONTROLLER_LQR:
        return LQRTracker(phy, params)
    if controller_id == CONTROLLER_MPC:
        return MPCTracker(phy, params)
    raise ValueError(f'Unknown controller: {controller_id}')


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
    tracker = _make_tracker(controller_id, phy, params)

    sim_dt, control_dt, steps_per_control = numerical_sim_timing(params)
    use_ff = bool(params.get('use_feedforward', True))
    clip_pos = float(params.get('clip_pos_error', 0.5))
    clip_vel = float(params.get('clip_vel_error', 0.5))

    actuator = ActuatorDynamics(actuator_config_from_params(params))

    if x0 is None:
        x13 = np.asarray(ref.xs[0, :13], dtype=float).copy()
    else:
        x13 = state12_to_state13(np.asarray(x0, dtype=float).reshape(12))

    t_end = ref.duration()
    if t_end <= 0:
        t_end = 0.1

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
    record_cascade = controller_id == CONTROLLER_PX4

    u_cmd_hold = None
    substep = 0

    t = 0.0
    while True:
        x12 = state13_to_state12(x13)
        t_clamped = min(t, ref.t[-1])

        if substep == 0:
            x_ref = ref.state12_at(t_clamped)
            acc_ref = ref.accel_at(t_clamped)
            u_ff = ref.control_lqr_at(t_clamped) if use_ff else None

            if controller_id == CONTROLLER_PX4:
                u_lqr, sig = tracker.compute(
                    x12, x_ref, acc_ref, control_dt, u_ff=u_ff, use_ff=use_ff,
                )
                sp_pos_hist.append(sig['pos'].copy())
                sp_vel_hist.append(sig['vel'].copy())
                sp_att_hist.append(sig['att_rad'].copy())
                sp_rate_hist.append(sig['rate_rad_s'].copy())
            elif controller_id == CONTROLLER_MPC:
                horizon = int(params.get('horizon', 20))
                mpc_dt = float(params.get('mpc_dt', 0.05))
                ref_h = ref.horizon_window(t_clamped, horizon, mpc_dt)
                u_lqr = tracker.compute(
                    x12, ref_h, u_ff=u_ff, use_ff=use_ff,
                    clip_pos=clip_pos, clip_vel=clip_vel,
                )
            else:
                u_lqr = tracker.compute(
                    x12, x_ref, u_ff=u_ff, use_ff=use_ff,
                    clip_pos=clip_pos, clip_vel=clip_vel,
                )

            u_cmd_hold = lqr_to_control_opt(u_lqr, mass, g)

        if actuator.any_enabled():
            u_plant = actuator.step(u_cmd_hold, sim_dt)
        else:
            u_plant = u_cmd_hold

        t_hist.append(t)
        x12_hist.append(x12.copy())
        u_opt_hist.append(u_plant.copy())
        if record_cascade or actuator.any_enabled():
            u_cmd_hist.append(u_cmd_hold.copy())

        if t >= t_end - 1e-12:
            break
        x13 = dyn_model.step(x13, u_plant, sim_dt)
        t += sim_dt
        substep = (substep + 1) % steps_per_control

    t_arr = np.asarray(t_hist, dtype=float)
    x_sim = np.asarray(x12_hist, dtype=float)
    x_ref_arr = np.array([ref.state12_at(min(ti, ref.t[-1])) for ti in t_arr])
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
        'dynamics_model': 'nonlinear',
        'actuator_dynamics_enabled': actuator.any_enabled(),
        'sim_dt': sim_dt,
        'control_dt': control_dt,
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
