# -*- coding: utf-8 -*-
"""Time-varying reference from optimized trajectory (13-state planner format)."""

from __future__ import annotations

import numpy as np

from .linear_model import control_opt_to_lqr, state13_to_state12


class TrajectoryReference:
    """
    Interpolate position, velocity, attitude, rates and planned controls along time.
    """

    def __init__(self, xs, us=None, time_states=None, mass=0.6, g=9.81):
        self.xs = np.asarray(xs, dtype=float)
        if self.xs.ndim != 2 or self.xs.shape[1] < 13:
            raise ValueError('xs must be (N, 13+) planner states')
        self.us = None if us is None else np.asarray(us, dtype=float)
        self.mass = float(mass)
        self.g = float(g)

        n = self.xs.shape[0]
        if time_states is not None:
            self.t = np.asarray(time_states, dtype=float).reshape(-1)
            if self.t.size != n:
                raise ValueError('time_states length must match xs')
        else:
            self.t = np.arange(n, dtype=float) * 0.05

        self.x12 = state13_to_state12(self.xs)
        self._build_derivatives()

    def _build_derivatives(self):
        """Numerical acceleration reference from velocity (same length as states)."""
        n = len(self.t)
        acc = np.zeros((n, 3), dtype=float)
        if n < 2:
            return
        for i in range(n):
            if i == 0:
                dt = max(self.t[1] - self.t[0], 1e-6)
                acc[i] = (self.x12[1, 3:6] - self.x12[0, 3:6]) / dt
            elif i == n - 1:
                dt = max(self.t[-1] - self.t[-2], 1e-6)
                acc[i] = (self.x12[-1, 3:6] - self.x12[-2, 3:6]) / dt
            else:
                dt = max(self.t[i + 1] - self.t[i - 1], 1e-6)
                acc[i] = (self.x12[i + 1, 3:6] - self.x12[i - 1, 3:6]) / dt
        self.acc = acc

    def duration(self):
        return float(self.t[-1] - self.t[0])

    def plan_end_time(self):
        """Absolute time of the last planned state sample."""
        return float(self.t[-1])

    def _interp_rows(self, arr, t_query):
        t_query = float(np.clip(t_query, self.t[0], self.t[-1]))
        out = np.zeros(arr.shape[1], dtype=float)
        for j in range(arr.shape[1]):
            out[j] = np.interp(t_query, self.t, arr[:, j])
        return out

    def state12_at(self, t_query):
        return self._interp_rows(self.x12, t_query)

    def accel_at(self, t_query):
        return self._interp_rows(self.acc, t_query)

    def control_lqr_at(self, t_query):
        if self.us is None or len(self.us) == 0:
            return np.zeros(4, dtype=float)
        t_u = self.t[: len(self.us)]
        t_query = float(np.clip(t_query, t_u[0], t_u[-1]))
        u = np.zeros(4, dtype=float)
        for j in range(min(4, self.us.shape[1])):
            u[j] = np.interp(t_query, t_u, self.us[:, j])
        return control_opt_to_lqr(u, self.mass, self.g)

    def control_opt_at(self, t_query):
        if self.us is None or len(self.us) == 0:
            return np.zeros(4, dtype=float)
        t_u = self.t[: len(self.us)]
        t_query = float(np.clip(t_query, t_u[0], t_u[-1]))
        u = np.zeros(4, dtype=float)
        for j in range(min(4, self.us.shape[1])):
            u[j] = np.interp(t_query, t_u, self.us[:, j])
        return u

    def in_terminal_hold(self, t_query):
        """True once simulation time has reached the end of the planned trajectory."""
        return float(t_query) >= self.plan_end_time() - 1e-9

    def plan_in_hover_segment(self, t_query):
        """Last 0.5 s of the plan while the reference is near the goal and slow."""
        t_q = float(t_query)
        if t_q >= self.plan_end_time() - 1e-9:
            return False
        if t_q + 1e-9 < self.plan_end_time() - 0.5:
            return False
        x_ref = self.state12_at(t_q)
        x_final = self.state12_at(self.t[-1])
        if float(np.linalg.norm(x_ref[0:3] - x_final[0:3])) > 0.08:
            return False
        return float(np.linalg.norm(x_ref[3:6])) < 0.12

    def use_terminal_gains(self, t_query):
        """Stronger hover gains: terminal mode, or last planned hover segment."""
        return self.in_terminal_hold(t_query) or self.plan_in_hover_segment(t_query)

    def terminal_hold_state12(self):
        """Final planned position, level attitude, zero velocity and body rates."""
        x = np.zeros(12, dtype=float)
        x[0:3] = self.state12_at(self.t[-1])[0:3]
        return x

    def terminal_hold_control_lqr(self):
        """Hover trim in LQR coords: level gimbal, thrust = weight, no yaw torque."""
        return np.zeros(4, dtype=float)

    def tracking_state12_at(self, t_query):
        """Closed-loop tracking target (terminal setpoint after plan time)."""
        if self.in_terminal_hold(t_query):
            return self.terminal_hold_state12()
        return self.state12_at(t_query)

    def horizon_window(self, t0, n_steps, dt):
        """Return (n_steps+1, 12) reference states starting at t0."""
        if self.in_terminal_hold(t0):
            x_hold = self.terminal_hold_state12()
            return np.tile(x_hold, (n_steps + 1, 1))
        times = t0 + np.arange(n_steps + 1) * dt
        return np.array([self.state12_at(min(t, self.plan_end_time())) for t in times])

    def control_lqr_horizon(self, t0, n_steps, dt):
        """Return (n_steps, 4) planned controls in LQR coords [qx, qy, T_delta, r]."""
        if self.in_terminal_hold(t0):
            u_hold = self.terminal_hold_control_lqr()
            return np.tile(u_hold, (n_steps, 1))
        times = t0 + np.arange(n_steps) * dt
        return np.array([self.control_lqr_at(min(t, self.plan_end_time())) for t in times])
