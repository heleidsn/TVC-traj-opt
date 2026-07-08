# -*- coding: utf-8 -*-
"""Flat-output tracking with PX4-style cascade on center-of-oscillation ξ."""

from __future__ import annotations

import numpy as np

from .flatness import (
    FlatnessParams,
    estimate_flat_outputs_from_state12,
    estimate_flat_state_from_state12,
    flat_state_control_at,
    min_snap_1d,
)
from .linear_model import control_opt_to_lqr
from .px4_cascade import PX4CascadeTracker


class FlatnessCascadeTracker:
    """
    PX4-style cascade with ξ (center-of-oscillation) position loops.

    Feedforward: planned trajectory controls (``ref.control_lqr_at``), identical to
    PX4 cascade — this carries the feasible attitude/gimbal/thrust from Method 8.

    Feedback: proportional position error on (ξ_x, ξ_y, z) instead of COM (x, y, z),
    so the outer loop avoids the lateral non-minimum-phase output map.
    """

    def __init__(
        self,
        phy_params,
        px4_gains,
        flat_outputs=None,
        flatness_physics=None,
        mass=0.6,
        g=9.81,
    ):
        # ξ measurement must use the same (m, I, l) as Method-8 planning when flat_outputs exist.
        if flatness_physics is not None:
            fp_phy = flatness_physics
            self.fp = FlatnessParams.from_gui(
                fp_phy['mass'], fp_phy['Ixx'], fp_phy['Iyy'], fp_phy['Izz'],
                fp_phy['r_thrust_z'], g=float(fp_phy.get('g', g)),
            )
            self.mass = float(fp_phy['mass'])
            self.g = float(fp_phy.get('g', g))
        else:
            self.fp = FlatnessParams.from_phy_dict(phy_params)
            self.mass = float(mass)
            self.g = float(g)
        self.cascade = PX4CascadeTracker(phy_params, px4_gains)
        self._flat = flat_outputs
        self._flatness_physics = flatness_physics

    def set_flat_outputs(self, flat_outputs):
        self._flat = flat_outputs

    def reset(self):
        self.cascade.reset()

    def _flat_derivatives_stored(self, t_query):
        """Numerical derivatives of stored flat samples via ``np.gradient``."""
        t = np.asarray(self._flat['t'], dtype=float)
        t_q = float(np.clip(t_query, t[0], t[-1]))
        if bool(self._flat.get('piecewise_constant', False)):
            return {
                'xi_x': float(np.interp(t_q, t, np.asarray(self._flat['xi_x'], dtype=float))),
                'dxi_x': 0.0, 'ddxi_x': 0.0, 'dddxi_x': 0.0, 'ddddxi_x': 0.0,
                'xi_y': float(np.interp(t_q, t, np.asarray(self._flat['xi_y'], dtype=float))),
                'dxi_y': 0.0, 'ddxi_y': 0.0, 'dddxi_y': 0.0, 'ddddxi_y': 0.0,
                'z': float(np.interp(t_q, t, np.asarray(self._flat['z'], dtype=float))),
                'dz': 0.0, 'ddz': 0.0,
                'psi': float(np.interp(t_q, t, np.asarray(self._flat['psi'], dtype=float))),
                'dpsi': 0.0, 'ddpsi': 0.0,
            }

        def series_deriv(key, order):
            v = np.asarray(self._flat[key], dtype=float)
            d = v
            for _ in range(order):
                d = np.gradient(d, t, edge_order=2)
            return float(np.interp(t_q, t, d))

        return {
            'xi_x': series_deriv('xi_x', 0), 'dxi_x': series_deriv('xi_x', 1),
            'ddxi_x': series_deriv('xi_x', 2), 'dddxi_x': series_deriv('xi_x', 3),
            'ddddxi_x': series_deriv('xi_x', 4),
            'xi_y': series_deriv('xi_y', 0), 'dxi_y': series_deriv('xi_y', 1),
            'ddxi_y': series_deriv('xi_y', 2), 'dddxi_y': series_deriv('xi_y', 3),
            'ddddxi_y': series_deriv('xi_y', 4),
            'z': series_deriv('z', 0), 'dz': series_deriv('z', 1), 'ddz': series_deriv('z', 2),
            'psi': series_deriv('psi', 0), 'dpsi': series_deriv('psi', 1), 'ddpsi': series_deriv('psi', 2),
        }

    def flat_reference_at(self, ref, t_query):
        """
        Flat-output reference at *t_query*.

        Primary source: planned COM state ``ref.state12_at`` mapped through the
        planning flatness model (consistent with the displayed trajectory).
        Falls back to stored ``flat_outputs`` only when the reference has no states.
        """
        t_q = float(t_query)
        if hasattr(ref, 'state12_at') and ref.x12 is not None and len(ref.x12) >= 2:
            x12 = ref.state12_at(t_q)
            est = estimate_flat_state_from_state12(self.fp, x12)
            out = {
                'xi_x': est['xi_x'], 'dxi_x': est['dxi_x'],
                'xi_y': est['xi_y'], 'dxi_y': est['dxi_y'],
                'z': est['z'], 'dz': est['dz'],
                'psi': est['psi'], 'dpsi': est['dpsi'],
            }
            if self._flat is not None and 't' in self._flat and len(self._flat['t']) >= 2:
                hi = self._flat_derivatives_stored(t_q)
                out.update({
                    'ddxi_x': hi['ddxi_x'], 'dddxi_x': hi['dddxi_x'], 'ddddxi_x': hi['ddddxi_x'],
                    'ddxi_y': hi['ddxi_y'], 'dddxi_y': hi['dddxi_y'], 'ddddxi_y': hi['ddddxi_y'],
                    'ddz': hi['ddz'], 'ddpsi': hi['ddpsi'],
                })
            else:
                out.update({
                    'ddxi_x': 0.0, 'dddxi_x': 0.0, 'ddddxi_x': 0.0,
                    'ddxi_y': 0.0, 'dddxi_y': 0.0, 'ddddxi_y': 0.0,
                    'ddz': 0.0, 'ddpsi': 0.0,
                })
            return out

        if self._flat is not None and 't' in self._flat and len(self._flat['t']) >= 2:
            return self._flat_derivatives_stored(t_q)

        t_arr = ref.t
        xi_x_arr = np.array([
            estimate_flat_outputs_from_state12(self.fp, ref.x12[i])[0]
            for i in range(len(t_arr))
        ])
        xi_y_arr = np.array([
            estimate_flat_outputs_from_state12(self.fp, ref.x12[i])[1]
            for i in range(len(t_arr))
        ])
        z_arr = ref.x12[:, 2]
        psi_arr = 2.0 * np.arcsin(np.clip(ref.x12[:, 8], -1.0, 1.0))

        t0, t1 = float(t_arr[0]), float(t_arr[-1])
        if t1 <= t0:
            t1 = t0 + 1e-3
        xi_x_v, dxi_x, ddxi_x, dddxi_x, ddddxi_x = min_snap_1d(
            t_q, t0, t1, float(xi_x_arr[0]), float(xi_x_arr[-1]),
        )
        xi_y_v, dxi_y, ddxi_y, dddxi_y, ddddxi_y = min_snap_1d(
            t_q, t0, t1, float(xi_y_arr[0]), float(xi_y_arr[-1]),
        )
        z, dz, ddz, _, _ = min_snap_1d(t_q, t0, t1, float(z_arr[0]), float(z_arr[-1]))
        psi, dpsi, ddpsi, _, _ = min_snap_1d(
            t_q, t0, t1, float(psi_arr[0]), float(psi_arr[-1]),
        )
        return dict(
            xi_x=xi_x_v, dxi_x=dxi_x, ddxi_x=ddxi_x, dddxi_x=dddxi_x, ddddxi_x=ddddxi_x,
            xi_y=xi_y_v, dxi_y=dxi_y, ddxi_y=ddxi_y, dddxi_y=dddxi_y, ddddxi_y=ddddxi_y,
            z=z, dz=dz, ddz=ddz, psi=psi, dpsi=dpsi, ddpsi=ddpsi,
        )

    def planned_feedforward_lqr_at(self, ref, t_query):
        """Use the planned trajectory control (same feedforward path as PX4 cascade)."""
        if hasattr(ref, 'control_lqr_at'):
            return ref.control_lqr_at(t_query)
        return self.feedforward_lqr_at(ref, t_query)

    def feedforward_lqr_at(self, ref, t_query):
        """Flatness-based feedforward in LQR control coordinates."""
        d = self.flat_reference_at(ref, t_query)
        _, u_opt = flat_state_control_at(
            self.fp,
            d['xi_x'], d['dxi_x'], d['ddxi_x'], d['dddxi_x'], d['ddddxi_x'],
            d['xi_y'], d['dxi_y'], d['ddxi_y'], d['dddxi_y'], d['ddddxi_y'],
            d['z'], d['dz'], d['ddz'],
            d['psi'], d['dpsi'], d['ddpsi'],
        )
        return control_opt_to_lqr(u_opt, self.mass, self.g)

    def compute(
        self,
        x12,
        ref,
        acc_ref,
        dt,
        u_ff=None,
        use_ff=True,
        t_query=0.0,
    ):
        """
        Cascade on ξ with optional flatness feedforward.

        Returns (u_lqr, cascade_signals) like :class:`PX4CascadeTracker`.
        """
        del acc_ref
        t_q = float(t_query)
        if hasattr(ref, 'in_terminal_hold') and ref.in_terminal_hold(t_q):
            x_ref = ref.terminal_hold_state12()
            u_ff_term = ref.terminal_hold_control_lqr() if use_ff else None
            return self.cascade.compute(
                x12, x_ref, np.zeros(3, dtype=float), dt,
                u_ff=u_ff_term, use_ff=use_ff,
            )

        d = self.flat_reference_at(ref, t_q)
        meas = estimate_flat_state_from_state12(self.fp, x12)

        pos_err = np.array([
            d['xi_x'] - meas['xi_x'],
            d['xi_y'] - meas['xi_y'],
            d['z'] - meas['z'],
        ], dtype=float)
        dt = max(float(dt), 1e-6)
        g = self.cascade.gains
        acc_ff = np.array([d['ddxi_x'], d['ddxi_y'], d['ddz']], dtype=float)
        ddxi_x_cmd = self.cascade._pos_vel_axis(
            0, pos_err[0], meas['dxi_x'], d['dxi_x'], acc_ff[0], dt, xy=True,
        )
        ddxi_y_cmd = self.cascade._pos_vel_axis(
            1, pos_err[1], meas['dxi_y'], d['dxi_y'], acc_ff[1], dt, xy=True,
        )
        ddz_cmd = self.cascade._pos_vel_axis(
            2, pos_err[2], meas['dz'], d['dz'], acc_ff[2], dt, xy=False,
        )
        x_ref_com, u_opt_ff = flat_state_control_at(
            self.fp,
            d['xi_x'], d['dxi_x'], ddxi_x_cmd, d.get('dddxi_x', 0.0), d.get('ddddxi_x', 0.0),
            d['xi_y'], d['dxi_y'], ddxi_y_cmd, d.get('dddxi_y', 0.0), d.get('ddddxi_y', 0.0),
            d['z'], d['dz'], ddz_cmd,
            d['psi'], d['dpsi'], d.get('ddpsi', 0.0),
        )

        vel_sp = np.array([
            g['Kp_pos_xy'] * pos_err[0] + d['dxi_x'],
            g['Kp_pos_xy'] * pos_err[1] + d['dxi_y'],
            g['Kp_pos_z'] * pos_err[2] + d['dz'],
        ], dtype=float)
        vel_lim = g.get('vel_limit_m_s')
        if vel_lim is not None and float(vel_lim) > 0.0:
            vel_sp = np.clip(vel_sp, -float(vel_lim), float(vel_lim))

        if use_ff:
            u_ff = control_opt_to_lqr(u_opt_ff, self.mass, self.g)

        u, signals = self.cascade.compute(
            x12, x_ref_com, np.zeros(3, dtype=float), dt,
            u_ff=u_ff, use_ff=use_ff,
        )
        signals['pos'] = np.array([d['xi_x'], d['xi_y'], d['z']], dtype=float)
        signals['vel'] = vel_sp
        return u, signals


# Backward-compatible alias
FlatnessTracker = FlatnessCascadeTracker
