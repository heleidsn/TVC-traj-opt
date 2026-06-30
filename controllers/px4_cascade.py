# -*- coding: utf-8 -*-
"""PX4-style cascaded position / velocity / attitude / rate controller (P + PID)."""

from __future__ import annotations

import math

import numpy as np

from .px4_params import normalize_px4_params

_DEG = math.pi / 180.0

TUNE_LEVEL_RATE = 'rate'
TUNE_LEVEL_ATTITUDE = 'attitude'
TUNE_LEVEL_VELOCITY = 'velocity'
TUNE_LEVEL_POSITION = 'position'
TUNE_LEVELS = (
    TUNE_LEVEL_RATE,
    TUNE_LEVEL_ATTITUDE,
    TUNE_LEVEL_VELOCITY,
    TUNE_LEVEL_POSITION,
)


class PX4CascadeTracker:
    """
    Multicopter-style cascade: pos P → vel PID → tilt → att P → rate PID → actuators.

    Position and attitude outer loops are proportional only; velocity and rate loops are PID.
    Attitude P gains are per-degree; limits are radians internally.
    Roll/pitch may share gains; yaw is always separate.
    """

    def __init__(self, phy_params, gains):
        self.m = phy_params['MASS']
        self.g = phy_params['G']
        self.l = phy_params['DIST_COM_2_THRUST']
        self.Ixx = phy_params['I_XX']
        self.Iyy = phy_params['I_YY']
        self.Izz = phy_params['I_ZZ']
        self.gains = normalize_px4_params(gains)
        self.reset()

    def reset(self):
        n = 3
        self._i_vel = np.zeros(n)
        self._i_rate = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self._prev_vel_err = np.zeros(n)
        self._prev_rate_err = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

    @staticmethod
    def _pid(err, prev_err, integral, dt, kp, ki, kd, i_limit=None):
        if dt > 0.0:
            integral += err * dt
            if i_limit is not None:
                integral = float(np.clip(integral, -i_limit, i_limit))
            deriv = (err - prev_err) / dt
        else:
            deriv = 0.0
        out = kp * err + ki * integral + kd * deriv
        return out, integral, err

    def _pos_vel_axis(self, axis, pos_err, vel, vel_ff, acc_ff, dt, xy=True):
        g = self.gains
        if xy:
            kp_p = g['Kp_pos_xy']
            kp_v, ki_v, kd_v = g['Kp_vel_xy'], g['Ki_vel_xy'], g['Kd_vel_xy']
        else:
            kp_p = g['Kp_pos_z']
            kp_v, ki_v, kd_v = g['Kp_vel_z'], g['Ki_vel_z'], g['Kd_vel_z']

        vel_sp = kp_p * float(pos_err) + vel_ff
        ve = vel_sp - float(vel)
        acc_sp, self._i_vel[axis], self._prev_vel_err[axis] = self._pid(
            ve, self._prev_vel_err[axis], self._i_vel[axis], dt,
            kp_v, ki_v, kd_v, i_limit=5.0,
        )
        return acc_sp + acc_ff

    def _vel_pid_axis(self, axis, vel_sp, vel_meas, dt, xy=True):
        """Velocity PID only (used for velocity-level cascade tuning)."""
        g = self.gains
        if xy:
            kp_v, ki_v, kd_v = g['Kp_vel_xy'], g['Ki_vel_xy'], g['Kd_vel_xy']
        else:
            kp_v, ki_v, kd_v = g['Kp_vel_z'], g['Ki_vel_z'], g['Kd_vel_z']
        ve = float(vel_sp) - float(vel_meas)
        acc_sp, self._i_vel[axis], self._prev_vel_err[axis] = self._pid(
            ve, self._prev_vel_err[axis], self._i_vel[axis], dt,
            kp_v, ki_v, kd_v, i_limit=5.0,
        )
        return acc_sp

    def _rate_pid_axis(self, name, rate_sp, rate_meas, dt, rate_gains):
        rate_err = float(rate_sp) - float(rate_meas)
        kp_r, ki_r, kd_r = rate_gains
        angacc, self._i_rate[name], self._prev_rate_err[name] = self._pid(
            rate_err, self._prev_rate_err[name], self._i_rate[name], dt,
            kp_r, ki_r, kd_r, i_limit=50.0,
        )
        return angacc

    def _att_rate_axis(self, name, att_err_rad, rate_meas, dt, kp_att_deg, rate_gains):
        """Attitude P (deg error) → rate PID (rad/s). Returns (angacc, rate_setpoint)."""
        rate_sp = kp_att_deg * math.degrees(att_err_rad) * _DEG
        angacc = self._rate_pid_axis(name, rate_sp, rate_meas, dt, rate_gains)
        return angacc, rate_sp

    def _tilt_sp_from_acc(self, ax, ay, tilt_max):
        """
        Map world-frame horizontal accel to body tilt setpoints.

        Signs match the nonlinear TVC model (ENU, thrust along body +Z):
        +ax → +pitch, +ay → -roll.
        """
        pitch_sp = float(np.clip(float(ax) / self.g, -tilt_max, tilt_max))
        roll_sp = float(np.clip(-float(ay) / self.g, -tilt_max, tilt_max))
        return roll_sp, pitch_sp

    def _attitude_from_acc(
        self, roll, pitch, yaw, p, q_rate, r, ax, ay, az, yaw_sp, dt,
    ):
        """Map horizontal/vertical accel commands to attitude + rate cascades."""
        g = self.gains
        roll_sp, pitch_sp = self._tilt_sp_from_acc(ax, ay, g['tilt_max'])
        angacc_x, rate_sp_p = self._att_rate_axis(
            'roll', roll_sp - roll, p, dt,
            g['Kp_att_roll_deg'],
            (g['Kp_rate_roll'], g['Ki_rate_roll'], g['Kd_rate_roll']),
        )
        angacc_y, rate_sp_q = self._att_rate_axis(
            'pitch', pitch_sp - pitch, q_rate, dt,
            g['Kp_att_pitch_deg'],
            (g['Kp_rate_pitch'], g['Ki_rate_pitch'], g['Kd_rate_pitch']),
        )
        angacc_z, rate_sp_r = self._att_rate_axis(
            'yaw', yaw_sp - yaw, r, dt,
            g['Kp_att_yaw_deg'],
            (g['Kp_rate_yaw'], g['Ki_rate_yaw'], g['Kd_rate_yaw']),
        )
        att_sp = np.array([roll_sp, pitch_sp, yaw_sp], dtype=float)
        rate_sp = np.array([rate_sp_p, rate_sp_q, rate_sp_r], dtype=float)
        return angacc_x, angacc_y, angacc_z, az, att_sp, rate_sp

    def _actuators_from_angacc(self, angacc_x, angacc_y, angacc_z, az, u_ff=None, use_ff=True):
        g = self.gains
        wz2_x = self.l * self.m * self.g / self.Ixx
        wz2_y = self.l * self.m * self.g / self.Iyy

        qx_cmd = float(np.clip(-angacc_x / wz2_x, -g['gimbal_max'], g['gimbal_max']))
        qy_cmd = float(np.clip(-angacc_y / wz2_y, -g['gimbal_max'], g['gimbal_max']))

        hover = self.m * self.g
        T_delta = float(np.clip(az * self.m, -0.5 * hover, 0.5 * hover))
        r_cmd = float(np.clip(angacc_z * self.Izz, -hover * 0.2, hover * 0.2))

        u = np.array([qx_cmd, qy_cmd, T_delta, r_cmd], dtype=float)
        if use_ff and u_ff is not None:
            u = u + np.asarray(u_ff, dtype=float)
        return self._clip_u(u)

    def compute(self, x12, ref, acc_ref, dt, u_ff=None, use_ff=True):
        g = self.gains
        dt = max(float(dt), 1e-6)
        pos = x12[0:3]
        vel = x12[3:6]
        qx, qy, qz = x12[6:9]
        p, q_rate, r = x12[9:12]

        pos_ref = ref[0:3]
        vel_ref = ref[3:6]
        pos_err = pos_ref - pos

        vel_sp = np.array([
            g['Kp_pos_xy'] * pos_err[0] + vel_ref[0],
            g['Kp_pos_xy'] * pos_err[1] + vel_ref[1],
            g['Kp_pos_z'] * pos_err[2] + vel_ref[2],
        ], dtype=float)

        ax = self._pos_vel_axis(0, pos_err[0], vel[0], vel_ref[0], acc_ref[0], dt, xy=True)
        ay = self._pos_vel_axis(1, pos_err[1], vel[1], vel_ref[1], acc_ref[1], dt, xy=True)
        az = self._pos_vel_axis(2, pos_err[2], vel[2], vel_ref[2], acc_ref[2], dt, xy=False)

        roll_sp, pitch_sp = self._tilt_sp_from_acc(ax, ay, g['tilt_max'])
        yaw_sp = float(2.0 * ref[8])

        pitch = float(2.0 * qy)
        roll = float(2.0 * qx)
        yaw = float(2.0 * qz)

        angacc_x, rate_sp_p = self._att_rate_axis(
            'roll', roll_sp - roll, p, dt,
            g['Kp_att_roll_deg'],
            (g['Kp_rate_roll'], g['Ki_rate_roll'], g['Kd_rate_roll']),
        )
        angacc_y, rate_sp_q = self._att_rate_axis(
            'pitch', pitch_sp - pitch, q_rate, dt,
            g['Kp_att_pitch_deg'],
            (g['Kp_rate_pitch'], g['Ki_rate_pitch'], g['Kd_rate_pitch']),
        )
        angacc_z, rate_sp_r = self._att_rate_axis(
            'yaw', yaw_sp - yaw, r, dt,
            g['Kp_att_yaw_deg'],
            (g['Kp_rate_yaw'], g['Ki_rate_yaw'], g['Kd_rate_yaw']),
        )
        signals = {
            'pos': np.asarray(pos_ref, dtype=float).reshape(3),
            'vel': vel_sp,
            'att_rad': np.array([roll_sp, pitch_sp, yaw_sp], dtype=float),
            'rate_rad_s': np.array([rate_sp_p, rate_sp_q, rate_sp_r], dtype=float),
        }
        u = self._actuators_from_angacc(
            angacc_x, angacc_y, angacc_z, az, u_ff=u_ff, use_ff=use_ff,
        )
        return u, signals

    def compute_tune(self, x12, dt, level, setpoints, u_ff=None, use_ff=False):
        """
        Isolated cascade level for step tuning.

        Returns (u_lqr, cascade_signals) where cascade_signals holds inner-loop
        setpoints: pos [m], vel [m/s], att_rad, rate_rad_s (nan if not used).
        """
        g = self.gains
        dt = max(float(dt), 1e-6)
        pos = x12[0:3]
        vel = x12[3:6]
        qx, qy, qz = x12[6:9]
        p, q_rate, r = x12[9:12]
        pitch = float(2.0 * qy)
        roll = float(2.0 * qx)
        yaw = float(2.0 * qz)
        sp = dict(setpoints or {})
        nan3 = np.full(3, np.nan, dtype=float)
        signals = {
            'pos': nan3.copy(),
            'vel': nan3.copy(),
            'att_rad': nan3.copy(),
            'rate_rad_s': nan3.copy(),
        }

        if level == TUNE_LEVEL_RATE:
            p_sp = float(sp.get('p_deg_s', 0.0)) * _DEG
            q_sp = float(sp.get('q_deg_s', 0.0)) * _DEG
            r_sp = float(sp.get('r_deg_s', 0.0)) * _DEG
            signals['rate_rad_s'] = np.array([p_sp, q_sp, r_sp], dtype=float)
            angacc_x = self._rate_pid_axis(
                'roll', p_sp, p, dt,
                (g['Kp_rate_roll'], g['Ki_rate_roll'], g['Kd_rate_roll']),
            )
            angacc_y = self._rate_pid_axis(
                'pitch', q_sp, q_rate, dt,
                (g['Kp_rate_pitch'], g['Ki_rate_pitch'], g['Kd_rate_pitch']),
            )
            angacc_z = self._rate_pid_axis(
                'yaw', r_sp, r, dt,
                (g['Kp_rate_yaw'], g['Ki_rate_yaw'], g['Kd_rate_yaw']),
            )
            az = 0.0
        elif level == TUNE_LEVEL_ATTITUDE:
            roll_sp = float(sp.get('roll_deg', 0.0)) * _DEG
            pitch_sp = float(sp.get('pitch_deg', 0.0)) * _DEG
            yaw_sp = float(sp.get('yaw_deg', 0.0)) * _DEG
            tilt_max = g['tilt_max']
            roll_sp = float(np.clip(roll_sp, -tilt_max, tilt_max))
            pitch_sp = float(np.clip(pitch_sp, -tilt_max, tilt_max))
            signals['att_rad'] = np.array([roll_sp, pitch_sp, yaw_sp], dtype=float)
            angacc_x, rate_sp_p = self._att_rate_axis(
                'roll', roll_sp - roll, p, dt,
                g['Kp_att_roll_deg'],
                (g['Kp_rate_roll'], g['Ki_rate_roll'], g['Kd_rate_roll']),
            )
            angacc_y, rate_sp_q = self._att_rate_axis(
                'pitch', pitch_sp - pitch, q_rate, dt,
                g['Kp_att_pitch_deg'],
                (g['Kp_rate_pitch'], g['Ki_rate_pitch'], g['Kd_rate_pitch']),
            )
            angacc_z, rate_sp_r = self._att_rate_axis(
                'yaw', yaw_sp - yaw, r, dt,
                g['Kp_att_yaw_deg'],
                (g['Kp_rate_yaw'], g['Ki_rate_yaw'], g['Kd_rate_yaw']),
            )
            signals['rate_rad_s'] = np.array([rate_sp_p, rate_sp_q, rate_sp_r], dtype=float)
            az = 0.0
        elif level == TUNE_LEVEL_POSITION:
            x_sp = float(sp.get('x', 0.0))
            y_sp = float(sp.get('y', 0.0))
            z_sp = float(sp.get('z', 0.0))
            vel_sp = np.array([
                g['Kp_pos_xy'] * (x_sp - float(pos[0])),
                g['Kp_pos_xy'] * (y_sp - float(pos[1])),
                g['Kp_pos_z'] * (z_sp - float(pos[2])),
            ], dtype=float)
            signals['pos'] = np.array([x_sp, y_sp, z_sp], dtype=float)
            signals['vel'] = vel_sp
            ax = self._vel_pid_axis(0, vel_sp[0], vel[0], dt, xy=True)
            ay = self._vel_pid_axis(1, vel_sp[1], vel[1], dt, xy=True)
            az = self._vel_pid_axis(2, vel_sp[2], vel[2], dt, xy=False)
            yaw_sp = float(sp.get('yaw_deg', 0.0)) * _DEG
            angacc_x, angacc_y, angacc_z, az, att_sp, rate_sp = self._attitude_from_acc(
                roll, pitch, yaw, p, q_rate, r, ax, ay, az, yaw_sp, dt,
            )
            signals['att_rad'] = att_sp
            signals['rate_rad_s'] = rate_sp
        elif level == TUNE_LEVEL_VELOCITY:
            vel_sp = np.array([
                float(sp.get('vx', 0.0)),
                float(sp.get('vy', 0.0)),
                float(sp.get('vz', 0.0)),
            ], dtype=float)
            signals['vel'] = vel_sp
            xy_active = abs(vel_sp[0]) > 1e-6 or abs(vel_sp[1]) > 1e-6
            if xy_active:
                ax = self._vel_pid_axis(0, vel_sp[0], vel[0], dt, xy=True)
                ay = self._vel_pid_axis(1, vel_sp[1], vel[1], dt, xy=True)
            else:
                ax, ay = 0.0, 0.0
            az = self._vel_pid_axis(2, vel_sp[2], vel[2], dt, xy=False)
            yaw_sp = float(sp.get('yaw_deg', 0.0)) * _DEG
            if xy_active:
                angacc_x, angacc_y, angacc_z, az, att_sp, rate_sp = self._attitude_from_acc(
                    roll, pitch, yaw, p, q_rate, r, ax, ay, az, yaw_sp, dt,
                )
            else:
                roll_sp, pitch_sp = self._tilt_sp_from_acc(ax, ay, g['tilt_max'])
                angacc_x, rate_sp_p = self._att_rate_axis(
                    'roll', roll_sp - roll, p, dt,
                    g['Kp_att_roll_deg'],
                    (g['Kp_rate_roll'], g['Ki_rate_roll'], g['Kd_rate_roll']),
                )
                angacc_y, rate_sp_q = self._att_rate_axis(
                    'pitch', pitch_sp - pitch, q_rate, dt,
                    g['Kp_att_pitch_deg'],
                    (g['Kp_rate_pitch'], g['Ki_rate_pitch'], g['Kd_rate_pitch']),
                )
                angacc_z, rate_sp_r = self._att_rate_axis(
                    'yaw', yaw_sp - yaw, r, dt,
                    g['Kp_att_yaw_deg'],
                    (g['Kp_rate_yaw'], g['Ki_rate_yaw'], g['Kd_rate_yaw']),
                )
                att_sp = np.array([roll_sp, pitch_sp, yaw_sp], dtype=float)
                rate_sp = np.array([rate_sp_p, rate_sp_q, rate_sp_r], dtype=float)
            signals['att_rad'] = att_sp
            signals['rate_rad_s'] = rate_sp
        else:
            raise ValueError(f'Unknown tune level: {level}')

        u = self._actuators_from_angacc(angacc_x, angacc_y, angacc_z, az, u_ff=u_ff, use_ff=use_ff)
        return u, signals

    def _clip_u(self, u):
        gmax = self.gains['gimbal_max']
        u[0] = np.clip(u[0], -gmax, gmax)
        u[1] = np.clip(u[1], -gmax, gmax)
        hover = self.m * self.g
        u[2] = np.clip(u[2], -0.5 * hover, 0.5 * hover)
        return u
