#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization GUI using PyQt5

Create user interface using PyQt5, supports:
- Waypoints (start and targets; initial state from first waypoint)
- Adjust cost weight parameters; save/load JSON parameters
- Real-time display of optimization process and results

Usage:
    python run_tvc_traj_opt.py          # preferred: project root entry
    python scripts/tvc_traj_opt_gui.py  # legacy / direct

Installation:
    If PyQt5 import error occurs, please install:
    - Using conda: conda install pyqt
    - Using pip: pip install PyQt5
    
Note: Need to activate conda environment first
    conda activate eagle_mpc
"""

import sys
import os
import json
import time

# Ensure tvc_traj_opt module can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

GUI_PARAMS_VERSION = 1
DEFAULT_GUI_PARAMS_FILENAME = 'tvc_traj_opt_gui_params.json'

import numpy as np
import matplotlib

# Check and import Qt backend
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                                 QGroupBox, QGridLayout, QTextEdit, QTabWidget,
                                 QDoubleSpinBox, QSpinBox, QMessageBox, QProgressBar, QComboBox, QCheckBox,
                                 QFileDialog, QScrollArea, QSizePolicy)
    from PyQt5.QtCore import QThread, pyqtSignal, Qt
    from PyQt5.QtGui import QFont
    QT_AVAILABLE = True
except ImportError:
    try:
        import PySide2
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                      QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                                      QGroupBox, QGridLayout, QTextEdit, QTabWidget,
                                      QDoubleSpinBox, QSpinBox, QMessageBox, QProgressBar, QComboBox, QCheckBox,
                                      QFileDialog, QScrollArea, QSizePolicy)
        from PySide2.QtCore import QThread, Signal as pyqtSignal, Qt
        from PySide2.QtGui import QFont
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        print("=" * 60)
        print("Error: PyQt5 or PySide2 not found")
        print("=" * 60)
        print("Please install PyQt5 or PySide2:")
        print("  Using pip:  pip install PyQt5")
        print("  Using conda: conda install pyqt")
        print("")
        print("If using conda environment, please run:")
        print("  conda activate eagle_mpc")
        print("  conda install pyqt")
        print("=" * 60)
        sys.exit(1)

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import crocoddyl
# Import TVC model and common utils (from same directory)
from tvc_traj_opt import TVCRocketActionModel
from tvc_common import quat_to_euler, yaw_to_quaternion
from tvc_traj_gui_plots import draw_trajectory_panels, draw_cost_panel
from tvc_traj_opt_pinocchio import (solve_with_pinocchio_waypoints, solve_with_pinocchio_waypoints_unified,
                                    convert_pinocchio_state_to_method1)
try:
    from tvc_traj_opt_acados import (solve_with_acados_waypoints, solve_with_acados_waypoints_unified,
                                     ACADOS_AVAILABLE)
except ImportError:
    solve_with_acados_waypoints = None
    solve_with_acados_waypoints_unified = None
    ACADOS_AVAILABLE = False
try:
    from tvc_traj_opt_acados_min_time import solve_spannagl_style_waypoints
except ImportError:
    solve_spannagl_style_waypoints = None

# Built-in trajectory presets: waypoint [x_m, y_m, z_m, yaw_deg, arrival_time_s]
TRAJECTORY_PRESETS = (
    (
        'Grasshopper',
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 5.0],
            [4.0, 0.0, 0.0, 0.0, 10.0],
        ],
    ),
    (
        'Platform hop (0→1 m, +2 m x, land)',
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 5.0],
            [2.0, 0.0, 1.0, 0.0, 10.0],
            [2.0, 0.0, 0.0, 0.0, 15.0],
        ],
    ),
)


def _normalize_waypoint_row(row):
    r = list(row)
    while len(r) < 5:
        r.append(0.0)
    return [float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])]


def trajectory_preset_match_index(waypoints, tolerance=1e-4):
    """Return TRAJECTORY_PRESETS index if waypoints match a preset, else None."""
    wn = [_normalize_waypoint_row(w) for w in waypoints]
    for i, (_, preset) in enumerate(TRAJECTORY_PRESETS):
        if len(wn) != len(preset):
            continue
        match = True
        for a, b in zip(wn, preset):
            if any(abs(a[j] - b[j]) > tolerance for j in range(5)):
                match = False
                break
        if match:
            return i
    return None


class OptimizationThread(QThread):
    """Optimization thread, runs optimization process in background"""
    # Signal definitions
    iteration_update = pyqtSignal(int, float, float, int)  # iteration number, cost, stop, segment_index
    state_update = pyqtSignal(list, list)  # xs, us
    finished = pyqtSignal(list, list, list, dict)  # xs, us, all_loggers, timing_info
    error = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.running = True
        
    def run(self):
        """Run optimization with multiple waypoints and time segments"""
        try:
            # Unpack parameters
            dt = self.params['dt']
            max_iter = self.params['max_iter']
            weights = self.params['weights']
            terminal_weights = self.params.get('terminal_weights', None)
            bounds = self.params['bounds']
            m = self.params['m']
            I = self.params['I']
            r_thrust = self.params['r_thrust']
            waypoints = self.params.get('waypoints', [])
            method = self.params.get('method', 3)  # 3–4,6 Acados; 5 = Method 6 Spannagl FFTF
            use_box_solver = (method == 2)  # BoxFDDP uses native control bounds
            unified = self.params.get('unified', False)  # Merge all segments into one problem
            
            # Method 6 (index 5): Spannagl et al. style minimum-thrust + free-final-time (CasADi/Acados FFTF)
            if method == 5:
                if solve_spannagl_style_waypoints is None:
                    self.error.emit(
                        "Method 6 unavailable: could not import tvc_traj_opt_acados_min_time (need CasADi)."
                    )
                    return
                if len(waypoints) < 2:
                    self.error.emit("Need at least 2 waypoints (start and at least one waypoint)")
                    return

                def callback_spannagl(solver, seg_idx, current_xs, current_us, completed_xs, completed_us):
                    if self.running and current_xs:
                        combined_xs, combined_us = [], []
                        for i, (seg_xs, seg_us) in enumerate(zip(completed_xs, completed_us)):
                            if i == 0:
                                combined_xs.extend(seg_xs)
                                combined_us.extend(seg_us)
                            else:
                                combined_xs.extend(seg_xs[1:])
                                combined_us.extend(seg_us)
                        if len(combined_xs) > 0 and seg_idx > 0:
                            combined_xs.extend(current_xs[1:])
                        else:
                            combined_xs.extend(current_xs)
                        combined_us.extend(current_us)
                        self.state_update.emit(combined_xs, combined_us)

                def iteration_callback_spannagl(iter, cost, stop, seg_idx):
                    if self.running:
                        self.iteration_update.emit(iter, cost, stop, seg_idx)

                x0_m1 = np.asarray(self.params.get("x0"), dtype=float).flatten()
                t0 = time.perf_counter()
                _pack = solve_spannagl_style_waypoints(
                    dt=dt,
                    waypoints=waypoints,
                    m=m,
                    I=I,
                    r_thrust=r_thrust,
                    weights=weights,
                    bounds=bounds,
                    x0_method1=x0_m1,
                    max_iter=max_iter,
                    callback=callback_spannagl,
                    running_flag=lambda: self.running,
                    iteration_callback=iteration_callback_spannagl,
                    verbose_solve=True,
                )
                combined_xs, combined_us, all_loggers, us_actual = _pack[:4]
                sp_meta = _pack[4] if len(_pack) >= 5 and isinstance(_pack[4], dict) else {}
                total_time = time.perf_counter() - t0
                total_iters = sum(len(logger.costs) for logger in all_loggers) if all_loggers else 0
                timing_info = {
                    "total_time": total_time,
                    "total_iters": total_iters,
                    "avg_time_per_iter": total_time / total_iters if total_iters > 0 else 0.0,
                    "method": "Method 6 (Spannagl min-fuel FFTF)",
                    "us_actual": us_actual,
                }
                timing_info.update(sp_meta)
                if self.running:
                    self.finished.emit(combined_xs, combined_us, all_loggers, timing_info)
                return

            # Method 4–5 & 7: Acados; 5=min-time LS; 6=Method7 free-tf state + EXTERNAL cost
            if method in (3, 4, 6):
                if not ACADOS_AVAILABLE:
                    self.error.emit("Acados not available. Install: pip install casadi; build acados from source.")
                    return
                def callback_acados(solver, seg_idx, current_xs, current_us, completed_xs, completed_us):
                    if self.running and current_xs:
                        # Same as Pinocchio: send completed segments + current, show full trajectory
                        combined_xs, combined_us = [], []
                        for i, (seg_xs, seg_us) in enumerate(zip(completed_xs, completed_us)):
                            if i == 0:
                                combined_xs.extend(seg_xs)
                                combined_us.extend(seg_us)
                            else:
                                combined_xs.extend(seg_xs[1:])
                                combined_us.extend(seg_us)
                        if len(combined_xs) > 0 and seg_idx > 0:
                            combined_xs.extend(current_xs[1:])
                        else:
                            combined_xs.extend(current_xs)
                        combined_us.extend(current_us)
                        self.state_update.emit(combined_xs, combined_us)
                def iteration_callback_acados(iter, cost, stop, seg_idx):
                    if self.running:
                        self.iteration_update.emit(iter, cost, stop, seg_idx)
                # When multi-waypoint and need waypoint terminal cost: use segment mode (terminal cost at each segment end)
                use_segment_for_waypoints = (
                    len(waypoints) > 2 and unified and
                    weights.get("waypoint_terminal_cost", True) and method == 3
                )
                if use_segment_for_waypoints:
                    unified = False  # Switch to segment mode for terminal cost at each intermediate point
                solver_fn = solve_with_acados_waypoints_unified if unified else solve_with_acados_waypoints
                t0 = time.perf_counter()
                _pack = solver_fn(
                    dt=dt, waypoints=waypoints, m=m, I=I, r_thrust=r_thrust,
                    weights=weights, terminal_weights=terminal_weights, bounds=bounds,
                    max_iter=max_iter, use_box_solver=False, callback=callback_acados,
                    running_flag=lambda: self.running,
                    iteration_callback=iteration_callback_acados,
                    verbose_solve=True  # For store_iterates and cost curve plotting
                )
                combined_xs, combined_us, all_loggers, us_actual = _pack[:4]
                acados_meta = _pack[4] if len(_pack) >= 5 and isinstance(_pack[4], dict) else {}
                total_time = time.perf_counter() - t0
                total_iters = sum(len(logger.costs) for logger in all_loggers) if all_loggers else 0
                if method == 6:
                    mname = "Method 7 (Acados free-tf EXTERNAL)"
                elif method == 4:
                    mname = "Method 5 (Acados min-time)"
                else:
                    mname = "Method 4 (Acados)"
                timing_info = {
                    "total_time": total_time,
                    "total_iters": total_iters,
                    "avg_time_per_iter": total_time / total_iters if total_iters > 0 else 0.0,
                    "method": mname,
                    "us_actual": us_actual,
                }
                if isinstance(acados_meta, dict):
                    timing_info.update(acados_meta)
                if self.running:
                    self.finished.emit(combined_xs, combined_us, all_loggers, timing_info)
                return
            
            # Check if we have waypoints with times
            if len(waypoints) < 2:
                self.error.emit("Need at least 2 waypoints (start and at least one waypoint)")
                return
            
            # Ensure all waypoints have all required fields and convert to lists
            # Format: [x, y, z, yaw_deg, time]
            waypoints_list = []
            for wp in waypoints:
                wp_list = list(wp) if isinstance(wp, (list, tuple, np.ndarray)) else [wp]
                # Ensure waypoint has 5 elements [x, y, z, yaw_deg, time]
                if len(wp_list) == 4:
                    # Old format: [x, y, z, time] -> convert to [x, y, z, yaw=0, time]
                    wp_list = [wp_list[0], wp_list[1], wp_list[2], 0.0, wp_list[3]]
                while len(wp_list) < 5:
                    wp_list.append(0.0)
                waypoints_list.append(wp_list[:5])  # Keep only first 5 elements
            
            # Sort waypoints by time (index 4)
            waypoints = sorted(waypoints_list, key=lambda x: float(x[4]))
            
            # Calculate segment durations
            durations = []
            for i in range(len(waypoints) - 1):
                duration = waypoints[i+1][4] - waypoints[i][4]  # time is at index 4
                if duration <= 0:
                    self.error.emit(f"Waypoint {i+1} time must be greater than waypoint {i} time")
                    return
                durations.append(duration)
            
            uref = np.array([0.0, 0.0, m*9.81, 0.0])
            
            # Check which method to use
            if method >= 1:  # Method 2: Pinocchio (FDDP or BoxFDDP)
                # Use Pinocchio-based optimization
                def callback_func(solver, seg_idx, current_xs, current_us, completed_xs, completed_us):
                    """Callback for real-time updates in Pinocchio method"""
                    if self.running:
                        # Emit iteration update
                        self.iteration_update.emit(solver.iter, solver.cost, solver.stop, seg_idx)
                        
                        # Update state display periodically
                        if solver.iter % 5 == 0:
                            # Combine with completed segments
                            combined_xs = []
                            combined_us = []
                            
                            # Add all completed segments
                            for i, (seg_xs, seg_us) in enumerate(zip(completed_xs, completed_us)):
                                if i == 0:
                                    combined_xs.extend(seg_xs)
                                    combined_us.extend(seg_us)
                                else:
                                    combined_xs.extend(seg_xs[1:])
                                    combined_us.extend(seg_us)
                            
                            # Add current segment
                            if len(combined_xs) > 0:
                                if seg_idx > 0:
                                    combined_xs.extend(current_xs[1:])
                                else:
                                    combined_xs.extend(current_xs)
                            else:
                                combined_xs.extend(current_xs)
                            combined_us.extend(current_us)
                            
                            if len(combined_xs) > 0:
                                self.state_update.emit(combined_xs, combined_us)
                
                # Use Pinocchio method (FDDP or BoxFDDP), segmented or unified
                solver_fn = solve_with_pinocchio_waypoints_unified if unified else solve_with_pinocchio_waypoints
                t0 = time.perf_counter()
                combined_xs, combined_us, all_loggers = solver_fn(
                    dt=dt,
                    waypoints=waypoints,
                    m=m,
                    I=I,
                    r_thrust=r_thrust,
                    weights=weights,
                    terminal_weights=terminal_weights,
                    bounds=bounds,
                    max_iter=max_iter,
                    use_box_solver=use_box_solver,
                    callback=callback_func,
                    running_flag=lambda: self.running
                )
                total_time = time.perf_counter() - t0
                total_iters = sum(len(logger.costs) for logger in all_loggers) if all_loggers else 0
                avg_time_per_iter = total_time / total_iters if total_iters > 0 else 0.0
                method_name = "Method 3 (BoxFDDP)" if use_box_solver else "Method 2 (FDDP)"
                timing_info = {"total_time": total_time, "total_iters": total_iters,
                              "avg_time_per_iter": avg_time_per_iter, "method": method_name}
                if self.running:
                    self.finished.emit(combined_xs, combined_us, all_loggers, timing_info)
                return
            
            # Method 1: Custom calcDiff (original implementation)
            # Store all segments' trajectories
            all_xs = []
            all_us = []
            all_loggers = []
            t0_method1 = time.perf_counter()
            total_iters_method1 = 0
            
            # Custom callback for real-time updates
            class RealTimeCallback(crocoddyl.CallbackAbstract):
                def __init__(self, thread, seg_idx, completed_segments_xs, completed_segments_us):
                    crocoddyl.CallbackAbstract.__init__(self)
                    self.thread = thread
                    self.seg_idx = seg_idx
                    self.completed_segments_xs = completed_segments_xs  # List of completed segment trajectories
                    self.completed_segments_us = completed_segments_us  # List of completed segment controls
                    self.last_update_iter = -1
                    
                def __call__(self, solver):
                    if self.thread.running:
                        # Emit iteration update with segment index
                        self.thread.iteration_update.emit(
                            solver.iter, solver.cost, solver.stop, self.seg_idx
                        )
                        
                        # Update state display periodically
                        if solver.iter % 5 == 0 and solver.iter != self.last_update_iter:
                            self.last_update_iter = solver.iter
                            # Try to get current solver's states for display
                            try:
                                if hasattr(solver, 'xs') and len(solver.xs) > 0:
                                    # Get current segment's trajectory
                                    current_xs = [np.array(x) for x in solver.xs]
                                    current_us = [np.array(u) for u in solver.us]
                                    
                                    # Combine with completed segments
                                    combined_xs = []
                                    combined_us = []
                                    
                                    # Add all completed segments
                                    for i, (seg_xs, seg_us) in enumerate(zip(self.completed_segments_xs, self.completed_segments_us)):
                                        if i == 0:
                                            # First segment: include all states and controls
                                            combined_xs.extend(seg_xs)
                                            combined_us.extend(seg_us)
                                        else:
                                            # Subsequent segments: skip first state (duplicate)
                                            combined_xs.extend(seg_xs[1:])
                                            combined_us.extend(seg_us)
                                    
                                    # Add current segment being optimized
                                    if len(combined_xs) > 0:
                                        # Skip first state of current segment if it's not the first segment
                                        if self.seg_idx > 0:
                                            combined_xs.extend(current_xs[1:])
                                        else:
                                            combined_xs.extend(current_xs)
                                    else:
                                        combined_xs.extend(current_xs)
                                    combined_us.extend(current_us)
                                    
                                    if len(combined_xs) > 0:
                                        self.thread.state_update.emit(combined_xs, combined_us)
                            except Exception:
                                pass  # Ignore errors in callback
            
            self.all_xs = []
            self.all_us = []
            
            # Initial state for first segment
            x0_seg = np.zeros(17)
            if len(waypoints) > 0:
                first_wp = waypoints[0]
                x0_seg[0:3] = [float(first_wp[0]), float(first_wp[1]), float(first_wp[2])]  # Position
                # Convert yaw to quaternion
                yaw_deg = float(first_wp[3]) if len(first_wp) > 3 else 0.0
                yaw_rad = np.radians(yaw_deg)
                x0_seg[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])  # Quaternion from yaw
                # Velocity and angular velocity start at zero
            
            # Solve each segment
            for seg_idx in range(len(durations)):
                if not self.running:
                    break
                
                duration = durations[seg_idx]
                start_wp = waypoints[seg_idx]
                end_wp = waypoints[seg_idx + 1]
                
                # Calculate number of time steps for this segment
                N = max(10, int(duration / dt))
                
                # Debug info
                start_yaw = float(start_wp[3]) if len(start_wp) > 3 else 0.0
                end_yaw = float(end_wp[3]) if len(end_wp) > 3 else 0.0
                print(f"Segment {seg_idx + 1}/{len(durations)}: {duration:.2f}s, {N} steps, "
                      f"from [{float(start_wp[0]):.2f}, {float(start_wp[1]):.2f}, {float(start_wp[2]):.2f}] yaw={start_yaw:.1f}° "
                      f"to [{float(end_wp[0]):.2f}, {float(end_wp[1]):.2f}, {float(end_wp[2]):.2f}] yaw={end_yaw:.1f}°")
                
                # For subsequent segments, use final state from previous segment
                # (x0_seg is already set from previous iteration)
                
                # Target state for this segment
                xg_seg = np.zeros(17)
                xg_seg[0:3] = [float(end_wp[0]), float(end_wp[1]), float(end_wp[2])]  # Target position
                # Convert yaw to quaternion
                yaw_deg = float(end_wp[3]) if len(end_wp) > 3 else 0.0
                yaw_rad = np.radians(yaw_deg)
                xg_seg[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])  # Quaternion from yaw
                # Target velocity and angular velocity are zero
                
                # Create models for this segment
                running = TVCRocketActionModel(dt, m, I, r_thrust,
                                             tvc_order="pitch_roll",
                                             x_goal=xg_seg, u_ref=uref,
                                             weights=weights,
                                             bounds=bounds)
                
                # Terminal model with weights from GUI (or fallback to defaults)
                tw = terminal_weights if terminal_weights else {
                    "p": 200.0, "v": 50.0, "R": 200.0, "w": 20.0, "u": 0.0, "du": 0.0
                }
                terminal = TVCRocketActionModel(dt, m, I, r_thrust,
                                              tvc_order="pitch_roll",
                                              x_goal=xg_seg, u_ref=uref,
                                              weights=tw,
                                              bounds=bounds)
                
                # Create problem for this segment
                problem = crocoddyl.ShootingProblem(x0_seg, [running]*N, terminal)
                solver = crocoddyl.SolverFDDP(problem)
                
                # Set callbacks
                logger = crocoddyl.CallbackLogger()
                # Create callback with access to completed segments for real-time updates
                callback = RealTimeCallback(self, seg_idx, all_xs.copy(), all_us.copy())
                # Use callback for all segments to show cumulative progress
                solver.setCallbacks([callback, logger])
                all_loggers.append(logger)
                
                # Initial guess
                xs_init = [x0_seg.copy() for _ in range(N+1)]
                us_init = [uref.copy() for _ in range(N)]
                
                # Solve this segment
                try:
                    solver.solve(xs_init, us_init, max_iter, False)
                    total_iters_method1 += solver.iter
                    print(f"  Segment {seg_idx + 1} solved: cost={solver.cost:.6e}, iter={solver.iter}")
                except Exception as e:
                    error_msg = f"Error solving segment {seg_idx + 1}: {str(e)}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                # Store results
                seg_xs = [np.array(x) for x in solver.xs]
                seg_us = [np.array(u) for u in solver.us]
                
                if len(seg_xs) == 0 or len(seg_us) == 0:
                    raise Exception(f"Segment {seg_idx + 1} produced empty trajectory")
                
                # Verify state continuity at connection point (for segments after the first)
                if seg_idx > 0 and len(all_xs) > 0:
                    prev_final = all_xs[-1][-1]  # Previous segment's final state
                    curr_initial = seg_xs[0]      # Current segment's initial state
                    state_diff = np.linalg.norm(prev_final - curr_initial)
                    if state_diff > 1e-6:  # Check if states match (allowing small numerical error)
                        print(f"  Warning: State discontinuity at segment {seg_idx + 1} connection: "
                              f"diff={state_diff:.2e}")
                        # Force continuity by using previous segment's final state
                        seg_xs[0] = prev_final.copy()
                
                self.all_xs.append(seg_xs)
                self.all_us.append(seg_us)
                all_xs.append(seg_xs)
                all_us.append(seg_us)
                
                # Update initial state for next segment (use final state of current segment)
                # This ensures state continuity: next segment starts exactly where current segment ends
                if seg_idx < len(durations) - 1:
                    x0_seg = seg_xs[-1].copy()  # Use final state as next segment's initial state
                    print(f"  Segment {seg_idx + 1} final state -> Segment {seg_idx + 2} initial state")
            
            if self.running:
                # Combine all segments
                combined_xs = []
                combined_us = []
                
                # Add all states and controls, connecting segments smoothly
                for i, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
                    if i == 0:
                        # First segment: include all states and controls
                        combined_xs.extend(seg_xs)
                        combined_us.extend(seg_us)
                    else:
                        # Subsequent segments: skip first state (duplicate of previous segment's last state)
                        # but keep all controls
                        combined_xs.extend(seg_xs[1:])  # Skip duplicate state
                        combined_us.extend(seg_us)  # Keep all controls
                
                # Validate combined trajectory
                if len(combined_xs) == 0 or len(combined_us) == 0:
                    raise Exception("Combined trajectory is empty")
                
                print(f"Combined trajectory: {len(combined_xs)} states, {len(combined_us)} controls")
                
                total_time_method1 = time.perf_counter() - t0_method1
                avg_time_per_iter = total_time_method1 / total_iters_method1 if total_iters_method1 > 0 else 0.0
                timing_info = {"total_time": total_time_method1, "total_iters": total_iters_method1,
                              "avg_time_per_iter": avg_time_per_iter, "method": "Method 1 (Custom calcDiff)"}
                self.finished.emit(combined_xs, combined_us, all_loggers, timing_info)
                
        except Exception as e:
            import traceback
            error_msg = f"Optimization error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            if self.running:
                self.error.emit(error_msg)
            print(error_msg)  # Also print to console for debugging
    
    def stop(self):
        """Stop optimization"""
        self.running = False


class MainWindow(QMainWindow):
    """Main window"""
    
    def __init__(self):
        super().__init__()
        self.opt_thread = None
        self.params_file_path = os.path.join(script_dir, DEFAULT_GUI_PARAMS_FILENAME)
        # Cached most-recent optimized trajectory for CSV export
        self.last_trajectory = None
        self.last_csv_path = None
        self.init_ui()
        self._update_params_file_label()
        if os.path.isfile(self.params_file_path):
            self._load_params_from_path(self.params_file_path, quiet=True)
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle('TVC Rocket Trajectory Optimization')

        # Cap the window at a standard 1080p screen, but never exceed the
        # actual available screen area (e.g. laptop 1366x768, screen with a
        # visible taskbar/dock, etc.). This guarantees every button stays
        # reachable regardless of the host display size.
        screen_w, screen_h = self._get_available_screen_size()
        max_width = min(1920, screen_w)
        max_height = min(1080, screen_h)
        self.setMaximumSize(max_width, max_height)

        # Allow the window to shrink well below the natural sizeHint of the
        # plot canvas / parameter panel (the scroll areas below handle the
        # overflow). Without this, a tall Figure sizeHint would block the
        # user from shrinking the window vertically.
        self.setMinimumSize(900, 600)

        # Default window size, capped by the maximum above
        default_width = min(1400, max_width)
        default_height = min(900, max_height)
        self.resize(default_width, default_height)

        # Center window on screen
        self.center_window()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel: parameter settings, wrapped in a scroll area so that
        # all controls remain reachable even when the window is shorter than
        # the panel's natural height.
        left_panel = self.create_parameter_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(380)
        left_scroll.setMaximumWidth(560)
        main_layout.addWidget(left_scroll, 0)

        # Right panel: display panel
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 1)

    def _get_available_screen_size(self):
        """Return the available screen size (width, height) in pixels.

        Falls back to 1920x1080 if the screen size cannot be determined.
        """
        try:
            app = QApplication.instance()
            if app is None:
                return 1920, 1080
            try:
                screen = app.desktop().availableGeometry()
                return screen.width(), screen.height()
            except AttributeError:
                try:
                    screen = app.primaryScreen().availableGeometry()
                    return screen.width(), screen.height()
                except AttributeError:
                    return 1920, 1080
        except Exception:
            return 1920, 1080

    def center_window(self):
        """Center the window on the screen"""
        try:
            screen_width, screen_height = self._get_available_screen_size()
            window_width = self.width()
            window_height = self.height()
            x = max(0, (screen_width - window_width) // 2)
            y = max(0, (screen_height - window_height) // 2)
            self.move(x, y)
        except Exception as e:
            print(f"Warning: Could not center window: {e}")
            self.move(100, 100)
        
    def create_parameter_panel(self):
        """Create parameter setting panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)  # Reduce spacing between widgets
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Title
        title = QLabel('Parameters')
        title.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(title)
        
        # Waypoints management
        waypoint_group = QGroupBox('Waypoints')
        waypoint_layout = QVBoxLayout()

        traj_pick_row = QHBoxLayout()
        traj_pick_row.addWidget(QLabel('Trajectory:'))
        self.trajectory_preset_combo = QComboBox()
        for name, _wps in TRAJECTORY_PRESETS:
            self.trajectory_preset_combo.addItem(name)
        self.trajectory_preset_combo.setToolTip(
            'Select a built-in waypoint sequence; the list below updates. Times are segment arrival times (s).')
        traj_pick_row.addWidget(self.trajectory_preset_combo, 1)
        waypoint_layout.addLayout(traj_pick_row)

        # Waypoint list widget
        if QT_AVAILABLE:
            try:
                from PyQt5.QtWidgets import QListWidget, QListWidgetItem
            except ImportError:
                from PySide2.QtWidgets import QListWidget, QListWidgetItem
        else:
            QListWidget = None
            QListWidgetItem = None
        
        self.waypoint_list = QListWidget()
        self.waypoint_list.setMaximumHeight(100)  # Reduce height
        waypoint_layout.addWidget(self.waypoint_list)
        
        # Buttons for waypoint management
        waypoint_btn_layout = QHBoxLayout()
        self.add_waypoint_btn = QPushButton('Add Waypoint')
        self.add_waypoint_btn.clicked.connect(self.add_waypoint)
        self.remove_waypoint_btn = QPushButton('Remove Selected')
        self.remove_waypoint_btn.clicked.connect(self.remove_waypoint)
        waypoint_btn_layout.addWidget(self.add_waypoint_btn)
        waypoint_btn_layout.addWidget(self.remove_waypoint_btn)
        waypoint_layout.addLayout(waypoint_btn_layout)
        
        # Current waypoint editor - arrange in 2 rows to save space
        current_wp_group = QGroupBox('Edit Waypoint')
        current_wp_layout = QGridLayout()
        current_wp_layout.setSpacing(3)  # Reduce spacing
        
        self.wp_x = QDoubleSpinBox()
        self.wp_x.setRange(-100, 100)
        self.wp_x.setValue(0.0)
        self.wp_x.setDecimals(2)
        self.wp_x.setMaximumHeight(25)
        self.wp_x.setMaximumWidth(80)
        
        self.wp_y = QDoubleSpinBox()
        self.wp_y.setRange(-100, 100)
        self.wp_y.setValue(0.0)
        self.wp_y.setDecimals(2)
        self.wp_y.setMaximumHeight(25)
        self.wp_y.setMaximumWidth(80)
        
        self.wp_z = QDoubleSpinBox()
        self.wp_z.setRange(-100, 100)
        self.wp_z.setValue(10.0)
        self.wp_z.setDecimals(2)
        self.wp_z.setMaximumHeight(25)
        self.wp_z.setMaximumWidth(80)
        
        self.wp_yaw = QDoubleSpinBox()
        self.wp_yaw.setRange(-180, 180)
        self.wp_yaw.setValue(0.0)
        self.wp_yaw.setDecimals(1)
        self.wp_yaw.setMaximumHeight(25)
        self.wp_yaw.setMaximumWidth(80)
        
        self.wp_time = QDoubleSpinBox()
        self.wp_time.setRange(0.0, 1000.0)
        self.wp_time.setValue(5.0)
        self.wp_time.setDecimals(2)
        self.wp_time.setMaximumHeight(25)
        self.wp_time.setMaximumWidth(80)
        
        # Two pairs per row (4 grid cols)
        current_wp_layout.addWidget(QLabel('X (m):'), 0, 0)
        current_wp_layout.addWidget(self.wp_x, 0, 1)
        current_wp_layout.addWidget(QLabel('Y (m):'), 0, 2)
        current_wp_layout.addWidget(self.wp_y, 0, 3)
        current_wp_layout.addWidget(QLabel('Z (m):'), 1, 0)
        current_wp_layout.addWidget(self.wp_z, 1, 1)
        current_wp_layout.addWidget(QLabel('Yaw (°):'), 1, 2)
        current_wp_layout.addWidget(self.wp_yaw, 1, 3)
        current_wp_layout.addWidget(QLabel('Arrival Time (s):'), 2, 0)
        current_wp_layout.addWidget(self.wp_time, 2, 1)
        
        self.update_waypoint_btn = QPushButton('Update Selected')
        self.update_waypoint_btn.clicked.connect(self.update_waypoint)
        self.update_waypoint_btn.setMaximumHeight(30)
        current_wp_layout.addWidget(self.update_waypoint_btn, 2, 2, 1, 2)
        
        current_wp_group.setLayout(current_wp_layout)
        waypoint_layout.addWidget(current_wp_group)
        
        # Connect waypoint list selection
        self.waypoint_list.itemSelectionChanged.connect(self.on_waypoint_selected)
        
        waypoint_group.setLayout(waypoint_layout)
        layout.addWidget(waypoint_group)
        
        # Default: Grasshopper preset. Format: [x, y, z, yaw_deg, arrival_time]
        _, grass = TRAJECTORY_PRESETS[0]
        self.waypoints = [_normalize_waypoint_row(w) for w in grass]
        self.update_waypoint_list()
        self.trajectory_preset_combo.setCurrentIndex(0)
        self.trajectory_preset_combo.currentIndexChanged.connect(self.on_trajectory_preset_changed)
        
        # Optimization method selection (method combo on first row; Unified options on second — saves width)
        method_group = QGroupBox('Optimization Method')
        method_outer = QVBoxLayout()
        method_outer.setSpacing(4)
        method_row1 = QHBoxLayout()
        method_row1.setSpacing(3)
        method_row2 = QHBoxLayout()
        method_row2.setSpacing(8)

        self.method_combo = QComboBox()
        self.method_combo.addItem('Method 1: Custom calcDiff (Slower, Numerical)')
        self.method_combo.addItem('Method 2: Pinocchio + FDDP (Penalty constraints)')
        self.method_combo.addItem('Method 3: Pinocchio + BoxFDDP (Native control bounds)')
        self.method_combo.addItem('Method 4: Acados (Native constraints)')
        self.method_combo.addItem('Method 5: Acados min-time (free segment duration)')
        self.method_combo.addItem('Method 6: Spannagl min-fuel FFTF (free t_f per leg)')
        self.method_combo.addItem('Method 7: Acados free t_f + EXTERNAL (thrust/TVC/time)')
        method_row1.addWidget(QLabel('Method:'))
        method_row1.addWidget(self.method_combo, 1)
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        self.unified_checkbox = QCheckBox('Unified (merge all segments)')
        self.unified_checkbox.setToolTip('Merge all waypoint segments into one optimization problem (Method 2/3/4)')
        self.unified_checkbox.setChecked(False)
        self.unified_checkbox.setEnabled(False)  # Enabled when Method 2/3/4 selected
        method_row2.addWidget(self.unified_checkbox)
        self.unified_interp_guess_checkbox = QCheckBox('Unified (Acados): interpolate x initial guess')
        self.unified_interp_guess_checkbox.setToolTip(
            'If checked, Method 4 unified uses linearly interpolated state over each segment as the initial '
            'guess. Default off: all nodes use the start state x0.'
        )
        self.unified_interp_guess_checkbox.setChecked(False)
        self.unified_interp_guess_checkbox.setEnabled(False)
        method_row2.addWidget(self.unified_interp_guess_checkbox)
        method_row2.addStretch(1)
        self.unified_checkbox.stateChanged.connect(self._refresh_unified_interp_guess_enabled)
        self.method_combo.currentIndexChanged.connect(self._update_unified_checkbox_state)

        method_outer.addLayout(method_row1)
        method_outer.addLayout(method_row2)
        method_group.setLayout(method_outer)
        layout.addWidget(method_group)
        self._update_unified_checkbox_state(self.method_combo.currentIndex())
        
        # Optimization parameters
        opt_group = QGroupBox('Optimization Parameters')
        opt_layout = QGridLayout()
        opt_layout.setSpacing(3)  # Reduce spacing
        
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 0.1)
        self.dt_spin.setValue(0.05)
        self.dt_spin.setSingleStep(0.01)
        self.dt_spin.setDecimals(3)
        self.dt_spin.setMaximumHeight(25)
        self.dt_spin.setMaximumWidth(100)
        
        self.N_spin = QSpinBox()
        self.N_spin.setRange(10, 500)
        self.N_spin.setValue(100)
        self.N_spin.setMaximumHeight(25)
        self.N_spin.setMaximumWidth(100)
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setValue(100)
        self.max_iter_spin.setMaximumHeight(25)
        self.max_iter_spin.setMaximumWidth(100)
        
        # Two pairs per row (4 grid cols)
        opt_layout.addWidget(QLabel('Time Step (s):'), 0, 0)
        opt_layout.addWidget(self.dt_spin, 0, 1)
        opt_layout.addWidget(QLabel('Time Steps:'), 0, 2)
        opt_layout.addWidget(self.N_spin, 0, 3)
        opt_layout.addWidget(QLabel('Max Iter:'), 1, 0)
        opt_layout.addWidget(self.max_iter_spin, 1, 1)
        
        opt_group.setLayout(opt_layout)
        
        # Tab widget for parameters
        params_tabs = QTabWidget()
        
        # Tab 1: Optimization + physical parameters (single tab)
        opt_tab = QWidget()
        opt_tab_layout = QVBoxLayout()
        opt_tab_layout.addWidget(opt_group)
        opt_tab.setLayout(opt_tab_layout)
        params_tabs.addTab(opt_tab, 'Optimization')
        
        # Running cost weights
        cost_group = QGroupBox('Running Cost')
        cost_layout = QGridLayout()
        cost_layout.setSpacing(3)  # Reduce spacing
        
        self.w_p = QDoubleSpinBox()
        self.w_p.setRange(0, 1000)
        self.w_p.setValue(1.0)
        self.w_p.setDecimals(3)
        self.w_p.setMaximumHeight(25)
        self.w_p.setMaximumWidth(100)  # Reduce width
        
        self.w_v = QDoubleSpinBox()
        self.w_v.setRange(0, 1000)
        self.w_v.setValue(01.0)
        self.w_v.setDecimals(3)
        self.w_v.setMaximumHeight(25)
        self.w_v.setMaximumWidth(100)
        
        self.w_R = QDoubleSpinBox()
        self.w_R.setRange(0, 1000)
        self.w_R.setValue(1.0)
        self.w_R.setDecimals(3)
        self.w_R.setMaximumHeight(25)
        self.w_R.setMaximumWidth(100)
        
        self.w_yaw = QDoubleSpinBox()
        self.w_yaw.setRange(0, 1000)
        self.w_yaw.setValue(1.0)
        self.w_yaw.setDecimals(3)
        self.w_yaw.setMaximumHeight(25)
        self.w_yaw.setMaximumWidth(100)
        self.w_yaw.setToolTip('Separate weight for yaw (default same as Attitude R)')
        
        self.w_w = QDoubleSpinBox()
        self.w_w.setRange(0, 1000)
        self.w_w.setValue(0.1)
        self.w_w.setDecimals(3)
        self.w_w.setMaximumHeight(25)
        self.w_w.setMaximumWidth(100)
        
        self.w_u = QDoubleSpinBox()
        self.w_u.setRange(0, 100)
        self.w_u.setValue(1.0)
        self.w_u.setDecimals(3)
        self.w_u.setMaximumHeight(25)
        self.w_u.setMaximumWidth(100)
        
        self.w_du = QDoubleSpinBox()
        self.w_du.setRange(0, 100)
        self.w_du.setValue(1.0)  # Default: enable control rate penalty
        self.w_du.setDecimals(3)
        self.w_du.setMaximumHeight(25)
        self.w_du.setMaximumWidth(100)
        
        # Constraint penalty coefficients
        self.k_bound = QDoubleSpinBox()
        self.k_bound.setRange(0, 10000)
        self.k_bound.setValue(200.0)
        self.k_bound.setDecimals(1)
        self.k_bound.setMaximumHeight(25)
        self.k_bound.setMaximumWidth(100)
        
        self.k_state_bound = QDoubleSpinBox()
        self.k_state_bound.setRange(0, 10000)
        self.k_state_bound.setValue(20.0)  # Lower default to avoid constraint gradient dominating position cost
        self.k_state_bound.setDecimals(1)
        self.k_state_bound.setMaximumHeight(25)
        self.k_state_bound.setMaximumWidth(100)
        
        # Two pairs per row (4 grid cols)
        cost_layout.addWidget(QLabel('Position (p):'), 0, 0)
        cost_layout.addWidget(self.w_p, 0, 1)
        cost_layout.addWidget(QLabel('Velocity (v):'), 0, 2)
        cost_layout.addWidget(self.w_v, 0, 3)
        cost_layout.addWidget(QLabel('Attitude (R):'), 1, 0)
        cost_layout.addWidget(self.w_R, 1, 1)
        cost_layout.addWidget(QLabel('Yaw:'), 1, 2)
        cost_layout.addWidget(self.w_yaw, 1, 3)
        cost_layout.addWidget(QLabel('Ang Vel (w):'), 2, 0)
        cost_layout.addWidget(self.w_w, 2, 1)
        cost_layout.addWidget(QLabel('Control (u):'), 3, 0)
        cost_layout.addWidget(self.w_u, 3, 1)
        cost_layout.addWidget(QLabel('Ctrl Chg (du):'), 3, 2)
        cost_layout.addWidget(self.w_du, 3, 3)
        cost_layout.addWidget(QLabel('Ctrl Bound (k_bound):'), 4, 0)
        cost_layout.addWidget(self.k_bound, 4, 1)
        cost_layout.addWidget(QLabel('State Bound (k_sb):'), 4, 2)
        cost_layout.addWidget(self.k_state_bound, 4, 3)
        self.schedule_ref_checkbox = QCheckBox('On-time arrival (schedule_ref)')
        self.schedule_ref_checkbox.setChecked(False)
        self.schedule_ref_checkbox.setToolTip('Checked: time-interpolated ref for on-time arrival; unchecked: constant goal ref, may arrive early')
        cost_layout.addWidget(self.schedule_ref_checkbox, 5, 0, 1, 4)
        
        # Acados: first-order actuator dynamics (per-channel time constants)
        self.actuator_dynamics_checkbox = QCheckBox('Actuator dynamics (first-order) [Acados]')
        self.actuator_dynamics_checkbox.setChecked(False)
        self.actuator_dynamics_checkbox.setToolTip('When enabled: model actuator as tau*u_dot = u_cmd - u_actual per channel. Acados only.')
        cost_layout.addWidget(self.actuator_dynamics_checkbox, 6, 0, 1, 4)
        self.tau_pitch_spin = QDoubleSpinBox()
        self.tau_pitch_spin.setRange(0.001, 1.0)
        self.tau_pitch_spin.setValue(0.05)
        self.tau_pitch_spin.setDecimals(3)
        self.tau_pitch_spin.setSingleStep(0.01)
        self.tau_pitch_spin.setMaximumHeight(25)
        self.tau_pitch_spin.setMaximumWidth(80)
        self.tau_pitch_spin.setToolTip('Pitch channel time constant (s)')
        self.tau_roll_spin = QDoubleSpinBox()
        self.tau_roll_spin.setRange(0.001, 1.0)
        self.tau_roll_spin.setValue(0.05)
        self.tau_roll_spin.setDecimals(3)
        self.tau_roll_spin.setSingleStep(0.01)
        self.tau_roll_spin.setMaximumHeight(25)
        self.tau_roll_spin.setMaximumWidth(80)
        self.tau_roll_spin.setToolTip('Roll channel time constant (s)')
        self.tau_T_spin = QDoubleSpinBox()
        self.tau_T_spin.setRange(0.001, 10.0)
        self.tau_T_spin.setValue(0.5)
        self.tau_T_spin.setDecimals(3)
        self.tau_T_spin.setSingleStep(0.01)
        self.tau_T_spin.setMaximumHeight(25)
        self.tau_T_spin.setMaximumWidth(80)
        self.tau_T_spin.setToolTip('Thrust channel time constant (s)')
        self.tau_yaw_spin = QDoubleSpinBox()
        self.tau_yaw_spin.setRange(0.001, 1.0)
        self.tau_yaw_spin.setValue(0.05)
        self.tau_yaw_spin.setDecimals(3)
        self.tau_yaw_spin.setSingleStep(0.01)
        self.tau_yaw_spin.setMaximumHeight(25)
        self.tau_yaw_spin.setMaximumWidth(80)
        self.tau_yaw_spin.setToolTip('Yaw channel time constant (s)')
        cost_layout.addWidget(QLabel('tau pitch (s):'), 7, 0)
        cost_layout.addWidget(self.tau_pitch_spin, 7, 1)
        cost_layout.addWidget(QLabel('tau roll (s):'), 7, 2)
        cost_layout.addWidget(self.tau_roll_spin, 7, 3)
        cost_layout.addWidget(QLabel('tau T (s):'), 8, 0)
        cost_layout.addWidget(self.tau_T_spin, 8, 1)
        cost_layout.addWidget(QLabel('tau yaw (s):'), 8, 2)
        cost_layout.addWidget(self.tau_yaw_spin, 8, 3)
        
        cost_group.setLayout(cost_layout)

        # Bounds on optimized segment duration T_seg / t_f (Method 5 min-time, 6 Spannagl, 7 free-tf)
        self.min_time_duration_group = QGroupBox(
            'Segment duration bounds (Methods 5–7: T_min, T_max scale)'
        )
        mt_dur_layout = QGridLayout()
        mt_dur_layout.setSpacing(3)
        self.min_time_T_min_spin = QDoubleSpinBox()
        self.min_time_T_min_spin.setRange(0.02, 300.0)
        self.min_time_T_min_spin.setValue(0.15)
        self.min_time_T_min_spin.setDecimals(3)
        self.min_time_T_min_spin.setSingleStep(0.05)
        self.min_time_T_min_spin.setMaximumHeight(25)
        self.min_time_T_min_spin.setMaximumWidth(100)
        self.min_time_T_min_spin.setToolTip(
            'Lower bound on physical segment duration per leg [s]. Acados: state T_seg; '
            'Spannagl/FFTF: lower bound on optimized t_f.'
        )
        self.min_time_T_max_scale_spin = QDoubleSpinBox()
        self.min_time_T_max_scale_spin.setRange(0.05, 10.0)
        self.min_time_T_max_scale_spin.setValue(1.0)
        self.min_time_T_max_scale_spin.setDecimals(3)
        self.min_time_T_max_scale_spin.setSingleStep(0.05)
        self.min_time_T_max_scale_spin.setMaximumHeight(25)
        self.min_time_T_max_scale_spin.setMaximumWidth(100)
        self.min_time_T_max_scale_spin.setToolTip(
            'Upper bound scale: T_max = (waypoint time gap) × this factor [—]. '
            'Larger allows longer segment; 1.0 uses nominal schedule gap as cap.'
        )
        mt_dur_layout.addWidget(QLabel('T_min (s):'), 0, 0)
        mt_dur_layout.addWidget(self.min_time_T_min_spin, 0, 1)
        mt_dur_layout.addWidget(QLabel('T_max scale (× gap):'), 0, 2)
        mt_dur_layout.addWidget(self.min_time_T_max_scale_spin, 0, 3)
        self.min_time_duration_group.setLayout(mt_dur_layout)
        self.min_time_duration_group.setVisible(False)
        
        # Terminal cost: same diagonal layout as running weights on physical state, × one multiplier.
        terminal_cost_group = QGroupBox('Terminal Cost')
        terminal_cost_layout = QGridLayout()
        terminal_cost_layout.setSpacing(3)

        terminal_cost_layout.addWidget(
            QLabel('Multiplier (terminal = running × k):'), 0, 0)
        self.terminal_cost_multiplier_spin = QDoubleSpinBox()
        self.terminal_cost_multiplier_spin.setRange(0.01, 10000.0)
        self.terminal_cost_multiplier_spin.setValue(200.0)
        self.terminal_cost_multiplier_spin.setDecimals(2)
        self.terminal_cost_multiplier_spin.setMaximumHeight(25)
        self.terminal_cost_multiplier_spin.setMaximumWidth(100)
        self.terminal_cost_multiplier_spin.setToolTip(
            'Terminal / waypoint-stage state weights use the same layout as Running Cost, multiplied by k. '
            'Typical: 100–1000 so the Mayer term dominates accumulated running cost.')
        terminal_cost_layout.addWidget(self.terminal_cost_multiplier_spin, 0, 1)
        self.terminal_constraint_checkbox = QCheckBox('Terminal position equality (p_N=p_g)')
        self.terminal_constraint_checkbox.setChecked(False)
        self.terminal_constraint_checkbox.setToolTip('Enforce exact terminal position. May cause infeasibility, use with caution')
        terminal_cost_layout.addWidget(self.terminal_constraint_checkbox, 1, 0, 1, 4)
        self.waypoint_terminal_checkbox = QCheckBox('Waypoint terminal cost (segment mode for multi-WP)')
        self.waypoint_terminal_checkbox.setChecked(True)
        self.waypoint_terminal_checkbox.setToolTip('Checked: terminal cost at each intermediate waypoint for arrival. Acados uses segment mode')
        terminal_cost_layout.addWidget(self.waypoint_terminal_checkbox, 2, 0, 1, 4)
        
        terminal_cost_group.setLayout(terminal_cost_layout)
        
        # Tab 2: Cost Weights
        cost_tab = QWidget()
        cost_tab_layout = QVBoxLayout()
        cost_tab_layout.addWidget(cost_group)
        cost_tab_layout.addWidget(self.min_time_duration_group)
        cost_tab_layout.addWidget(terminal_cost_group)
        cost_tab_layout.addStretch()
        cost_tab.setLayout(cost_tab_layout)
        params_tabs.addTab(cost_tab, 'Cost Weights')
        
        # Constraints (control + state limits in one tab)
        constraints_group = QGroupBox('Constraints')
        constraints_layout = QGridLayout()
        constraints_layout.setSpacing(3)

        self.th_p_max = QDoubleSpinBox()
        self.th_p_max.setRange(0, 90)  # Range in degrees
        self.th_p_max.setValue(10.0)  # 10 degrees
        self.th_p_max.setDecimals(1)
        self.th_p_max.setMaximumHeight(25)
        self.th_p_max.setMaximumWidth(100)

        self.th_r_max = QDoubleSpinBox()
        self.th_r_max.setRange(0, 90)  # Range in degrees
        self.th_r_max.setValue(10.0)  # 10 degrees
        self.th_r_max.setDecimals(1)
        self.th_r_max.setMaximumHeight(25)
        self.th_r_max.setMaximumWidth(100)

        self.T_max = QDoubleSpinBox()
        self.T_max.setRange(0, 100)
        self.T_max.setValue(25.0)
        self.T_max.setDecimals(2)
        self.T_max.setMaximumHeight(25)
        self.T_max.setMaximumWidth(100)

        self.tau_yaw_max = QDoubleSpinBox()
        self.tau_yaw_max.setRange(0, 10)
        self.tau_yaw_max.setValue(1.0)
        self.tau_yaw_max.setDecimals(2)
        self.tau_yaw_max.setMaximumHeight(25)
        self.tau_yaw_max.setMaximumWidth(100)

        self.v_horizontal_max = QDoubleSpinBox()
        self.v_horizontal_max.setRange(0, 100)
        self.v_horizontal_max.setValue(1.0)
        self.v_horizontal_max.setDecimals(1)
        self.v_horizontal_max.setMaximumHeight(25)
        self.v_horizontal_max.setMaximumWidth(100)

        self.v_vertical_max = QDoubleSpinBox()
        self.v_vertical_max.setRange(0, 100)
        self.v_vertical_max.setValue(3.0)
        self.v_vertical_max.setDecimals(1)
        self.v_vertical_max.setMaximumHeight(25)
        self.v_vertical_max.setMaximumWidth(100)

        self.roll_max = QDoubleSpinBox()
        self.roll_max.setRange(0, 180)
        self.roll_max.setValue(10.0)
        self.roll_max.setDecimals(1)
        self.roll_max.setMaximumHeight(25)
        self.roll_max.setMaximumWidth(100)

        self.pitch_max = QDoubleSpinBox()
        self.pitch_max.setRange(0, 180)
        self.pitch_max.setValue(10.0)
        self.pitch_max.setDecimals(1)
        self.pitch_max.setMaximumHeight(25)
        self.pitch_max.setMaximumWidth(100)

        self.yaw_max = QDoubleSpinBox()
        self.yaw_max.setRange(0, 180)
        self.yaw_max.setValue(180.0)
        self.yaw_max.setDecimals(1)
        self.yaw_max.setMaximumHeight(25)
        self.yaw_max.setMaximumWidth(100)

        self.w_max = QDoubleSpinBox()
        self.w_max.setRange(0, 10)
        self.w_max.setValue(2.0)
        self.w_max.setDecimals(2)
        self.w_max.setMaximumHeight(25)
        self.w_max.setMaximumWidth(100)

        row = 0
        constraints_layout.addWidget(QLabel('TVC Pitch (°):'), row, 0)
        constraints_layout.addWidget(self.th_p_max, row, 1)
        constraints_layout.addWidget(QLabel('TVC Roll (°):'), row, 2)
        constraints_layout.addWidget(self.th_r_max, row, 3)
        row += 1
        constraints_layout.addWidget(QLabel('Thrust (N):'), row, 0)
        constraints_layout.addWidget(self.T_max, row, 1)
        constraints_layout.addWidget(QLabel('Yaw torque (N·m):'), row, 2)
        constraints_layout.addWidget(self.tau_yaw_max, row, 3)
        row += 1
        constraints_layout.addWidget(QLabel('V_xy (m/s):'), row, 0)
        constraints_layout.addWidget(self.v_horizontal_max, row, 1)
        constraints_layout.addWidget(QLabel('V_z (m/s):'), row, 2)
        constraints_layout.addWidget(self.v_vertical_max, row, 3)
        row += 1
        constraints_layout.addWidget(QLabel('Roll (°):'), row, 0)
        constraints_layout.addWidget(self.roll_max, row, 1)
        constraints_layout.addWidget(QLabel('Pitch (°):'), row, 2)
        constraints_layout.addWidget(self.pitch_max, row, 3)
        row += 1
        constraints_layout.addWidget(QLabel('Yaw (°):'), row, 0)
        constraints_layout.addWidget(self.yaw_max, row, 1)
        constraints_layout.addWidget(QLabel('Ang Vel (rad/s):'), row, 2)
        constraints_layout.addWidget(self.w_max, row, 3)

        constraints_group.setLayout(constraints_layout)

        constraints_tab = QWidget()
        constraints_tab_layout = QVBoxLayout()
        constraints_tab_layout.addWidget(constraints_group)
        constraints_tab_layout.addStretch()
        constraints_tab.setLayout(constraints_tab_layout)
        params_tabs.addTab(constraints_tab, 'Constraints')
        
        # Physical parameters
        physics_group = QGroupBox('Physical Parameters')
        physics_layout = QGridLayout()
        physics_layout.setSpacing(3)
        
        # Mass
        self.mass = QDoubleSpinBox()
        self.mass.setRange(0.01, 10.0)
        self.mass.setValue(0.6)
        self.mass.setDecimals(3)
        self.mass.setMaximumHeight(25)
        self.mass.setMaximumWidth(100)
        
        # Moment of inertia (diagonal components)
        self.Ixx = QDoubleSpinBox()
        self.Ixx.setRange(0.0001, 1.0)
        self.Ixx.setValue(0.02)
        self.Ixx.setDecimals(2)
        self.Ixx.setMaximumHeight(25)
        self.Ixx.setMaximumWidth(100)
        
        self.Iyy = QDoubleSpinBox()
        self.Iyy.setRange(0.0001, 1.0)
        self.Iyy.setValue(0.02)
        self.Iyy.setDecimals(2)
        self.Iyy.setMaximumHeight(25)
        self.Iyy.setMaximumWidth(100)
        
        self.Izz = QDoubleSpinBox()
        self.Izz.setRange(0.0001, 1.0)
        self.Izz.setValue(0.01)
        self.Izz.setDecimals(2)
        self.Izz.setMaximumHeight(25)
        self.Izz.setMaximumWidth(100)
        
        # Thrust position (r_thrust)
        self.r_thrust_x = QDoubleSpinBox()
        self.r_thrust_x.setRange(-1.0, 1.0)
        self.r_thrust_x.setValue(0.0)
        self.r_thrust_x.setDecimals(2)
        self.r_thrust_x.setMaximumHeight(25)
        self.r_thrust_x.setMaximumWidth(100)
        
        self.r_thrust_y = QDoubleSpinBox()
        self.r_thrust_y.setRange(-1.0, 1.0)
        self.r_thrust_y.setValue(0.0)
        self.r_thrust_y.setDecimals(2)
        self.r_thrust_y.setMaximumHeight(25)
        self.r_thrust_y.setMaximumWidth(100)
        
        self.r_thrust_z = QDoubleSpinBox()
        self.r_thrust_z.setRange(-1.0, 1.0)
        self.r_thrust_z.setValue(-0.2)
        self.r_thrust_z.setDecimals(2)
        self.r_thrust_z.setMaximumHeight(25)
        self.r_thrust_z.setMaximumWidth(100)
        
        # Two pairs per row (4 grid cols)
        physics_layout.addWidget(QLabel('Mass (kg):'), 0, 0)
        physics_layout.addWidget(self.mass, 0, 1)
        physics_layout.addWidget(QLabel('Ixx (kg·m²):'), 0, 2)
        physics_layout.addWidget(self.Ixx, 0, 3)
        physics_layout.addWidget(QLabel('Iyy (kg·m²):'), 1, 0)
        physics_layout.addWidget(self.Iyy, 1, 1)
        physics_layout.addWidget(QLabel('Izz (kg·m²):'), 1, 2)
        physics_layout.addWidget(self.Izz, 1, 3)
        physics_layout.addWidget(QLabel('Thrust X (m):'), 2, 0)
        physics_layout.addWidget(self.r_thrust_x, 2, 1)
        physics_layout.addWidget(QLabel('Thrust Y (m):'), 2, 2)
        physics_layout.addWidget(self.r_thrust_y, 2, 3)
        physics_layout.addWidget(QLabel('Thrust Z (m):'), 3, 0)
        physics_layout.addWidget(self.r_thrust_z, 3, 1)
        
        physics_group.setLayout(physics_layout)
        opt_tab_layout.addWidget(physics_group)
        opt_tab_layout.addStretch()

        layout.addWidget(params_tabs)
        
        # Default parameters per method (state/control constraints equal across all methods)
        self.DEFAULT_PARAMS = {
            0: {  # Method 1: Custom calcDiff
                "w_p": 1.0, "w_v": 0.2, "w_R": 0.5, "w_yaw": 0.5, "w_w": 0.1,
                "w_u": 0.5, "w_du": 0.5,
                "terminal_cost_multiplier": 200.0,
                "k_bound": 200.0, "k_state_bound": 200.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 1.0, "v_vertical_max": 3.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 180.0, "w_max": 2.0,
            },
            1: {  # Method 2: FDDP - higher w_R and w_yaw to suppress yaw drift (Pinocchio cost structure)
                "w_p": 1.0, "w_v": 0.2, "w_R": 2.0, "w_yaw": 2.0, "w_w": 0.1,
                "w_u": 0.5, "w_du": 0.5,
                "terminal_cost_multiplier": 200.0,
                "k_bound": 200.0, "k_state_bound": 200.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 1.0, "v_vertical_max": 3.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
            2: {  # Method 3: BoxFDDP - higher w_R and w_yaw to suppress yaw drift
                "w_p": 1.0, "w_v": 0.2, "w_R": 2.0, "w_yaw": 2.0, "w_w": 0.1,
                "w_u": 0.5, "w_du": 0.5,
                "terminal_cost_multiplier": 200.0,
                "k_bound": 200.0, "k_state_bound": 200.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 1.0, "v_vertical_max": 3.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
            3: {  # Method 4: Acados - native constraints
                "w_p": 1.0, "w_v": 0.2, "w_R": 0.5, "w_yaw": 0.5, "w_w": 0.1,
                "actuator_dynamics": False, "actuator_tau": [0.05, 0.05, 0.05, 0.05],
                "w_u": 0.5, "w_du": 0.5, "schedule_ref": False,
                "terminal_cost_multiplier": 200.0, "terminal_constraint": False, "waypoint_terminal_cost": True,
                "unified_interp_initial_guess": False,
                "k_bound": 200.0, "k_state_bound": 20.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 2.5, "v_vertical_max": 2.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
            4: {  # Method 5: Acados minimum-time (same knobs as Method 4 + min-time NLP weights)
                "w_p": 1.0, "w_v": 0.2, "w_R": 0.5, "w_yaw": 0.5, "w_w": 0.1,
                "actuator_dynamics": False, "actuator_tau": [0.05, 0.05, 0.05, 0.05],
                "w_u": 0.5, "w_du": 0.5, "schedule_ref": False,
                "terminal_cost_multiplier": 200.0, "terminal_constraint": False, "waypoint_terminal_cost": True,
                "unified_interp_initial_guess": False,
                "min_time_weight": 1.0,
                "min_time_T_min": 0.15,
                "min_time_T_max_scale": 1.0,
                "k_bound": 200.0, "k_state_bound": 20.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 2.5, "v_vertical_max": 2.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
            5: {  # Method 6: Spannagl-style FFTF (weights mostly from GUI; extras for NLP)
                "w_p": 1.0, "w_v": 0.2, "w_R": 0.5, "w_yaw": 0.5, "w_w": 0.1,
                "actuator_dynamics": False, "actuator_tau": [0.05, 0.05, 0.05, 0.05],
                "w_u": 0.5, "w_du": 0.5, "schedule_ref": False,
                "terminal_cost_multiplier": 200.0, "terminal_constraint": False, "waypoint_terminal_cost": True,
                "unified_interp_initial_guess": False,
                "min_time_T_min": 0.15,
                "min_time_T_max_scale": 1.0,
                "g": 9.81,
                "spannagl_exact_terminal": True,
                "spannagl_ptol": 0.05,
                "spannagl_vtol": 0.05,
                "spannagl_min_tf_reg": 0.0,
                "spannagl_glideslope": False,
                "spannagl_gamma_deg": 15.0,
                "spannagl_udot_max": 500.0,
                "spannagl_lambda_yaw": 200.0,
                "spannagl_nlp_solver": "acados",
                "k_bound": 200.0, "k_state_bound": 20.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 2.5, "v_vertical_max": 2.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
            6: {  # Method 7: pseudo-time + tf state, EXTERNAL running/terminal cost (reference-style)
                "w_p": 1.0, "w_v": 0.2, "w_R": 0.5, "w_yaw": 0.5, "w_w": 0.1,
                "actuator_dynamics": False, "actuator_tau": [0.05, 0.05, 0.05, 0.05],
                "w_u": 0.5, "w_du": 0.5, "schedule_ref": False,
                "terminal_cost_multiplier": 200.0, "terminal_constraint": False, "waypoint_terminal_cost": True,
                "unified_interp_initial_guess": False,
                "min_time_T_min": 0.15,
                "min_time_T_max_scale": 1.0,
                "free_tf_w_T": 1.0,
                "free_tf_w_tvc": 0.01,
                "free_tf_w_tau_yaw": 0.01,
                "free_tf_w_terminal_time": 10.0,
                "free_tf_include_state_terminal": True,
                "k_bound": 200.0, "k_state_bound": 20.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 2.5, "v_vertical_max": 2.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
        }
        
        # Default optimization method: Method 4 (Acados); applied after all widgets exist
        self.method_combo.setCurrentIndex(3)
        
        # Start optimization + load parameters (one row)
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton('Start Optimization')
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.run_btn.setMaximumHeight(35)  # Reduce button height
        self.run_btn.clicked.connect(self.start_optimization)
        button_layout.addWidget(self.run_btn)
        self.btn_load_params = QPushButton('Load parameters...')
        self.btn_load_params.setToolTip('Load settings from a JSON file; subsequent Save writes to this file')
        self.btn_load_params.clicked.connect(self.load_parameters)
        self.btn_load_params.setMaximumHeight(35)
        button_layout.addWidget(self.btn_load_params)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)

        params_io_layout = QHBoxLayout()
        self.btn_save_params = QPushButton('Save parameters')
        self.btn_save_params.setToolTip(f'Write current settings to the active file (default: {DEFAULT_GUI_PARAMS_FILENAME})')
        self.btn_save_params.clicked.connect(self.save_parameters)
        self.btn_save_params_as = QPushButton('Save parameters as...')
        self.btn_save_params_as.setToolTip('Save to a new path; that path becomes the active file for Save')
        self.btn_save_params_as.clicked.connect(self.save_parameters_as)
        params_io_layout.addWidget(self.btn_save_params)
        params_io_layout.addWidget(self.btn_save_params_as)
        layout.addLayout(params_io_layout)
        self.params_file_label = QLabel()
        self.params_file_label.setWordWrap(True)
        self.params_file_label.setStyleSheet('color: #555;')
        layout.addWidget(self.params_file_label)

        # Trajectory export row: write the optimized trajectory to a CSV file
        # suitable for PX4 offboard / setpoint-trajectory tracking.
        traj_io_layout = QHBoxLayout()
        self.btn_save_traj_csv = QPushButton('Save trajectory (CSV)...')
        self.btn_save_traj_csv.setToolTip(
            'Export the most recent optimized trajectory as a CSV file '
            '(time, position, velocity, quaternion, body rates, Euler, control). '
            'The CSV can be replayed by a PX4 offboard / trajectory tracker.'
        )
        self.btn_save_traj_csv.setEnabled(False)
        self.btn_save_traj_csv.clicked.connect(self.save_trajectory_csv)
        traj_io_layout.addWidget(self.btn_save_traj_csv)

        self.frame_combo = QComboBox()
        self.frame_combo.addItem('Frame: ENU (as planned)')
        self.frame_combo.addItem('Frame: NED (PX4 default)')
        self.frame_combo.setToolTip(
            'Coordinate frame for the exported CSV.\n'
            'ENU = as used by the planner (x East, y North, z Up).\n'
            'NED = PX4 internal convention (x North, y East, z Down).'
        )
        traj_io_layout.addWidget(self.frame_combo)
        layout.addLayout(traj_io_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(20)  # Reduce height
        layout.addWidget(self.progress)
        
        # Status information
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)  # Reduce height
        self.status_text.setReadOnly(True)
        layout.addWidget(QLabel('Status:'))
        layout.addWidget(self.status_text)
        
        layout.addStretch()
        
        return panel
    
    def create_display_panel(self):
        """Create display panel - all states, controls and cost on one page"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create single canvas with all subplots. figsize is only used to
        # set the figure's natural aspect ratio; the canvas is allowed to
        # shrink below it so the whole window can stay within the screen.
        self.fig = Figure(figsize=(12, 7))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(400, 300)
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.35, wspace=0.3)
        self.fig.suptitle('TVC Rocket Trajectory Optimization', 
                         fontsize=16, fontweight='bold', y=0.995)
        
        # First row: 3D trajectory and convergence curve
        # 1. 3D position trajectory (occupies 2 positions)
        self.ax_3d = self.fig.add_subplot(gs[0, 0:2], projection='3d')
        self.ax_3d.set_xlabel('X (m)', fontsize=10)
        self.ax_3d.set_ylabel('Y (m)', fontsize=10)
        self.ax_3d.set_zlabel('Z (m)', fontsize=10)
        self.ax_3d.set_title('3D Position Trajectory', fontsize=11, fontweight='bold')
        self.ax_3d.grid(True, alpha=0.3)
        
        # 2. Cost convergence curve (occupies 2 positions)
        self.ax_cost = self.fig.add_subplot(gs[0, 2:4])
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        
        # Second row: position states
        # 3. Position
        self.ax_pos = self.fig.add_subplot(gs[1, 0])
        self.ax_pos.set_xlabel('Time (s)', fontsize=9)
        self.ax_pos.set_ylabel('Position (m)', fontsize=9)
        self.ax_pos.set_title('Position', fontsize=10, fontweight='bold')
        self.ax_pos.grid(True, alpha=0.3)
        
        # 4. Velocity
        self.ax_vel = self.fig.add_subplot(gs[1, 1])
        self.ax_vel.set_xlabel('Time (s)', fontsize=9)
        self.ax_vel.set_ylabel('Velocity (m/s)', fontsize=9)
        self.ax_vel.set_title('Linear Velocity', fontsize=10, fontweight='bold')
        self.ax_vel.grid(True, alpha=0.3)
        
        # 5. Euler angles (left of angular velocity)
        self.ax_euler = self.fig.add_subplot(gs[1, 2])
        self.ax_euler.set_xlabel('Time (s)', fontsize=9)
        self.ax_euler.set_ylabel('Euler Angles (deg)', fontsize=9)
        self.ax_euler.set_title('Attitude (Euler)', fontsize=10, fontweight='bold')
        self.ax_euler.grid(True, alpha=0.3)
        
        # 6. Angular velocity
        self.ax_angvel = self.fig.add_subplot(gs[1, 3])
        self.ax_angvel.set_xlabel('Time (s)', fontsize=9)
        self.ax_angvel.set_ylabel('Angular Vel (°/s)', fontsize=9)
        self.ax_angvel.set_title('Angular Velocity', fontsize=10, fontweight='bold')
        self.ax_angvel.grid(True, alpha=0.3)
        
        # Third row: control inputs
        # 7. TVC Pitch angle
        self.ax_pitch = self.fig.add_subplot(gs[2, 0])
        self.ax_pitch.set_xlabel('Time (s)', fontsize=9)
        self.ax_pitch.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_pitch.set_title('TVC Pitch Angle', fontsize=10, fontweight='bold')
        self.ax_pitch.grid(True, alpha=0.3)
        
        # 8. TVC Roll angle
        self.ax_roll = self.fig.add_subplot(gs[2, 1])
        self.ax_roll.set_xlabel('Time (s)', fontsize=9)
        self.ax_roll.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_roll.set_title('TVC Roll Angle', fontsize=10, fontweight='bold')
        self.ax_roll.grid(True, alpha=0.3)
        
        # 9. Thrust
        self.ax_thrust = self.fig.add_subplot(gs[2, 2])
        self.ax_thrust.set_xlabel('Time (s)', fontsize=9)
        self.ax_thrust.set_ylabel('Thrust (N)', fontsize=9)
        self.ax_thrust.set_title('Thrust', fontsize=10, fontweight='bold')
        self.ax_thrust.grid(True, alpha=0.3)
        
        # 10. Yaw torque
        self.ax_yaw = self.fig.add_subplot(gs[2, 3])
        self.ax_yaw.set_xlabel('Time (s)', fontsize=9)
        self.ax_yaw.set_ylabel('Torque (N·m)', fontsize=9)
        self.ax_yaw.set_title('Yaw Torque', fontsize=10, fontweight='bold')
        self.ax_yaw.grid(True, alpha=0.3)

        fig_actions = QHBoxLayout()
        self.btn_save_figure = QPushButton('Save figure…')
        self.btn_save_figure.setToolTip('Export the full plot grid (PNG, PDF, or SVG)')
        self.btn_save_figure.clicked.connect(self.save_figure)
        self.btn_save_figure.setMaximumHeight(30)
        fig_actions.addWidget(self.btn_save_figure)
        fig_actions.addStretch(1)
        layout.addLayout(fig_actions)
        layout.addWidget(self.canvas)
        
        # Data storage
        self.iterations = []
        self.costs = []
        self.stops = []
        self.current_xs = None
        self.current_us = None
        # Multi-segment cost tracking
        self.segment_costs = {}  # {segment_idx: [costs]}
        self.segment_iterations = {}  # {segment_idx: [iterations]}
        self.current_segment_idx = 0
        self._last_figure_save_path = ''

        return panel

    def save_figure(self):
        """Save ``self.fig`` to disk via file dialog (PNG / PDF / SVG)."""
        start = self._last_figure_save_path or os.path.join(
            os.path.expanduser('~'), 'tvc_traj_opt_figure.png'
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save figure',
            start,
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)',
        )
        if not path:
            return
        try:
            self.canvas.draw()
            self.fig.savefig(path, dpi=150, bbox_inches='tight')
        except Exception as e:
            QMessageBox.critical(self, 'Save figure failed', str(e))
            return
        self._last_figure_save_path = path
        if hasattr(self, 'status_text') and self.status_text is not None:
            self.status_text.append(f'Saved figure: {path}')
    
    def quat_to_euler(self, q):
        """Convert quaternion to Euler angles (ZYX order), q=[w,x,y,z]"""
        return quat_to_euler(q, format='wxyz')
    
    def yaw_to_quaternion(self, yaw_deg):
        """Convert yaw angle (degrees) to quaternion [w, x, y, z], roll=0, pitch=0"""
        return yaw_to_quaternion(yaw_deg)
    
    def _refresh_unified_interp_guess_enabled(self):
        idx = self.method_combo.currentIndex()
        acados = idx in (3, 4, 6)
        self.unified_interp_guess_checkbox.setEnabled(
            acados and idx >= 1 and self.unified_checkbox.isChecked()
        )

    def _update_unified_checkbox_state(self, index):
        """Enable unified for Method 2–4 (1–3); Methods 6–7 (5–6) are per-segment only."""
        self.unified_checkbox.setEnabled(index >= 1 and index not in (5, 6))
        if index == 0 or index in (5, 6):
            self.unified_checkbox.setChecked(False)
        self._refresh_unified_interp_guess_enabled()

    def _refresh_min_time_duration_group_visible(self, index=None):
        """Show T_min / T_max scale when Method 5–7 (combo indices 4–6)."""
        if not hasattr(self, "min_time_duration_group"):
            return
        if index is None:
            index = self.method_combo.currentIndex()
        self.min_time_duration_group.setVisible(index in (4, 5, 6))
    
    def on_method_changed(self, index):
        """Load parameter defaults when optimization method changes"""
        params = self.DEFAULT_PARAMS.get(index, self.DEFAULT_PARAMS[0])
        self.w_p.setValue(params["w_p"])
        self.w_v.setValue(params["w_v"])
        self.w_R.setValue(params["w_R"])
        self.w_yaw.setValue(params.get("w_yaw", params["w_R"]))
        self.w_w.setValue(params["w_w"])
        self.w_u.setValue(params["w_u"])
        self.w_du.setValue(params["w_du"])
        self.k_bound.setValue(params["k_bound"])
        self.k_state_bound.setValue(params["k_state_bound"])
        self.th_p_max.setValue(params["th_p_max"])
        self.th_r_max.setValue(params["th_r_max"])
        self.T_max.setValue(params["T_max"])
        self.tau_yaw_max.setValue(params["tau_yaw_max"])
        self.v_horizontal_max.setValue(params["v_horizontal_max"])
        self.v_vertical_max.setValue(params["v_vertical_max"])
        self.roll_max.setValue(params["roll_max"])
        self.pitch_max.setValue(params["pitch_max"])
        self.yaw_max.setValue(params["yaw_max"])
        self.w_max.setValue(params["w_max"])
        if "schedule_ref" in params:
            self.schedule_ref_checkbox.setChecked(params["schedule_ref"])
        if "terminal_cost_multiplier" in params:
            self.terminal_cost_multiplier_spin.setValue(params["terminal_cost_multiplier"])
        elif "terminal_scale" in params:
            self.terminal_cost_multiplier_spin.setValue(params["terminal_scale"])
        if "terminal_constraint" in params:
            self.terminal_constraint_checkbox.setChecked(params["terminal_constraint"])
        if "waypoint_terminal_cost" in params:
            self.waypoint_terminal_checkbox.setChecked(params["waypoint_terminal_cost"])
        if "unified_interp_initial_guess" in params:
            self.unified_interp_guess_checkbox.setChecked(params["unified_interp_initial_guess"])
        if "actuator_dynamics" in params:
            self.actuator_dynamics_checkbox.setChecked(params["actuator_dynamics"])
        if "actuator_tau" in params:
            tau = params["actuator_tau"]
            if len(tau) >= 4:
                self.tau_pitch_spin.setValue(tau[0])
                self.tau_roll_spin.setValue(tau[1])
                self.tau_T_spin.setValue(tau[2])
                self.tau_yaw_spin.setValue(tau[3])
        if "min_time_T_min" in params:
            self.min_time_T_min_spin.setValue(float(params["min_time_T_min"]))
        if "min_time_T_max_scale" in params:
            self.min_time_T_max_scale_spin.setValue(float(params["min_time_T_max_scale"]))
        method_names = [
            "Method 1 (Custom calcDiff)", "Method 2 (FDDP)", "Method 3 (BoxFDDP)",
            "Method 4 (Acados)", "Method 5 (Acados min-time)", "Method 6 (Spannagl FFTF)",
            "Method 7 (Acados free-tf EXTERNAL)",
        ]
        if hasattr(self, "status_text") and self.status_text is not None:
            self.status_text.append(
                f"Parameters loaded for {method_names[index] if index < len(method_names) else 'Unknown'}"
            )
        self._refresh_min_time_duration_group_visible(index)
    
    def update_waypoint_list(self):
        """Update waypoint list display"""
        self.waypoint_list.clear()
        for i, wp in enumerate(self.waypoints):
            # Ensure waypoint has all required fields (for backward compatibility)
            # Format: [x, y, z, yaw_deg, time]
            if len(wp) < 5:
                # Old format: [x, y, z, time] -> add yaw=0
                if len(wp) == 4:
                    wp = [wp[0], wp[1], wp[2], 0.0, wp[3]]  # Insert yaw=0 before time
                else:
                    wp = list(wp) + [0.0] * (5 - len(wp))
            
            yaw = wp[3] if len(wp) > 3 else 0.0
            time = wp[4] if len(wp) > 4 else (wp[3] if len(wp) > 3 else 0.0)
            
            if i == 0:
                item_text = f"Start: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}] yaw={yaw:.1f}° @ t={time:.2f}s"
            else:
                item_text = f"WP {i}: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}] yaw={yaw:.1f}° @ t={time:.2f}s"
            try:
                from PyQt5.QtWidgets import QListWidgetItem
            except ImportError:
                from PySide2.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            self.waypoint_list.addItem(item)
    
    def add_waypoint(self):
        """Add a new waypoint"""
        new_wp = [self.wp_x.value(), self.wp_y.value(), self.wp_z.value(), 
                  self.wp_yaw.value(), self.wp_time.value()]
        self.waypoints.append(new_wp)
        self.update_waypoint_list()
        # Select the newly added waypoint
        self.waypoint_list.setCurrentRow(len(self.waypoints) - 1)
        self._sync_trajectory_preset_combo_from_waypoints()
    
    def remove_waypoint(self):
        """Remove selected waypoint (cannot remove start point)"""
        current_row = self.waypoint_list.currentRow()
        if current_row >= 0 and current_row < len(self.waypoints):
            if current_row == 0:
                QMessageBox.warning(self, 'Warning', 'Cannot remove start point')
                return
            self.waypoints.pop(current_row)
            self.update_waypoint_list()
            # Select previous item if available
            if current_row > 0:
                self.waypoint_list.setCurrentRow(current_row - 1)
            self._sync_trajectory_preset_combo_from_waypoints()

    def update_waypoint(self):
        """Update selected waypoint with current values"""
        current_row = self.waypoint_list.currentRow()
        if current_row >= 0 and current_row < len(self.waypoints):
            self.waypoints[current_row] = [self.wp_x.value(), self.wp_y.value(), self.wp_z.value(), 
                                          self.wp_yaw.value(), self.wp_time.value()]
            self.update_waypoint_list()
            self.waypoint_list.setCurrentRow(current_row)
            self._sync_trajectory_preset_combo_from_waypoints()

    def on_waypoint_selected(self):
        """Handle waypoint selection"""
        current_row = self.waypoint_list.currentRow()
        if current_row >= 0 and current_row < len(self.waypoints):
            wp = self.waypoints[current_row]
            # Ensure waypoint has all required fields (for backward compatibility)
            # Format: [x, y, z, yaw_deg, time]
            if len(wp) < 5:
                # Old format: [x, y, z, time] -> add yaw=0
                if len(wp) == 4:
                    wp = [wp[0], wp[1], wp[2], 0.0, wp[3]]  # Insert yaw=0 before time
                else:
                    wp = list(wp) + [0.0] * (5 - len(wp))
            self.wp_x.setValue(wp[0])
            self.wp_y.setValue(wp[1])
            self.wp_z.setValue(wp[2])
            self.wp_yaw.setValue(wp[3] if len(wp) > 3 else 0.0)
            self.wp_time.setValue(wp[4] if len(wp) > 4 else (wp[3] if len(wp) > 3 else 0.0))

    def on_trajectory_preset_changed(self, index):
        """Apply built-in waypoint sequence for the selected trajectory preset."""
        if index < 0 or index >= len(TRAJECTORY_PRESETS):
            return
        _, wps = TRAJECTORY_PRESETS[index]
        self.waypoints = [_normalize_waypoint_row(w) for w in wps]
        self.update_waypoint_list()
        if self.waypoints:
            self.waypoint_list.setCurrentRow(0)
            self.on_waypoint_selected()

    def _sync_trajectory_preset_combo_from_waypoints(self):
        """Align trajectory combo with current waypoints after load (or leave index 0 if no preset matches)."""
        if not hasattr(self, 'trajectory_preset_combo'):
            return
        idx = trajectory_preset_match_index(self.waypoints)
        self.trajectory_preset_combo.blockSignals(True)
        self.trajectory_preset_combo.setCurrentIndex(idx if idx is not None else 0)
        self.trajectory_preset_combo.blockSignals(False)

    def gui_config_to_dict(self):
        """Serialize GUI settings to a JSON-friendly dict (waypoints + all parameters)."""
        wp = []
        for w in self.waypoints:
            w = list(w)
            while len(w) < 5:
                w.append(0.0)
            wp.append([float(w[0]), float(w[1]), float(w[2]), float(w[3]), float(w[4])])
        return {
            'version': GUI_PARAMS_VERSION,
            'waypoints': wp,
            'trajectory_preset': self.trajectory_preset_combo.currentIndex(),
            'method': self.method_combo.currentIndex(),
            'unified': self.unified_checkbox.isChecked(),
            'unified_interp_initial_guess': self.unified_interp_guess_checkbox.isChecked(),
            'dt': self.dt_spin.value(),
            'N': self.N_spin.value(),
            'max_iter': self.max_iter_spin.value(),
            'w_p': self.w_p.value(), 'w_v': self.w_v.value(), 'w_R': self.w_R.value(),
            'w_yaw': self.w_yaw.value(), 'w_w': self.w_w.value(),
            'w_u': self.w_u.value(), 'w_du': self.w_du.value(),
            'k_bound': self.k_bound.value(), 'k_state_bound': self.k_state_bound.value(),
            'schedule_ref': self.schedule_ref_checkbox.isChecked(),
            'actuator_dynamics': self.actuator_dynamics_checkbox.isChecked(),
            'actuator_tau': [self.tau_pitch_spin.value(), self.tau_roll_spin.value(),
                            self.tau_T_spin.value(), self.tau_yaw_spin.value()],
            'terminal_cost_multiplier': self.terminal_cost_multiplier_spin.value(),
            'terminal_constraint': self.terminal_constraint_checkbox.isChecked(),
            'waypoint_terminal_cost': self.waypoint_terminal_checkbox.isChecked(),
            'th_p_max': self.th_p_max.value(), 'th_r_max': self.th_r_max.value(),
            'T_max': self.T_max.value(), 'tau_yaw_max': self.tau_yaw_max.value(),
            'v_horizontal_max': self.v_horizontal_max.value(),
            'v_vertical_max': self.v_vertical_max.value(),
            'roll_max': self.roll_max.value(), 'pitch_max': self.pitch_max.value(),
            'yaw_max': self.yaw_max.value(), 'w_max': self.w_max.value(),
            'mass': self.mass.value(), 'Ixx': self.Ixx.value(), 'Iyy': self.Iyy.value(), 'Izz': self.Izz.value(),
            'r_thrust_x': self.r_thrust_x.value(), 'r_thrust_y': self.r_thrust_y.value(),
            'r_thrust_z': self.r_thrust_z.value(),
            'min_time_T_min': self.min_time_T_min_spin.value(),
            'min_time_T_max_scale': self.min_time_T_max_scale_spin.value(),
        }

    def apply_gui_config(self, cfg):
        """Apply settings from dict (e.g. loaded JSON). Does not call on_method_changed."""
        if not cfg:
            return
        self.method_combo.blockSignals(True)
        try:
            if 'waypoints' in cfg:
                self.waypoints = []
                for w in cfg['waypoints']:
                    w = list(w)
                    while len(w) < 5:
                        w.append(0.0)
                    self.waypoints.append([float(w[0]), float(w[1]), float(w[2]), float(w[3]), float(w[4])])
                self.update_waypoint_list()
                if self.waypoints:
                    self.waypoint_list.setCurrentRow(0)
                    self.on_waypoint_selected()

            def _set_spin(spin, key):
                if key in cfg:
                    v = cfg[key]
                    spin.setValue(float(v) if not isinstance(v, bool) else v)

            def _set_int_spin(spin, key):
                if key in cfg:
                    spin.setValue(int(cfg[key]))

            def _set_check(cb, key):
                if key in cfg:
                    cb.setChecked(bool(cfg[key]))

            if 'dt' in cfg:
                self.dt_spin.setValue(float(cfg['dt']))
            _set_int_spin(self.N_spin, 'N')
            _set_int_spin(self.max_iter_spin, 'max_iter')
            if 'method' in cfg:
                idx = int(cfg['method'])
                idx = max(0, min(self.method_combo.count() - 1, idx))
                self.method_combo.setCurrentIndex(idx)
            _set_check(self.unified_checkbox, 'unified')
            _set_check(self.unified_interp_guess_checkbox, 'unified_interp_initial_guess')

            for k, sp in [('w_p', self.w_p), ('w_v', self.w_v), ('w_R', self.w_R), ('w_yaw', self.w_yaw),
                          ('w_w', self.w_w), ('w_u', self.w_u), ('w_du', self.w_du),
                          ('k_bound', self.k_bound), ('k_state_bound', self.k_state_bound)]:
                _set_spin(sp, k)
            _set_check(self.schedule_ref_checkbox, 'schedule_ref')
            _set_check(self.actuator_dynamics_checkbox, 'actuator_dynamics')
            if 'actuator_tau' in cfg:
                t = cfg['actuator_tau']
                if isinstance(t, (list, tuple)) and len(t) >= 4:
                    self.tau_pitch_spin.setValue(float(t[0]))
                    self.tau_roll_spin.setValue(float(t[1]))
                    self.tau_T_spin.setValue(float(t[2]))
                    self.tau_yaw_spin.setValue(float(t[3]))
            if 'terminal_cost_multiplier' in cfg:
                self.terminal_cost_multiplier_spin.setValue(float(cfg['terminal_cost_multiplier']))
            elif 'terminal_scale' in cfg:
                self.terminal_cost_multiplier_spin.setValue(float(cfg['terminal_scale']))
            _set_check(self.terminal_constraint_checkbox, 'terminal_constraint')
            _set_check(self.waypoint_terminal_checkbox, 'waypoint_terminal_cost')

            for k, sp in [('th_p_max', self.th_p_max), ('th_r_max', self.th_r_max),
                          ('T_max', self.T_max), ('tau_yaw_max', self.tau_yaw_max),
                          ('v_horizontal_max', self.v_horizontal_max), ('v_vertical_max', self.v_vertical_max),
                          ('roll_max', self.roll_max), ('pitch_max', self.pitch_max),
                          ('yaw_max', self.yaw_max), ('w_max', self.w_max)]:
                _set_spin(sp, k)

            if 'mass' in cfg:
                self.mass.setValue(float(cfg['mass']))
            for k, sp in [('Ixx', self.Ixx), ('Iyy', self.Iyy), ('Izz', self.Izz)]:
                _set_spin(sp, k)
            for k, sp in [('r_thrust_x', self.r_thrust_x), ('r_thrust_y', self.r_thrust_y),
                          ('r_thrust_z', self.r_thrust_z)]:
                _set_spin(sp, k)
            _set_spin(self.min_time_T_min_spin, 'min_time_T_min')
            _set_spin(self.min_time_T_max_scale_spin, 'min_time_T_max_scale')
        finally:
            self.method_combo.blockSignals(False)
        self._update_unified_checkbox_state(self.method_combo.currentIndex())
        self._refresh_min_time_duration_group_visible(self.method_combo.currentIndex())
        self._sync_trajectory_preset_combo_from_waypoints()

    def _update_params_file_label(self):
        if hasattr(self, 'params_file_label'):
            self.params_file_label.setText(f'Parameters file: {self.params_file_path}')

    def _save_params_to_path(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.gui_config_to_dict(), f, indent=2, ensure_ascii=False)

    def save_parameters(self):
        """Overwrite the current parameters file."""
        try:
            self._save_params_to_path(self.params_file_path)
            self.status_text.append(f'Saved parameters to {self.params_file_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save failed', str(e))

    def save_parameters_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save parameters as', self.params_file_path,
            'JSON (*.json);;All files (*)')
        if not path:
            return
        try:
            self._save_params_to_path(path)
            self.params_file_path = path
            self._update_params_file_label()
            self.status_text.append(f'Saved parameters to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save failed', str(e))

    def load_parameters(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load parameters', self.params_file_path,
            'JSON (*.json);;All files (*)')
        if not path:
            return
        self._load_params_from_path(path, quiet=False)

    def _load_params_from_path(self, path, quiet=False):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:
            if not quiet:
                QMessageBox.critical(self, 'Load failed', str(e))
            return
        self.apply_gui_config(cfg)
        self.params_file_path = os.path.abspath(path)
        self._update_params_file_label()
        if not quiet:
            self.status_text.append(f'Loaded parameters from {self.params_file_path}')

    def get_parameters(self):
        """Get optimization parameters"""
        # Initial / goal pose from waypoints (requires ≥2 waypoints to run optimization)
        x0 = np.zeros(17)
        if len(self.waypoints) > 0:
            first_wp = self.waypoints[0]
            x0[0] = first_wp[0]
            x0[1] = first_wp[1]
            x0[2] = first_wp[2]
            yaw_deg = first_wp[3] if len(first_wp) > 3 else 0.0
            yaw_rad = np.radians(yaw_deg)
            x0[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])
        else:
            x0[6:10] = np.array([1., 0., 0., 0.])

        xg = np.zeros(17)
        if len(self.waypoints) > 0:
            last_wp = self.waypoints[-1]
            xg[0] = last_wp[0]
            xg[1] = last_wp[1]
            xg[2] = last_wp[2]
            yaw_deg = last_wp[3] if len(last_wp) > 3 else 0.0
            yaw_rad = np.radians(yaw_deg)
            xg[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])
        else:
            xg[6:10] = np.array([1., 0., 0., 0.])
        
        k_term = self.terminal_cost_multiplier_spin.value()

        # Cost weights (running cost + Acados terminal multiplier)
        weights = {
            "p": self.w_p.value(),
            "v": self.w_v.value(),
            "R": self.w_R.value(),
            "yaw": self.w_yaw.value(),
            "w": self.w_w.value(),
            "u": self.w_u.value(),
            "du": self.w_du.value(),
            "schedule_ref": self.schedule_ref_checkbox.isChecked(),
            "terminal_cost_multiplier": k_term,
            "terminal_constraint": self.terminal_constraint_checkbox.isChecked(),
            "waypoint_terminal_cost": self.waypoint_terminal_checkbox.isChecked(),
            "unified_interp_initial_guess": self.unified_interp_guess_checkbox.isChecked(),
            "actuator_dynamics": self.actuator_dynamics_checkbox.isChecked(),
            "actuator_tau": [self.tau_pitch_spin.value(), self.tau_roll_spin.value(),
                            self.tau_T_spin.value(), self.tau_yaw_spin.value()],
        }
        mc = self.method_combo.currentIndex()
        if mc == 4:
            weights["acados_objective"] = "min_time"
            d_mt = self.DEFAULT_PARAMS.get(4, self.DEFAULT_PARAMS[3])
            weights["min_time_weight"] = float(d_mt.get("min_time_weight", 1.0))
            weights["min_time_T_min"] = float(self.min_time_T_min_spin.value())
            weights["min_time_T_max_scale"] = float(self.min_time_T_max_scale_spin.value())
        elif mc == 5:
            weights["acados_objective"] = "spannagl_fftf"
            d6 = self.DEFAULT_PARAMS.get(5, self.DEFAULT_PARAMS[3])
            for key in (
                "min_time_T_min",
                "min_time_T_max_scale",
                "g",
                "spannagl_exact_terminal",
                "spannagl_ptol",
                "spannagl_vtol",
                "spannagl_min_tf_reg",
                "spannagl_glideslope",
                "spannagl_gamma_deg",
                "spannagl_udot_max",
                "spannagl_lambda_yaw",
                "spannagl_nlp_solver",
            ):
                if key in d6:
                    weights[key] = d6[key]
            weights["min_time_T_min"] = float(self.min_time_T_min_spin.value())
            weights["min_time_T_max_scale"] = float(self.min_time_T_max_scale_spin.value())
        elif mc == 6:
            weights["acados_objective"] = "free_tf"
            d7 = self.DEFAULT_PARAMS.get(6, self.DEFAULT_PARAMS[4])
            for key in (
                "free_tf_w_T",
                "free_tf_w_tvc",
                "free_tf_w_tau_yaw",
                "free_tf_w_terminal_time",
                "free_tf_include_state_terminal",
            ):
                if key in d7:
                    weights[key] = d7[key]
            weights["min_time_T_min"] = float(self.min_time_T_min_spin.value())
            weights["min_time_T_max_scale"] = float(self.min_time_T_max_scale_spin.value())
        else:
            weights["acados_objective"] = "tracking"

        # Terminal weights for Method 1–3: running × k (same policy as Acados Mayer scaling)
        terminal_weights = {
            "p": self.w_p.value() * k_term,
            "v": self.w_v.value() * k_term,
            "R": self.w_R.value() * k_term,
            "yaw": self.w_yaw.value() * k_term,
            "w": self.w_w.value() * k_term,
            "u": self.w_u.value() * k_term,
            "du": self.w_du.value() * k_term,
        }
        
        # Control constraints - convert degrees to radians for optimization
        th_p_max_rad = np.radians(self.th_p_max.value())
        th_r_max_rad = np.radians(self.th_r_max.value())
        tau_yaw_max_val = self.tau_yaw_max.value()
        bounds = {
            "th_p": (-th_p_max_rad, th_p_max_rad),
            "th_r": (-th_r_max_rad, th_r_max_rad),
            "T": (0.0, self.T_max.value()),
            "tau_yaw": (-tau_yaw_max_val, tau_yaw_max_val),
            "k_bound": self.k_bound.value(),  # Control constraint penalty coefficient
            # State constraints - convert degrees to radians
            "state_v_horizontal_max": self.v_horizontal_max.value(),
            "state_v_vertical_max": self.v_vertical_max.value(),
            "state_roll_max": np.radians(self.roll_max.value()),
            "state_pitch_max": np.radians(self.pitch_max.value()),
            "state_yaw_max": np.radians(self.yaw_max.value()),
            "state_w_max": self.w_max.value(),
            "state_k_state_bound": self.k_state_bound.value(),  # State constraint penalty coefficient
            "state_constraint_lxx_scale": 0.0,  # 0=Method1 style (recommended), state constraint not in Lxx
        }
        
        # Physical parameters - from GUI settings
        m = self.mass.value()
        I = np.diag([self.Ixx.value(), self.Iyy.value(), self.Izz.value()])
        r_thrust = np.array([self.r_thrust_x.value(), self.r_thrust_y.value(), self.r_thrust_z.value()])
        
        return {
            'dt': self.dt_spin.value(),
            'N': self.N_spin.value(),
            'max_iter': self.max_iter_spin.value(),
            'x0': x0,
            'xg': xg,
            'weights': weights,
            'terminal_weights': terminal_weights,
            'bounds': bounds,
            'm': m,
            'I': I,
            'r_thrust': r_thrust,
            'waypoints': self.waypoints.copy(),  # Include waypoints for plotting
            'method': self.method_combo.currentIndex(),  # 3–4,6 Acados; 5 Spannagl; 7=index6 free_tf
            'unified': self.unified_checkbox.isChecked()  # Merge segments (Method 2/3/4)
        }
    
    def start_optimization(self):
        """Start optimization"""
        if self.opt_thread and self.opt_thread.isRunning():
            QMessageBox.warning(self, 'Warning', 'Optimization already in progress')
            return
        
        # Validate waypoints and times
        if len(self.waypoints) < 2:
            QMessageBox.warning(self, 'Warning', 'Need at least 2 waypoints (start and at least one waypoint)')
            return
        
        # Ensure all waypoints have all required fields and validate time order
        # Format: [x, y, z, yaw_deg, time]
        for i, wp in enumerate(self.waypoints):
            if len(wp) == 4:
                # Old format: [x, y, z, time] -> convert to [x, y, z, yaw=0, time]
                self.waypoints[i] = [wp[0], wp[1], wp[2], 0.0, wp[3]]
            elif len(wp) < 5:
                self.waypoints[i] = list(wp) + [0.0] * (5 - len(wp))
        
        # Check time order (time is at index 4)
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i][4] >= self.waypoints[i+1][4]:
                QMessageBox.warning(self, 'Warning', 
                                  f'Waypoint {i+1} arrival time ({self.waypoints[i+1][4]:.2f}s) must be greater than waypoint {i} time ({self.waypoints[i][4]:.2f}s)')
                return
        
        # Reset display
        self.iterations = []
        self.costs = []
        self.stops = []
        # Reset segment tracking
        self.segment_costs = {}
        self.segment_iterations = {}
        self.current_segment_idx = 0
        
        # Clear all plots
        self.ax_3d.clear()
        self.ax_cost.clear()
        self.ax_pos.clear()
        self.ax_vel.clear()
        self.ax_angvel.clear()
        self.ax_euler.clear()
        self.ax_pitch.clear()
        self.ax_roll.clear()
        self.ax_thrust.clear()
        self.ax_yaw.clear()
        
        # Reset titles and labels
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
        # Get parameters
        params = self.get_parameters()
        
        # Create optimization thread
        self.opt_thread = OptimizationThread(params)
        self.opt_thread.iteration_update.connect(self.update_iteration)
        self.opt_thread.state_update.connect(self.update_state)
        self.opt_thread.finished.connect(self.optimization_finished)
        self.opt_thread.error.connect(self.optimization_error)
        
        # Update button state
        self.run_btn.setEnabled(False)
        self.progress.setMaximum(params['max_iter'])
        self.progress.setValue(0)
        
        # Start optimization
        self.status_text.append('Starting optimization...')
        wps = params.get('waypoints', [])
        if (params.get('method') == 3 and len(wps) > 2 and params.get('unified') and
                params.get('weights', {}).get('waypoint_terminal_cost', True)):
            self.status_text.append('(Multi-WP + waypoint terminal cost: using segment mode)')
        if params.get('method') == 4 and params.get('unified') and len(wps) > 2:
            self.status_text.append('(Method 5: min-time uses per-segment Acados; unified merges are not applied)')
        if params.get('method') == 6 and params.get('unified') and len(wps) > 2:
            self.status_text.append('(Method 7: free-tf uses per-segment Acados; unified merges are not applied)')
        if params.get('method') == 5:
            self.status_text.append(
                '(Method 6: Spannagl-style FFTF — one NLP per waypoint leg; unified is not used)'
            )
        if params.get('method') == 6:
            self.status_text.append(
                '(Method 7: free physical t_f as state, τ∈[0,1], EXTERNAL cost; per-segment Acados)'
            )
        self.opt_thread.start()
    
    def update_iteration(self, iter_num, cost, stop, segment_idx):
        """Update iteration information"""
        self.iterations.append(iter_num)
        self.costs.append(cost)
        self.stops.append(stop)
        self.current_segment_idx = segment_idx
        
        # Track costs per segment
        if segment_idx not in self.segment_costs:
            self.segment_costs[segment_idx] = []
            self.segment_iterations[segment_idx] = []
        
        self.segment_costs[segment_idx].append(cost)
        # Calculate cumulative iteration number (total iterations so far)
        cumulative_iter = len(self.iterations) - 1
        self.segment_iterations[segment_idx].append(cumulative_iter)
        
        # Update progress bar
        self.progress.setValue(iter_num)
        
        # Update status text
        self.status_text.clear()
        self.status_text.append(f'Segment: {segment_idx + 1}')
        self.status_text.append(f'Iteration: {iter_num}')
        self.status_text.append(f'Cost: {cost:.6e}')
        self.status_text.append(f'Stop Condition: {stop:.6e}')
        
        # Update Cost curve with different colors for each segment
        self.ax_cost.clear()
        
        # Define colors for different segments
        colors = ['b', 'r', 'g', 'm', 'c', 'orange', 'purple', 'brown']
        
        # Plot each segment's cost with different color
        for seg_idx in sorted(self.segment_costs.keys()):
            if len(self.segment_costs[seg_idx]) > 0:
                color = colors[seg_idx % len(colors)]
                label = f'Segment {seg_idx + 1}'
                self.ax_cost.semilogy(self.segment_iterations[seg_idx], 
                                     self.segment_costs[seg_idx], 
                                     color=color, linewidth=2.5, 
                                     marker='o', markersize=3, label=label)
        
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        if len(self.segment_costs) > 1:
            self.ax_cost.legend(fontsize=8, loc='best')
        
        # Add current cost text
        if len(self.costs) > 0:
            final_cost = self.costs[-1]
            self.ax_cost.text(0.02, 0.98, f'Current Cost: {final_cost:.4e}', 
                            transform=self.ax_cost.transAxes, fontsize=9,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.canvas.draw()
    
    def update_state(self, xs, us, us_actual=None, plot_dt=None, segment_boundaries_override=None,
                     time_states=None):
        """Update all state and control displays.

        ``time_states``: optional physical time [s] per state (multi-segment min-time); when set,
        ``plot_dt`` is not used as a uniform spacing (see ``draw_trajectory_panels``).
        """
        self.current_xs = xs
        self.current_us = us
        
        if xs is None or len(xs) == 0:
            return
        
        dt = self.dt_spin.value()
        waypoints = self.waypoints if hasattr(self, 'waypoints') else None
        bd = self._bounds_display_from_widgets()
        draw_trajectory_panels({
            'ax_3d': self.ax_3d, 'ax_pos': self.ax_pos, 'ax_vel': self.ax_vel,
            'ax_angvel': self.ax_angvel, 'ax_euler': self.ax_euler,
            'ax_pitch': self.ax_pitch, 'ax_roll': self.ax_roll,
            'ax_thrust': self.ax_thrust, 'ax_yaw': self.ax_yaw,
        }, xs, us, dt, waypoints, bd, quat_to_euler_fn=self.quat_to_euler,
           us_actual=us_actual,
           time_step_override=plot_dt if time_states is None else None,
           segment_boundaries_override=segment_boundaries_override,
           time_states=time_states)
        self.canvas.draw()
    
    def _bounds_display_from_widgets(self):
        """Same numeric limits as GUI constraint spin boxes (for shared plot style)."""
        return {
            'v_horizontal_max': self.v_horizontal_max.value(),
            'v_vertical_max': self.v_vertical_max.value(),
            'roll_max': self.roll_max.value(),
            'pitch_max': self.pitch_max.value(),
            'yaw_max': self.yaw_max.value(),
            'w_max': self.w_max.value(),
            'th_p_max': self.th_p_max.value(),
            'th_r_max': self.th_r_max.value(),
            'T_max': self.T_max.value(),
            'tau_yaw_max': self.tau_yaw_max.value(),
        }
    
    def optimization_finished(self, xs, us, all_loggers, timing_info=None):
        """Optimization finished"""
        self.status_text.append('Optimization completed!')
        self.run_btn.setEnabled(True)
        self.progress.setValue(self.progress.maximum())
        
        # Show timing: average time per iteration
        if timing_info:
            avg_ms = timing_info.get("avg_time_per_iter", 0) * 1000
            total_time = timing_info.get("total_time", 0)
            total_iters = timing_info.get("total_iters", 0)
            method_name = timing_info.get("method", "")
            self.status_text.append(f'{method_name}')
            self.status_text.append(f'Total time: {total_time:.3f}s, Total iterations: {total_iters}')
            self.status_text.append(f'Average time per iteration: {avg_ms:.2f} ms')
            print(f"[{method_name}] Total: {total_time:.3f}s, Iters: {total_iters}, Avg/iter: {avg_ms:.2f} ms")
            ots = timing_info.get("optimal_segment_times")
            if ots and (
                timing_info.get("min_time")
                or timing_info.get("spannagl_fftf")
                or timing_info.get("free_tf_acados")
            ):
                self.status_text.append(f'Optimal segment times [s]: {[round(t, 4) for t in ots]}')
        
        # Show final results from all segments
        if all_loggers and len(all_loggers) > 0:
            total_iterations = 0
            for i, logger in enumerate(all_loggers):
                if logger and len(logger.costs) > 0:
                    final_cost = logger.costs[-1]
                    total_iterations += len(logger.costs)
                    self.status_text.append(f'Segment {i+1} Final Cost: {final_cost:.6e}')
            if not timing_info:
                self.status_text.append(f'Total Iterations: {total_iterations}')
        
        draw_cost_panel(self.ax_cost, all_loggers)
        self.canvas.draw()
        
        # Update final state (us_actual: actuator model x[12:16] per shooting node, from Acados)
        u_act = timing_info.get("us_actual") if timing_info else None
        pdt = timing_info.get("plot_dt") if timing_info else None
        sbo = timing_info.get("segment_boundary_indices") if timing_info else None
        ots = timing_info.get("optimal_segment_times") if timing_info else None
        time_axis = None
        if (
            ots
            and sbo
            and len(ots) == len(sbo)
            and len(xs) == int(sbo[-1]) + 1
        ):
            from tvc_common import physical_time_grid_per_shooting_segment

            time_axis = physical_time_grid_per_shooting_segment(ots, sbo)
        self.update_state(
            xs,
            us,
            u_act,
            plot_dt=pdt,
            segment_boundaries_override=sbo,
            time_states=time_axis,
        )

        # Cache the optimized trajectory so the user can export it as CSV later.
        self.last_trajectory = {
            'xs': xs,
            'us': us,
            'us_actual': u_act,
            'plot_dt': pdt,
            'dt': self.dt_spin.value(),
            'time_states': time_axis,
            'segment_boundary_indices': sbo,
            'optimal_segment_times': ots,
            'method': self.method_combo.currentIndex(),
            'method_name': (timing_info or {}).get('method', ''),
        }
        if hasattr(self, 'btn_save_traj_csv'):
            self.btn_save_traj_csv.setEnabled(True)
    
    def save_trajectory_csv(self):
        """Export the most recently optimized trajectory to a CSV file.

        The CSV format is designed to be consumed by a PX4 offboard /
        setpoint-trajectory tracker. Columns:

            t, x, y, z, vx, vy, vz,
            qw, qx, qy, qz, wx, wy, wz,
            roll_deg, pitch_deg, yaw_deg,
            th_p_cmd, th_r_cmd, T_cmd, tau_yaw_cmd,
            th_p_act, th_r_act, T_act, tau_yaw_act

        The state and command columns are aligned by node index (length
        ``N+1`` for state, ``N`` for command). The last command row repeats
        the previous one (zero-order hold) so the file has uniform length.

        Coordinate frame is selected by ``self.frame_combo``:
          - ENU: as used internally by the planner (x East, y North, z Up).
          - NED: converted for PX4 (x North, y East, z Down, yaw Z-down).
        """
        if not self.last_trajectory or self.last_trajectory.get('xs') is None:
            QMessageBox.warning(
                self, 'No trajectory',
                'Please run an optimization first; there is no trajectory to export yet.'
            )
            return

        traj = self.last_trajectory
        xs = np.asarray(traj['xs'], dtype=float)
        us = np.asarray(traj['us'], dtype=float) if traj.get('us') is not None else None
        us_act = (
            np.asarray(traj['us_actual'], dtype=float)
            if traj.get('us_actual') is not None
            else None
        )

        if xs.ndim != 2 or xs.shape[1] < 13:
            QMessageBox.critical(
                self, 'Invalid trajectory',
                f'Unexpected state shape {xs.shape}; expected (N+1, >=13).'
            )
            return

        N_states = xs.shape[0]

        # Physical time axis per state
        time_states = traj.get('time_states')
        if time_states is not None:
            t = np.asarray(time_states, dtype=float).reshape(-1)
            if t.size != N_states:
                t = None
        else:
            t = None
        if t is None:
            dt_vis = traj.get('plot_dt') or traj.get('dt') or 0.02
            t = np.arange(N_states, dtype=float) * float(dt_vis)

        # State decomposition
        p = xs[:, 0:3]
        v = xs[:, 3:6]
        q = xs[:, 6:10]  # [qw, qx, qy, qz]
        w = xs[:, 10:13]

        # Euler ZYX (rad) per node; convert to deg for the CSV
        euler_rad = np.array([quat_to_euler(qq, format='wxyz') for qq in q])
        euler_deg = np.degrees(euler_rad)

        # Pad commands to N+1 with zero-order hold so columns line up
        def _pad_to_state_length(arr):
            if arr is None:
                return np.full((N_states, 4), np.nan, dtype=float)
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] < 4:
                pad = np.full((arr.shape[0], 4 - arr.shape[1]), np.nan)
                arr = np.hstack([arr, pad])
            arr = arr[:, :4]
            if arr.shape[0] == N_states:
                return arr
            if arr.shape[0] == N_states - 1:
                return np.vstack([arr, arr[-1:]])  # ZOH last row
            if arr.shape[0] > N_states:
                return arr[:N_states]
            pad_rows = np.repeat(arr[-1:], N_states - arr.shape[0], axis=0)
            return np.vstack([arr, pad_rows])

        u_cmd = _pad_to_state_length(us)
        u_act_out = _pad_to_state_length(us_act)

        # Optional NED conversion: planner uses ENU (x East, y North, z Up).
        # PX4 internally uses NED (x North, y East, z Down). The conversion is:
        #   x_NED = y_ENU, y_NED = x_ENU, z_NED = -z_ENU
        # Velocities, angular rates and Euler angles transform the same way.
        frame_is_ned = (
            hasattr(self, 'frame_combo') and self.frame_combo.currentIndex() == 1
        )
        if frame_is_ned:
            p = np.column_stack([p[:, 1], p[:, 0], -p[:, 2]])
            v = np.column_stack([v[:, 1], v[:, 0], -v[:, 2]])
            w = np.column_stack([w[:, 1], w[:, 0], -w[:, 2]])
            # Rotate quaternion from ENU body->world to NED body->world by
            # composing with R_enu2ned = diag(...). Simpler: rebuild from the
            # NED Euler angles (ZYX with sign flips on y and z components).
            euler_ned = np.column_stack([
                 euler_rad[:, 0],          # roll
                -euler_rad[:, 1],          # pitch (sign flip: Up -> Down)
                -euler_rad[:, 2],          # yaw   (sign flip: Up -> Down)
            ])
            from tvc_common import euler_to_quat_wxyz
            q_out = np.array([
                euler_to_quat_wxyz(r, pi, ya) for r, pi, ya in euler_ned
            ])
            euler_deg = np.degrees(euler_ned)
        else:
            q_out = q

        # Default file name based on method and timestamp
        method_name = traj.get('method_name') or f"method{traj.get('method', 0) + 1}"
        safe_method = ''.join(c if c.isalnum() else '_' for c in str(method_name)).strip('_')
        default_dir = (
            os.path.dirname(self.last_csv_path) if self.last_csv_path
            else os.path.dirname(self.params_file_path) or script_dir
        )
        default_name = time.strftime(
            f"tvc_traj_{safe_method}_%Y%m%d_%H%M%S.csv"
        )
        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save trajectory as CSV',
            default_path,
            'CSV files (*.csv);;All files (*.*)',
        )
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'

        frame_tag = 'NED' if frame_is_ned else 'ENU'
        header_lines = [
            f"# TVC trajectory exported from tvc_traj_opt_gui",
            f"# generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# method: {method_name}",
            f"# frame: {frame_tag} "
            f"({'x N, y E, z D (PX4)' if frame_is_ned else 'x E, y N, z U (planner native)'})",
            f"# N_states: {N_states}, duration: {float(t[-1] - t[0]):.6f} s",
            "# Commands repeat the last row (zero-order hold) so all columns "
            "have length N_states.",
            "# th_p, th_r: gimbal pitch/roll angles [rad]; T: thrust [N]; "
            "tau_yaw: reaction-wheel torque [N*m].",
            "# *_act columns are the actuator's actual states (NaN when the "
            "method has no actuator dynamics).",
        ]
        columns = [
            't',
            'x', 'y', 'z',
            'vx', 'vy', 'vz',
            'qw', 'qx', 'qy', 'qz',
            'wx', 'wy', 'wz',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'th_p_cmd', 'th_r_cmd', 'T_cmd', 'tau_yaw_cmd',
            'th_p_act', 'th_r_act', 'T_act', 'tau_yaw_act',
        ]

        try:
            with open(path, 'w', encoding='utf-8') as f:
                for line in header_lines:
                    f.write(line + '\n')
                f.write(','.join(columns) + '\n')
                for i in range(N_states):
                    row = [
                        f"{t[i]:.9f}",
                        f"{p[i, 0]:.9f}", f"{p[i, 1]:.9f}", f"{p[i, 2]:.9f}",
                        f"{v[i, 0]:.9f}", f"{v[i, 1]:.9f}", f"{v[i, 2]:.9f}",
                        f"{q_out[i, 0]:.9f}", f"{q_out[i, 1]:.9f}",
                        f"{q_out[i, 2]:.9f}", f"{q_out[i, 3]:.9f}",
                        f"{w[i, 0]:.9f}", f"{w[i, 1]:.9f}", f"{w[i, 2]:.9f}",
                        f"{euler_deg[i, 0]:.6f}",
                        f"{euler_deg[i, 1]:.6f}",
                        f"{euler_deg[i, 2]:.6f}",
                        f"{u_cmd[i, 0]:.9f}", f"{u_cmd[i, 1]:.9f}",
                        f"{u_cmd[i, 2]:.9f}", f"{u_cmd[i, 3]:.9f}",
                        f"{u_act_out[i, 0]:.9f}", f"{u_act_out[i, 1]:.9f}",
                        f"{u_act_out[i, 2]:.9f}", f"{u_act_out[i, 3]:.9f}",
                    ]
                    f.write(','.join(row) + '\n')
        except OSError as e:
            QMessageBox.critical(self, 'Save failed', f'Could not write {path}:\n{e}')
            return

        self.last_csv_path = path
        self.status_text.append(
            f'Saved trajectory ({N_states} samples, {frame_tag}) to:\n  {path}'
        )

    def optimization_error(self, error_msg):
        """Optimization error"""
        QMessageBox.critical(self, 'Error', f'Error during optimization:\n{error_msg}')
        self.status_text.append(f'Error: {error_msg}')
        self.run_btn.setEnabled(True)


def run_gui(argv=None) -> int:
    """Create Qt application, show main window, run event loop."""
    if argv is not None:
        sys.argv = list(argv)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec_())


def main() -> int:
    """Backward-compatible alias for :func:`run_gui`."""
    return run_gui()


if __name__ == '__main__':
    raise SystemExit(main())
