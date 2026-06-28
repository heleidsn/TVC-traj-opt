#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization GUI using PyQt5

Create user interface using PyQt5, supports:
- Waypoints (start and targets; initial state from first waypoint)
- Adjust cost weight parameters; save/load JSON parameters
- Real-time display of optimization process and results

Usage:
    python tvc_traj_opt_gui.py

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
import signal
import subprocess

# Project root (this file) and scripts/ (solver modules)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

GUI_PARAMS_VERSION = 1
DEFAULT_GUI_PARAMS_FILENAME = 'tvc_traj_opt_gui_params.json'
LEFT_STATUS_TEXT_HEIGHT = 72
DEFAULT_TRAJ_CSV_DIR = os.path.join(ROOT_DIR, 'trajs')
DEFAULT_TRAJ_CSV_PATH = os.path.join(DEFAULT_TRAJ_CSV_DIR, 'latest.csv')
DEFAULT_SAVED_WAYPOINTS_PATH = os.path.join(DEFAULT_TRAJ_CSV_DIR, 'saved_waypoints.json')
TRAJ_PRESETS_DIR = os.path.join(DEFAULT_TRAJ_CSV_DIR, 'presets')
CUSTOM_WAYPOINTS_PATH = os.path.join(TRAJ_PRESETS_DIR, 'custom_waypoints.json')

# Table columns in the Trajectory tab waypoint editor
WP_COL_IDX = 0
WP_COL_X = 1
WP_COL_Y = 2
WP_COL_Z = 3
WP_COL_YAW = 4
WP_COL_LEG_DT = 5
WP_COL_T_ARR = 6
WP_TABLE_HEADERS = ('#', 'X (m)', 'Y (m)', 'Z (m)', 'Yaw (°)', 'Δt (s)', 't (s)')

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
                                 QFileDialog, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
                                 QAbstractItemView, QHeaderView)
    from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
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
                                      QFileDialog, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
                                      QAbstractItemView, QHeaderView)
        from PySide2.QtCore import QThread, Signal as pyqtSignal, Qt, QTimer
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
TRAJECTORY_PRESET_STORAGE_SLUGS = ('grasshopper', 'platform_hop')
TRAJECTORY_PRESETS = (
    (
        'Traj1: Grasshopper',
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


# JSON ``trajectory_preset`` ids (built-in presets use 0 .. len-1).
TRAJECTORY_PRESET_SAVED = -2
TRAJECTORY_PRESET_CUSTOM = -1
TRAJECTORY_SAVED_COMBO_LABEL = 'Traj2: Waypoints(optimal)'

# QComboBox order (decoupled from preset_id numeric order).
TRAJECTORY_COMBO_SLOTS = (
    {'label': 'Traj1: Grasshopper', 'preset_id': 0},
    {'label': TRAJECTORY_SAVED_COMBO_LABEL, 'preset_id': TRAJECTORY_PRESET_SAVED},
    {'label': 'Platform hop (0→1 m, +2 m x, land)', 'preset_id': 1},
    {'label': 'Custom waypoints', 'preset_id': TRAJECTORY_PRESET_CUSTOM},
)

# Trajectory optimization mode (below Trajectory combo).
TRAJ_OPT_MODE_NORMAL = 0
TRAJ_OPT_MODE_MIN_TIME = 1
METHOD_NORMAL_DEFAULT_INDEX = 3   # Method 4: Acados
METHOD_MIN_TIME_INDEX = 4         # Method 5: Acados min-time


def trajectory_saved_combo_index():
    """QComboBox index for the user-saved waypoint slot."""
    for i, slot in enumerate(TRAJECTORY_COMBO_SLOTS):
        if slot['preset_id'] == TRAJECTORY_PRESET_SAVED:
            return i
    return 1


def trajectory_custom_combo_index():
    """QComboBox index for ephemeral / params-embedded custom waypoints."""
    for i, slot in enumerate(TRAJECTORY_COMBO_SLOTS):
        if slot['preset_id'] == TRAJECTORY_PRESET_CUSTOM:
            return i
    return len(TRAJECTORY_COMBO_SLOTS) - 1


def combo_index_to_trajectory_preset_id(combo_index):
    """Map Trajectory combo index to JSON ``trajectory_preset`` id."""
    combo_index = int(combo_index)
    if 0 <= combo_index < len(TRAJECTORY_COMBO_SLOTS):
        return TRAJECTORY_COMBO_SLOTS[combo_index]['preset_id']
    return TRAJECTORY_PRESET_CUSTOM


def trajectory_preset_id_to_combo_index(preset_id):
    """Map JSON ``trajectory_preset`` id to Trajectory combo index."""
    if preset_id is None:
        return trajectory_custom_combo_index()
    preset_id = int(preset_id)
    for i, slot in enumerate(TRAJECTORY_COMBO_SLOTS):
        if slot['preset_id'] == preset_id:
            return i
    return trajectory_custom_combo_index()


def trajectory_waypoints_path_for_preset_id(preset_id):
    """JSON file that stores waypoints for a trajectory preset id."""
    preset_id = int(preset_id)
    if preset_id == TRAJECTORY_PRESET_CUSTOM:
        return CUSTOM_WAYPOINTS_PATH
    if preset_id == TRAJECTORY_PRESET_SAVED:
        return DEFAULT_SAVED_WAYPOINTS_PATH
    if 0 <= preset_id < len(TRAJECTORY_PRESET_STORAGE_SLUGS):
        return os.path.join(
            TRAJ_PRESETS_DIR,
            f'{TRAJECTORY_PRESET_STORAGE_SLUGS[preset_id]}.json',
        )
    return CUSTOM_WAYPOINTS_PATH


def trajectory_waypoints_path_for_combo_index(combo_index):
    """JSON file that stores waypoints for a Trajectory combo slot."""
    return trajectory_waypoints_path_for_preset_id(
        combo_index_to_trajectory_preset_id(combo_index)
    )


def trajectory_combo_label(combo_index):
    """Human-readable Trajectory combo label."""
    combo_index = int(combo_index)
    if 0 <= combo_index < len(TRAJECTORY_COMBO_SLOTS):
        return TRAJECTORY_COMBO_SLOTS[combo_index]['label']
    return 'Custom waypoints'


TRAJ_JSON_VERSION = 3


def traj_opt_mode_key(mode_index=None):
    """Return ``'normal'`` or ``'min_time'`` for artifact file suffixes."""
    if mode_index is None:
        return 'normal'
    if int(mode_index) == TRAJ_OPT_MODE_MIN_TIME:
        return 'min_time'
    return 'normal'


def trajectory_artifact_paths(combo_index, mode_key='normal'):
    """JSON / CSV / NPZ paths for a trajectory slot + optimization mode."""
    json_path = trajectory_waypoints_path_for_combo_index(combo_index)
    base, _ = os.path.splitext(json_path)
    if mode_key not in ('normal', 'min_time'):
        mode_key = traj_opt_mode_key(mode_key)
    return {
        'json': json_path,
        'csv': f'{base}_{mode_key}.csv',
        'npz': f'{base}_{mode_key}_traj.npz',
        'mode_key': mode_key,
    }


def resolve_trajectory_artifact_paths(combo_index, mode_key='normal'):
    """Like ``trajectory_artifact_paths`` but fall back to legacy unsuffixed files."""
    paths = trajectory_artifact_paths(combo_index, mode_key)
    if os.path.isfile(paths['npz']) or os.path.isfile(paths['csv']):
        return paths
    if mode_key == 'normal':
        base, _ = os.path.splitext(paths['json'])
        leg_csv, leg_npz = f'{base}.csv', f'{base}_traj.npz'
        if os.path.isfile(leg_npz):
            return {**paths, 'csv': leg_csv, 'npz': leg_npz}
        if os.path.isfile(leg_csv):
            return {**paths, 'csv': leg_csv, 'npz': leg_npz if os.path.isfile(leg_npz) else paths['npz']}
    return paths


def optimization_summary_from_json(data, mode_key='normal'):
    """Extract saved optimization summary for one mode from trajectory JSON."""
    if not isinstance(data, dict):
        return None
    opts = data.get('optimizations')
    if isinstance(opts, dict) and opts.get(mode_key):
        return opts[mode_key]
    legacy = data.get('optimization')
    if not legacy:
        return None
    method_idx = legacy.get('method')
    if mode_key == 'min_time':
        return legacy if method_idx == METHOD_MIN_TIME_INDEX else None
    if method_idx is None or method_idx != METHOD_MIN_TIME_INDEX:
        return legacy
    return None


def optimization_summary_matches_mode(summary, mode_key='normal'):
    """True when a summary dict belongs to the requested optimization mode."""
    if not summary:
        return False
    saved_mode = summary.get('mode_key')
    if saved_mode in ('normal', 'min_time'):
        return saved_mode == mode_key
    method_idx = summary.get('method')
    if mode_key == 'min_time':
        return method_idx == METHOD_MIN_TIME_INDEX
    if method_idx is None:
        return mode_key == 'normal'
    return method_idx != METHOD_MIN_TIME_INDEX


def load_trajectory_from_export_csv(csv_path):
    """Rebuild ``xs`` / ``time_states`` from a GUI-exported trajectory CSV."""
    numeric_rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if s.lower().startswith('t,') and 'x,' in s.lower():
                    continue
                parts = [p.strip() for p in s.split(',')]
                if len(parts) < 13:
                    continue
                try:
                    numeric_rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    except OSError:
        return None
    if not numeric_rows:
        return None
    data = np.asarray(numeric_rows, dtype=float)
    n = int(data.shape[0])
    t = data[:, 0]
    xs = np.zeros((n, 17), dtype=float)
    xs[:, 0:3] = data[:, 1:4]
    xs[:, 3:6] = data[:, 4:7]
    xs[:, 6:10] = data[:, 7:11]
    xs[:, 10:13] = data[:, 11:14]
    us = None
    if data.shape[1] >= 21:
        u = data[:, 17:21]
        us = np.vstack([u, u[-1:]])
    dt = float(np.median(np.diff(t))) if n > 1 else 0.05
    return {
        'xs': xs,
        'us': us,
        'us_actual': None,
        'plot_dt': dt,
        'dt': dt,
        'time_states': t,
        'segment_boundary_indices': None,
        'optimal_segment_times': None,
    }


def trajectory_path_length_m(xs):
    """Total path length [m] along optimized position samples."""
    arr = np.asarray(xs, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
        return 0.0
    p = arr[:, 0:3]
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


class _SavedCostLogger:
    """Minimal stand-in for crocoddyl.CallbackLogger when reloading saved costs."""

    def __init__(self, costs):
        self.costs = list(costs or [])


def nominal_segment_duration(dt, N):
    """Default leg duration [s] when adding a waypoint (dt × N)."""
    return float(dt) * int(N)


def waypoints_to_json_list(waypoints):
    """Normalize waypoint rows for JSON export."""
    out = []
    for w in waypoints:
        row = _normalize_waypoint_row(w)
        out.append([
            float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]),
        ])
    return out


def waypoints_for_optimizer(waypoints):
    """Normalize to [x,y,z,yaw,time_s] for solvers."""
    return [_normalize_waypoint_row(w) for w in waypoints]


def leg_duration_s(waypoints, index):
    """Duration [s] of the leg ending at waypoint ``index``."""
    if index <= 0 or index >= len(waypoints):
        return 0.0
    a = _normalize_waypoint_row(waypoints[index - 1])
    b = _normalize_waypoint_row(waypoints[index])
    return float(b[4]) - float(a[4])


OPTIMIZATION_METHOD_ITEMS = (
    'Method 1: Custom calcDiff (Slower, Numerical)',
    'Method 2: Pinocchio + FDDP (Penalty constraints)',
    'Method 3: Pinocchio + BoxFDDP (Native control bounds)',
    'Method 4: Acados (Native constraints)',
    'Method 5: Acados min-time (free segment duration)',
    'Method 6: Spannagl min-fuel FFTF (free t_f per leg)',
    'Method 7: Acados free t_f + EXTERNAL (thrust/TVC/time)',
)


def _populate_optimization_method_combo(combo):
    """Fill a QComboBox with the standard optimization method list."""
    combo.clear()
    for label in OPTIMIZATION_METHOD_ITEMS:
        combo.addItem(label)


def _build_optimization_method_group(combo):
    """Return a group box with label + method combo (shared by both tabs)."""
    group = QGroupBox('Optimization Method')
    outer = QVBoxLayout()
    outer.setSpacing(4)
    row = QHBoxLayout()
    row.setSpacing(3)
    _populate_optimization_method_combo(combo)
    row.addWidget(QLabel('Method:'))
    row.addWidget(combo, 1)
    outer.addLayout(row)
    group.setLayout(outer)
    return group


class TabScrollArea(QScrollArea):
    """Keep tab content at natural height; scroll vertically when needed."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self._page = page
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setWidget(page)
        self.setWidgetResizable(False)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        vw = self.viewport().width()
        if vw > 0:
            self._page.setFixedWidth(vw)


def _normalize_waypoint_row(row, default_t=0.0):
    """Return [x, y, z, yaw_deg, arrival_time_s]."""
    r = list(row)
    while len(r) < 4:
        r.append(0.0)
    t = float(r[4]) if len(r) >= 5 else float(default_t)
    return [float(r[0]), float(r[1]), float(r[2]), float(r[3]), t]


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
            waypoints_raw = self.params.get('waypoints', [])
            waypoints = waypoints_for_optimizer(waypoints_raw)
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
            
            # Check if we have waypoints
            if len(waypoints) < 2:
                self.error.emit("Need at least 2 waypoints (start and at least one waypoint)")
                return

            for i in range(len(waypoints) - 1):
                if waypoints[i][4] >= waypoints[i + 1][4]:
                    self.error.emit(
                        f"Waypoint {i + 1} arrival time ({waypoints[i + 1][4]:.2f}s) must be "
                        f"greater than waypoint {i} time ({waypoints[i][4]:.2f}s)"
                    )
                    return

            durations = [
                waypoints[i + 1][4] - waypoints[i][4] for i in range(len(waypoints) - 1)
            ]
            
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
        self.params_file_path = os.path.join(ROOT_DIR, DEFAULT_GUI_PARAMS_FILENAME)
        # Cached most-recent optimized trajectory for CSV export
        self.last_trajectory = None
        self.opt_summary = None
        self._trajectory_mode_cache = {}
        self.last_csv_path = None
        self.default_traj_csv_path = DEFAULT_TRAJ_CSV_PATH
        self.saved_waypoints_path = DEFAULT_SAVED_WAYPOINTS_PATH
        self._tvc_launch_proc = None
        self._traj_player_proc = None
        self._method_sync_guard = False
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

        # Left panel: tabbed sidebar (no outer scroll; Parameters tab scrolls internally)
        left_panel = self.create_left_panel()
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(480)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        main_layout.addWidget(left_panel, 0)

        # Right panel: display panel
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 1)

        # Load default trajectory after plot axes exist (restore may redraw panels).
        self._load_trajectory_for_combo_index(0, quiet=True)
        self._traj_combo_prev_index = 0
        if hasattr(self, 'traj_opt_mode_combo'):
            self._traj_opt_mode_prev_index = self.traj_opt_mode_combo.currentIndex()
        if hasattr(self, 'tracking_source_combo'):
            self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())

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
        
    def create_left_panel(self):
        """Create tabbed left sidebar (Trajectory / Parameters)."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setSpacing(4)
        outer.setContentsMargins(0, 0, 0, 0)

        traj_tab, params_tab = self.create_parameter_panels()
        traj_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        params_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(True)
        self.left_tabs.addTab(TabScrollArea(traj_tab), 'Trajectory')
        self.left_tabs.addTab(TabScrollArea(params_tab), 'Parameters')
        self.left_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        outer.addWidget(self.left_tabs, 1)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(20)
        outer.addWidget(self.progress)
        outer.addWidget(QLabel('Status:'))
        self.status_text = QTextEdit()
        self.status_text.setFixedHeight(LEFT_STATUS_TEXT_HEIGHT)
        self.status_text.setReadOnly(True)
        outer.addWidget(self.status_text)

        return container

    def create_parameter_panels(self):
        """Build content widgets for each left-sidebar tab."""
        traj_tab = QWidget()
        layout = QVBoxLayout(traj_tab)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        params_tab = QWidget()
        adv_layout = QVBoxLayout(params_tab)
        adv_layout.setSpacing(4)
        adv_layout.setContentsMargins(4, 4, 4, 4)
        
        # Waypoints management
        waypoint_group = QGroupBox('Waypoints')
        waypoint_layout = QVBoxLayout()

        traj_pick_row = QHBoxLayout()
        traj_pick_row.addWidget(QLabel('Trajectory:'))
        self.trajectory_preset_combo = QComboBox()
        for slot in TRAJECTORY_COMBO_SLOTS:
            self.trajectory_preset_combo.addItem(slot['label'])
        self.trajectory_preset_combo.setToolTip(
            'Traj1 / Traj2 load from trajs/presets/*.json or saved_waypoints.json when saved.\n'
            f'Traj2 — {DEFAULT_SAVED_WAYPOINTS_PATH}\n'
            'Custom — trajs/presets/custom_waypoints.json or parameters file.\n'
            'After optimization, use Save Trajectory (next to Start Optimization) '
            'to store results for this slot.'
        )
        traj_pick_row.addWidget(self.trajectory_preset_combo, 1)
        waypoint_layout.addLayout(traj_pick_row)

        traj_mode_row = QHBoxLayout()
        traj_mode_row.addWidget(QLabel('Optimization:'))
        self.traj_opt_mode_combo = QComboBox()
        self.traj_opt_mode_combo.addItem('Normal')
        self.traj_opt_mode_combo.addItem('Minimum time')
        self.traj_opt_mode_combo.setToolTip(
            'Normal and Minimum time are separate trajectories for each slot.\n'
            'Each mode has its own saved CSV/NPZ; switching modes reloads or clears the plots.'
        )
        self.traj_opt_mode_combo.currentIndexChanged.connect(self._on_traj_opt_mode_changed)
        traj_mode_row.addWidget(self.traj_opt_mode_combo, 1)
        waypoint_layout.addLayout(traj_mode_row)
        self._normal_method_index = METHOD_NORMAL_DEFAULT_INDEX
        self._traj_opt_mode_guard = False

        self.trajectory_storage_path_label = QLabel()
        self.trajectory_storage_path_label.setWordWrap(True)
        self.trajectory_storage_path_label.setStyleSheet('color: #555;')
        waypoint_layout.addWidget(self.trajectory_storage_path_label)

        self.waypoint_table = QTableWidget()
        self.waypoint_table.setColumnCount(len(WP_TABLE_HEADERS))
        self.waypoint_table.setHorizontalHeaderLabels(list(WP_TABLE_HEADERS))
        self.waypoint_table.verticalHeader().setVisible(False)
        self.waypoint_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.waypoint_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.waypoint_table.setAlternatingRowColors(True)
        self.waypoint_table.setMinimumHeight(120)
        self.waypoint_table.setMaximumHeight(220)
        hdr = self.waypoint_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        hdr.setSectionResizeMode(WP_COL_IDX, QHeaderView.ResizeToContents)
        self.waypoint_table.itemChanged.connect(self._on_waypoint_table_item_changed)
        waypoint_layout.addWidget(self.waypoint_table)

        waypoint_btn_layout = QHBoxLayout()
        self.add_waypoint_btn = QPushButton('Add row')
        self.add_waypoint_btn.clicked.connect(self.add_waypoint)
        self.remove_waypoint_btn = QPushButton('Remove row')
        self.remove_waypoint_btn.clicked.connect(self.remove_waypoint)
        waypoint_btn_layout.addWidget(self.add_waypoint_btn)
        waypoint_btn_layout.addWidget(self.remove_waypoint_btn)
        waypoint_layout.addLayout(waypoint_btn_layout)

        waypoint_group.setLayout(waypoint_layout)
        layout.addWidget(waypoint_group)

        self._waypoint_table_updating = False
        _, grass = TRAJECTORY_PRESETS[0]
        self.waypoints = [_normalize_waypoint_row(w) for w in grass]
        self._populate_waypoint_table()
        self.trajectory_preset_combo.setCurrentIndex(0)
        self._refresh_trajectory_storage_path_label()
        self._traj_combo_prev_index = 0
        self._traj_opt_mode_prev_index = TRAJ_OPT_MODE_NORMAL
        self.trajectory_preset_combo.currentIndexChanged.connect(self.on_trajectory_preset_changed)
        
        # Optimization method (Trajectory tab; synced with Parameters tab)
        self.method_combo = QComboBox()
        layout.addWidget(_build_optimization_method_group(self.method_combo))
        self.method_combo.currentIndexChanged.connect(
            lambda idx: self._on_method_combo_changed(idx, self.method_combo)
        )

        # Parameters tab: duplicate method selector at top (linked to Trajectory tab)
        self.method_combo_params = QComboBox()
        adv_layout.addWidget(_build_optimization_method_group(self.method_combo_params))
        self.method_combo_params.currentIndexChanged.connect(
            lambda idx: self._on_method_combo_changed(idx, self.method_combo_params)
        )

        self.unified_checkbox = QCheckBox('Unified (merge all segments)')
        self.unified_checkbox.setToolTip(
            'Merge all waypoint segments into one optimization problem (Method 2/3/4)'
        )
        self.unified_checkbox.setChecked(False)
        self.unified_checkbox.setEnabled(False)
        self.unified_interp_guess_checkbox = QCheckBox('Unified (Acados)')
        self.unified_interp_guess_checkbox.setToolTip(
            'If checked, Method 4 unified uses linearly interpolated state over each segment as the initial '
            'guess. Default off: all nodes use the start state x0.'
        )
        self.unified_interp_guess_checkbox.setChecked(False)
        self.unified_interp_guess_checkbox.setEnabled(False)

        self.unified_checkbox.stateChanged.connect(self._refresh_unified_interp_guess_enabled)
        self.method_combo.currentIndexChanged.connect(self._update_unified_checkbox_state)
        self.method_combo_params.currentIndexChanged.connect(self._update_unified_checkbox_state)
        self._update_unified_checkbox_state(self.method_combo.currentIndex())

        params_tabs = QTabWidget()
        params_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Optimization parameters (+ method options inside same group)
        opt_group = QGroupBox('Optimization Parameters')
        opt_outer = QVBoxLayout()
        opt_outer.setSpacing(4)

        method_row2 = QHBoxLayout()
        method_row2.setSpacing(8)
        method_row2.addWidget(self.unified_checkbox)
        method_row2.addWidget(self.unified_interp_guess_checkbox)
        method_row2.addStretch(1)
        opt_outer.addLayout(method_row2)

        opt_layout = QGridLayout()
        opt_layout.setSpacing(3)  # Reduce spacing
        
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 0.1)
        self.dt_spin.setValue(0.05)
        self.dt_spin.setSingleStep(0.01)
        self.dt_spin.setDecimals(3)
        self.dt_spin.setMaximumWidth(100)
        self.dt_spin.setToolTip('Discretization step [s] for shooting methods.')

        self.N_spin = QSpinBox()
        self.N_spin.setRange(10, 500)
        self.N_spin.setValue(100)
        self.N_spin.setMaximumWidth(100)
        self.N_spin.setToolTip('Number of shooting intervals per segment (when not overridden by leg duration / dt).')
        self.dt_spin.valueChanged.connect(self._on_default_leg_duration_hint_changed)
        self.N_spin.valueChanged.connect(self._on_default_leg_duration_hint_changed)
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setValue(100)
        self.max_iter_spin.setMaximumWidth(100)
        
        # Two pairs per row (4 grid cols)
        opt_layout.addWidget(QLabel('Time Step (s):'), 0, 0)
        opt_layout.addWidget(self.dt_spin, 0, 1)
        opt_layout.addWidget(QLabel('Time Steps:'), 0, 2)
        opt_layout.addWidget(self.N_spin, 0, 3)
        opt_layout.addWidget(QLabel('Max Iter:'), 1, 0)
        opt_layout.addWidget(self.max_iter_spin, 1, 1)

        opt_outer.addLayout(opt_layout)
        opt_group.setLayout(opt_outer)

        opt_tab = QWidget()
        opt_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
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
        self.w_p.setMaximumWidth(100)  # Reduce width
        
        self.w_v = QDoubleSpinBox()
        self.w_v.setRange(0, 1000)
        self.w_v.setValue(01.0)
        self.w_v.setDecimals(3)
        self.w_v.setMaximumWidth(100)
        
        self.w_R = QDoubleSpinBox()
        self.w_R.setRange(0, 1000)
        self.w_R.setValue(1.0)
        self.w_R.setDecimals(3)
        self.w_R.setMaximumWidth(100)
        
        self.w_yaw = QDoubleSpinBox()
        self.w_yaw.setRange(0, 1000)
        self.w_yaw.setValue(1.0)
        self.w_yaw.setDecimals(3)
        self.w_yaw.setMaximumWidth(100)
        self.w_yaw.setToolTip('Separate weight for yaw (default same as Attitude R)')
        
        self.w_w = QDoubleSpinBox()
        self.w_w.setRange(0, 1000)
        self.w_w.setValue(0.1)
        self.w_w.setDecimals(3)
        self.w_w.setMaximumWidth(100)
        
        self.w_u = QDoubleSpinBox()
        self.w_u.setRange(0, 100)
        self.w_u.setValue(1.0)
        self.w_u.setDecimals(3)
        self.w_u.setMaximumWidth(100)
        
        self.w_du = QDoubleSpinBox()
        self.w_du.setRange(0, 100)
        self.w_du.setValue(1.0)  # Default: enable control rate penalty
        self.w_du.setDecimals(3)
        self.w_du.setMaximumWidth(100)
        
        # Constraint penalty coefficients
        self.k_bound = QDoubleSpinBox()
        self.k_bound.setRange(0, 10000)
        self.k_bound.setValue(200.0)
        self.k_bound.setDecimals(1)
        self.k_bound.setMaximumWidth(100)
        
        self.k_state_bound = QDoubleSpinBox()
        self.k_state_bound.setRange(0, 10000)
        self.k_state_bound.setValue(20.0)  # Lower default to avoid constraint gradient dominating position cost
        self.k_state_bound.setDecimals(1)
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
        self.tau_pitch_spin.setMaximumWidth(80)
        self.tau_pitch_spin.setToolTip('Pitch channel time constant (s)')
        self.tau_roll_spin = QDoubleSpinBox()
        self.tau_roll_spin.setRange(0.001, 1.0)
        self.tau_roll_spin.setValue(0.05)
        self.tau_roll_spin.setDecimals(3)
        self.tau_roll_spin.setSingleStep(0.01)
        self.tau_roll_spin.setMaximumWidth(80)
        self.tau_roll_spin.setToolTip('Roll channel time constant (s)')
        self.tau_T_spin = QDoubleSpinBox()
        self.tau_T_spin.setRange(0.001, 10.0)
        self.tau_T_spin.setValue(0.5)
        self.tau_T_spin.setDecimals(3)
        self.tau_T_spin.setSingleStep(0.01)
        self.tau_T_spin.setMaximumWidth(80)
        self.tau_T_spin.setToolTip('Thrust channel time constant (s)')
        self.tau_yaw_spin = QDoubleSpinBox()
        self.tau_yaw_spin.setRange(0.001, 1.0)
        self.tau_yaw_spin.setValue(0.05)
        self.tau_yaw_spin.setDecimals(3)
        self.tau_yaw_spin.setSingleStep(0.01)
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
        self.min_time_T_max_scale_spin.setMaximumWidth(100)
        self.min_time_T_max_scale_spin.setToolTip(
            'Upper bound scale: T_max = (waypoint leg Δt) × this factor [—]. '
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
        cost_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        cost_tab_layout = QVBoxLayout()
        cost_tab_layout.addWidget(cost_group)
        cost_tab_layout.addWidget(self.min_time_duration_group)
        cost_tab_layout.addWidget(terminal_cost_group)
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
        self.th_p_max.setMaximumWidth(100)

        self.th_r_max = QDoubleSpinBox()
        self.th_r_max.setRange(0, 90)  # Range in degrees
        self.th_r_max.setValue(10.0)  # 10 degrees
        self.th_r_max.setDecimals(1)
        self.th_r_max.setMaximumWidth(100)

        self.T_max = QDoubleSpinBox()
        self.T_max.setRange(0, 100)
        self.T_max.setValue(25.0)
        self.T_max.setDecimals(2)
        self.T_max.setMaximumWidth(100)

        self.tau_yaw_max = QDoubleSpinBox()
        self.tau_yaw_max.setRange(0, 10)
        self.tau_yaw_max.setValue(1.0)
        self.tau_yaw_max.setDecimals(2)
        self.tau_yaw_max.setMaximumWidth(100)

        self.v_horizontal_max = QDoubleSpinBox()
        self.v_horizontal_max.setRange(0, 100)
        self.v_horizontal_max.setValue(1.0)
        self.v_horizontal_max.setDecimals(1)
        self.v_horizontal_max.setMaximumWidth(100)

        self.v_vertical_max = QDoubleSpinBox()
        self.v_vertical_max.setRange(0, 100)
        self.v_vertical_max.setValue(3.0)
        self.v_vertical_max.setDecimals(1)
        self.v_vertical_max.setMaximumWidth(100)

        self.roll_max = QDoubleSpinBox()
        self.roll_max.setRange(0, 180)
        self.roll_max.setValue(10.0)
        self.roll_max.setDecimals(1)
        self.roll_max.setMaximumWidth(100)

        self.pitch_max = QDoubleSpinBox()
        self.pitch_max.setRange(0, 180)
        self.pitch_max.setValue(10.0)
        self.pitch_max.setDecimals(1)
        self.pitch_max.setMaximumWidth(100)

        self.yaw_max = QDoubleSpinBox()
        self.yaw_max.setRange(0, 180)
        self.yaw_max.setValue(180.0)
        self.yaw_max.setDecimals(1)
        self.yaw_max.setMaximumWidth(100)

        self.w_max = QDoubleSpinBox()
        self.w_max.setRange(0, 10)
        self.w_max.setValue(2.0)
        self.w_max.setDecimals(2)
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
        constraints_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        constraints_tab_layout = QVBoxLayout()
        constraints_tab_layout.addWidget(constraints_group)
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
        self.mass.setMaximumWidth(100)
        
        # Moment of inertia (diagonal components)
        self.Ixx = QDoubleSpinBox()
        self.Ixx.setRange(0.0001, 1.0)
        self.Ixx.setValue(0.02)
        self.Ixx.setDecimals(2)
        self.Ixx.setMaximumWidth(100)
        
        self.Iyy = QDoubleSpinBox()
        self.Iyy.setRange(0.0001, 1.0)
        self.Iyy.setValue(0.02)
        self.Iyy.setDecimals(2)
        self.Iyy.setMaximumWidth(100)
        
        self.Izz = QDoubleSpinBox()
        self.Izz.setRange(0.0001, 1.0)
        self.Izz.setValue(0.01)
        self.Izz.setDecimals(2)
        self.Izz.setMaximumWidth(100)
        
        # Thrust position (r_thrust)
        self.r_thrust_x = QDoubleSpinBox()
        self.r_thrust_x.setRange(-1.0, 1.0)
        self.r_thrust_x.setValue(0.0)
        self.r_thrust_x.setDecimals(2)
        self.r_thrust_x.setMaximumWidth(100)
        
        self.r_thrust_y = QDoubleSpinBox()
        self.r_thrust_y.setRange(-1.0, 1.0)
        self.r_thrust_y.setValue(0.0)
        self.r_thrust_y.setDecimals(2)
        self.r_thrust_y.setMaximumWidth(100)
        
        self.r_thrust_z = QDoubleSpinBox()
        self.r_thrust_z.setRange(-1.0, 1.0)
        self.r_thrust_z.setValue(-0.2)
        self.r_thrust_z.setDecimals(2)
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

        physics_tab = QWidget()
        physics_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        physics_tab_layout = QVBoxLayout()
        physics_tab_layout.addWidget(physics_group)
        physics_tab.setLayout(physics_tab_layout)
        params_tabs.insertTab(1, physics_tab, 'Physical')

        adv_layout.addWidget(params_tabs)
        
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
        self._set_method_combo_index(3)
        self.on_method_changed(3)
        self._update_method_combo_enabled_for_traj_mode()
        
        # Start optimization + save + load parameters (one row)
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton('Start Optimization')
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.run_btn.clicked.connect(self.start_optimization)
        button_layout.addWidget(self.run_btn)
        self.btn_save_trajectory = QPushButton('Save Trajectory')
        self.btn_save_trajectory.setToolTip(
            'Save waypoints, optimized CSV, and optimization summary for the '
            'currently selected trajectory slot (run optimization first).'
        )
        self.btn_save_trajectory.setEnabled(False)
        self.btn_save_trajectory.clicked.connect(self.save_trajectory)
        button_layout.addWidget(self.btn_save_trajectory)
        self.btn_load_params = QPushButton('Load parameters...')
        self.btn_load_params.setToolTip('Load settings from a JSON file; subsequent Save writes to this file')
        self.btn_load_params.clicked.connect(self.load_parameters)
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

        # Trajectory export row: quick save + save-as dialog
        traj_io_layout = QHBoxLayout()
        self.btn_save_traj = QPushButton('Save traj CSV')
        self.btn_save_traj.setToolTip(
            f'Write the latest optimized trajectory to the default path:\n{DEFAULT_TRAJ_CSV_PATH}'
        )
        self.btn_save_traj.setEnabled(False)
        self.btn_save_traj.clicked.connect(self.save_trajectory_default)
        traj_io_layout.addWidget(self.btn_save_traj)

        self.btn_save_traj_csv = QPushButton('Save traj CSV...')
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

        self.traj_default_path_label = QLabel()
        self.traj_default_path_label.setWordWrap(True)
        self.traj_default_path_label.setStyleSheet('color: #555;')
        self.traj_default_path_label.setText(f'Default: {DEFAULT_TRAJ_CSV_PATH}')
        layout.addWidget(self.traj_default_path_label)

        # ROS2 simulation / tracking controls (label + button rows)
        sim_group = QGroupBox('Simulation & Tracking')
        sim_layout = QGridLayout()
        sim_layout.setSpacing(3)

        self.btn_start_px4_sitl = QPushButton('Start')
        self.btn_start_px4_sitl.setToolTip('ros2 launch tvc_controller tvc.launch.py')
        self.btn_start_px4_sitl.clicked.connect(self.start_px4_sitl)
        self.btn_stop_px4_sitl = QPushButton('Stop')
        self.btn_stop_px4_sitl.setToolTip('Terminate the TVC launch stack (PX4 SITL + agents)')
        self.btn_stop_px4_sitl.clicked.connect(self.stop_px4_sitl)
        self.btn_stop_px4_sitl.setEnabled(False)
        self.lbl_px4_sitl_status = QLabel('Stopped')
        self.lbl_px4_sitl_status.setStyleSheet('color: #888;')
        sim_layout.addWidget(QLabel('PX4 SITL:'), 0, 0)
        sim_layout.addWidget(self.btn_start_px4_sitl, 0, 1)
        sim_layout.addWidget(self.btn_stop_px4_sitl, 0, 2)
        sim_layout.addWidget(self.lbl_px4_sitl_status, 0, 3)

        self.btn_start_tracking = QPushButton('Start')
        self.btn_start_tracking.setToolTip(
            'ros2 launch tvc_controller tvc_traj_player.launch.py '
            '(uses last saved CSV when available)'
        )
        self.btn_start_tracking.clicked.connect(self.start_tracking_node)
        self.btn_stop_tracking = QPushButton('Stop')
        self.btn_stop_tracking.setToolTip('Terminate tvc_traj_player launch')
        self.btn_stop_tracking.clicked.connect(self.stop_tracking_node)
        self.btn_stop_tracking.setEnabled(False)
        self.lbl_tracking_status = QLabel('Stopped')
        self.lbl_tracking_status.setStyleSheet('color: #888;')
        sim_layout.addWidget(QLabel('Tracking:'), 1, 0)
        sim_layout.addWidget(self.btn_start_tracking, 1, 1)
        sim_layout.addWidget(self.btn_stop_tracking, 1, 2)
        sim_layout.addWidget(self.lbl_tracking_status, 1, 3)

        self.tracking_source_combo = QComboBox()
        self.tracking_source_combo.addItem('Current trajectory')
        self.tracking_source_combo.addItem('CSV file')
        self.tracking_source_combo.addItem('GUI waypoints')
        self.tracking_source_combo.setToolTip(
            'Trajectory source for tvc_traj_player:\n'
            'Current trajectory — CSV saved for the selected trajectory slot;\n'
            'CSV file — pick an existing trajectory CSV;\n'
            'GUI waypoints — GotoSetpoint through current waypoint list.'
        )
        self.tracking_source_combo.currentIndexChanged.connect(self._on_tracking_source_changed)
        sim_layout.addWidget(QLabel('Track source:'), 2, 0)
        sim_layout.addWidget(self.tracking_source_combo, 2, 1, 1, 3)

        self.tracking_csv_widget = QWidget()
        tracking_csv_row = QHBoxLayout(self.tracking_csv_widget)
        tracking_csv_row.setContentsMargins(0, 0, 0, 0)
        tracking_csv_row.setSpacing(4)
        self.tracking_csv_edit = QLineEdit()
        self.tracking_csv_edit.setPlaceholderText('Path to trajectory CSV')
        self.tracking_csv_edit.setText(DEFAULT_TRAJ_CSV_PATH)
        self.tracking_csv_edit.textChanged.connect(self._on_tracking_csv_edited)
        tracking_csv_row.addWidget(self.tracking_csv_edit, 1)
        self.tracking_csv_browse = QPushButton('Browse…')
        self.tracking_csv_browse.clicked.connect(self.browse_tracking_csv)
        tracking_csv_row.addWidget(self.tracking_csv_browse)
        sim_layout.addWidget(self.tracking_csv_widget, 3, 0, 1, 4)
        self.tracking_csv_widget.setVisible(False)

        self.lbl_tracking_source_info = QLabel()
        self.lbl_tracking_source_info.setWordWrap(True)
        self.lbl_tracking_source_info.setStyleSheet('color: #555;')
        sim_layout.addWidget(self.lbl_tracking_source_info, 4, 0, 1, 4)

        self.btn_clear_rviz_traj = QPushButton('Clear executed path')
        self.btn_clear_rviz_traj.setToolTip(
            'Clear RViz executed setpoint trail (/tvc_traj_player/executed_path) and '
            'current setpoint marker. Planned trajectory is kept; it is (re)loaded when '
            'you start the tracking node.'
        )
        self.btn_clear_rviz_traj.clicked.connect(
            lambda: self.clear_rviz_trajectory_display(quiet=False)
        )
        sim_layout.addWidget(self.btn_clear_rviz_traj, 5, 0, 1, 4)
        self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        return traj_tab, params_tab
    
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
        gs = GridSpec(
            4, 4, figure=self.fig, height_ratios=[0.28, 1.0, 1.0, 1.0],
            hspace=0.38, wspace=0.3,
        )
        self.fig.suptitle('TVC Rocket Trajectory Optimization', 
                         fontsize=16, fontweight='bold', y=0.995)

        self.ax_opt_info = self.fig.add_subplot(gs[0, :])
        self.ax_opt_info.axis('off')
        
        # Second row: 3D trajectory and convergence curve
        # 1. 3D position trajectory (occupies 2 positions)
        self.ax_3d = self.fig.add_subplot(gs[1, 0:2], projection='3d')
        self.ax_3d.set_xlabel('X (m)', fontsize=10)
        self.ax_3d.set_ylabel('Y (m)', fontsize=10)
        self.ax_3d.set_zlabel('Z (m)', fontsize=10)
        self.ax_3d.set_title('3D Position Trajectory', fontsize=11, fontweight='bold')
        self.ax_3d.grid(True, alpha=0.3)
        
        # 2. Cost convergence curve (occupies 2 positions)
        self.ax_cost = self.fig.add_subplot(gs[1, 2:4])
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        
        # Third row: position states
        # 3. Position
        self.ax_pos = self.fig.add_subplot(gs[2, 0])
        self.ax_pos.set_xlabel('Time (s)', fontsize=9)
        self.ax_pos.set_ylabel('Position (m)', fontsize=9)
        self.ax_pos.set_title('Position', fontsize=10, fontweight='bold')
        self.ax_pos.grid(True, alpha=0.3)
        
        # 4. Velocity
        self.ax_vel = self.fig.add_subplot(gs[2, 1])
        self.ax_vel.set_xlabel('Time (s)', fontsize=9)
        self.ax_vel.set_ylabel('Velocity (m/s)', fontsize=9)
        self.ax_vel.set_title('Linear Velocity', fontsize=10, fontweight='bold')
        self.ax_vel.grid(True, alpha=0.3)
        
        # 5. Euler angles (left of angular velocity)
        self.ax_euler = self.fig.add_subplot(gs[2, 2])
        self.ax_euler.set_xlabel('Time (s)', fontsize=9)
        self.ax_euler.set_ylabel('Euler Angles (deg)', fontsize=9)
        self.ax_euler.set_title('Attitude (Euler)', fontsize=10, fontweight='bold')
        self.ax_euler.grid(True, alpha=0.3)
        
        # 6. Angular velocity
        self.ax_angvel = self.fig.add_subplot(gs[2, 3])
        self.ax_angvel.set_xlabel('Time (s)', fontsize=9)
        self.ax_angvel.set_ylabel('Angular Vel (°/s)', fontsize=9)
        self.ax_angvel.set_title('Angular Velocity', fontsize=10, fontweight='bold')
        self.ax_angvel.grid(True, alpha=0.3)
        
        # Fourth row: control inputs
        # 7. TVC Pitch angle
        self.ax_pitch = self.fig.add_subplot(gs[3, 0])
        self.ax_pitch.set_xlabel('Time (s)', fontsize=9)
        self.ax_pitch.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_pitch.set_title('TVC Pitch Angle', fontsize=10, fontweight='bold')
        self.ax_pitch.grid(True, alpha=0.3)
        
        # 8. TVC Roll angle
        self.ax_roll = self.fig.add_subplot(gs[3, 1])
        self.ax_roll.set_xlabel('Time (s)', fontsize=9)
        self.ax_roll.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_roll.set_title('TVC Roll Angle', fontsize=10, fontweight='bold')
        self.ax_roll.grid(True, alpha=0.3)
        
        # 9. Thrust
        self.ax_thrust = self.fig.add_subplot(gs[3, 2])
        self.ax_thrust.set_xlabel('Time (s)', fontsize=9)
        self.ax_thrust.set_ylabel('Thrust (N)', fontsize=9)
        self.ax_thrust.set_title('Thrust', fontsize=10, fontweight='bold')
        self.ax_thrust.grid(True, alpha=0.3)
        
        # 10. Yaw torque
        self.ax_yaw = self.fig.add_subplot(gs[3, 3])
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
        self._refresh_opt_info_display()

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
    
    def _method_combos(self):
        """Both linked optimization-method selectors (Trajectory + Parameters tabs)."""
        combos = [self.method_combo]
        if hasattr(self, 'method_combo_params'):
            combos.append(self.method_combo_params)
        return combos

    def _set_method_combo_index(self, index):
        """Set method index on all linked combos without triggering handlers."""
        for combo in self._method_combos():
            combo.blockSignals(True)
            idx = max(0, min(combo.count() - 1, int(index)))
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _current_traj_opt_mode_key(self):
        if not hasattr(self, 'traj_opt_mode_combo'):
            return 'normal'
        return traj_opt_mode_key(self.traj_opt_mode_combo.currentIndex())

    def _mode_cache_key(self, combo_index=None, mode_key=None):
        if combo_index is None:
            combo_index = self.trajectory_preset_combo.currentIndex()
        if mode_key is None:
            mode_key = self._current_traj_opt_mode_key()
        return (int(combo_index), str(mode_key))

    def _clone_trajectory_for_cache(self, traj, summary):
        if not traj or traj.get('xs') is None:
            return None
        cloned = {}
        for key, val in traj.items():
            if val is None:
                cloned[key] = None
            elif isinstance(val, np.ndarray):
                cloned[key] = np.asarray(val, dtype=float).copy()
            elif isinstance(val, list):
                cloned[key] = list(val)
            else:
                cloned[key] = val
        return {
            'last_trajectory': cloned,
            'opt_summary': dict(summary) if summary else None,
        }

    def _cache_results_for_slot_mode(self, combo_index, mode_key):
        """Store in-memory results under an explicit trajectory slot + mode key."""
        if not hasattr(self, '_trajectory_mode_cache'):
            self._trajectory_mode_cache = {}
        entry = self._clone_trajectory_for_cache(self.last_trajectory, self.opt_summary)
        if entry is not None:
            if not optimization_summary_matches_mode(entry.get('opt_summary'), mode_key):
                return
            self._trajectory_mode_cache[(int(combo_index), str(mode_key))] = entry

    def _cache_current_mode_results(self):
        """Keep in-memory results for the current trajectory slot + mode."""
        self._cache_results_for_slot_mode(
            self.trajectory_preset_combo.currentIndex(),
            self._current_traj_opt_mode_key(),
        )

    def _enable_trajectory_export_buttons(self, enabled=True):
        if hasattr(self, 'btn_save_trajectory'):
            self.btn_save_trajectory.setEnabled(enabled)
        if hasattr(self, 'btn_save_traj'):
            self.btn_save_traj.setEnabled(enabled)
        if hasattr(self, 'btn_save_traj_csv'):
            self.btn_save_traj_csv.setEnabled(enabled)

    def _set_traj_opt_mode_combo_index(self, mode_index):
        if not hasattr(self, 'traj_opt_mode_combo'):
            return
        mode_index = max(0, min(self.traj_opt_mode_combo.count() - 1, int(mode_index)))
        self.traj_opt_mode_combo.blockSignals(True)
        self.traj_opt_mode_combo.setCurrentIndex(mode_index)
        self.traj_opt_mode_combo.blockSignals(False)

    def _update_method_combo_enabled_for_traj_mode(self):
        """Lock method selector when Minimum time mode forces Method 5."""
        if not hasattr(self, 'traj_opt_mode_combo'):
            return
        lock = self.traj_opt_mode_combo.currentIndex() == TRAJ_OPT_MODE_MIN_TIME
        for combo in self._method_combos():
            combo.setEnabled(not lock)

    def _on_traj_opt_mode_changed(self, mode_index):
        """Normal vs minimum-time; min-time selects Method 5 (Acados min-time)."""
        if self._traj_opt_mode_guard:
            return
        self._traj_opt_mode_guard = True
        try:
            old_mode_key = traj_opt_mode_key(
                getattr(self, '_traj_opt_mode_prev_index', TRAJ_OPT_MODE_NORMAL)
            )
            combo_idx = self.trajectory_preset_combo.currentIndex()
            self._cache_results_for_slot_mode(combo_idx, old_mode_key)
            if int(mode_index) == TRAJ_OPT_MODE_MIN_TIME:
                cur = self.method_combo.currentIndex()
                if cur != METHOD_MIN_TIME_INDEX:
                    self._normal_method_index = cur
                self._set_method_combo_index(METHOD_MIN_TIME_INDEX)
                self.on_method_changed(METHOD_MIN_TIME_INDEX)
            else:
                restore = getattr(self, '_normal_method_index', METHOD_NORMAL_DEFAULT_INDEX)
                if restore == METHOD_MIN_TIME_INDEX:
                    restore = METHOD_NORMAL_DEFAULT_INDEX
                self._set_method_combo_index(restore)
                self.on_method_changed(restore)
            self._update_method_combo_enabled_for_traj_mode()
            combo_idx = self.trajectory_preset_combo.currentIndex()
            if not self._restore_optimization_for_slot_and_mode(combo_idx, quiet=True):
                self._clear_optimization_results(clear_plots=True)
            self._traj_opt_mode_prev_index = int(mode_index)
            self._refresh_trajectory_storage_path_label()
            if hasattr(self, 'tracking_source_combo'):
                self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())
        finally:
            self._traj_opt_mode_guard = False

    def _apply_traj_opt_mode_from_json(self, data):
        """Restore Normal / Minimum time from trajectory JSON."""
        if not isinstance(data, dict) or not hasattr(self, 'traj_opt_mode_combo'):
            return
        mode = data.get('traj_opt_mode')
        if mode == 'min_time':
            target = TRAJ_OPT_MODE_MIN_TIME
        elif mode == 'normal':
            target = TRAJ_OPT_MODE_NORMAL
        else:
            opt = data.get('optimization') or {}
            method_idx = opt.get('method')
            target = (
                TRAJ_OPT_MODE_MIN_TIME
                if method_idx == METHOD_MIN_TIME_INDEX
                else TRAJ_OPT_MODE_NORMAL
            )
        if self.traj_opt_mode_combo.currentIndex() != target:
            self._set_traj_opt_mode_combo_index(target)
            self._on_traj_opt_mode_changed(target)
        else:
            self._update_method_combo_enabled_for_traj_mode()

    def _on_method_combo_changed(self, index, source):
        """Keep Trajectory / Parameters method combos in sync and apply defaults."""
        if self._method_sync_guard:
            return
        if (
            hasattr(self, 'traj_opt_mode_combo')
            and self.traj_opt_mode_combo.currentIndex() == TRAJ_OPT_MODE_MIN_TIME
            and index != METHOD_MIN_TIME_INDEX
        ):
            self._set_method_combo_index(METHOD_MIN_TIME_INDEX)
            return
        if (
            hasattr(self, 'traj_opt_mode_combo')
            and self.traj_opt_mode_combo.currentIndex() == TRAJ_OPT_MODE_NORMAL
        ):
            self._normal_method_index = int(index)
        self._method_sync_guard = True
        try:
            for combo in self._method_combos():
                if combo is source:
                    continue
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
            self.on_method_changed(index)
        finally:
            self._method_sync_guard = False

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
    
    def _default_leg_duration(self):
        if hasattr(self, 'dt_spin') and hasattr(self, 'N_spin'):
            return nominal_segment_duration(self.dt_spin.value(), self.N_spin.value())
        return 5.0

    def _on_default_leg_duration_hint_changed(self, *_args):
        """No-op placeholder; new rows use dt×N as default Δt."""
        pass

    def _refresh_trajectory_storage_path_label(self):
        if not hasattr(self, 'trajectory_storage_path_label'):
            return
        idx = self.trajectory_preset_combo.currentIndex() if hasattr(self, 'trajectory_preset_combo') else 0
        mode_key = self._current_traj_opt_mode_key()
        mode_label = 'Minimum time' if mode_key == 'min_time' else 'Normal'
        paths = resolve_trajectory_artifact_paths(idx, mode_key)
        label = trajectory_combo_label(idx)
        self.trajectory_storage_path_label.setText(
            f'{label} ({mode_label}) — waypoints: {paths["json"]}\n'
            f'  CSV: {paths["csv"]}'
        )

    def _opt_info_empty_message(self):
        mode_key = self._current_traj_opt_mode_key()
        mode_label = 'Minimum time' if mode_key == 'min_time' else 'Normal'
        return (
            f'No saved {mode_label} result for this trajectory — '
            f'run Start Optimization, then Save Trajectory.'
        )

    def _make_waypoint_table_item(self, text, editable=True):
        item = QTableWidgetItem(str(text))
        if editable:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
        else:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _populate_waypoint_table(self):
        """Fill table from ``self.waypoints``."""
        if not hasattr(self, 'waypoint_table'):
            return
        self._waypoint_table_updating = True
        self.waypoint_table.blockSignals(True)
        try:
            self.waypoint_table.setRowCount(len(self.waypoints))
            for i, wp in enumerate(self.waypoints):
                row = _normalize_waypoint_row(wp)
                x, y, z, yaw, t_arr = row
                leg = leg_duration_s(self.waypoints, i)
                self.waypoint_table.setItem(i, WP_COL_IDX, self._make_waypoint_table_item(i, editable=False))
                self.waypoint_table.setItem(i, WP_COL_X, self._make_waypoint_table_item(f'{x:.3f}'))
                self.waypoint_table.setItem(i, WP_COL_Y, self._make_waypoint_table_item(f'{y:.3f}'))
                self.waypoint_table.setItem(i, WP_COL_Z, self._make_waypoint_table_item(f'{z:.3f}'))
                self.waypoint_table.setItem(i, WP_COL_YAW, self._make_waypoint_table_item(f'{yaw:.1f}'))
                if i == 0:
                    self.waypoint_table.setItem(
                        i, WP_COL_LEG_DT, self._make_waypoint_table_item('—', editable=False),
                    )
                else:
                    self.waypoint_table.setItem(
                        i, WP_COL_LEG_DT, self._make_waypoint_table_item(f'{leg:.3f}'),
                    )
                self.waypoint_table.setItem(
                    i, WP_COL_T_ARR, self._make_waypoint_table_item(f'{t_arr:.3f}', editable=(i > 0)),
                )
        finally:
            self.waypoint_table.blockSignals(False)
            self._waypoint_table_updating = False
        if (
            hasattr(self, 'tracking_source_combo')
            and self.tracking_source_combo.currentIndex() == 2
        ):
            self._on_tracking_source_changed(2)

    def _sync_waypoints_from_table(self):
        """Read table into ``self.waypoints`` (used before optimize/save)."""
        if not hasattr(self, 'waypoint_table'):
            return
        self._waypoint_table_updating = True
        try:
            rows = []
            n = self.waypoint_table.rowCount()
            for i in range(n):
                def _cell(col, default='0'):
                    it = self.waypoint_table.item(i, col)
                    return it.text().strip() if it else default

                x = float(_cell(WP_COL_X))
                y = float(_cell(WP_COL_Y))
                z = float(_cell(WP_COL_Z))
                yaw = float(_cell(WP_COL_YAW))
                t_item = self.waypoint_table.item(i, WP_COL_T_ARR)
                t_arr = float(t_item.text()) if t_item and i > 0 else 0.0
                rows.append([x, y, z, yaw, t_arr])
            self.waypoints = [_normalize_waypoint_row(r) for r in rows]
        finally:
            self._waypoint_table_updating = False

    def _on_waypoint_table_item_changed(self, item):
        if self._waypoint_table_updating or item is None:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.waypoints):
            return
        self._sync_waypoints_from_table()
        try:
            if col in (WP_COL_X, WP_COL_Y, WP_COL_Z, WP_COL_YAW):
                pass
            elif col == WP_COL_LEG_DT:
                if row <= 0:
                    return
                leg = float(item.text())
                if leg <= 0:
                    raise ValueError('leg duration must be positive')
                prev_t = self.waypoints[row - 1][4]
                old_t = self.waypoints[row][4]
                shift = (prev_t + leg) - old_t
                self.waypoints[row][4] = prev_t + leg
                for j in range(row + 1, len(self.waypoints)):
                    self.waypoints[j][4] += shift
            elif col == WP_COL_T_ARR:
                if row <= 0:
                    return
                new_t = float(item.text())
                if new_t <= self.waypoints[row - 1][4]:
                    raise ValueError('arrival time must increase')
                old_t = self.waypoints[row][4]
                shift = new_t - old_t
                self.waypoints[row][4] = new_t
                for j in range(row + 1, len(self.waypoints)):
                    self.waypoints[j][4] += shift
            else:
                return
        except ValueError:
            self._populate_waypoint_table()
            return
        self._populate_waypoint_table()
        self.waypoint_table.selectRow(row)

    def add_waypoint(self):
        """Append a waypoint row (default Δt = dt×N)."""
        self._sync_waypoints_from_table()
        leg = self._default_leg_duration()
        if self.waypoints:
            last = _normalize_waypoint_row(self.waypoints[-1])
            new_t = last[4] + leg
            new_wp = [last[0], last[1], last[2] + 1.0, last[3], new_t]
        else:
            new_wp = [0.0, 0.0, 1.0, 0.0, leg]
        self.waypoints.append(new_wp)
        self._populate_waypoint_table()
        self.waypoint_table.selectRow(len(self.waypoints) - 1)

    def remove_waypoint(self):
        """Remove selected table row (cannot remove start)."""
        self._sync_waypoints_from_table()
        row = self.waypoint_table.currentRow()
        if row < 0 or row >= len(self.waypoints):
            return
        if row == 0:
            QMessageBox.warning(self, 'Warning', 'Cannot remove start point')
            return
        self.waypoints.pop(row)
        self._populate_waypoint_table()
        if row > 0:
            self.waypoint_table.selectRow(row - 1)

    def on_trajectory_preset_changed(self, index):
        """Load waypoints and saved optimization for the selected slot + current mode."""
        old_combo = getattr(self, '_traj_combo_prev_index', int(index))
        old_mode_key = traj_opt_mode_key(
            getattr(self, '_traj_opt_mode_prev_index', TRAJ_OPT_MODE_NORMAL)
        )
        self._cache_results_for_slot_mode(old_combo, old_mode_key)
        self._load_trajectory_for_combo_index(index)
        self._traj_combo_prev_index = int(index)
        self._refresh_trajectory_storage_path_label()
        if hasattr(self, 'tracking_source_combo'):
            self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())
        if hasattr(self, 'tracking_source_combo'):
            self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())

    def _apply_waypoints_from_list(self, waypoints):
        """Replace GUI waypoints and refresh the table."""
        self.waypoints = [_normalize_waypoint_row(w) for w in waypoints]
        self._populate_waypoint_table()
        if self.waypoints:
            self.waypoint_table.selectRow(0)

    def _builtin_waypoints_for_combo_index(self, combo_index):
        preset_id = combo_index_to_trajectory_preset_id(combo_index)
        if 0 <= preset_id < len(TRAJECTORY_PRESETS):
            _, wps = TRAJECTORY_PRESETS[preset_id]
            return [_normalize_waypoint_row(w) for w in wps]
        return []

    def _load_trajectory_for_combo_index(self, combo_index, quiet=False):
        """Load waypoints from the JSON file for a slot, or built-in defaults."""
        combo_index = int(combo_index)
        preset_id = combo_index_to_trajectory_preset_id(combo_index)
        path = trajectory_waypoints_path_for_preset_id(preset_id)
        if preset_id == TRAJECTORY_PRESET_CUSTOM:
            if os.path.isfile(path):
                if not self._load_waypoints_file(path, quiet=quiet):
                    return False
            elif not quiet and hasattr(self, 'status_text'):
                self.status_text.append('Custom waypoints — edit table or save to create file.')
            self._restore_optimization_for_slot_and_mode(combo_index, quiet=quiet)
            return True
        if os.path.isfile(path):
            if not self._load_waypoints_file(path, quiet=quiet):
                return False
            self._restore_optimization_for_slot_and_mode(combo_index, quiet=quiet)
            return True
        builtin = self._builtin_waypoints_for_combo_index(combo_index)
        if builtin:
            self._apply_waypoints_from_list(builtin)
            self._restore_optimization_for_slot_and_mode(combo_index, quiet=quiet)
            if not quiet and hasattr(self, 'status_text'):
                self.status_text.append(
                    f'Loaded built-in defaults for {trajectory_combo_label(combo_index)} '
                    f'(no saved file at {path}).'
                )
            return True
        if not quiet:
            QMessageBox.warning(self, 'Load failed', f'No waypoints for slot index {combo_index}')
        return False

    def _load_waypoints_file(self, path, quiet=False, combo_index=None):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            if not quiet:
                QMessageBox.critical(self, 'Load failed', f'Could not read {path}:\n{e}')
            return False
        raw = data.get('waypoints', data if isinstance(data, list) else [])
        if not raw:
            if not quiet:
                QMessageBox.warning(self, 'Load failed', f'No waypoints in {path}')
            return False
        self._apply_waypoints_from_list(raw)
        if not quiet and hasattr(self, 'status_text'):
            self.status_text.append(
                f'Loaded trajectory ({len(self.waypoints)} pts) from:\n  {path}'
            )
        return True

    def _format_opt_summary_lines(self, summary):
        """Multi-line optimization summary for the plot info bar."""
        if not summary:
            return ['No optimization results yet.']
        method = summary.get('method_name') or f"method {summary.get('method', '?')}"
        total_time = float(summary.get('total_time_s', 0.0))
        total_iters = int(summary.get('total_iters', 0))
        avg_ms = float(summary.get('avg_time_per_iter_ms', 0.0))
        path_len = float(summary.get('path_length_m', 0.0))
        duration = float(summary.get('trajectory_duration_s', 0.0))
        line1 = (
            f'Method: {method}   |   Total: {total_time:.2f} s   |   '
            f'Iters: {total_iters}   |   Avg: {avg_ms:.1f} ms/iter   |   '
            f'Path: {path_len:.2f} m   |   Duration: {duration:.2f} s'
        )
        seg_iters = summary.get('segment_iters') or []
        seg_costs = summary.get('segment_final_costs') or []
        seg_parts = []
        for i, n in enumerate(seg_iters):
            part = f'Seg {i + 1}: {int(n)} opt'
            if i < len(seg_costs):
                part += f' (cost {float(seg_costs[i]):.3e})'
            seg_parts.append(part)
        ots = summary.get('optimal_segment_times')
        if ots:
            seg_parts.append('Seg times [s]: ' + ', '.join(f'{float(t):.3f}' for t in ots))
        line2 = '   |   '.join(seg_parts) if seg_parts else ''
        saved_at = summary.get('saved_at')
        line3 = f'Saved: {saved_at}' if saved_at else ''
        lines = [line1]
        if line2:
            lines.append(line2)
        if line3:
            lines.append(line3)
        return lines

    def _refresh_opt_info_display(self):
        """Update the top plot row with optimization statistics."""
        if not hasattr(self, 'ax_opt_info'):
            return
        self.ax_opt_info.clear()
        self.ax_opt_info.axis('off')
        summary = getattr(self, 'opt_summary', None)
        if not summary:
            self.ax_opt_info.text(
                0.5, 0.5,
                self._opt_info_empty_message(),
                ha='center', va='center', fontsize=10, color='#666',
                transform=self.ax_opt_info.transAxes,
            )
        else:
            lines = self._format_opt_summary_lines(summary)
            self.ax_opt_info.text(
                0.01, 0.92, '\n'.join(lines),
                ha='left', va='top', fontsize=9.5, family='monospace',
                transform=self.ax_opt_info.transAxes,
            )
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _clear_optimization_plots(self):
        """Clear trajectory / cost axes (keep titles)."""
        if not hasattr(self, 'ax_3d'):
            return
        plot_axes = (
            self.ax_3d, self.ax_cost, self.ax_pos, self.ax_vel, self.ax_euler,
            self.ax_angvel, self.ax_pitch, self.ax_roll, self.ax_thrust, self.ax_yaw,
        )
        for ax in plot_axes:
            ax.clear()
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _clear_optimization_results(self, clear_plots=False):
        """Drop cached optimization; optionally clear plot panels."""
        self.last_trajectory = None
        self.opt_summary = None
        self._enable_trajectory_export_buttons(False)
        self._refresh_opt_info_display()
        if clear_plots:
            self._clear_optimization_plots()

    def _apply_cached_trajectory_entry(self, entry, combo_index, mode_key, quiet=False):
        self.last_trajectory = entry['last_trajectory']
        self.opt_summary = entry.get('opt_summary')
        paths = resolve_trajectory_artifact_paths(combo_index, mode_key)
        if os.path.isfile(paths['csv']):
            self.last_csv_path = paths['csv']
        self._enable_trajectory_export_buttons(True)
        self._restore_plot_from_last_trajectory()
        if not quiet and hasattr(self, 'status_text'):
            label = trajectory_combo_label(combo_index)
            mode_label = 'Minimum time' if mode_key == 'min_time' else 'Normal'
            iters = (self.opt_summary or {}).get('total_iters', '?')
            path_len = (self.opt_summary or {}).get('path_length_m', 0.0)
            self.status_text.append(
                f'Loaded {label} ({mode_label}): {iters} iters, {path_len:.2f} m path.'
            )
        return True

    def _restore_optimization_for_slot_and_mode(self, combo_index, mode_key=None, quiet=False):
        """Load in-memory or on-disk optimization for one trajectory slot + mode."""
        combo_index = int(combo_index)
        mode_key = mode_key or self._current_traj_opt_mode_key()
        cache_key = (combo_index, mode_key)
        cached = self._trajectory_mode_cache.get(cache_key)
        if cached and cached.get('last_trajectory', {}).get('xs') is not None:
            if optimization_summary_matches_mode(cached.get('opt_summary'), mode_key):
                return self._apply_cached_trajectory_entry(
                    cached, combo_index, mode_key, quiet=quiet,
                )
            self._trajectory_mode_cache.pop(cache_key, None)
        return self._try_restore_saved_optimization(
            combo_index, mode_key=mode_key, quiet=quiet,
        )

    def _load_trajectory_arrays_from_paths(self, paths, summary, mode_key):
        """Load trajectory arrays from NPZ, else from exported CSV."""
        npz_path = paths['npz']
        csv_path = paths['csv']
        if os.path.isfile(npz_path):
            z = np.load(npz_path, allow_pickle=False)
            xs = np.asarray(z['xs'], dtype=float)
            us = np.asarray(z['us'], dtype=float) if 'us' in z.files else None
            us_actual = np.asarray(z['us_actual'], dtype=float) if 'us_actual' in z.files else None
            time_states = (
                np.asarray(z['time_states'], dtype=float) if 'time_states' in z.files else None
            )
            sbo = z['segment_boundary_indices'] if 'segment_boundary_indices' in z.files else None
            if sbo is not None:
                sbo = [int(x) for x in np.asarray(sbo).reshape(-1)]
            ots = z['optimal_segment_times'] if 'optimal_segment_times' in z.files else None
            if ots is not None:
                ots = [float(x) for x in np.asarray(ots).reshape(-1)]
            return {
                'xs': xs,
                'us': us,
                'us_actual': us_actual,
                'plot_dt': summary.get('plot_dt'),
                'dt': summary.get('dt', self.dt_spin.value()),
                'time_states': time_states,
                'segment_boundary_indices': sbo or summary.get('segment_boundary_indices'),
                'optimal_segment_times': ots or summary.get('optimal_segment_times'),
                'method': summary.get('method'),
                'method_name': summary.get('method_name', ''),
            }
        if os.path.isfile(csv_path):
            loaded = load_trajectory_from_export_csv(csv_path)
            if loaded is None:
                return None
            loaded['method'] = summary.get('method')
            loaded['method_name'] = summary.get('method_name', '')
            loaded['plot_dt'] = summary.get('plot_dt') or loaded.get('plot_dt')
            loaded['dt'] = summary.get('dt') or loaded.get('dt')
            loaded['segment_boundary_indices'] = summary.get('segment_boundary_indices')
            loaded['optimal_segment_times'] = summary.get('optimal_segment_times')
            return loaded
        alt_csv = (summary or {}).get('csv_path')
        if alt_csv and alt_csv != csv_path and os.path.isfile(alt_csv):
            loaded = load_trajectory_from_export_csv(alt_csv)
            if loaded is None:
                return None
            loaded['method'] = summary.get('method')
            loaded['method_name'] = summary.get('method_name', '')
            loaded['plot_dt'] = summary.get('plot_dt') or loaded.get('plot_dt')
            loaded['dt'] = summary.get('dt') or loaded.get('dt')
            loaded['segment_boundary_indices'] = summary.get('segment_boundary_indices')
            loaded['optimal_segment_times'] = summary.get('optimal_segment_times')
            return loaded
        return None

    def _make_optimization_summary(self, xs, us, all_loggers, timing_info=None):
        """Build JSON-serializable optimization summary dict."""
        timing_info = timing_info or {}
        xs_arr = np.asarray(xs, dtype=float) if xs is not None else np.zeros((0, 17))
        segment_iters = []
        segment_final_costs = []
        segment_cost_histories = []
        if all_loggers:
            for logger in all_loggers:
                costs = [float(c) for c in (getattr(logger, 'costs', []) or [])]
                segment_cost_histories.append(costs)
                segment_iters.append(len(costs))
                if costs:
                    segment_final_costs.append(float(costs[-1]))
        total_iters = int(timing_info.get('total_iters', sum(segment_iters)))
        total_time = float(timing_info.get('total_time', 0.0))
        avg_ms = float(timing_info.get('avg_time_per_iter', 0.0)) * 1000.0
        if total_iters > 0 and avg_ms <= 0.0 and total_time > 0:
            avg_ms = (total_time / total_iters) * 1000.0
        traj_duration = 0.0
        ots = timing_info.get('optimal_segment_times')
        if ots:
            traj_duration = float(sum(float(t) for t in ots))
        elif xs_arr.shape[0] > 1:
            traj_duration = float(
                (xs_arr.shape[0] - 1) * float(timing_info.get('plot_dt') or self.dt_spin.value())
            )
        return {
            'method': int(self.method_combo.currentIndex()),
            'method_name': timing_info.get('method', ''),
            'total_time_s': total_time,
            'total_iters': total_iters,
            'avg_time_per_iter_ms': avg_ms,
            'path_length_m': trajectory_path_length_m(xs_arr),
            'trajectory_duration_s': traj_duration,
            'segment_iters': segment_iters,
            'segment_final_costs': segment_final_costs,
            'segment_cost_histories': segment_cost_histories,
            'optimal_segment_times': [float(t) for t in ots] if ots else None,
            'segment_boundary_indices': timing_info.get('segment_boundary_indices'),
            'plot_dt': timing_info.get('plot_dt'),
            'dt': float(self.dt_spin.value()),
        }

    def _restore_plot_from_last_trajectory(self):
        """Redraw all panels from ``self.last_trajectory``."""
        traj = self.last_trajectory
        if not traj or traj.get('xs') is None:
            return
        xs = traj['xs']
        us = traj.get('us')
        histories = (self.opt_summary or {}).get('segment_cost_histories') or []
        if histories:
            loggers = [_SavedCostLogger(h) for h in histories]
            draw_cost_panel(self.ax_cost, loggers)
        self._refresh_opt_info_display()
        self.update_state(
            xs,
            us,
            traj.get('us_actual'),
            plot_dt=traj.get('plot_dt'),
            segment_boundaries_override=traj.get('segment_boundary_indices'),
            time_states=traj.get('time_states'),
        )

    def _try_restore_saved_optimization(
        self, combo_index, json_data=None, mode_key=None, quiet=False,
    ):
        """Load saved CSV/NPZ optimization for one trajectory slot + mode."""
        combo_index = int(combo_index)
        mode_key = mode_key or self._current_traj_opt_mode_key()
        paths = resolve_trajectory_artifact_paths(combo_index, mode_key)
        if json_data is None:
            if not os.path.isfile(paths['json']):
                self._clear_optimization_results(clear_plots=True)
                return False
            try:
                with open(paths['json'], 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._clear_optimization_results(clear_plots=True)
                return False
        summary = optimization_summary_from_json(json_data, mode_key)
        if not summary or not optimization_summary_matches_mode(summary, mode_key):
            self._trajectory_mode_cache.pop((combo_index, mode_key), None)
            self._clear_optimization_results(clear_plots=True)
            return False
        loaded = self._load_trajectory_arrays_from_paths(paths, summary, mode_key)
        if loaded is None or loaded.get('xs') is None:
            self._trajectory_mode_cache.pop((combo_index, mode_key), None)
            self._clear_optimization_results(clear_plots=True)
            return False
        try:
            self.opt_summary = dict(summary)
            self.last_trajectory = loaded
            if os.path.isfile(paths['csv']):
                self.last_csv_path = paths['csv']
            self._cache_results_for_slot_mode(combo_index, mode_key)
            return self._apply_cached_trajectory_entry(
                self._trajectory_mode_cache[(combo_index, mode_key)],
                combo_index,
                mode_key,
                quiet=quiet,
            )
        except (OSError, ValueError, KeyError) as e:
            if not quiet and hasattr(self, 'status_text'):
                self.status_text.append(f'Could not load saved trajectory: {e}')
            self._trajectory_mode_cache.pop((combo_index, mode_key), None)
            self._clear_optimization_results(clear_plots=True)
            return False

    def _current_trajectory_csv_path(self):
        """CSV path for the current trajectory slot + optimization mode."""
        if not hasattr(self, 'trajectory_preset_combo'):
            return DEFAULT_TRAJ_CSV_PATH
        return resolve_trajectory_artifact_paths(
            self.trajectory_preset_combo.currentIndex(),
            self._current_traj_opt_mode_key(),
        )['csv']

    def save_trajectory(self):
        """Save waypoints, optimized CSV, NPZ, and summary for the current trajectory slot."""
        self._sync_waypoints_from_table()
        if len(self.waypoints) < 2:
            QMessageBox.warning(
                self,
                'Not enough waypoints',
                'Need at least 2 waypoints (start + one target) before saving.',
            )
            return
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i][4] >= self.waypoints[i + 1][4]:
                QMessageBox.warning(
                    self,
                    'Invalid times',
                    f'Waypoint {i + 1} arrival time must be greater than waypoint {i}.',
                )
                return
        if not self.last_trajectory or self.last_trajectory.get('xs') is None:
            QMessageBox.warning(
                self,
                'No optimization',
                'Run Start Optimization first, then click Save Trajectory.',
            )
            return
        payload = self._build_trajectory_csv_payload()
        if payload is None:
            QMessageBox.warning(self, 'No trajectory', 'Could not build trajectory export.')
            return

        combo_index = self.trajectory_preset_combo.currentIndex()
        mode_key = self._current_traj_opt_mode_key()
        paths = trajectory_artifact_paths(combo_index, mode_key)
        label = trajectory_combo_label(combo_index)
        mode_label = 'Minimum time' if mode_key == 'min_time' else 'Normal'
        summary = dict(self.opt_summary or self._make_optimization_summary(
            self.last_trajectory['xs'],
            self.last_trajectory.get('us'),
            [],
            {'total_time': 0.0, 'total_iters': 0, 'method': self.last_trajectory.get('method_name', '')},
        ))
        summary['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        summary['mode_key'] = mode_key
        summary['csv_path'] = paths['csv']
        summary['npz_path'] = paths['npz']
        self.opt_summary = summary

        try:
            os.makedirs(os.path.dirname(os.path.abspath(paths['json'])), exist_ok=True)
            self._write_trajectory_csv(paths['csv'], payload)
            traj = self.last_trajectory
            npz_kw = {'xs': np.asarray(traj['xs'], dtype=float)}
            if traj.get('us') is not None:
                npz_kw['us'] = np.asarray(traj['us'], dtype=float)
            if traj.get('us_actual') is not None:
                npz_kw['us_actual'] = np.asarray(traj['us_actual'], dtype=float)
            if traj.get('time_states') is not None:
                npz_kw['time_states'] = np.asarray(traj['time_states'], dtype=float)
            if traj.get('segment_boundary_indices') is not None:
                npz_kw['segment_boundary_indices'] = np.asarray(
                    traj['segment_boundary_indices'], dtype=int
                )
            if traj.get('optimal_segment_times') is not None:
                npz_kw['optimal_segment_times'] = np.asarray(
                    traj['optimal_segment_times'], dtype=float
                )
            np.savez_compressed(paths['npz'], **npz_kw)
            optimizations = {}
            if os.path.isfile(paths['json']):
                try:
                    with open(paths['json'], 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    opts = existing.get('optimizations')
                    if isinstance(opts, dict):
                        optimizations.update(opts)
                    legacy = existing.get('optimization')
                    if legacy and 'normal' not in optimizations and 'min_time' not in optimizations:
                        if legacy.get('method') == METHOD_MIN_TIME_INDEX:
                            optimizations['min_time'] = legacy
                        else:
                            optimizations['normal'] = legacy
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
            optimizations[mode_key] = summary
            json_payload = {
                'version': TRAJ_JSON_VERSION,
                'name': label,
                'waypoints': waypoints_to_json_list(self.waypoints),
                'optimizations': optimizations,
            }
            with open(paths['json'], 'w', encoding='utf-8') as f:
                json.dump(json_payload, f, indent=2, ensure_ascii=False, default=str)
        except OSError as e:
            QMessageBox.critical(self, 'Save failed', f'Could not write trajectory files:\n{e}')
            return

        self.last_csv_path = paths['csv']
        self.default_traj_csv_path = paths['csv']
        self._cache_current_mode_results()
        self._refresh_trajectory_storage_path_label()
        self._refresh_opt_info_display()
        if hasattr(self, 'tracking_source_combo'):
            self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())
        self.status_text.append(
            f'Saved {label} ({mode_label}): waypoints + optimization to\n'
            f'  JSON: {paths["json"]}\n'
            f'  CSV:  {paths["csv"]}'
        )

    def _load_saved_waypoints_from_file(self, quiet=False):
        """Load waypoints from the Saved trajectory file."""
        path = DEFAULT_SAVED_WAYPOINTS_PATH
        if not os.path.isfile(path):
            if not quiet:
                QMessageBox.warning(
                    self,
                    'No saved trajectory',
                    f'File not found:\n{path}\n\n'
                    'Edit waypoints and click Save trajectory first.',
                )
            return False
        return self._load_waypoints_file(path, quiet=quiet)

    def _set_trajectory_preset_combo_index(self, combo_index):
        """Set Trajectory combo without firing preset replacement."""
        if not hasattr(self, 'trajectory_preset_combo'):
            return
        combo_index = max(0, min(self.trajectory_preset_combo.count() - 1, int(combo_index)))
        self.trajectory_preset_combo.blockSignals(True)
        self.trajectory_preset_combo.setCurrentIndex(combo_index)
        self.trajectory_preset_combo.blockSignals(False)

    def _sync_trajectory_preset_combo_from_waypoints(self):
        """Align Trajectory combo with waypoints (preset / saved file / custom)."""
        if not hasattr(self, 'trajectory_preset_combo'):
            return
        if (
            os.path.isfile(self.saved_waypoints_path)
            and trajectory_preset_match_index(self.waypoints)
            is None
        ):
            try:
                with open(self.saved_waypoints_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                saved_wps = waypoints_to_json_list(saved.get('waypoints', []))
                current_wps = waypoints_to_json_list(self.waypoints)
                if saved_wps == current_wps:
                    self._set_trajectory_preset_combo_index(trajectory_saved_combo_index())
                    return
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        idx = trajectory_preset_match_index(self.waypoints)
        self._set_trajectory_preset_combo_index(
            idx if idx is not None else trajectory_custom_combo_index()
        )

    def _apply_trajectory_from_config(self, cfg):
        """Load waypoints according to ``trajectory_preset`` in a params file."""
        preset_raw = cfg.get('trajectory_preset', TRAJECTORY_PRESET_CUSTOM)
        try:
            preset_id = int(preset_raw)
        except (TypeError, ValueError):
            preset_id = TRAJECTORY_PRESET_CUSTOM

        combo_idx = trajectory_preset_id_to_combo_index(preset_id)
        self._set_trajectory_preset_combo_index(combo_idx)
        if not self._load_trajectory_for_combo_index(combo_idx, quiet=True):
            if 'waypoints' in cfg:
                self._apply_waypoints_from_list(cfg['waypoints'])
        self._refresh_trajectory_storage_path_label()

    def gui_config_to_dict(self):
        """Serialize GUI settings to a JSON-friendly dict."""
        preset_id = combo_index_to_trajectory_preset_id(
            self.trajectory_preset_combo.currentIndex()
        )
        cfg = {
            'version': GUI_PARAMS_VERSION,
            'trajectory_preset': preset_id,
            'traj_opt_mode': (
                'min_time'
                if hasattr(self, 'traj_opt_mode_combo')
                and self.traj_opt_mode_combo.currentIndex() == TRAJ_OPT_MODE_MIN_TIME
                else 'normal'
            ),
            'normal_method': getattr(self, '_normal_method_index', METHOD_NORMAL_DEFAULT_INDEX),
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
        if preset_id == TRAJECTORY_PRESET_CUSTOM:
            cfg['waypoints'] = waypoints_to_json_list(self.waypoints)
        return cfg

    def apply_gui_config(self, cfg):
        """Apply settings from dict (e.g. loaded JSON). Does not call on_method_changed."""
        if not cfg:
            return
        for combo in self._method_combos():
            combo.blockSignals(True)
        try:
            self._apply_trajectory_from_config(cfg)

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
                for combo in self._method_combos():
                    combo.setCurrentIndex(idx)
            if 'normal_method' in cfg:
                self._normal_method_index = int(cfg['normal_method'])
            if 'traj_opt_mode' in cfg:
                mode = TRAJ_OPT_MODE_MIN_TIME if cfg['traj_opt_mode'] == 'min_time' else TRAJ_OPT_MODE_NORMAL
                self._set_traj_opt_mode_combo_index(mode)
                self._on_traj_opt_mode_changed(mode)
                self._traj_opt_mode_prev_index = int(mode)
            elif 'method' in cfg:
                self._update_method_combo_enabled_for_traj_mode()
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
            for combo in self._method_combos():
                combo.blockSignals(False)
        self._update_unified_checkbox_state(self.method_combo.currentIndex())
        self._refresh_min_time_duration_group_visible(self.method_combo.currentIndex())

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
        self._sync_waypoints_from_table()
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
        
        self._sync_waypoints_from_table()
        if len(self.waypoints) < 2:
            QMessageBox.warning(self, 'Warning', 'Need at least 2 waypoints (start and at least one waypoint)')
            return

        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i][4] >= self.waypoints[i + 1][4]:
                QMessageBox.warning(
                    self,
                    'Warning',
                    f'Waypoint {i + 1} arrival time ({self.waypoints[i + 1][4]:.2f}s) must be '
                    f'greater than waypoint {i} time ({self.waypoints[i][4]:.2f}s)',
                )
                return

        self.waypoints = [_normalize_waypoint_row(w) for w in self.waypoints]

        self._trajectory_mode_cache.pop(self._mode_cache_key(), None)
        self.last_trajectory = None
        
        # Reset display
        self.iterations = []
        self.costs = []
        self.stops = []
        self.opt_summary = None
        self._refresh_opt_info_display()
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
        if wps:
            legs = [
                f'{wps[i + 1][4] - wps[i][4]:.2f}s' for i in range(len(wps) - 1)
            ]
            self.status_text.append(f'Segment Δt [s]: {legs}')
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
        self._sync_waypoints_from_table()
        waypoints = waypoints_for_optimizer(
            self.waypoints if hasattr(self, 'waypoints') else []
        )
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
        self.opt_summary = self._make_optimization_summary(xs, us, all_loggers, timing_info)
        self._cache_current_mode_results()
        self._refresh_opt_info_display()
        self._enable_trajectory_export_buttons(True)
        if hasattr(self, 'tracking_source_combo') and self.tracking_source_combo.currentIndex() == 0:
            self._on_tracking_source_changed(0)
    
    def _frame_is_ned(self):
        return hasattr(self, 'frame_combo') and self.frame_combo.currentIndex() == 1

    def _build_trajectory_csv_payload(self):
        """Prepare arrays and metadata for CSV export; None if no valid trajectory."""
        if not self.last_trajectory or self.last_trajectory.get('xs') is None:
            return None

        traj = self.last_trajectory
        xs = np.asarray(traj['xs'], dtype=float)
        us = np.asarray(traj['us'], dtype=float) if traj.get('us') is not None else None
        us_act = (
            np.asarray(traj['us_actual'], dtype=float)
            if traj.get('us_actual') is not None
            else None
        )

        if xs.ndim != 2 or xs.shape[1] < 13:
            return None

        N_states = xs.shape[0]
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

        p = xs[:, 0:3]
        v = xs[:, 3:6]
        q = xs[:, 6:10]
        w = xs[:, 10:13]
        euler_rad = np.array([quat_to_euler(qq, format='wxyz') for qq in q])
        euler_deg = np.degrees(euler_rad)

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
                return np.vstack([arr, arr[-1:]])
            if arr.shape[0] > N_states:
                return arr[:N_states]
            pad_rows = np.repeat(arr[-1:], N_states - arr.shape[0], axis=0)
            return np.vstack([arr, pad_rows])

        u_cmd = _pad_to_state_length(us)
        u_act_out = _pad_to_state_length(us_act)

        if self._frame_is_ned():
            p = np.column_stack([p[:, 1], p[:, 0], -p[:, 2]])
            v = np.column_stack([v[:, 1], v[:, 0], -v[:, 2]])
            w = np.column_stack([w[:, 1], w[:, 0], -w[:, 2]])
            euler_ned = np.column_stack([
                euler_rad[:, 0],
                -euler_rad[:, 1],
                -euler_rad[:, 2],
            ])
            from tvc_common import euler_to_quat_wxyz
            q_out = np.array([
                euler_to_quat_wxyz(r, pi, ya) for r, pi, ya in euler_ned
            ])
            euler_deg = np.degrees(euler_ned)
            frame_tag = 'NED'
        else:
            q_out = q
            frame_tag = 'ENU'

        method_name = traj.get('method_name') or f"method{traj.get('method', 0) + 1}"
        return {
            't': t,
            'p': p,
            'v': v,
            'q_out': q_out,
            'w': w,
            'euler_deg': euler_deg,
            'u_cmd': u_cmd,
            'u_act_out': u_act_out,
            'N_states': N_states,
            'frame_tag': frame_tag,
            'method_name': method_name,
        }

    def _execution_waypoints_csv_line(self, payload):
        """Segment-end poses on the saved trajectory (matches executed CSV, not GUI table)."""
        traj = self.last_trajectory
        if not traj:
            return ''
        p = payload['p']
        euler_deg = payload['euler_deg']
        n = int(p.shape[0])
        idxs = [0]
        sbo = traj.get('segment_boundary_indices') or []
        for bi in sbo:
            bi = int(bi)
            if 0 <= bi < n and bi not in idxs:
                idxs.append(bi)
        last = n - 1
        if last >= 0 and last not in idxs:
            idxs.append(last)
        idxs.sort()
        parts = []
        for i in idxs:
            parts.append(
                f'{p[i, 0]:.6f},{p[i, 1]:.6f},{p[i, 2]:.6f},{euler_deg[i, 2]:.6f}'
            )
        return ';'.join(parts)

    def _write_trajectory_csv(self, path, payload):
        """Write prepared trajectory payload to ``path``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        t = payload['t']
        p = payload['p']
        v = payload['v']
        q_out = payload['q_out']
        w = payload['w']
        euler_deg = payload['euler_deg']
        u_cmd = payload['u_cmd']
        u_act_out = payload['u_act_out']
        N_states = payload['N_states']
        frame_tag = payload['frame_tag']
        method_name = payload['method_name']

        header_lines = [
            '# TVC trajectory exported from tvc_traj_opt_gui',
            f"# generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f'# method: {method_name}',
            f"# frame: {frame_tag} "
            f"({'x N, y E, z D (PX4)' if frame_tag == 'NED' else 'x E, y N, z U (planner native)'})",
            f'# N_states: {N_states}, duration: {float(t[-1] - t[0]):.6f} s',
        ]
        wp_line = self._execution_waypoints_csv_line(payload)
        if wp_line:
            header_lines.append(f'# waypoints_enu: {wp_line}')
        header_lines.extend([
            '# Commands repeat the last row (zero-order hold) so all columns have length N_states.',
            '# th_p, th_r: gimbal pitch/roll angles [rad]; T: thrust [N]; tau_yaw: reaction-wheel torque [N*m].',
            '# *_act columns are the actuator actual states (NaN when the method has no actuator dynamics).',
        ])
        columns = [
            't', 'x', 'y', 'z', 'vx', 'vy', 'vz',
            'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'th_p_cmd', 'th_r_cmd', 'T_cmd', 'tau_yaw_cmd',
            'th_p_act', 'th_r_act', 'T_act', 'tau_yaw_act',
        ]
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

    def _after_trajectory_saved(self, path, payload):
        """Update paths/labels after a successful CSV write."""
        self.last_csv_path = path
        self.default_traj_csv_path = path
        if hasattr(self, 'traj_default_path_label'):
            self.traj_default_path_label.setText(f'Default: {path}')
        if hasattr(self, 'tracking_csv_edit'):
            if self.tracking_source_combo.currentIndex() == 0:
                self.tracking_csv_edit.setText(path)
            self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())
        self.status_text.append(
            f'Saved trajectory ({payload["N_states"]} samples, {payload["frame_tag"]}) to:\n  {path}'
        )

    def save_trajectory_default(self):
        """Save the latest optimized trajectory to the default CSV path."""
        payload = self._build_trajectory_csv_payload()
        if payload is None:
            QMessageBox.warning(
                self, 'No trajectory',
                'Please run an optimization first; there is no trajectory to export yet.'
            )
            return
        path = DEFAULT_TRAJ_CSV_PATH
        try:
            self._write_trajectory_csv(path, payload)
        except OSError as e:
            QMessageBox.critical(self, 'Save failed', f'Could not write {path}:\n{e}')
            return
        self._after_trajectory_saved(path, payload)

    def save_trajectory_csv(self):
        """Export the most recently optimized trajectory via file dialog."""
        payload = self._build_trajectory_csv_payload()
        if payload is None:
            QMessageBox.warning(
                self, 'No trajectory',
                'Please run an optimization first; there is no trajectory to export yet.'
            )
            return

        method_name = payload['method_name']
        safe_method = ''.join(c if c.isalnum() else '_' for c in str(method_name)).strip('_')
        default_dir = (
            os.path.dirname(self.last_csv_path) if self.last_csv_path
            else DEFAULT_TRAJ_CSV_DIR
        )
        default_name = time.strftime(f'tvc_traj_{safe_method}_%Y%m%d_%H%M%S.csv')
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

        try:
            self._write_trajectory_csv(path, payload)
        except OSError as e:
            QMessageBox.critical(self, 'Save failed', f'Could not write {path}:\n{e}')
            return
        self._after_trajectory_saved(path, payload)

    def _effective_last_saved_csv_path(self):
        """CSV path for the current trajectory slot (legacy alias)."""
        return self._current_trajectory_csv_path()

    def _encode_waypoints_for_tracking(self):
        """Encode GUI waypoint list as tvc_traj_player waypoints_enu string."""
        self._sync_waypoints_from_table()
        if not self.waypoints:
            return ''
        parts = []
        for i, wp in enumerate(self.waypoints):
            row = _normalize_waypoint_row(wp)
            x, y, z, yaw, _t = row
            hover_s = 2.0 if i == 0 else max(0.5, leg_duration_s(self.waypoints, i))
            parts.append(f'{x},{y},{z},{yaw},{hover_s:.3f}')
        return ';'.join(parts)

    def _launch_arg(self, key, value):
        """Format a ros2 launch key:=value argument with safe shell quoting."""
        v = str(value).replace("'", "'\\''")
        return f"{key}:='{v}'"

    def _on_tracking_source_changed(self, index):
        """Show/hide CSV picker and update tracking source summary."""
        if hasattr(self, 'tracking_csv_widget'):
            self.tracking_csv_widget.setVisible(index == 1)
        if not hasattr(self, 'lbl_tracking_source_info'):
            return
        if index == 0:
            label = trajectory_combo_label(
                self.trajectory_preset_combo.currentIndex()
                if hasattr(self, 'trajectory_preset_combo') else 0
            )
            mode_label = (
                'Minimum time' if self._current_traj_opt_mode_key() == 'min_time' else 'Normal'
            )
            path = self._current_trajectory_csv_path()
            exists = os.path.isfile(path)
            slot_label = f'{label} ({mode_label})'
            if exists:
                self.lbl_tracking_source_info.setText(f'{slot_label}: {path}')
            elif (
                getattr(self, 'last_trajectory', None)
                and self.last_trajectory.get('xs') is not None
            ):
                self.lbl_tracking_source_info.setText(
                    f'{slot_label}: optimized in memory — click Save Trajectory before tracking.'
                )
            else:
                self.lbl_tracking_source_info.setText(
                    f'{slot_label}: not saved — run Start Optimization, then Save Trajectory.'
                )
            if hasattr(self, 'tracking_csv_edit'):
                self.tracking_csv_edit.setText(path)
        elif index == 1:
            path = self.tracking_csv_edit.text().strip() if hasattr(self, 'tracking_csv_edit') else ''
            ok = bool(path) and os.path.isfile(path)
            self.lbl_tracking_source_info.setText(
                f'CSV: {path or "(choose a file)"}' + ('' if ok else ' — file not found')
            )
        else:
            n = len(self.waypoints) if hasattr(self, 'waypoints') else 0
            self.lbl_tracking_source_info.setText(
                f'Using {n} waypoint(s) from the Trajectory tab (GotoSetpoint mode).'
            )

    def _on_tracking_csv_edited(self, _text=''):
        if self.tracking_source_combo.currentIndex() == 1:
            self._on_tracking_source_changed(1)

    def browse_tracking_csv(self):
        """Pick a trajectory CSV for tracking."""
        start = self.tracking_csv_edit.text().strip() or DEFAULT_TRAJ_CSV_DIR
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select trajectory CSV',
            start,
            'CSV files (*.csv);;All files (*.*)',
        )
        if path:
            self.tracking_csv_edit.setText(path)
            self._on_tracking_source_changed(1)

    def _ros2_workspace_root(self):
        """Resolve TVC_ws root (parent of TVC-traj-opt)."""
        return os.path.abspath(os.path.join(ROOT_DIR, '..'))

    def _ros2_shell_command(self, launch_file, extra_args=None):
        """Build a bash command that sources the workspace then runs ros2 launch."""
        ws = self._ros2_workspace_root()
        setup = os.path.join(ws, 'install', 'setup.bash')
        args = ' '.join(extra_args or [])
        launch_cmd = f'ros2 launch tvc_controller {launch_file}'
        if args:
            launch_cmd = f'{launch_cmd} {args}'
        if os.path.isfile(setup):
            return ['bash', '-lc', f'source "{setup}" && {launch_cmd}']
        return ['ros2', 'launch', 'tvc_controller', launch_file] + (extra_args or [])

    def _ros2_shell_run_script(self, script_path, *script_args):
        """Run a Python script with workspace ROS env (one-shot)."""
        ws = self._ros2_workspace_root()
        setup = os.path.join(ws, 'install', 'setup.bash')
        py = sys.executable
        quoted_args = ' '.join(f'"{a}"' for a in script_args)
        run = f'{py} "{script_path}"'
        if quoted_args:
            run = f'{run} {quoted_args}'
        if os.path.isfile(setup):
            return ['bash', '-lc', f'source "{setup}" && {run}']
        return [py, script_path, *script_args]

    def clear_rviz_trajectory_display(self, quiet=False, include_planned=False):
        """Clear RViz trajectory display (executed only, or planned + executed)."""
        script = os.path.join(SCRIPTS_DIR, 'tvc_clear_traj_viz.py')
        if not os.path.isfile(script):
            if not quiet:
                QMessageBox.warning(self, 'Clear failed', f'Script not found:\n{script}')
            return False
        mode = 'all' if include_planned else 'executed'
        script_args = ['world', '--mode', mode]
        if include_planned:
            script_args.extend(['--hold', '2.0'])
        cmd = self._ros2_shell_run_script(script, *script_args)
        timeout = 12.0 if include_planned else 10.0
        try:
            result = subprocess.run(
                cmd,
                cwd=self._ros2_workspace_root(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            if not quiet:
                QMessageBox.warning(self, 'Clear failed', str(e))
                self.status_text.append(f'Clear RViz trajectory failed: {e}')
            return False
        if result.returncode != 0:
            err = (result.stderr or result.stdout or '').strip()
            if not quiet:
                QMessageBox.warning(
                    self,
                    'Clear failed',
                    err or f'Exit code {result.returncode}',
                )
                self.status_text.append(f'Clear RViz trajectory failed: {err}')
            return False
        if not quiet:
            if include_planned:
                self.status_text.append(
                    'Cleared RViz planned path, waypoint markers, executed path, '
                    'and current setpoint.'
                )
            else:
                self.status_text.append(
                    'Cleared RViz executed path and current setpoint.'
                )
        return True

    def _start_background_process(self, attr_name, cmd, label):
        """Start a detached process group; store handle on self.<attr_name>."""
        proc = getattr(self, attr_name, None)
        if proc is not None and proc.poll() is None:
            QMessageBox.information(self, label, f'{label} is already running.')
            return
        try:
            new_proc = subprocess.Popen(
                cmd,
                cwd=self._ros2_workspace_root(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                text=True,
            )
        except OSError as e:
            QMessageBox.critical(self, f'{label} failed', str(e))
            self.status_text.append(f'{label} failed: {e}')
            return
        setattr(self, attr_name, new_proc)
        self.status_text.append(f'{label} started (pid {new_proc.pid})')
        self._refresh_sim_button_states()

    def _stop_background_process(self, attr_name, label):
        """Send SIGTERM to the process group, then SIGKILL if needed."""
        proc = getattr(self, attr_name, None)
        if proc is None or proc.poll() is not None:
            setattr(self, attr_name, None)
            self._refresh_sim_button_states()
            self.status_text.append(f'{label}: not running')
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as e:
            QMessageBox.warning(self, f'{label}', f'Could not stop process: {e}')
            return
        try:
            proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=3.0)
        setattr(self, attr_name, None)
        self.status_text.append(f'{label} stopped')
        self._refresh_sim_button_states()

    def _refresh_sim_button_states(self):
        """Enable start/stop buttons and status labels according to subprocess state."""
        tvc_running = (
            self._tvc_launch_proc is not None and self._tvc_launch_proc.poll() is None
        )
        track_running = (
            self._traj_player_proc is not None and self._traj_player_proc.poll() is None
        )
        if hasattr(self, 'btn_start_px4_sitl'):
            self.btn_start_px4_sitl.setEnabled(not tvc_running)
            self.btn_stop_px4_sitl.setEnabled(tvc_running)
        if hasattr(self, 'lbl_px4_sitl_status'):
            if tvc_running:
                self.lbl_px4_sitl_status.setText('Running')
                self.lbl_px4_sitl_status.setStyleSheet('color: #2e7d32; font-weight: bold;')
            else:
                self.lbl_px4_sitl_status.setText('Stopped')
                self.lbl_px4_sitl_status.setStyleSheet('color: #888;')
        if hasattr(self, 'btn_start_tracking'):
            self.btn_start_tracking.setEnabled(not track_running)
            self.btn_stop_tracking.setEnabled(track_running)
        if hasattr(self, 'lbl_tracking_status'):
            if track_running:
                self.lbl_tracking_status.setText('Running')
                self.lbl_tracking_status.setStyleSheet('color: #2e7d32; font-weight: bold;')
            else:
                self.lbl_tracking_status.setText('Stopped')
                self.lbl_tracking_status.setStyleSheet('color: #888;')

    def start_px4_sitl(self):
        """Launch full TVC stack (PX4 SITL + micro-XRCE + controller)."""
        self._start_background_process(
            '_tvc_launch_proc',
            self._ros2_shell_command('tvc.launch.py'),
            'PX4 SITL (tvc.launch.py)',
        )
        self.status_text.append(
            'PX4 SITL starting — planned trajectory appears only after Tracking starts.'
        )

    def stop_px4_sitl(self):
        """Stop tvc.launch.py process group."""
        self._stop_background_process('_tvc_launch_proc', 'PX4 SITL')

    def start_tracking_node(self):
        """Launch tvc_traj_player with the selected tracking source."""
        if (
            self._traj_player_proc is not None
            and self._traj_player_proc.poll() is None
        ):
            self.stop_tracking_node()

        self.clear_rviz_trajectory_display(quiet=True)

        source = self.tracking_source_combo.currentIndex()
        extra = []

        if source == 0:
            label = trajectory_combo_label(self.trajectory_preset_combo.currentIndex())
            path = self._current_trajectory_csv_path()
            if not os.path.isfile(path):
                if (
                    getattr(self, 'last_trajectory', None)
                    and self.last_trajectory.get('xs') is not None
                ):
                    QMessageBox.warning(
                        self,
                        'Save required',
                        f'"{label}" is optimized but not saved yet.\n\n'
                        'Click Save Trajectory, then start Tracking again.',
                    )
                else:
                    QMessageBox.warning(
                        self,
                        'Trajectory not optimized',
                        f'No saved optimization CSV for "{label}".\n\n'
                        'Run Start Optimization, then Save Trajectory, '
                        'or choose CSV file / GUI waypoints.',
                    )
                return
            extra.extend([
                self._launch_arg('play_mode', 'trajectory'),
                self._launch_arg('csv_path', path),
            ])
            self.status_text.append(f'Tracking: {label}\n  {path}')
        elif source == 1:
            path = self.tracking_csv_edit.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(
                    self,
                    'CSV not found',
                    'Select a valid trajectory CSV file for tracking.',
                )
                return
            extra.extend([
                self._launch_arg('play_mode', 'trajectory'),
                self._launch_arg('csv_path', path),
            ])
            self.status_text.append(f'Tracking: CSV file\n  {path}')
        else:
            if len(self.waypoints) < 1:
                QMessageBox.warning(
                    self,
                    'No waypoints',
                    'Add at least one waypoint on the Trajectory tab.',
                )
                return
            wp_str = self._encode_waypoints_for_tracking()
            extra.extend([
                self._launch_arg('play_mode', 'waypoint'),
                self._launch_arg('waypoints_enu', wp_str),
            ])
            self.status_text.append(
                f'Tracking: GUI waypoints ({len(self.waypoints)} pts, GotoSetpoint mode)'
            )

        self._start_background_process(
            '_traj_player_proc',
            self._ros2_shell_command('tvc_traj_player.launch.py', extra),
            'Tracking node (tvc_traj_player.launch.py)',
        )

    def stop_tracking_node(self):
        """Stop tvc_traj_player and clear all trajectory layers in RViz."""
        self._stop_background_process('_traj_player_proc', 'Tracking node')
        self.clear_rviz_trajectory_display(quiet=True, include_planned=True)
        self.status_text.append('RViz planned and executed trajectories cleared.')

    def closeEvent(self, event):
        """Terminate background ROS2 launches when the GUI exits."""
        for attr, label in (
            ('_traj_player_proc', 'Tracking node'),
            ('_tvc_launch_proc', 'PX4 SITL'),
        ):
            proc = getattr(self, attr, None)
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5.0)
                except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                setattr(self, attr, None)
        super().closeEvent(event)

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
    """Bootstrap acados/runtime, then launch the GUI."""
    from tvc_runtime import bootstrap

    bootstrap()
    return run_gui()


if __name__ == '__main__':
    raise SystemExit(main())
