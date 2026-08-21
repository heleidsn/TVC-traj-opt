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
import hashlib

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
NMP_SAVED_WAYPOINTS_PATH = os.path.join(DEFAULT_TRAJ_CSV_DIR, 'nmp_saved_waypoints.json')
NMP_MODEL_PARAMS_PATH = os.path.join(DEFAULT_TRAJ_CSV_DIR, 'nmp_model_params.json')
NMP_MODEL_PARAM_KEYS = (
    'mass', 'Ixx', 'Iyy', 'Izz', 'r_thrust_z', 'body_length', 'com_from_bottom',
)
CONTROLLERS_DIR = os.path.join(ROOT_DIR, 'controllers')
DEFAULT_TRACKING_PARAMS_PATH = os.path.join(CONTROLLERS_DIR, 'tracking_params.json')
DEFAULT_TRACKING_GIF_PATH = os.path.join(DEFAULT_TRAJ_CSV_DIR, 'tracking_anim.gif')
APP_ICON_PATH = os.path.join(ROOT_DIR, 'assets', 'icon', 'rocket.jpg')
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
                                 QAbstractItemView, QHeaderView, QRadioButton, QButtonGroup, QStackedWidget)
    from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize
    from PyQt5.QtGui import QFont, QIcon, QMovie
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
                                      QAbstractItemView, QHeaderView, QRadioButton, QButtonGroup, QStackedWidget)
        from PySide2.QtCore import QThread, Signal as pyqtSignal, Qt, QTimer, QSize
        from PySide2.QtGui import QFont, QIcon, QMovie
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
from tvc_traj_gui_plot_layout import apply_responsive_layout, install_responsive_canvas
from tvc_traj_gui_tracking import (
    draw_tracking_state_panels,
    draw_tracking_3d_panel,
    draw_tracking_metrics_panels,
    tracking_state_axes_dict,
    tracking_metrics_axes_dict,
    tracking_summary_text,
)
from tvc_traj_gui_nmp import build_nmp_series, draw_nmp_panels, nmp_axes_dict
from tvc_traj_gui_margins import draw_stability_margins_panels
from controllers.stability_margins import (
    AXIS_PITCH,
    LOOP_ATTITUDE,
    LOOP_IDS,
    LOOP_POSITION,
    LOOP_RATE,
    LOOP_VELOCITY,
    analyze_all_loops,
)
from controllers.actuator_dynamics import (
    BW_MAX_HZ,
    BW_MIN_HZ,
    SCALE_MAX,
    SCALE_MIN,
    TAU_MAX,
    TAU_MIN,
    THRUST_RES_MAX,
    THRUST_RES_MIN,
    actuator_config_from_params,
    bandwidth_hz_to_tau,
    default_actuator_tracking_config,
    default_thrust_resolution_for_platform,
    tau_to_bandwidth_hz,
)
from controllers.params import (
    CONTROLLER_IDS,
    CONTROLLER_LABELS,
    CONTROLLER_ACADOS_NMPC,
    CONTROLLER_FLATNESS,
    CONTROLLER_LQR,
    CONTROLLER_MPC,
    CONTROLLER_PX4,
    SIM_NUMERICAL,
    SIM_SITL,
    all_default_tracking_config,
    default_controller_params_map,
    default_numerical_sim_config,
    default_params_for,
    migrate_numerical_sim_config,
    migrate_params_by_platform,
    params_map_for_platform,
)
from controllers.param_groups import param_groups_for
from controllers.px4_params import migrate_px4_params
from controllers.simulator import run_tracking_simulation
from controllers.waypoint_ref import build_waypoint_flight_reference
from controllers.acados_nmpc_tracker import acados_nmpc_available
from controllers.px4_tune import (
    TUNE_LEVEL_LABELS,
    TUNE_LEVELS,
    default_px4_tune_config,
    run_px4_cascade_tune_sim,
)
from tvc_rocket_platforms import (
    PLATFORM_REAL,
    PLATFORM_PROXY,
    default_constraints,
    default_physics,
    default_thrust_quantization_resolution,
    normalize_platform_id,
    constraint_spin_ranges,
    physics_spin_ranges,
    rocket_visual_geometry,
    sitl_launch_kwargs,
    platform_description,
    platform_label,
)
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
try:
    from tvc_traj_opt_flatness import (
        solve_with_flatness_waypoints,
        solve_with_flatness_waypoints_unified,
    )
except ImportError:
    solve_with_flatness_waypoints = None
    solve_with_flatness_waypoints_unified = None

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


TRAJ_JSON_VERSION = 4


def _normalize_platform_id_for_cache(platform_id):
    pid = (platform_id or 'proxy').strip().lower()
    if pid in ('flight',):
        return 'real'
    return pid if pid in ('proxy', 'real') else 'proxy'


def optimization_cache_key(platform_id, combo_index, mode_key, method_index):
    """In-memory cache key: platform × trajectory slot × mode × method."""
    return (
        _normalize_platform_id_for_cache(platform_id),
        int(combo_index),
        str(mode_key),
        int(method_index),
    )


def _artifact_name_suffix(platform_id, mode_key, method_index):
    """Filename suffix for per-platform / per-method optimization artifacts."""
    pid = _normalize_platform_id_for_cache(platform_id)
    mk = mode_key if mode_key in ('normal', 'min_time') else traj_opt_mode_key(mode_key)
    return f'_{pid}_{mk}_m{int(method_index)}'


def trajectory_artifact_paths(combo_index, mode_key='normal', platform_id='proxy', method_index=0):
    """JSON / CSV / NPZ paths for platform + trajectory slot + mode + method."""
    json_path = trajectory_waypoints_path_for_combo_index(combo_index)
    base, _ = os.path.splitext(json_path)
    if mode_key not in ('normal', 'min_time'):
        mode_key = traj_opt_mode_key(mode_key)
    suffix = _artifact_name_suffix(platform_id, mode_key, method_index)
    return {
        'json': json_path,
        'csv': f'{base}{suffix}.csv',
        'npz': f'{base}{suffix}_traj.npz',
        'mode_key': mode_key,
        'platform_id': _normalize_platform_id_for_cache(platform_id),
        'method_index': int(method_index),
    }


def resolve_trajectory_artifact_paths(
    combo_index, mode_key='normal', platform_id='proxy', method_index=0,
):
    """Resolve artifact paths; fall back to older naming schemes when present."""
    paths = trajectory_artifact_paths(combo_index, mode_key, platform_id, method_index)
    if os.path.isfile(paths['npz']) or os.path.isfile(paths['csv']):
        return paths
    base, _ = os.path.splitext(paths['json'])
    mk = paths['mode_key']
    # Previous naming: only mode suffix (no platform / method).
    leg_csv = f'{base}_{mk}.csv'
    leg_npz = f'{base}_{mk}_traj.npz'
    if os.path.isfile(leg_npz) or os.path.isfile(leg_csv):
        return {**paths, 'csv': leg_csv, 'npz': leg_npz}
    if mk == 'normal':
        leg_csv, leg_npz = f'{base}.csv', f'{base}_traj.npz'
        if os.path.isfile(leg_npz):
            return {**paths, 'csv': leg_csv, 'npz': leg_npz}
        if os.path.isfile(leg_csv):
            return {**paths, 'csv': leg_csv, 'npz': leg_npz if os.path.isfile(leg_npz) else paths['npz']}
    return paths


def _json_safe_for_fingerprint(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe_for_fingerprint(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_for_fingerprint(v) for v in obj]
    return obj


def optimization_input_fingerprint(params):
    """Stable hash of optimization inputs (for cache validity)."""
    weights = dict(params.get('weights') or {})
    bounds = dict(params.get('bounds') or {})
    I = params.get('I')
    r_thrust = params.get('r_thrust')
    payload = {
        'rocket_platform': _normalize_platform_id_for_cache(params.get('rocket_platform')),
        'waypoints': waypoints_to_json_list(params.get('waypoints') or []),
        'method': int(params.get('method', 0)),
        'dt': float(params.get('dt', 0.0)),
        'N': int(params.get('N', 0)),
        'max_iter': int(params.get('max_iter', 0)),
        'unified': bool(params.get('unified', False)),
        'm': float(params.get('m', 0.0)),
        'I': np.asarray(I, dtype=float).reshape(-1).tolist() if I is not None else None,
        'r_thrust': np.asarray(r_thrust, dtype=float).reshape(-1).tolist() if r_thrust is not None else None,
        'weights': _json_safe_for_fingerprint(weights),
        'terminal_weights': _json_safe_for_fingerprint(params.get('terminal_weights') or {}),
        'bounds': _json_safe_for_fingerprint(bounds),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def optimization_summary_lookup(data, platform_id, mode_key, method_index):
    """Read optimization summary for platform + mode + method from trajectory JSON."""
    if not isinstance(data, dict):
        return None
    pid = _normalize_platform_id_for_cache(platform_id)
    mk = mode_key if mode_key in ('normal', 'min_time') else traj_opt_mode_key(mode_key)
    mid = str(int(method_index))
    v2 = data.get('optimizations_v2')
    if isinstance(v2, dict):
        plat = v2.get(pid)
        if isinstance(plat, dict):
            mode_bucket = plat.get(mk)
            if isinstance(mode_bucket, dict) and mid in mode_bucket:
                return mode_bucket[mid]
    return optimization_summary_from_json(data, mk, method_index=int(method_index))


def optimization_summary_matches_context(summary, platform_id, mode_key, method_index):
    """True when summary belongs to the requested optimization context."""
    if not summary:
        return False
    pid = _normalize_platform_id_for_cache(platform_id)
    mk = mode_key if mode_key in ('normal', 'min_time') else traj_opt_mode_key(mode_key)
    mid = int(method_index)
    if summary.get('platform_id') and summary.get('platform_id') != pid:
        return False
    if not optimization_summary_matches_mode(summary, mk):
        return False
    saved_method = summary.get('method')
    if saved_method is not None and int(saved_method) != mid:
        return False
    return True


def merge_optimization_summary_into_json(
    existing_data, platform_id, mode_key, method_index, summary,
):
    """Insert/update one optimization summary in trajectory JSON (v2 schema)."""
    data = dict(existing_data) if isinstance(existing_data, dict) else {}
    pid = _normalize_platform_id_for_cache(platform_id)
    mk = mode_key if mode_key in ('normal', 'min_time') else traj_opt_mode_key(mode_key)
    mid = str(int(method_index))
    v2 = data.get('optimizations_v2')
    if not isinstance(v2, dict):
        v2 = {}
    plat = dict(v2.get(pid) or {})
    mode_bucket = dict(plat.get(mk) or {})
    mode_bucket[mid] = dict(summary)
    plat[mk] = mode_bucket
    v2[pid] = plat
    data['optimizations_v2'] = v2
    data['version'] = max(int(data.get('version', 0)), TRAJ_JSON_VERSION)
    return data


def traj_opt_mode_key(mode_index=None):
    """Return ``'normal'`` or ``'min_time'`` for artifact file suffixes."""
    if mode_index is None:
        return 'normal'
    if int(mode_index) == TRAJ_OPT_MODE_MIN_TIME:
        return 'min_time'
    return 'normal'


def optimization_summary_from_json(data, mode_key='normal', method_index=None):
    """Extract saved optimization summary (legacy JSON schema)."""
    if not isinstance(data, dict):
        return None
    opts = data.get('optimizations')
    if isinstance(opts, dict) and opts.get(mode_key):
        legacy_summary = opts[mode_key]
        if method_index is None:
            return legacy_summary
        saved_method = legacy_summary.get('method') if legacy_summary else None
        if saved_method is None or int(saved_method) == int(method_index):
            return legacy_summary
        return None
    legacy = data.get('optimization')
    if not legacy:
        return None
    method_idx = legacy.get('method')
    if method_index is not None and method_idx is not None and int(method_idx) != int(method_index):
        return None
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


def _load_nmp_waypoints_from_file(path):
    """Return normalized waypoint rows from a saved NMP waypoints JSON file, or None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get('waypoints', data if isinstance(data, list) else [])
    if not raw:
        return None
    return [_normalize_waypoint_row(w) for w in raw]


def _load_nmp_model_overrides():
    """Return {platform_id: {mass, Ixx, Iyy, Izz, r_thrust_z, body_length,
    com_from_bottom}} saved overrides for the NMP tab, or {}."""
    if not os.path.isfile(NMP_MODEL_PARAMS_PATH):
        return {}
    try:
        with open(NMP_MODEL_PARAMS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for pid in (PLATFORM_PROXY, PLATFORM_REAL):
        entry = data.get(pid)
        if not isinstance(entry, dict):
            continue
        row = {}
        for key in NMP_MODEL_PARAM_KEYS:
            if key in entry:
                try:
                    row[key] = float(entry[key])
                except (TypeError, ValueError):
                    pass
        if row:
            out[pid] = row
    return out


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
    'Method 8: Differential flatness (min-snap on ξ, z, ψ)',
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

    def refresh_content_geometry(self):
        """Recompute embedded page height from its layout (after expand/collapse)."""
        vw = max(1, self.viewport().width())
        self._page.setFixedWidth(vw)
        lay = self._page.layout()
        if lay is not None:
            lay.invalidate()
            lay.activate()
        self._page.adjustSize()
        h = self._page.sizeHint().height()
        if lay is not None:
            h = max(h, lay.sizeHint().height(), lay.minimumSize().height())
        h = max(h, self._page.minimumSizeHint().height(), 1)
        self._page.setFixedHeight(h)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_content_geometry()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_content_geometry()


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
            
            # Method 8 (index 7): differential flatness (min-snap on flat outputs)
            if method == 7:
                if solve_with_flatness_waypoints is None:
                    self.error.emit(
                        "Method 8 unavailable: could not import tvc_traj_opt_flatness."
                    )
                    return
                if len(waypoints) < 2:
                    self.error.emit("Need at least 2 waypoints (start and at least one waypoint)")
                    return

                def callback_flat(_solver, seg_idx, current_xs, current_us, completed_xs, completed_us):
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

                def iteration_callback_flat(iter, cost, stop, seg_idx):
                    if self.running:
                        self.iteration_update.emit(iter, cost, stop, seg_idx)

                solver_fn = (
                    solve_with_flatness_waypoints_unified
                    if unified else solve_with_flatness_waypoints
                )
                t0 = time.perf_counter()
                _pack = solver_fn(
                    dt=dt, waypoints=waypoints, m=m, I=I, r_thrust=r_thrust,
                    weights=weights, bounds=bounds, max_iter=max_iter,
                    callback=callback_flat, running_flag=lambda: self.running,
                    iteration_callback=iteration_callback_flat, verbose_solve=True,
                )
                combined_xs, combined_us, all_loggers, us_actual = _pack[:4]
                flat_meta = _pack[4] if len(_pack) >= 5 and isinstance(_pack[4], dict) else {}
                total_time = time.perf_counter() - t0
                total_iters = sum(len(logger.costs) for logger in all_loggers) if all_loggers else 0
                timing_info = {
                    "total_time": total_time,
                    "total_iters": total_iters,
                    "avg_time_per_iter": total_time / total_iters if total_iters > 0 else 0.0,
                    "method": "Method 8 (Differential flatness)",
                    "us_actual": us_actual,
                }
                if isinstance(flat_meta, dict):
                    timing_info.update(flat_meta)
                if self.running:
                    self.finished.emit(combined_xs, combined_us, all_loggers, timing_info)
                return

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
                    running_flag=lambda: self.running,
                    rocket_platform=self.params.get('rocket_platform', PLATFORM_PROXY),
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


class Px4TuneSimulationThread(QThread):
    """Background PX4 cascade level tuning (step response)."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, params, phy_gui, tune_config):
        super().__init__()
        self.params = dict(params)
        self.phy_gui = dict(phy_gui)
        self.tune_config = dict(tune_config)

    def run(self):
        try:
            result = run_px4_cascade_tune_sim(
                self.params, self.phy_gui, self.tune_config,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n\n{traceback.format_exc()}')


class TrackingSimulationThread(QThread):
    """Background numerical closed-loop tracking simulation."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self, xs, us, time_states, controller_id, params, phy_gui,
        flat_outputs=None, flatness_physics=None, x0=None,
    ):
        super().__init__()
        self.xs = xs
        self.us = us
        self.time_states = time_states
        self.controller_id = controller_id
        self.params = dict(params)
        self.phy_gui = dict(phy_gui)
        self.flat_outputs = flat_outputs
        self.flatness_physics = flatness_physics
        self.x0 = x0

    def run(self):
        try:
            result = run_tracking_simulation(
                self.xs,
                self.us,
                self.time_states,
                self.controller_id,
                self.params,
                self.phy_gui,
                x0=self.x0,
                flat_outputs=self.flat_outputs,
                flatness_physics=self.flatness_physics,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n\n{traceback.format_exc()}')


class NmpPlanningThread(QThread):
    """Background differential-flatness (Method 8) planning for the NMP tab."""

    finished = pyqtSignal(list, list, dict)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = dict(params)
        self.running = True

    def run(self):
        try:
            if solve_with_flatness_waypoints is None:
                self.error.emit(
                    'Method 8 unavailable: could not import tvc_traj_opt_flatness.',
                )
                return
            p = self.params
            waypoints = waypoints_for_optimizer(p.get('waypoints', []))
            if len(waypoints) < 2:
                self.error.emit('Need at least 2 waypoints.')
                return
            t0 = time.perf_counter()
            pack = solve_with_flatness_waypoints(
                dt=float(p['dt']),
                waypoints=waypoints,
                m=float(p['m']),
                I=tuple(p['I']),
                r_thrust=tuple(p['r_thrust']),
                weights=p.get('weights') or {},
                bounds=p.get('bounds') or {},
                max_iter=1,
                running_flag=lambda: self.running,
                verbose_solve=False,
            )
            combined_xs, combined_us, _loggers, us_actual = pack[:4]
            flat_meta = pack[4] if len(pack) >= 5 and isinstance(pack[4], dict) else {}
            timing_info = dict(flat_meta)
            timing_info.update({
                'total_time': time.perf_counter() - t0,
                'method': 'Method 8 (NMP / differential flatness)',
                'us_actual': us_actual,
            })
            if self.running:
                self.finished.emit(combined_xs, combined_us, timing_info)
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n\n{traceback.format_exc()}')


class TrackingGifThread(QThread):
    """Background tracking GIF renderer (3D or 2D)."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, result, plan, output_path, view_mode='3d', playback_speed=1.0):
        super().__init__()
        self.result = result
        self.plan = plan
        self.output_path = output_path
        self.view_mode = view_mode
        self.playback_speed = float(playback_speed)

    def run(self):
        try:
            from tvc_traj_gui_gif import generate_tracking_gif
            path = generate_tracking_gif(
                self.result,
                self.plan,
                output_path=self.output_path,
                view_mode=self.view_mode,
                playback_speed=self.playback_speed,
            )
            self.finished.emit(path)
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n\n{traceback.format_exc()}')


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
        self._online_planner_proc = None
        self._bg_proc_log_handles = {}
        self._online_planner_diag_timer = QTimer(self)
        self._online_planner_diag_timer.setInterval(500)
        self._online_planner_diag_timer.timeout.connect(
            self._poll_online_planner_diagnostics
        )
        self._sitl_nodes_timer = QTimer(self)
        self._sitl_nodes_timer.setInterval(2000)
        self._sitl_nodes_timer.timeout.connect(self._poll_sitl_nodes_status)
        self._online_planner_advance_retries_left = 0
        self._method_sync_guard = False
        self._platform_phys_cache = {}
        self._cached_platform_id = PLATFORM_PROXY
        self._rocket_platform_guard = False
        self.tracking_params_file_path = DEFAULT_TRACKING_PARAMS_PATH
        self._tracking_config = all_default_tracking_config()
        self._px4_tune_config = default_px4_tune_config()
        self._px4_tune_sp_widgets = {}
        self._tracking_param_widgets = {}
        self._tracking_sim_thread = None
        self._px4_tune_sim_thread = None
        self._tracking_gif_thread = None
        self._tracking_gif_movie = None
        self._tracking_gif_path = DEFAULT_TRACKING_GIF_PATH
        self._tracking_gif_regen_pending = False
        self._last_tracking_result = None
        self.nmp_last_trajectory = None
        self._nmp_plan_thread = None
        self._nmp_tracking_sim_thread = None
        self._nmp_last_tracking_result = None
        self._nmp_active_tracking_traj = None
        self.nmp_waypoints = _load_nmp_waypoints_from_file(NMP_SAVED_WAYPOINTS_PATH) or [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 5.0],
        ]
        self._nmp_waypoint_table_updating = False
        self._nmp_platform_guard = False
        self._cached_nmp_platform_id = PLATFORM_PROXY
        self._nmp_model_overrides = _load_nmp_model_overrides()
        self.init_ui()
        self._update_params_file_label()
        if os.path.isfile(self.params_file_path):
            self._load_params_from_path(self.params_file_path, quiet=True)
        if os.path.isfile(self.tracking_params_file_path):
            self._load_tracking_params_from_path(self.tracking_params_file_path, quiet=True)
        self._restore_optimization_for_current_context(quiet=True)
        
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

        traj_tab, params_tab, tracking_tab, nmp_tab = self.create_parameter_panels()
        traj_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        params_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        tracking_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        nmp_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(True)
        self.left_tabs.addTab(TabScrollArea(traj_tab), 'Trajectory')
        self.left_tabs.addTab(TabScrollArea(params_tab), 'Parameters')
        self.left_tabs.addTab(TabScrollArea(tracking_tab), 'Tracking')
        self.left_tabs.addTab(TabScrollArea(nmp_tab), 'NMP')
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

        # Rocket platform (Proxy validation rig vs real 20 kg vehicle)
        platform_group = QGroupBox('Rocket platform')
        platform_layout = QVBoxLayout()
        platform_radio_row = QHBoxLayout()
        self.rocket_proxy_radio = QRadioButton(platform_label(PLATFORM_PROXY))
        self.rocket_real_radio = QRadioButton(platform_label(PLATFORM_REAL))
        self.rocket_proxy_radio.setChecked(True)
        self.rocket_proxy_radio.setToolTip(platform_description(PLATFORM_PROXY))
        self.rocket_real_radio.setToolTip(platform_description(PLATFORM_REAL))
        self._rocket_platform_button_group = QButtonGroup(self)
        self._rocket_platform_button_group.addButton(self.rocket_proxy_radio, 0)
        self._rocket_platform_button_group.addButton(self.rocket_real_radio, 1)
        self._rocket_platform_button_group.buttonClicked.connect(self._on_rocket_platform_changed)
        platform_radio_row.addWidget(self.rocket_proxy_radio)
        platform_radio_row.addWidget(self.rocket_real_radio)
        platform_radio_row.addStretch(1)
        platform_layout.addLayout(platform_radio_row)
        self.rocket_platform_desc_label = QLabel(platform_description(PLATFORM_PROXY))
        self.rocket_platform_desc_label.setWordWrap(True)
        self.rocket_platform_desc_label.setStyleSheet('color: #555;')
        platform_layout.addWidget(self.rocket_platform_desc_label)
        platform_group.setLayout(platform_layout)
        layout.addWidget(platform_group)
        
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
        self._traj_combo_prev_index = 0
        self._traj_opt_mode_prev_index = TRAJ_OPT_MODE_NORMAL
        self.trajectory_preset_combo.currentIndexChanged.connect(self.on_trajectory_preset_changed)
        
        # Optimization method (Trajectory tab; synced with Parameters tab)
        self.method_combo = QComboBox()
        layout.addWidget(_build_optimization_method_group(self.method_combo))
        self.method_combo.currentIndexChanged.connect(
            lambda idx: self._on_method_combo_changed(idx, self.method_combo)
        )
        self._refresh_trajectory_storage_path_label()

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

        self.T_min_thrust = QDoubleSpinBox()
        self.T_min_thrust.setRange(0, 500)
        self.T_min_thrust.setValue(0.0)
        self.T_min_thrust.setDecimals(1)
        self.T_min_thrust.setMaximumWidth(100)
        self.T_min_thrust.setToolTip('Minimum thrust magnitude [N]')

        self.T_max = QDoubleSpinBox()
        self.T_max.setRange(0, 500)
        self.T_max.setValue(25.0)
        self.T_max.setDecimals(2)
        self.T_max.setMaximumWidth(100)

        self.tau_yaw_max = QDoubleSpinBox()
        self.tau_yaw_max.setRange(0, 20)
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
        constraints_layout.addWidget(QLabel('Thrust min (N):'), row, 0)
        constraints_layout.addWidget(self.T_min_thrust, row, 1)
        constraints_layout.addWidget(QLabel('Thrust max (N):'), row, 2)
        constraints_layout.addWidget(self.T_max, row, 3)
        row += 1
        constraints_layout.addWidget(QLabel('Yaw torque (N·m):'), row, 0)
        constraints_layout.addWidget(self.tau_yaw_max, row, 1)
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

        self.physics_platform_hint = QLabel()
        self.physics_platform_hint.setWordWrap(True)
        self.physics_platform_hint.setStyleSheet('color: #555;')
        physics_layout.addWidget(self.physics_platform_hint, 4, 0, 1, 4)
        
        physics_group.setLayout(physics_layout)

        physics_tab = QWidget()
        physics_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        physics_tab_layout = QVBoxLayout()
        physics_tab_layout.addWidget(physics_group)
        physics_tab.setLayout(physics_tab_layout)
        params_tabs.insertTab(1, physics_tab, 'Physical')

        adv_layout.addWidget(params_tabs)
        self._update_platform_spin_ranges(PLATFORM_PROXY)
        self._refresh_physics_platform_hint()
        
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
            7: {  # Method 8: differential flatness (bounds only; no NLP weights)
                "w_p": 1.0, "w_v": 0.2, "w_R": 0.5, "w_yaw": 0.5, "w_w": 0.1,
                "w_u": 0.5, "w_du": 0.5,
                "terminal_cost_multiplier": 200.0,
                "k_bound": 200.0, "k_state_bound": 200.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 1.0, "v_vertical_max": 3.0,
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

        tracking_tab = self._create_tracking_panel()
        nmp_tab = self._create_nmp_panel()

        return traj_tab, params_tab, tracking_tab, nmp_tab
    
    def _refresh_tab_scroll_areas(self):
        """Reflow sidebar tab pages after dynamic widgets change size."""
        if not hasattr(self, 'left_tabs'):
            return
        for i in range(self.left_tabs.count()):
            scroll = self.left_tabs.widget(i)
            if isinstance(scroll, TabScrollArea):
                scroll.refresh_content_geometry()

    def _make_collapsible_group(self, title, parent_layout, expanded=False):
        """Collapsible section: click the arrow title row (no checkbox)."""
        section = QGroupBox()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(8, 6, 8, 6)
        section_layout.setSpacing(4)

        header_btn = QPushButton()
        header_btn.setFlat(True)
        header_btn.setCursor(Qt.PointingHandCursor)
        header_btn.setStyleSheet(
            'QPushButton { text-align: left; font-weight: bold; border: none; padding: 2px 4px; }'
            'QPushButton:hover { color: #1565c0; }'
        )
        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 0, 0, 0)
        inner_layout.setSpacing(4)

        section_layout.addWidget(header_btn)
        section_layout.addWidget(inner)
        parent_layout.addWidget(section)

        state = {'open': bool(expanded), 'title': str(title)}

        def _sync_header():
            arrow = '▾' if state['open'] else '▸'
            header_btn.setText(f'{arrow}  {state["title"]}')

        def _set_title(new_title):
            state['title'] = str(new_title)
            _sync_header()

        def _toggle():
            state['open'] = not state['open']
            inner.setVisible(state['open'])
            inner.setMaximumHeight(16777215 if state['open'] else 0)
            _sync_header()
            self._refresh_tab_scroll_areas()
            if state['open']:
                # Nested scroll areas report a stale sizeHint the instant they're
                # unhidden (Qt hasn't laid them out while hidden); redo the geometry
                # pass once the event loop catches up so the section doesn't stay
                # clipped on its first expansion.
                QTimer.singleShot(0, self._refresh_tab_scroll_areas)

        header_btn.clicked.connect(_toggle)
        _sync_header()
        inner.setVisible(expanded)
        inner.setMaximumHeight(16777215 if expanded else 0)
        section._tvc_set_title = _set_title

        return section, inner_layout

    _ACT_DYN_CHANNEL_ROWS = (
        ('tau_gimbal', 'Gimbal (th_p, th_r)'),
        ('tau_thrust', 'Thrust'),
        ('tau_yaw_torque', 'Yaw torque'),
    )

    _ACT_MISMATCH_ROWS = (
        ('gimbal', 'Gimbal (th_p, th_r)', 'rad'),
        ('thrust', 'Thrust', 'N'),
        ('yaw_torque', 'Yaw torque', 'N·m'),
    )

    def _create_actuator_dynamics_panel(self, parent_layout):
        """Actuator lag, scale/bias mismatch, and thrust quantization."""
        self.actuator_dynamics_group = QGroupBox('Actuator dynamics')
        act_group = self.actuator_dynamics_group
        act_layout = QVBoxLayout(act_group)
        act_layout.setSpacing(4)

        self.act_dyn_enable_cb = QCheckBox('Enable first-order actuator lag')
        self.act_dyn_enable_cb.setToolTip(
            'When enabled, TVC gimbal, thrust, and yaw torque follow a first-order lag\n'
            'before entering the nonlinear dynamics model.'
        )
        self.act_dyn_enable_cb.stateChanged.connect(self._on_act_dyn_enable_changed)
        act_layout.addWidget(self.act_dyn_enable_cb)

        self.act_dyn_lag_detail = QWidget()
        lag_layout = QVBoxLayout(self.act_dyn_lag_detail)
        lag_layout.setContentsMargins(12, 0, 0, 0)
        lag_layout.setSpacing(4)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.addWidget(QLabel(''), 0, 0)
        grid.addWidget(QLabel('τ [s]'), 0, 1)
        grid.addWidget(QLabel('f_c [Hz]'), 0, 2)
        self._act_dyn_tau_spins = {}
        self._act_dyn_bw_spins = {}
        for row, (key, label) in enumerate(self._ACT_DYN_CHANNEL_ROWS, start=1):
            grid.addWidget(QLabel(label), row, 0)
            tau_spin = QDoubleSpinBox()
            tau_spin.setRange(TAU_MIN, TAU_MAX)
            tau_spin.setDecimals(3)
            tau_spin.setSingleStep(0.005)
            tau_spin.valueChanged.connect(
                lambda _v, k=key: self._on_act_dyn_tau_changed(k)
            )
            bw_spin = QDoubleSpinBox()
            bw_spin.setRange(BW_MIN_HZ, BW_MAX_HZ)
            bw_spin.setDecimals(2)
            bw_spin.setSingleStep(0.1)
            bw_spin.valueChanged.connect(
                lambda _v, k=key: self._on_act_dyn_bw_changed(k)
            )
            grid.addWidget(tau_spin, row, 1)
            grid.addWidget(bw_spin, row, 2)
            self._act_dyn_tau_spins[key] = tau_spin
            self._act_dyn_bw_spins[key] = bw_spin
        lag_layout.addLayout(grid)
        lag_hint = QLabel('τ and f_c are linked: τ = 1 / (2π f_c)')
        lag_hint.setWordWrap(True)
        lag_hint.setStyleSheet('color: #555;')
        lag_layout.addWidget(lag_hint)
        act_layout.addWidget(self.act_dyn_lag_detail)

        # Scale / bias mismatch (error tolerance tests)
        self.act_mismatch_enable_cb = QCheckBox('Enable scale / bias mismatch')
        self.act_mismatch_enable_cb.setToolTip(
            'Plant-side calibration error after lag:\n'
            '  u_plant = scale · u_lag + bias\n'
            'Gimbal scale/bias applies to both th_p and th_r.\n'
            'Use for thrust / gimbal / yaw-torque error-tolerance sweeps.'
        )
        self.act_mismatch_enable_cb.stateChanged.connect(self._on_act_mismatch_enable_changed)
        act_layout.addWidget(self.act_mismatch_enable_cb)

        self.act_mismatch_detail = QWidget()
        mm_layout = QVBoxLayout(self.act_mismatch_detail)
        mm_layout.setContentsMargins(12, 0, 0, 0)
        mm_layout.setSpacing(4)
        mm_grid = QGridLayout()
        mm_grid.setHorizontalSpacing(8)
        mm_grid.setVerticalSpacing(4)
        mm_grid.addWidget(QLabel(''), 0, 0)
        mm_grid.addWidget(QLabel('scale [—]'), 0, 1)
        mm_grid.addWidget(QLabel('bias'), 0, 2)
        self._act_scale_spins = {}
        self._act_bias_spins = {}
        for row, (key, label, bias_unit) in enumerate(self._ACT_MISMATCH_ROWS, start=1):
            mm_grid.addWidget(QLabel(label), row, 0)
            scale_spin = QDoubleSpinBox()
            scale_spin.setRange(SCALE_MIN, SCALE_MAX)
            scale_spin.setDecimals(3)
            scale_spin.setSingleStep(0.01)
            scale_spin.setValue(1.0)
            scale_spin.setToolTip(f'Multiplying scale for {label} (1.0 = ideal)')
            scale_spin.valueChanged.connect(self._on_act_mismatch_spin_changed)
            bias_spin = QDoubleSpinBox()
            bias_lo, bias_hi = (-1.0, 1.0) if key == 'gimbal' else (
                (-200.0, 200.0) if key == 'thrust' else (-50.0, 50.0)
            )
            bias_spin.setRange(bias_lo, bias_hi)
            bias_spin.setDecimals(4 if key == 'gimbal' else 2)
            bias_spin.setSingleStep(0.001 if key == 'gimbal' else 0.5)
            bias_spin.setValue(0.0)
            bias_spin.setToolTip(f'Additive bias for {label} [{bias_unit}]')
            bias_spin.setSuffix(f' {bias_unit}')
            bias_spin.valueChanged.connect(self._on_act_mismatch_spin_changed)
            mm_grid.addWidget(scale_spin, row, 1)
            mm_grid.addWidget(bias_spin, row, 2)
            self._act_scale_spins[key] = scale_spin
            self._act_bias_spins[key] = bias_spin
        mm_layout.addLayout(mm_grid)
        mm_hint = QLabel(
            'Pipeline: cmd → lag → scale·u+bias → thrust quantization → plant'
        )
        mm_hint.setWordWrap(True)
        mm_hint.setStyleSheet('color: #555;')
        mm_layout.addWidget(mm_hint)
        act_layout.addWidget(self.act_mismatch_detail)

        self.act_thrust_quant_cb = QCheckBox('Thrust quantization (discrete steps)')
        self.act_thrust_quant_cb.setToolTip(
            'Round commanded thrust to the nearest multiple of the resolution.\n'
            'Proxy ≈ 0.5 N; real platform ≈ 10 N (adjustable).'
        )
        self.act_thrust_quant_cb.stateChanged.connect(self._on_act_thrust_quant_changed)
        act_layout.addWidget(self.act_thrust_quant_cb)

        res_row = QHBoxLayout()
        res_row.setContentsMargins(12, 0, 0, 0)
        res_row.addWidget(QLabel('Thrust resolution [N]:'))
        self.act_thrust_resolution_spin = QDoubleSpinBox()
        self.act_thrust_resolution_spin.setRange(THRUST_RES_MIN, THRUST_RES_MAX)
        self.act_thrust_resolution_spin.setDecimals(2)
        self.act_thrust_resolution_spin.setSingleStep(0.1)
        self.act_thrust_resolution_spin.setToolTip(
            'Smallest thrust increment after quantization (0 = continuous).'
        )
        self.act_thrust_resolution_spin.valueChanged.connect(self._on_act_thrust_resolution_changed)
        res_row.addWidget(self.act_thrust_resolution_spin)
        res_row.addStretch(1)
        act_layout.addLayout(res_row)

        parent_layout.addWidget(act_group)

        self._act_dyn_updating = False
        plat = self._current_rocket_platform_id() if hasattr(self, 'rocket_proxy_radio') else None
        self._apply_actuator_config(
            self._tracking_config.setdefault(
                'actuator', default_actuator_tracking_config(plat),
            )
        )

    def _on_act_dyn_enable_changed(self, _state=None):
        enabled = self.act_dyn_enable_cb.isChecked()
        self.act_dyn_lag_detail.setEnabled(enabled)
        self.act_dyn_lag_detail.setVisible(enabled)
        if getattr(self, '_act_dyn_updating', False):
            return
        self._store_actuator_config()
        self._refresh_tab_scroll_areas()

    def _on_act_mismatch_enable_changed(self, _state=None):
        on = self.act_mismatch_enable_cb.isChecked()
        self.act_mismatch_detail.setEnabled(on)
        self.act_mismatch_detail.setVisible(on)
        if getattr(self, '_act_dyn_updating', False):
            return
        self._store_actuator_config()
        self._refresh_tab_scroll_areas()

    def _on_act_mismatch_spin_changed(self, _value=None):
        if getattr(self, '_act_dyn_updating', False):
            return
        self._store_actuator_config()

    def _on_act_thrust_quant_changed(self, _state=None):
        on = self.act_thrust_quant_cb.isChecked()
        self.act_thrust_resolution_spin.setEnabled(on)
        if getattr(self, '_act_dyn_updating', False):
            return
        self._store_actuator_config()
        self._refresh_tab_scroll_areas()

    def _on_act_thrust_resolution_changed(self, _value=None):
        if getattr(self, '_act_dyn_updating', False):
            return
        act = self._tracking_config.setdefault('actuator', default_actuator_tracking_config())
        act['thrust_resolution_N'] = float(self.act_thrust_resolution_spin.value())

    def _on_act_dyn_tau_changed(self, key):
        if getattr(self, '_act_dyn_updating', False):
            return
        tau_spin = self._act_dyn_tau_spins.get(key)
        bw_spin = self._act_dyn_bw_spins.get(key)
        if tau_spin is None or bw_spin is None:
            return
        tau = float(tau_spin.value())
        act = self._tracking_config.setdefault('actuator', default_actuator_tracking_config())
        act[key] = tau
        self._act_dyn_updating = True
        try:
            bw_spin.setValue(tau_to_bandwidth_hz(tau))
        finally:
            self._act_dyn_updating = False

    def _on_act_dyn_bw_changed(self, key):
        if getattr(self, '_act_dyn_updating', False):
            return
        tau_spin = self._act_dyn_tau_spins.get(key)
        bw_spin = self._act_dyn_bw_spins.get(key)
        if tau_spin is None or bw_spin is None:
            return
        tau = bandwidth_hz_to_tau(float(bw_spin.value()))
        act = self._tracking_config.setdefault('actuator', default_actuator_tracking_config())
        act[key] = tau
        self._act_dyn_updating = True
        try:
            tau_spin.setValue(tau)
        finally:
            self._act_dyn_updating = False

    def _refresh_act_dyn_spins(self):
        if not hasattr(self, '_act_dyn_tau_spins'):
            return
        act = self._tracking_config.get('actuator') or default_actuator_tracking_config()
        self._act_dyn_updating = True
        try:
            for key, tau_spin in self._act_dyn_tau_spins.items():
                tau = float(act.get(key, 0.05))
                bw_spin = self._act_dyn_bw_spins[key]
                tau_spin.setValue(tau)
                bw_spin.setValue(tau_to_bandwidth_hz(tau))
            if hasattr(self, '_act_scale_spins'):
                for key, spin in self._act_scale_spins.items():
                    spin.setValue(float(act.get(f'scale_{key}', 1.0)))
                for key, spin in self._act_bias_spins.items():
                    spin.setValue(float(act.get(f'bias_{key}', 0.0)))
        finally:
            self._act_dyn_updating = False

    def _store_actuator_config(self):
        if not hasattr(self, 'act_dyn_enable_cb'):
            return
        act = self._tracking_config.setdefault('actuator', default_actuator_tracking_config())
        act['enabled'] = self.act_dyn_enable_cb.isChecked()
        act['thrust_quant_enabled'] = self.act_thrust_quant_cb.isChecked()
        act['thrust_resolution_N'] = float(self.act_thrust_resolution_spin.value())
        for key, tau_spin in self._act_dyn_tau_spins.items():
            act[key] = float(tau_spin.value())
        if hasattr(self, 'act_mismatch_enable_cb'):
            act['mismatch_enable'] = self.act_mismatch_enable_cb.isChecked()
            for key, spin in self._act_scale_spins.items():
                act[f'scale_{key}'] = float(spin.value())
            for key, spin in self._act_bias_spins.items():
                act[f'bias_{key}'] = float(spin.value())

    def _apply_actuator_config(self, cfg=None):
        if not hasattr(self, 'act_dyn_enable_cb'):
            return
        plat = self._current_rocket_platform_id() if hasattr(self, 'rocket_proxy_radio') else None
        defaults = default_actuator_tracking_config(plat)
        raw = dict(cfg or defaults)
        act = default_actuator_tracking_config(plat)
        act['enabled'] = bool(raw.get('enabled', False))
        act['thrust_quant_enabled'] = bool(raw.get(
            'thrust_quant_enabled', raw.get('thrust_quant_enable', False),
        ))
        if 'thrust_resolution_N' in raw:
            act['thrust_resolution_N'] = float(raw['thrust_resolution_N'])
        else:
            act['thrust_resolution_N'] = default_thrust_quantization_resolution(plat or 'proxy')
        act['tau_gimbal'] = float(raw.get('tau_gimbal', 0.05))
        act['tau_thrust'] = float(raw.get('tau_thrust', 0.05))
        act['tau_yaw_torque'] = float(raw.get('tau_yaw_torque', 0.05))
        act['mismatch_enable'] = bool(raw.get('mismatch_enable', False))
        for key in ('gimbal', 'thrust', 'yaw_torque'):
            act[f'scale_{key}'] = float(raw.get(f'scale_{key}', 1.0))
            act[f'bias_{key}'] = float(raw.get(f'bias_{key}', 0.0))
        # Commit desired config first, then push into widgets under a guard so
        # checkbox/spin signals cannot overwrite with stale spin values.
        self._tracking_config['actuator'] = act
        self._act_dyn_updating = True
        try:
            for key, tau_spin in self._act_dyn_tau_spins.items():
                tau = float(act.get(key, 0.05))
                self._act_dyn_bw_spins[key].setValue(tau_to_bandwidth_hz(tau))
                tau_spin.setValue(tau)
            self.act_thrust_resolution_spin.setValue(act['thrust_resolution_N'])
            self.act_dyn_enable_cb.setChecked(act['enabled'])
            self.act_thrust_quant_cb.setChecked(act['thrust_quant_enabled'])
            if hasattr(self, 'act_mismatch_enable_cb'):
                self.act_mismatch_enable_cb.setChecked(act['mismatch_enable'])
                for key, spin in self._act_scale_spins.items():
                    spin.setValue(float(act.get(f'scale_{key}', 1.0)))
                for key, spin in self._act_bias_spins.items():
                    spin.setValue(float(act.get(f'bias_{key}', 0.0)))
        finally:
            self._act_dyn_updating = False
        # Re-assert stored config (signals may have run during setChecked).
        self._tracking_config['actuator'] = dict(act)
        self.act_dyn_lag_detail.setEnabled(act['enabled'])
        self.act_dyn_lag_detail.setVisible(act['enabled'])
        self.act_thrust_resolution_spin.setEnabled(act['thrust_quant_enabled'])
        if hasattr(self, 'act_mismatch_detail'):
            self.act_mismatch_detail.setEnabled(act['mismatch_enable'])
            self.act_mismatch_detail.setVisible(act['mismatch_enable'])

    def _actuator_params_for_sim(self):
        act = self._tracking_config.get('actuator') or default_actuator_tracking_config()
        return {
            'act_dyn_enable': bool(act.get('enabled', False)),
            'thrust_quant_enable': bool(act.get('thrust_quant_enabled', False)),
            'thrust_resolution_N': float(act.get('thrust_resolution_N', 0.5)),
            'tau_gimbal': float(act.get('tau_gimbal', 0.05)),
            'tau_thrust': float(act.get('tau_thrust', 0.05)),
            'tau_yaw_torque': float(act.get('tau_yaw_torque', 0.05)),
            'mismatch_enable': bool(act.get('mismatch_enable', False)),
            'scale_gimbal': float(act.get('scale_gimbal', 1.0)),
            'bias_gimbal': float(act.get('bias_gimbal', 0.0)),
            'scale_thrust': float(act.get('scale_thrust', 1.0)),
            'bias_thrust': float(act.get('bias_thrust', 0.0)),
            'scale_yaw_torque': float(act.get('scale_yaw_torque', 1.0)),
            'bias_yaw_torque': float(act.get('bias_yaw_torque', 0.0)),
        }

    def _migrate_actuator_from_controller_params(self, cfg):
        """Lift legacy per-controller actuator keys into top-level actuator block."""
        if cfg.get('actuator'):
            return cfg.get('actuator')
        params_map = cfg.get('params') or {}
        for _cid, raw in params_map.items():
            if not isinstance(raw, dict):
                continue
            if not any(k in raw for k in (
                'act_dyn_enable', 'act_dyn_gimbal', 'act_dyn_thrust', 'act_dyn_yaw', 'tau_gimbal',
                'mismatch_enable', 'scale_thrust', 'bias_thrust',
            )):
                continue
            act = default_actuator_tracking_config()
            act['enabled'] = bool(raw.get(
                'act_dyn_enable',
                any(raw.get(k) for k in ('act_dyn_gimbal', 'act_dyn_thrust', 'act_dyn_yaw')),
            ))
            act['tau_gimbal'] = float(raw.get('tau_gimbal', 0.05))
            act['tau_thrust'] = float(raw.get('tau_thrust', 0.05))
            act['tau_yaw_torque'] = float(raw.get('tau_yaw_torque', 0.05))
            act['mismatch_enable'] = bool(raw.get('mismatch_enable', False))
            for key in ('gimbal', 'thrust', 'yaw_torque'):
                act[f'scale_{key}'] = float(raw.get(f'scale_{key}', 1.0))
                act[f'bias_{key}'] = float(raw.get(f'bias_{key}', 0.0))
            return act
        return default_actuator_tracking_config()

    def _create_numerical_sim_timing_panel(self, parent_layout):
        """Plant step and controller rate for numerical closed-loop simulation."""
        self.numerical_sim_group = QGroupBox('Numerical simulation timing')
        grid = QGridLayout(self.numerical_sim_group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        self.num_sim_dt_spin = QDoubleSpinBox()
        self.num_sim_dt_spin.setRange(0.001, 0.05)
        self.num_sim_dt_spin.setDecimals(4)
        self.num_sim_dt_spin.setSingleStep(0.001)
        self.num_sim_dt_spin.setToolTip(
            'Plant integration step [s]. Smaller values improve accuracy for fast dynamics.'
        )
        self.num_sim_dt_spin.valueChanged.connect(self._on_numerical_sim_timing_changed)

        self.num_control_dt_spin = QDoubleSpinBox()
        self.num_control_dt_spin.setRange(0.005, 0.2)
        self.num_control_dt_spin.setDecimals(4)
        self.num_control_dt_spin.setSingleStep(0.005)
        self.num_control_dt_spin.setToolTip(
            'Controller update period [s] (zero-order hold between updates).'
        )
        self.num_control_dt_spin.valueChanged.connect(self._on_numerical_sim_timing_changed)

        self.num_terminal_hold_spin = QDoubleSpinBox()
        self.num_terminal_hold_spin.setRange(0.0, 120.0)
        self.num_terminal_hold_spin.setDecimals(2)
        self.num_terminal_hold_spin.setSingleStep(0.5)
        self.num_terminal_hold_spin.setToolTip(
            'Extra hover time after the planned trajectory ends (terminal hold gains). '
            'Ignored when Total duration > 0.'
        )
        self.num_terminal_hold_spin.valueChanged.connect(self._on_numerical_sim_timing_changed)

        self.num_total_duration_spin = QDoubleSpinBox()
        self.num_total_duration_spin.setRange(0.0, 600.0)
        self.num_total_duration_spin.setDecimals(2)
        self.num_total_duration_spin.setSingleStep(1.0)
        self.num_total_duration_spin.setSpecialValueText('Auto (plan + hold)')
        self.num_total_duration_spin.setToolTip(
            'Fixed simulation length from t=0 [s]. 0 = automatic: planned duration '
            '+ terminal hold.'
        )
        self.num_total_duration_spin.valueChanged.connect(self._on_numerical_sim_timing_changed)

        self.lbl_num_control_hz = QLabel()
        self.lbl_num_control_hz.setStyleSheet('color: #555;')
        self.lbl_num_substeps = QLabel()
        self.lbl_num_substeps.setStyleSheet('color: #555;')
        self.lbl_num_sim_duration = QLabel()
        self.lbl_num_sim_duration.setStyleSheet('color: #555;')
        self.lbl_num_sim_duration.setWordWrap(True)

        grid.addWidget(QLabel('Plant step sim_dt [s]:'), 0, 0)
        grid.addWidget(self.num_sim_dt_spin, 0, 1)
        grid.addWidget(QLabel('Control period [s]:'), 1, 0)
        grid.addWidget(self.num_control_dt_spin, 1, 1)
        grid.addWidget(QLabel('Terminal hold after plan [s]:'), 2, 0)
        grid.addWidget(self.num_terminal_hold_spin, 2, 1)
        grid.addWidget(QLabel('Total duration [s]:'), 3, 0)
        grid.addWidget(self.num_total_duration_spin, 3, 1)
        grid.addWidget(self.lbl_num_control_hz, 4, 0, 1, 2)
        grid.addWidget(self.lbl_num_substeps, 5, 0, 1, 2)
        grid.addWidget(self.lbl_num_sim_duration, 6, 0, 1, 2)

        parent_layout.addWidget(self.numerical_sim_group)
        self._apply_numerical_sim_config(
            self._tracking_config.get('numerical_sim') or default_numerical_sim_config(),
        )

    def _on_numerical_sim_timing_changed(self, _value=None):
        if getattr(self, '_num_sim_timing_guard', False):
            return
        sim_dt = float(self.num_sim_dt_spin.value())
        control_dt = float(self.num_control_dt_spin.value())
        if control_dt < sim_dt:
            self._num_sim_timing_guard = True
            self.num_control_dt_spin.setValue(sim_dt)
            self._num_sim_timing_guard = False
            control_dt = sim_dt
        ratio = max(1, int(round(control_dt / sim_dt)))
        aligned_control_dt = ratio * sim_dt
        if abs(aligned_control_dt - control_dt) > 1e-9:
            self._num_sim_timing_guard = True
            self.num_control_dt_spin.setValue(aligned_control_dt)
            self._num_sim_timing_guard = False
            control_dt = aligned_control_dt
        hz = 1.0 / control_dt if control_dt > 0 else 0.0
        self.lbl_num_control_hz.setText(f'Control rate: {hz:.1f} Hz')
        self.lbl_num_substeps.setText(
            f'Plant substeps per control update: {ratio}'
        )
        hold = float(self.num_terminal_hold_spin.value())
        total = float(self.num_total_duration_spin.value())
        plan_s = self._planned_trajectory_duration_s()
        if total > 0.0:
            sim_s = total
            dur_txt = f'Simulation length: {sim_s:.2f} s (fixed total)'
        elif plan_s is not None:
            sim_s = plan_s + hold
            dur_txt = (
                f'Simulation length: {sim_s:.2f} s '
                f'(plan {plan_s:.2f} s + hold {hold:.2f} s)'
            )
        else:
            dur_txt = (
                f'Simulation length: plan duration + {hold:.2f} s hold '
                f'(optimize a trajectory to preview)'
            )
        self.lbl_num_sim_duration.setText(dur_txt)
        self._tracking_config['numerical_sim'] = {
            'sim_dt': sim_dt,
            'control_dt': control_dt,
            'terminal_hold_duration_s': hold,
            'total_duration_s': total,
        }

    def _planned_trajectory_duration_s(self):
        """Planned trajectory span [s] from cached result, or None."""
        traj = getattr(self, 'last_trajectory', None)
        if not traj or traj.get('xs') is None:
            return None
        ts = traj.get('time_states')
        if ts is not None and len(ts) >= 2:
            return float(np.asarray(ts, dtype=float)[-1] - np.asarray(ts, dtype=float)[0])
        dt = traj.get('dt')
        xs = traj.get('xs')
        if dt is not None and xs is not None and len(xs) >= 2:
            return float((len(xs) - 1) * float(dt))
        return None

    def _store_numerical_sim_config(self):
        if not hasattr(self, 'num_sim_dt_spin'):
            return
        self._on_numerical_sim_timing_changed()

    def _apply_numerical_sim_config(self, cfg):
        if not hasattr(self, 'num_sim_dt_spin'):
            return
        cfg = dict(cfg or default_numerical_sim_config())
        self._num_sim_timing_guard = True
        self.num_sim_dt_spin.setValue(float(cfg.get('sim_dt', 0.005)))
        self.num_control_dt_spin.setValue(float(cfg.get('control_dt', 0.02)))
        self.num_terminal_hold_spin.setValue(
            float(cfg.get('terminal_hold_duration_s', 3.0)),
        )
        self.num_total_duration_spin.setValue(float(cfg.get('total_duration_s', 0.0)))
        self._num_sim_timing_guard = False
        self._tracking_config['numerical_sim'] = dict(cfg)
        self._on_numerical_sim_timing_changed()

    def _numerical_sim_params_for_sim(self):
        cfg = self._tracking_config.get('numerical_sim') or default_numerical_sim_config()
        return {
            'sim_dt': float(cfg.get('sim_dt', 0.005)),
            'control_dt': float(cfg.get('control_dt', 0.02)),
            'terminal_hold_duration_s': float(cfg.get('terminal_hold_duration_s', 3.0)),
            'total_duration_s': float(cfg.get('total_duration_s', 0.0)),
        }

    def _migrate_numerical_sim_from_controller_params(self, cfg):
        return migrate_numerical_sim_config(cfg)

    def _create_px4_cascade_tune_panel(self, parent_layout):
        """Step-response tuning for isolated PX4 cascade levels."""
        self.px4_tune_group, tune_outer = self._make_collapsible_group(
            'PX4 cascade tuning (step response)',
            parent_layout,
            expanded=False,
        )
        self.px4_tune_group.setToolTip(
            'Click the section title (▸/▾) to expand or collapse cascade step-response tuning.\n'
            'Uses the same Actuator dynamics panel above (first-order lag / thrust quantization).'
        )

        level_row = QGridLayout()
        self.px4_tune_level_combo = QComboBox()
        for level in TUNE_LEVELS:
            self.px4_tune_level_combo.addItem(TUNE_LEVEL_LABELS[level], level)
        self.px4_tune_level_combo.currentIndexChanged.connect(self._on_px4_tune_level_changed)
        level_row.addWidget(QLabel('Tune level:'), 0, 0)
        level_row.addWidget(self.px4_tune_level_combo, 0, 1)

        self.px4_tune_duration = QDoubleSpinBox()
        self.px4_tune_duration.setRange(0.5, 60.0)
        self.px4_tune_duration.setDecimals(1)
        self.px4_tune_duration.setSingleStep(0.5)
        self.px4_tune_duration.setValue(5.0)
        self.px4_tune_duration.setToolTip('Simulation duration for the step response.')
        level_row.addWidget(QLabel('Duration [s]:'), 1, 0)
        level_row.addWidget(self.px4_tune_duration, 1, 1)
        tune_outer.addLayout(level_row)

        self.px4_tune_sp_stack = QStackedWidget()
        self._px4_tune_sp_widgets = {}
        sp_defs = {
            'rate': [
                ('p_deg_s', 'Roll rate p [deg/s]', 20.0, -180.0, 180.0),
                ('q_deg_s', 'Pitch rate q [deg/s]', 0.0, -180.0, 180.0),
                ('r_deg_s', 'Yaw rate r [deg/s]', 0.0, -180.0, 180.0),
            ],
            'attitude': [
                ('roll_deg', 'Roll [deg]', 5.0, -45.0, 45.0),
                ('pitch_deg', 'Pitch [deg]', 0.0, -45.0, 45.0),
                ('yaw_deg', 'Yaw [deg]', 0.0, -180.0, 180.0),
            ],
            'velocity': [
                ('vx', 'Vx [m/s]', 0.0, -5.0, 5.0),
                ('vy', 'Vy [m/s]', 0.0, -5.0, 5.0),
                ('vz', 'Vz [m/s]', 0.0, -3.0, 3.0),
                ('yaw_deg', 'Yaw hold [deg]', 0.0, -180.0, 180.0),
            ],
            'position': [
                ('x', 'X [m]', 0.5, -5.0, 5.0),
                ('y', 'Y [m]', 0.0, -5.0, 5.0),
                ('z', 'Z [m]', 0.2, -3.0, 3.0),
                ('yaw_deg', 'Yaw hold [deg]', 0.0, -180.0, 180.0),
            ],
        }
        defaults = default_px4_tune_config()['setpoints']
        for level in TUNE_LEVELS:
            page = QWidget()
            grid = QGridLayout(page)
            grid.setContentsMargins(0, 0, 0, 0)
            self._px4_tune_sp_widgets[level] = {}
            for row, (key, label, default, min_v, max_v) in enumerate(sp_defs[level]):
                grid.addWidget(QLabel(label), row, 0)
                spin = QDoubleSpinBox()
                spin.setRange(min_v, max_v)
                spin.setDecimals(2)
                spin.setSingleStep(0.1)
                spin.setValue(float(defaults.get(level, {}).get(key, default)))
                grid.addWidget(spin, row, 1)
                self._px4_tune_sp_widgets[level][key] = spin
            self.px4_tune_sp_stack.addWidget(page)
        tune_outer.addWidget(self.px4_tune_sp_stack)

        self.btn_run_px4_tune = QPushButton('Run cascade tune sim')
        self.btn_run_px4_tune.setToolTip(
            'Hover step response at the selected cascade level.\n'
            'Outer loops are bypassed; only the active level and inner loops run.\n'
            'Plant input includes Actuator dynamics (τ / thrust quantization) when enabled above.'
        )
        self.btn_run_px4_tune.clicked.connect(self.run_px4_cascade_tune_sim)
        tune_outer.addWidget(self.btn_run_px4_tune)

        self.lbl_px4_tune_result = QLabel(
            'Tune: rate → attitude → velocity → position. '
            'Uses Actuator dynamics when enabled. '
            'States tab: dashed cmd = inner-loop setpoints; controls show cmd vs act.'
        )
        self.lbl_px4_tune_result.setWordWrap(True)
        self.lbl_px4_tune_result.setStyleSheet('color: #555;')
        tune_outer.addWidget(self.lbl_px4_tune_result)

    def _update_tracking_panel_visibility(self):
        """Show sidebar blocks only for the active controller / simulation mode."""
        if not hasattr(self, 'tracking_controller_combo'):
            return
        cid = self._current_tracking_controller_id()
        sitl = (
            hasattr(self, 'tracking_sim_sitl_radio')
            and self.tracking_sim_sitl_radio.isChecked()
        )
        numerical = not sitl

        if hasattr(self, 'numerical_sim_group'):
            self.numerical_sim_group.setVisible(numerical)
        if hasattr(self, 'actuator_dynamics_group'):
            self.actuator_dynamics_group.setVisible(numerical)
        if hasattr(self, 'tracking_params_group'):
            self.tracking_params_group.setVisible(True)
        if hasattr(self, 'px4_tune_group'):
            show_px4_tune = cid in (CONTROLLER_PX4, CONTROLLER_FLATNESS)
            self.px4_tune_group.setVisible(show_px4_tune)
            if hasattr(self, 'btn_run_px4_tune'):
                self.btn_run_px4_tune.setEnabled(show_px4_tune and numerical)

        if hasattr(self, 'btn_run_numerical_tracking'):
            self.btn_run_numerical_tracking.setVisible(numerical)
        if hasattr(self, 'sitl_run_widget'):
            numerical_only = cid in (CONTROLLER_MPC, CONTROLLER_ACADOS_NMPC)
            self.sitl_run_widget.setVisible(sitl and not numerical_only)

        self._refresh_tab_scroll_areas()

    def _update_px4_tune_visibility(self):
        self._update_tracking_panel_visibility()

    def _on_px4_tune_level_changed(self, _index=None):
        if not hasattr(self, 'px4_tune_sp_stack'):
            return
        idx = self.px4_tune_level_combo.currentIndex()
        self.px4_tune_sp_stack.setCurrentIndex(idx)
        level = self.px4_tune_level_combo.itemData(idx)
        if level:
            self._px4_tune_config['level'] = level

    def _collect_px4_tune_config(self):
        if not hasattr(self, 'px4_tune_level_combo'):
            return dict(self._px4_tune_config)
        level = self.px4_tune_level_combo.currentData()
        setpoints = {}
        for lvl in TUNE_LEVELS:
            sp = {}
            for key, widget in self._px4_tune_sp_widgets.get(lvl, {}).items():
                sp[key] = float(widget.value())
            setpoints[lvl] = sp
        cfg = {
            'level': level or TUNE_LEVELS[0],
            'duration_s': float(self.px4_tune_duration.value()),
            'setpoints': setpoints,
        }
        self._px4_tune_config = cfg
        return cfg

    def _apply_px4_tune_config(self, cfg):
        if not cfg or not hasattr(self, 'px4_tune_level_combo'):
            return
        self._px4_tune_config = dict(cfg)
        level = cfg.get('level', TUNE_LEVELS[0])
        for i in range(self.px4_tune_level_combo.count()):
            if self.px4_tune_level_combo.itemData(i) == level:
                self.px4_tune_level_combo.setCurrentIndex(i)
                break
        self.px4_tune_duration.setValue(float(cfg.get('duration_s', 5.0)))
        sp_map = cfg.get('setpoints') or {}
        for lvl in TUNE_LEVELS:
            sp = sp_map.get(lvl, {})
            for key, widget in self._px4_tune_sp_widgets.get(lvl, {}).items():
                if key in sp:
                    widget.setValue(float(sp[key]))
        self._on_px4_tune_level_changed()

    def run_px4_cascade_tune_sim(self):
        if self._current_tracking_controller_id() != CONTROLLER_PX4:
            return
        if self._px4_tune_sim_thread is not None and self._px4_tune_sim_thread.isRunning():
            return
        params = self._collect_tracking_params()
        tune_config = self._collect_px4_tune_config()
        self.btn_run_px4_tune.setEnabled(False)
        self.lbl_px4_tune_result.setText('Running cascade tune simulation…')
        self._px4_tune_sim_thread = Px4TuneSimulationThread(
            params, self._physics_dict_for_tracking(), tune_config,
        )
        self._px4_tune_sim_thread.finished.connect(self._on_px4_tune_sim_finished)
        self._px4_tune_sim_thread.error.connect(self._on_px4_tune_sim_error)
        self._px4_tune_sim_thread.start()

    def _on_px4_tune_sim_finished(self, result):
        self._update_px4_tune_visibility()
        self._last_tracking_result = result
        summary = tracking_summary_text(result)
        self.lbl_px4_tune_result.setText(summary)
        self.lbl_tracking_result.setText(summary)
        self.status_text.append(summary)
        if self._display_panels_ready():
            self._draw_tracking_plot_tabs(result)
            if hasattr(self, 'plot_tabs'):
                self.plot_tabs.setCurrentWidget(self._plot_tab_states)

    def _on_px4_tune_sim_error(self, msg):
        self._update_px4_tune_visibility()
        self.lbl_px4_tune_result.setText('Cascade tune simulation failed.')
        QMessageBox.critical(self, 'PX4 cascade tune error', msg)
        self.status_text.append(f'PX4 tune error: {msg}')

    def _create_tracking_panel(self):
        """Build the Tracking sidebar tab (controllers, params, simulation)."""
        tracking_tab = QWidget()
        layout = QVBoxLayout(tracking_tab)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        src_group = QGroupBox('Trajectory source')
        src_layout = QGridLayout()
        self.tracking_source_combo = QComboBox()
        self.tracking_source_combo.addItem('Current trajectory')
        self.tracking_source_combo.addItem('CSV file')
        self.tracking_source_combo.addItem('GUI waypoints')
        self.tracking_source_combo.setToolTip(
            'Reference for tracking:\n'
            'Current trajectory — saved CSV for the active slot;\n'
            'CSV file — pick an existing trajectory CSV;\n'
            'GUI waypoints — GotoSetpoint through waypoint list (SITL only).'
        )
        self.tracking_source_combo.currentIndexChanged.connect(self._on_tracking_source_changed)
        src_layout.addWidget(QLabel('Source:'), 0, 0)
        src_layout.addWidget(self.tracking_source_combo, 0, 1, 1, 2)

        self.tracking_csv_widget = QWidget()
        tracking_csv_row = QHBoxLayout(self.tracking_csv_widget)
        tracking_csv_row.setContentsMargins(0, 0, 0, 0)
        self.tracking_csv_edit = QLineEdit()
        self.tracking_csv_edit.setPlaceholderText('Path to trajectory CSV')
        self.tracking_csv_edit.setText(DEFAULT_TRAJ_CSV_PATH)
        self.tracking_csv_edit.textChanged.connect(self._on_tracking_csv_edited)
        tracking_csv_row.addWidget(self.tracking_csv_edit, 1)
        self.tracking_csv_browse = QPushButton('Browse…')
        self.tracking_csv_browse.clicked.connect(self.browse_tracking_csv)
        tracking_csv_row.addWidget(self.tracking_csv_browse)
        src_layout.addWidget(self.tracking_csv_widget, 1, 0, 1, 3)
        self.tracking_csv_widget.setVisible(False)

        self.lbl_tracking_source_info = QLabel()
        self.lbl_tracking_source_info.setWordWrap(True)
        self.lbl_tracking_source_info.setStyleSheet('color: #555;')
        src_layout.addWidget(self.lbl_tracking_source_info, 2, 0, 1, 3)
        src_group.setLayout(src_layout)
        layout.addWidget(src_group)

        ctrl_group = QGroupBox('Controller')
        ctrl_layout = QVBoxLayout()
        self.tracking_controller_combo = QComboBox()
        for cid in CONTROLLER_IDS:
            label = CONTROLLER_LABELS[cid]
            if cid == CONTROLLER_ACADOS_NMPC and not acados_nmpc_available():
                label = f'{label} (requires acados)'
            self.tracking_controller_combo.addItem(label, cid)
        self.tracking_controller_combo.currentIndexChanged.connect(self._on_tracking_controller_changed)
        ctrl_layout.addWidget(self.tracking_controller_combo)

        sim_mode_row = QHBoxLayout()
        self.tracking_sim_numerical_radio = QRadioButton('Numerical simulation')
        self.tracking_sim_sitl_radio = QRadioButton('PX4 SITL')
        self.tracking_sim_numerical_radio.setChecked(True)
        self._tracking_sim_mode_group = QButtonGroup(self)
        self._tracking_sim_mode_group.addButton(self.tracking_sim_numerical_radio, 0)
        self._tracking_sim_mode_group.addButton(self.tracking_sim_sitl_radio, 1)
        self._tracking_sim_mode_group.buttonClicked.connect(self._on_tracking_sim_mode_changed)
        sim_mode_row.addWidget(self.tracking_sim_numerical_radio)
        sim_mode_row.addWidget(self.tracking_sim_sitl_radio)
        ctrl_layout.addLayout(sim_mode_row)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        self._create_numerical_sim_timing_panel(layout)

        self._create_actuator_dynamics_panel(layout)

        self.tracking_params_group, param_outer = self._make_collapsible_group(
            'Controller parameters',
            layout,
            expanded=False,
        )
        self.tracking_params_group.setToolTip(
            'Click the section title (▸/▾) to expand or collapse controller gains.\n'
            'Gains are stored separately per rocket platform (Proxy vs Real).'
        )
        self.lbl_tracking_params_platform = QLabel()
        self.lbl_tracking_params_platform.setWordWrap(True)
        self.lbl_tracking_params_platform.setStyleSheet(
            'color: #1565c0; font-weight: bold;'
        )
        param_outer.addWidget(self.lbl_tracking_params_platform)
        self.tracking_params_scroll = QScrollArea()
        self.tracking_params_scroll.setWidgetResizable(True)
        self.tracking_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tracking_params_scroll.setMinimumHeight(180)
        self.tracking_params_scroll.setMaximumHeight(420)
        self.tracking_params_grid_host = QWidget()
        self.tracking_params_groups_layout = QVBoxLayout(self.tracking_params_grid_host)
        self.tracking_params_groups_layout.setSpacing(6)
        self.tracking_params_groups_layout.setContentsMargins(2, 2, 2, 2)
        self.tracking_params_scroll.setWidget(self.tracking_params_grid_host)
        param_outer.addWidget(self.tracking_params_scroll)

        param_io = QHBoxLayout()
        self.btn_save_tracking_params = QPushButton('Save params')
        self.btn_save_tracking_params.setToolTip(
            f'Save controller parameters to {DEFAULT_TRACKING_PARAMS_PATH}'
        )
        self.btn_save_tracking_params.clicked.connect(self.save_tracking_params)
        self.btn_load_tracking_params = QPushButton('Load params…')
        self.btn_load_tracking_params.clicked.connect(self.load_tracking_params)
        param_io.addWidget(self.btn_save_tracking_params)
        param_io.addWidget(self.btn_load_tracking_params)
        param_outer.addLayout(param_io)

        self.tracking_params_file_label = QLabel()
        self.tracking_params_file_label.setWordWrap(True)
        self.tracking_params_file_label.setStyleSheet('color: #555;')
        param_outer.addWidget(self.tracking_params_file_label)

        self._create_px4_cascade_tune_panel(layout)

        run_group = QGroupBox('Run')
        run_layout = QGridLayout()
        self.btn_run_numerical_tracking = QPushButton('Run numerical sim')
        self.btn_run_numerical_tracking.setToolTip(
            'Closed-loop simulation on the nonlinear TVC dynamics model (same as '
            'trajectory optimization) with the selected controller.'
        )
        self.btn_run_numerical_tracking.clicked.connect(self.run_numerical_tracking)
        run_layout.addWidget(self.btn_run_numerical_tracking, 0, 0, 1, 2)

        self.lbl_tracking_result = QLabel('No simulation yet.')
        self.lbl_tracking_result.setWordWrap(True)
        self.lbl_tracking_result.setStyleSheet('color: #555;')
        run_layout.addWidget(self.lbl_tracking_result, 1, 0, 1, 2)

        self.sitl_run_widget = QWidget()
        sitl_run_layout = QGridLayout(self.sitl_run_widget)
        sitl_run_layout.setContentsMargins(0, 0, 0, 0)
        sitl_run_layout.setHorizontalSpacing(6)
        sitl_run_layout.setVerticalSpacing(4)
        self.btn_start_px4_sitl = QPushButton('Start SITL')
        self.btn_start_px4_sitl.setToolTip(
            'Launch PX4 SITL + Gazebo (+ RViz). Uncheck "Show Gazebo GUI" to run '
            'Gazebo headless and visualize only in RViz.'
        )
        self.btn_start_px4_sitl.clicked.connect(self.start_px4_sitl_for_tracking)
        self.btn_stop_px4_sitl = QPushButton('Stop SITL')
        self.btn_stop_px4_sitl.clicked.connect(self.stop_px4_sitl)
        self.btn_stop_px4_sitl.setEnabled(False)
        self.lbl_px4_sitl_status = QLabel('Stopped')
        self.lbl_px4_sitl_status.setStyleSheet('color: #888;')
        sitl_run_layout.addWidget(QLabel('PX4 SITL:'), 0, 0)
        sitl_run_layout.addWidget(self.btn_start_px4_sitl, 0, 1)
        sitl_run_layout.addWidget(self.btn_stop_px4_sitl, 1, 0)
        sitl_run_layout.addWidget(self.lbl_px4_sitl_status, 1, 1)

        self.lbl_sitl_nodes_status = QLabel('Nodes: —')
        self.lbl_sitl_nodes_status.setWordWrap(True)
        self.lbl_sitl_nodes_status.setTextFormat(Qt.RichText)
        self.lbl_sitl_nodes_status.setStyleSheet(
            'color: #888; font-family: monospace; font-size: 11px;'
        )
        self.lbl_sitl_nodes_status.setToolTip(
            'Live ROS2 / process status (● up, ○ down). Refreshes every 2 s while SITL runs.'
        )
        sitl_run_layout.addWidget(QLabel('Stack:'), 2, 0, Qt.AlignTop)
        sitl_run_layout.addWidget(self.lbl_sitl_nodes_status, 2, 1)

        self.chk_show_gazebo_gui = QCheckBox('Show Gazebo GUI')
        self.chk_show_gazebo_gui.setToolTip(
            'When unchecked, Gazebo runs headless (no 3D window). Physics and '
            'ros_gz_bridge still run — RViz is enough for most SITL tracking.'
        )
        self.chk_show_gazebo_gui.setChecked(False)
        self.chk_show_gazebo_gui.stateChanged.connect(
            self._on_show_gazebo_gui_changed
        )
        sitl_run_layout.addWidget(self.chk_show_gazebo_gui, 3, 0, 1, 2)

        self.chk_enable_online_planner = QCheckBox('Online safety planner (RViz)')
        self.chk_enable_online_planner.setToolTip(
            'When checked, Start SITL also launches the online_planner node '
            '(Acados replanning from current state to the next waypoint(s), '
            'visualized in RViz as /online_planner/planned_path). '
            'During tracking, targets follow Trajectory-tab arrival times.'
        )
        self.chk_enable_online_planner.setChecked(True)
        self.chk_enable_online_planner.stateChanged.connect(
            self._on_online_planner_checkbox_changed
        )
        sitl_run_layout.addWidget(self.chk_enable_online_planner, 4, 0)

        self.spin_online_planner_rate = QDoubleSpinBox()
        self.spin_online_planner_rate.setRange(0.2, 50.0)
        self.spin_online_planner_rate.setSingleStep(0.5)
        self.spin_online_planner_rate.setDecimals(1)
        self.spin_online_planner_rate.setSuffix(' Hz')
        self.spin_online_planner_rate.setToolTip(
            'Online planner replanning rate (0.2–50 Hz). Applied at launch; '
            'if the planner is already running, the rate is updated live via ros2 param set.'
        )
        self.spin_online_planner_rate.setValue(
            float(self._tracking_config.get('online_planner_rate_hz', 10.0))
        )
        self.spin_online_planner_rate.valueChanged.connect(
            self._on_online_planner_rate_changed
        )
        self.lbl_online_planner_actual_hz = QLabel('Actual: —')
        self.lbl_online_planner_actual_hz.setStyleSheet(
            'color: #888; font-family: monospace; font-size: 11px;'
        )
        self.lbl_online_planner_actual_hz.setToolTip(
            'Measured replan completion rate over the last 5 s '
            '(actual/target). Drops below target when Acados solve is slower '
            'than the timer period (shown as busy %).'
        )
        rate_col = QWidget()
        rate_col_layout = QVBoxLayout(rate_col)
        rate_col_layout.setContentsMargins(0, 0, 0, 0)
        rate_col_layout.setSpacing(2)
        rate_col_layout.addWidget(self.spin_online_planner_rate)
        rate_col_layout.addWidget(self.lbl_online_planner_actual_hz)
        sitl_run_layout.addWidget(rate_col, 4, 1)

        self.lbl_online_planner_diag = QLabel('Planner: —')
        self.lbl_online_planner_diag.setStyleSheet('color: #888;')
        self.lbl_online_planner_diag.setWordWrap(True)
        self.lbl_online_planner_diag.setToolTip(
            'Live metrics from /online_planner/diagnostics: replan rate, '
            'solve time, start/end ENU, active waypoint.'
        )
        sitl_run_layout.addWidget(self.lbl_online_planner_diag, 5, 0, 1, 2)

        self.btn_online_planner_next_wp = QPushButton('Planner: next WP')
        self.btn_online_planner_next_wp.setToolTip(
            'Advance the online planner to the next mission waypoint '
            '(cycles back to WP0 after the last). '
            'Calls service /online_planner/advance_waypoint.'
        )
        self.btn_online_planner_next_wp.clicked.connect(
            self._online_planner_advance_waypoint
        )
        self.lbl_online_planner_wp_info = QLabel('')
        self.lbl_online_planner_wp_info.setStyleSheet('color: #888;')
        self.lbl_online_planner_wp_info.setWordWrap(True)
        sitl_run_layout.addWidget(self.btn_online_planner_next_wp, 6, 0)
        sitl_run_layout.addWidget(self.lbl_online_planner_wp_info, 6, 1)

        self.btn_start_tracking = QPushButton('Start tracking')
        self.btn_start_tracking.setToolTip('Launch tvc_traj_player with the selected trajectory source.')
        self.btn_start_tracking.clicked.connect(self.start_tracking_node)
        self.btn_stop_tracking = QPushButton('Stop tracking')
        self.btn_stop_tracking.clicked.connect(self.stop_tracking_node)
        self.btn_stop_tracking.setEnabled(False)
        self.lbl_tracking_status = QLabel('Stopped')
        self.lbl_tracking_status.setStyleSheet('color: #888;')
        sitl_run_layout.addWidget(self.btn_start_tracking, 7, 0)
        sitl_run_layout.addWidget(self.btn_stop_tracking, 7, 1)
        sitl_run_layout.addWidget(self.lbl_tracking_status, 8, 0, 1, 2)
        run_layout.addWidget(self.sitl_run_widget, 2, 0, 1, 2)

        self.btn_clear_rviz_traj = QPushButton('Clear RViz executed path')
        self.btn_clear_rviz_traj.clicked.connect(
            lambda: self.clear_rviz_trajectory_display(quiet=False)
        )
        run_layout.addWidget(self.btn_clear_rviz_traj, 3, 0, 1, 2)
        run_group.setLayout(run_layout)
        layout.addWidget(run_group)

        self._update_tracking_params_file_label()
        self._rebuild_tracking_param_widgets()
        self._apply_px4_tune_config(self._px4_tune_config)
        self._update_tracking_panel_visibility()
        self._on_tracking_sim_mode_changed()
        self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())
        QTimer.singleShot(0, self._refresh_tab_scroll_areas)
        return tracking_tab

    def _create_nmp_panel(self):
        """Non-minimum-phase workflow: platform → waypoints → flatness planning."""
        nmp_tab = QWidget()
        layout = QVBoxLayout(nmp_tab)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        intro = QLabel(
            'Plan on the center-of-oscillation flat output ξ (min-snap), then compare '
            'ξ with the reconstructed COM state. Waypoint X/Y are ξ_x/ξ_y, not COM.'
        )
        intro.setWordWrap(True)
        intro.setStyleSheet('color: #444;')
        layout.addWidget(intro)

        model_group = QGroupBox('1. Model')
        model_layout = QVBoxLayout(model_group)
        plat_row = QHBoxLayout()
        self.nmp_proxy_radio = QRadioButton(platform_label(PLATFORM_PROXY))
        self.nmp_real_radio = QRadioButton(platform_label(PLATFORM_REAL))
        self.nmp_proxy_radio.setChecked(True)
        self.nmp_proxy_radio.setToolTip(platform_description(PLATFORM_PROXY))
        self.nmp_real_radio.setToolTip(platform_description(PLATFORM_REAL))
        self._nmp_platform_button_group = QButtonGroup(self)
        self._nmp_platform_button_group.addButton(self.nmp_proxy_radio, 0)
        self._nmp_platform_button_group.addButton(self.nmp_real_radio, 1)
        self._nmp_platform_button_group.buttonClicked.connect(self._on_nmp_platform_changed)
        plat_row.addWidget(self.nmp_proxy_radio)
        plat_row.addWidget(self.nmp_real_radio)
        plat_row.addStretch(1)
        model_layout.addLayout(plat_row)
        model_param_grid = QGridLayout()
        model_param_grid.setHorizontalSpacing(8)
        model_param_grid.setVerticalSpacing(4)

        def _add_model_spin(row, col, label, tooltip):
            model_param_grid.addWidget(QLabel(label), row, col * 2)
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setSingleStep(0.001)
            spin.setToolTip(tooltip)
            spin.valueChanged.connect(self._on_nmp_model_param_changed)
            model_param_grid.addWidget(spin, row, col * 2 + 1)
            return spin

        self.nmp_mass_spin = _add_model_spin(
            0, 0, 'Mass [kg]:', 'Vehicle mass used for planning/tracking/GIF.',
        )
        self.nmp_ixx_spin = _add_model_spin(
            0, 1, 'Ixx [kg·m²]:', 'Roll-axis moment of inertia.',
        )
        self.nmp_iyy_spin = _add_model_spin(
            1, 0, 'Iyy [kg·m²]:', 'Pitch-axis moment of inertia.',
        )
        self.nmp_izz_spin = _add_model_spin(
            1, 1, 'Izz [kg·m²]:', 'Yaw-axis moment of inertia.',
        )
        self.nmp_body_len_spin = _add_model_spin(
            2, 0, 'Body length [m]:',
            'Total rocket length (bottom to nose tip), for the GIF/3D rocket model.',
        )
        self.nmp_com_from_bottom_spin = _add_model_spin(
            2, 1, 'COM from bottom [m]:',
            'Distance from the rocket\'s bottom to the center of mass, for the GIF/3D rocket model.',
        )
        self.nmp_r_thrust_dist_spin = _add_model_spin(
            3, 0, 'Thrust point below COM [m]:',
            'Distance from COM to the TVC gimbal/thrust point.',
        )
        model_layout.addLayout(model_param_grid)

        model_btn_row = QHBoxLayout()
        self.btn_nmp_save_model_params = QPushButton('Save model')
        self.btn_nmp_save_model_params.setToolTip(
            'Save mass/inertia/geometry above as the default for the selected '
            f'platform.\n{NMP_MODEL_PARAMS_PATH}'
        )
        self.btn_nmp_save_model_params.clicked.connect(self.save_nmp_model_params)
        self.btn_nmp_reset_model_params = QPushButton('Reset to platform default')
        self.btn_nmp_reset_model_params.setToolTip(
            'Discard any saved override and reload the built-in default for this platform.'
        )
        self.btn_nmp_reset_model_params.clicked.connect(self.reset_nmp_model_params)
        model_btn_row.addWidget(self.btn_nmp_save_model_params)
        model_btn_row.addWidget(self.btn_nmp_reset_model_params)
        model_btn_row.addStretch(1)
        model_layout.addLayout(model_btn_row)
        self.lbl_nmp_model_desc = QLabel(platform_description(PLATFORM_PROXY))
        self.lbl_nmp_model_desc.setWordWrap(True)
        self.lbl_nmp_model_desc.setStyleSheet('color: #555;')
        model_layout.addWidget(self.lbl_nmp_model_desc)
        self.lbl_nmp_nmp_params = QLabel()
        self.lbl_nmp_nmp_params.setWordWrap(True)
        self.lbl_nmp_nmp_params.setStyleSheet('color: #333; font-family: monospace;')
        model_layout.addWidget(self.lbl_nmp_nmp_params)
        layout.addWidget(model_group)

        wp_group = QGroupBox('2. Waypoints (ξ_x, ξ_y, z, yaw)')
        wp_layout = QVBoxLayout(wp_group)
        self.nmp_waypoint_table = QTableWidget()
        self.nmp_waypoint_table.setColumnCount(len(WP_TABLE_HEADERS))
        self.nmp_waypoint_table.setHorizontalHeaderLabels(list(WP_TABLE_HEADERS))
        self.nmp_waypoint_table.verticalHeader().setVisible(False)
        self.nmp_waypoint_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.nmp_waypoint_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.nmp_waypoint_table.setAlternatingRowColors(True)
        self.nmp_waypoint_table.setMinimumHeight(88)
        self.nmp_waypoint_table.setMaximumHeight(130)
        nmp_wp_hdr = self.nmp_waypoint_table.horizontalHeader()
        nmp_wp_hdr.setSectionResizeMode(QHeaderView.Interactive)
        nmp_wp_widths = {
            WP_COL_IDX: 28,
            WP_COL_X: 54,
            WP_COL_Y: 54,
            WP_COL_Z: 54,
            WP_COL_YAW: 58,
            WP_COL_LEG_DT: 54,
            WP_COL_T_ARR: 54,
        }
        for col, width in nmp_wp_widths.items():
            self.nmp_waypoint_table.setColumnWidth(col, width)
        self.nmp_waypoint_table.itemChanged.connect(self._on_nmp_waypoint_table_item_changed)
        wp_layout.addWidget(self.nmp_waypoint_table)
        wp_btn_row = QHBoxLayout()
        self.btn_nmp_add_wp = QPushButton('Add')
        self.btn_nmp_add_wp.clicked.connect(self.add_nmp_waypoint)
        self.btn_nmp_remove_wp = QPushButton('Remove')
        self.btn_nmp_remove_wp.clicked.connect(self.remove_nmp_waypoint)
        self.btn_nmp_save_wp = QPushButton('Save')
        self.btn_nmp_save_wp.setToolTip(
            'Save these waypoints so the NMP tab opens with them next time.\n'
            f'{NMP_SAVED_WAYPOINTS_PATH}'
        )
        self.btn_nmp_save_wp.clicked.connect(self.save_nmp_waypoints)
        wp_btn_row.addWidget(self.btn_nmp_add_wp)
        wp_btn_row.addWidget(self.btn_nmp_remove_wp)
        wp_btn_row.addWidget(self.btn_nmp_save_wp)
        wp_btn_row.addStretch(1)
        wp_layout.addLayout(wp_btn_row)
        plan_row = QGridLayout()
        self.nmp_dt_spin = QDoubleSpinBox()
        self.nmp_dt_spin.setRange(0.01, 0.2)
        self.nmp_dt_spin.setValue(0.05)
        self.nmp_dt_spin.setDecimals(3)
        self.nmp_dt_spin.setSingleStep(0.01)
        self.nmp_dt_spin.setToolTip('Sample period for flat-output reconstruction [s].')
        plan_row.addWidget(QLabel('Sample dt [s]:'), 0, 0)
        plan_row.addWidget(self.nmp_dt_spin, 0, 1)
        self.btn_nmp_plan = QPushButton('Plan flat trajectory')
        self.btn_nmp_plan.setToolTip(
            'Min-snap on ξ_x, ξ_y, z, ψ between waypoints; reconstruct COM state and controls.'
        )
        self.btn_nmp_plan.clicked.connect(self.start_nmp_planning)
        plan_row.addWidget(self.btn_nmp_plan, 1, 0, 1, 2)
        self.lbl_nmp_plan_status = QLabel('No NMP plan yet.')
        self.lbl_nmp_plan_status.setWordWrap(True)
        self.lbl_nmp_plan_status.setStyleSheet('color: #555;')
        plan_row.addWidget(self.lbl_nmp_plan_status, 2, 0, 1, 2)
        wp_layout.addLayout(plan_row)
        layout.addWidget(wp_group)

        track_group = QGroupBox('3. Tracking (PX4 cascade)')
        track_layout = QVBoxLayout(track_group)
        track_intro = QLabel(
            'Closed-loop waypoint tracking with PX4 cascade. Choose whether the '
            'position loop tracks COM (center of mass) or ξ (center of oscillation).'
        )
        track_intro.setWordWrap(True)
        track_intro.setStyleSheet('color: #444;')
        track_layout.addWidget(track_intro)
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel('Reference:'))
        self.nmp_ref_planned_radio = QRadioButton('Flat planned trajectory')
        self.nmp_ref_waypoint_radio = QRadioButton('Direct waypoint flight')
        self.nmp_ref_planned_radio.setChecked(True)
        self.nmp_ref_planned_radio.setToolTip(
            'Track the min-snap flat trajectory from “Plan flat trajectory”.'
        )
        self.nmp_ref_waypoint_radio.setToolTip(
            'No planning: GotoSetpoint-style holds — switch target at each arrival time.'
        )
        self._nmp_ref_mode_group = QButtonGroup(self)
        self._nmp_ref_mode_group.addButton(self.nmp_ref_planned_radio, 0)
        self._nmp_ref_mode_group.addButton(self.nmp_ref_waypoint_radio, 1)
        self.nmp_ref_planned_radio.toggled.connect(self._on_nmp_tracking_timing_changed)
        self.nmp_ref_waypoint_radio.toggled.connect(self._on_nmp_tracking_timing_changed)
        ref_row.addWidget(self.nmp_ref_planned_radio)
        ref_row.addWidget(self.nmp_ref_waypoint_radio)
        ref_row.addStretch(1)
        track_layout.addLayout(ref_row)
        plot_mode_row = QHBoxLayout()
        plot_mode_row.addWidget(QLabel('NMP plot:'))
        self.nmp_plot_all_radio = QRadioButton('All state')
        self.nmp_plot_2d_radio = QRadioButton('2D (x-only)')
        self.nmp_plot_all_radio.setChecked(True)
        self.nmp_plot_all_radio.setToolTip('Show x/y/z states and all attitude/gimbal channels.')
        self.nmp_plot_2d_radio.setToolTip(
            'Hide y-axis channels (y, vy, roll, p, roll gimbal) for x-direction motion.'
        )
        self._nmp_plot_mode_group = QButtonGroup(self)
        self._nmp_plot_mode_group.addButton(self.nmp_plot_all_radio, 0)
        self._nmp_plot_mode_group.addButton(self.nmp_plot_2d_radio, 1)
        self.nmp_plot_all_radio.toggled.connect(self._on_nmp_plot_mode_changed)
        plot_mode_row.addWidget(self.nmp_plot_all_radio)
        plot_mode_row.addWidget(self.nmp_plot_2d_radio)
        plot_mode_row.addStretch(1)
        track_layout.addLayout(plot_mode_row)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel('Control point:'))
        self.nmp_ctrl_com_radio = QRadioButton('COM (center of mass)')
        self.nmp_ctrl_xi_radio = QRadioButton('ξ (oscillation point)')
        self.nmp_ctrl_com_radio.setChecked(True)
        self.nmp_ctrl_com_radio.setToolTip('PX4 cascade on COM position (default).')
        self.nmp_ctrl_xi_radio.setToolTip(
            'Flatness cascade: outer loop on ξ_x/ξ_y, inner loop still PX4 cascade.'
        )
        self._nmp_ctrl_point_group = QButtonGroup(self)
        self._nmp_ctrl_point_group.addButton(self.nmp_ctrl_com_radio, 0)
        self._nmp_ctrl_point_group.addButton(self.nmp_ctrl_xi_radio, 1)
        self.nmp_ctrl_com_radio.toggled.connect(self._on_nmp_control_point_changed)
        self.nmp_ctrl_xi_radio.toggled.connect(self._on_nmp_control_point_changed)
        ctrl_row.addWidget(self.nmp_ctrl_com_radio)
        ctrl_row.addWidget(self.nmp_ctrl_xi_radio)
        ctrl_row.addStretch(1)
        track_layout.addLayout(ctrl_row)
        self.nmp_params_group, nmp_param_outer = self._make_collapsible_group(
            'Controller parameters',
            track_layout,
            expanded=False,
        )
        self.nmp_params_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.lbl_nmp_params_platform = QLabel()
        self.lbl_nmp_params_platform.setWordWrap(True)
        self.lbl_nmp_params_platform.setStyleSheet(
            'color: #1565c0; font-weight: bold;'
        )
        nmp_param_outer.addWidget(self.lbl_nmp_params_platform)
        self._create_nmp_tracking_timing_panel(nmp_param_outer)
        nmp_param_io = QHBoxLayout()
        self.btn_save_nmp_tracking_params = QPushButton('Save params')
        self.btn_save_nmp_tracking_params.setToolTip(
            f'Save NMP controller/timing parameters to {DEFAULT_TRACKING_PARAMS_PATH}'
        )
        self.btn_save_nmp_tracking_params.clicked.connect(self.save_tracking_params)
        nmp_param_io.addWidget(self.btn_save_nmp_tracking_params)
        nmp_param_io.addStretch(1)
        nmp_param_outer.addLayout(nmp_param_io)
        self.nmp_params_scroll = QScrollArea()
        self.nmp_params_scroll.setWidgetResizable(True)
        self.nmp_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nmp_params_scroll.setMinimumHeight(240)
        self.nmp_params_scroll.setMaximumHeight(420)
        self.nmp_params_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.nmp_params_grid_host = QWidget()
        self.nmp_params_groups_layout = QVBoxLayout(self.nmp_params_grid_host)
        self.nmp_params_groups_layout.setSpacing(5)
        self.nmp_params_groups_layout.setContentsMargins(2, 2, 2, 2)
        self.nmp_params_scroll.setWidget(self.nmp_params_grid_host)
        nmp_param_outer.addWidget(self.nmp_params_scroll)
        self._rebuild_nmp_tracking_param_widgets()
        self.btn_nmp_run_tracking = QPushButton('Run tracking sim')
        self.btn_nmp_run_tracking.setToolTip(
            'Numerical closed-loop tracking with PX4 cascade '
            '(planned trajectory or direct waypoint flight).'
        )
        self.btn_nmp_run_tracking.clicked.connect(self.run_nmp_tracking_sim)
        track_layout.addWidget(self.btn_nmp_run_tracking)
        self.lbl_nmp_tracking_status = QLabel(
            'Choose reference mode, then run tracking sim '
            '(planned mode needs “Plan flat trajectory” first).'
        )
        self.lbl_nmp_tracking_status.setWordWrap(True)
        self.lbl_nmp_tracking_status.setStyleSheet('color: #555;')
        track_layout.addWidget(self.lbl_nmp_tracking_status)
        layout.addWidget(track_group)

        layout.addStretch(1)
        self._apply_nmp_model_params_for_platform(PLATFORM_PROXY)
        self._populate_nmp_waypoint_table()
        self._refresh_nmp_model_info()
        QTimer.singleShot(0, self._refresh_tab_scroll_areas)
        return nmp_tab

    def _current_nmp_platform_id(self):
        if hasattr(self, 'nmp_real_radio') and self.nmp_real_radio.isChecked():
            return PLATFORM_REAL
        return PLATFORM_PROXY

    def _nmp_model_params_builtin_default(self, platform_id):
        """Built-in (non-override) mass/inertia/geometry defaults for a platform."""
        phy = default_physics(platform_id)
        geo = rocket_visual_geometry(platform_id)
        return {
            'mass': float(phy['mass']),
            'Ixx': float(phy['Ixx']),
            'Iyy': float(phy['Iyy']),
            'Izz': float(phy['Izz']),
            'r_thrust_z': abs(float(phy['r_thrust_z'])),
            'body_length': float(geo['nose_tip_z'] - geo['body_bottom_z']),
            'com_from_bottom': float(-geo['body_bottom_z']),
        }

    def _nmp_model_params_default(self, platform_id):
        """Saved override merged over the platform's built-in default."""
        row = self._nmp_model_params_builtin_default(platform_id)
        row.update(self._nmp_model_overrides.get(platform_id, {}))
        return row

    def _apply_nmp_model_params_for_platform(self, platform_id):
        if not hasattr(self, 'nmp_mass_spin'):
            return
        values = self._nmp_model_params_default(platform_id)
        mass_lo, mass_hi, mass_dec = physics_spin_ranges(platform_id)['mass']
        inertia_lo, inertia_hi, inertia_dec = physics_spin_ranges(platform_id)['inertia']
        _, thrust_hi, thrust_dec = physics_spin_ranges(platform_id)['r_thrust']
        specs = (
            (self.nmp_mass_spin, 'mass', mass_lo, mass_hi, mass_dec),
            (self.nmp_ixx_spin, 'Ixx', inertia_lo, inertia_hi, inertia_dec),
            (self.nmp_iyy_spin, 'Iyy', inertia_lo, inertia_hi, inertia_dec),
            (self.nmp_izz_spin, 'Izz', inertia_lo, inertia_hi, inertia_dec),
            (self.nmp_body_len_spin, 'body_length', 0.05, 10.0, 3),
            (self.nmp_com_from_bottom_spin, 'com_from_bottom', 0.0, 10.0, 3),
            (self.nmp_r_thrust_dist_spin, 'r_thrust_z', 0.0, abs(float(thrust_hi)), thrust_dec),
        )
        for spin, key, lo, hi, dec in specs:
            spin.blockSignals(True)
            spin.setDecimals(dec)
            spin.setSingleStep(10 ** (-dec))
            spin.setRange(float(lo), float(hi))
            spin.setValue(float(values[key]))
            spin.blockSignals(False)

    def _on_nmp_model_param_changed(self, _value=None):
        self._refresh_nmp_model_info()

    def _nmp_model_params_from_spins(self):
        return {
            'mass': float(self.nmp_mass_spin.value()),
            'Ixx': float(self.nmp_ixx_spin.value()),
            'Iyy': float(self.nmp_iyy_spin.value()),
            'Izz': float(self.nmp_izz_spin.value()),
            'r_thrust_z': abs(float(self.nmp_r_thrust_dist_spin.value())),
            'body_length': float(self.nmp_body_len_spin.value()),
            'com_from_bottom': float(self.nmp_com_from_bottom_spin.value()),
        }

    def save_nmp_model_params(self):
        pid = self._current_nmp_platform_id()
        self._nmp_model_overrides[pid] = self._nmp_model_params_from_spins()
        os.makedirs(os.path.dirname(NMP_MODEL_PARAMS_PATH), exist_ok=True)
        try:
            with open(NMP_MODEL_PARAMS_PATH, 'w', encoding='utf-8') as f:
                json.dump(self._nmp_model_overrides, f, indent=2)
        except OSError as e:
            QMessageBox.critical(self, 'NMP', f'Could not save model parameters:\n{e}')
            return
        if hasattr(self, 'status_text'):
            self.status_text.append(
                f'NMP: saved model parameters for {platform_label(pid)} to:\n  '
                f'{NMP_MODEL_PARAMS_PATH}'
            )

    def reset_nmp_model_params(self):
        pid = self._current_nmp_platform_id()
        self._nmp_model_overrides.pop(pid, None)
        self._apply_nmp_model_params_for_platform(pid)
        self._refresh_nmp_model_info()
        if hasattr(self, 'status_text'):
            self.status_text.append(f'NMP: reset {platform_label(pid)} model to built-in default.')

    def _nmp_rocket_geometry_dict(self):
        if not hasattr(self, 'nmp_body_len_spin'):
            return None
        body_length = float(self.nmp_body_len_spin.value())
        com_from_bottom = float(self.nmp_com_from_bottom_spin.value())
        body_bottom_z = -com_from_bottom
        base_geo = rocket_visual_geometry(self._current_nmp_platform_id())
        return {
            'body_bottom_z': body_bottom_z,
            'nose_tip_z': body_bottom_z + body_length,
            'fin_z': body_bottom_z + 0.33 * body_length,
            'shaft_lw': base_geo.get('shaft_lw', 4.5),
            'fin_lw': base_geo.get('fin_lw', 2.6),
        }

    def _nmp_physics_dict(self):
        phy = dict(default_physics(self._current_nmp_platform_id()))
        phy['g'] = 9.81
        if hasattr(self, 'nmp_mass_spin'):
            phy.update(self._nmp_model_params_from_spins())
            phy['r_thrust_z'] = -abs(phy['r_thrust_z'])
            phy.pop('body_length', None)
            phy.pop('com_from_bottom', None)
        return phy

    def _nmp_bounds_dict(self):
        pid = self._current_nmp_platform_id()
        cons = dict(default_constraints(pid))
        cons['g'] = 9.81
        cons['th_p_max_deg'] = 15.0
        cons['th_r_max_deg'] = 15.0
        cons['th_p_max'] = cons['th_p_max_deg']
        cons['th_r_max'] = cons['th_r_max_deg']
        return cons

    def _on_nmp_platform_changed(self, _button=None):
        if self._nmp_platform_guard:
            return
        previous_id = getattr(self, '_cached_nmp_platform_id', PLATFORM_PROXY)
        pid = self._current_nmp_platform_id()
        if previous_id != pid:
            self._store_nmp_tracking_params_for_controller()
            self._cached_nmp_platform_id = pid
            self._rebuild_nmp_tracking_param_widgets()
        self.lbl_nmp_model_desc.setText(platform_description(pid))
        self._apply_nmp_model_params_for_platform(pid)
        self._refresh_nmp_model_info()
        self._refresh_tracking_params_platform_labels()
        if hasattr(self, 'status_text'):
            self.status_text.append(f'NMP model: {platform_label(pid)}')

    def _refresh_nmp_model_info(self):
        if not hasattr(self, 'lbl_nmp_nmp_params'):
            return
        phy = self._nmp_physics_dict()
        from controllers.flatness import FlatnessParams
        fp = FlatnessParams.from_gui(
            phy['mass'], phy['Ixx'], phy['Iyy'], phy['Izz'], phy['r_thrust_z'], g=phy['g'],
        )
        self.lbl_nmp_nmp_params.setText(
            f"m={fp.mass:.3g} kg  l={fp.lever_arm:.3f} m  "
            f"ω_z(pitch)={fp.omega_z_pitch:.2f} rad/s ({fp.omega_z_pitch / (2 * np.pi):.2f} Hz)  "
            f"d_pitch={fp.flat_offset_pitch * 100:.1f} cm  d_roll={fp.flat_offset_roll * 100:.1f} cm"
        )

    def _populate_nmp_waypoint_table(self):
        if not hasattr(self, 'nmp_waypoint_table'):
            return
        self._nmp_waypoint_table_updating = True
        self.nmp_waypoint_table.blockSignals(True)
        try:
            self.nmp_waypoint_table.setRowCount(len(self.nmp_waypoints))
            for i, wp in enumerate(self.nmp_waypoints):
                row = _normalize_waypoint_row(wp)
                x, y, z, yaw, t_arr = row
                leg = leg_duration_s(self.nmp_waypoints, i)
                self.nmp_waypoint_table.setItem(
                    i, WP_COL_IDX, self._make_waypoint_table_item(i, editable=False),
                )
                self.nmp_waypoint_table.setItem(i, WP_COL_X, self._make_waypoint_table_item(f'{x:.1f}'))
                self.nmp_waypoint_table.setItem(i, WP_COL_Y, self._make_waypoint_table_item(f'{y:.1f}'))
                self.nmp_waypoint_table.setItem(i, WP_COL_Z, self._make_waypoint_table_item(f'{z:.1f}'))
                self.nmp_waypoint_table.setItem(i, WP_COL_YAW, self._make_waypoint_table_item(f'{yaw:.1f}'))
                if i == 0:
                    self.nmp_waypoint_table.setItem(
                        i, WP_COL_LEG_DT, self._make_waypoint_table_item('—', editable=False),
                    )
                else:
                    self.nmp_waypoint_table.setItem(
                        i, WP_COL_LEG_DT, self._make_waypoint_table_item(f'{leg:.1f}'),
                    )
                self.nmp_waypoint_table.setItem(
                    i, WP_COL_T_ARR,
                    self._make_waypoint_table_item(f'{t_arr:.1f}', editable=(i > 0)),
                )
        finally:
            self.nmp_waypoint_table.blockSignals(False)
            self._nmp_waypoint_table_updating = False

    def _sync_nmp_waypoints_from_table(self):
        if not hasattr(self, 'nmp_waypoint_table'):
            return
        self._nmp_waypoint_table_updating = True
        try:
            rows = []
            n = self.nmp_waypoint_table.rowCount()
            for i in range(n):
                def _cell(col, default='0'):
                    it = self.nmp_waypoint_table.item(i, col)
                    return it.text().strip() if it else default

                t_item = self.nmp_waypoint_table.item(i, WP_COL_T_ARR)
                rows.append([
                    float(_cell(WP_COL_X)), float(_cell(WP_COL_Y)), float(_cell(WP_COL_Z)),
                    float(_cell(WP_COL_YAW)),
                    float(t_item.text()) if t_item and i > 0 else 0.0,
                ])
            self.nmp_waypoints = [_normalize_waypoint_row(r) for r in rows]
        finally:
            self._nmp_waypoint_table_updating = False

    def _on_nmp_waypoint_table_item_changed(self, item):
        if self._nmp_waypoint_table_updating or item is None:
            return
        self._sync_nmp_waypoints_from_table()
        self._populate_nmp_waypoint_table()
        if hasattr(self, 'nmp_sim_dt_spin'):
            self._on_nmp_tracking_timing_changed()

    def add_nmp_waypoint(self):
        self._sync_nmp_waypoints_from_table()
        if not self.nmp_waypoints:
            self.nmp_waypoints = [[0.0, 0.0, 0.0, 0.0, 0.0]]
        last = _normalize_waypoint_row(self.nmp_waypoints[-1])
        dt_leg = nominal_segment_duration(self.nmp_dt_spin.value(), 100)
        self.nmp_waypoints.append([
            last[0], last[1], last[2] + 0.5, last[3], last[4] + dt_leg,
        ])
        self._populate_nmp_waypoint_table()
        if hasattr(self, 'nmp_sim_dt_spin'):
            self._on_nmp_tracking_timing_changed()

    def remove_nmp_waypoint(self):
        self._sync_nmp_waypoints_from_table()
        if len(self.nmp_waypoints) <= 2:
            QMessageBox.warning(self, 'NMP', 'Need at least 2 waypoints.')
            return
        row = self.nmp_waypoint_table.currentRow()
        if row < 1:
            row = len(self.nmp_waypoints) - 1
        if 0 <= row < len(self.nmp_waypoints):
            del self.nmp_waypoints[row]
        self._populate_nmp_waypoint_table()
        if hasattr(self, 'nmp_sim_dt_spin'):
            self._on_nmp_tracking_timing_changed()

    def save_nmp_waypoints(self):
        """Persist current NMP waypoints so they auto-load next time the GUI starts."""
        self._sync_nmp_waypoints_from_table()
        os.makedirs(os.path.dirname(NMP_SAVED_WAYPOINTS_PATH), exist_ok=True)
        try:
            with open(NMP_SAVED_WAYPOINTS_PATH, 'w', encoding='utf-8') as f:
                json.dump({'waypoints': waypoints_to_json_list(self.nmp_waypoints)}, f, indent=2)
        except OSError as e:
            QMessageBox.critical(self, 'NMP', f'Could not save waypoints:\n{e}')
            return
        if hasattr(self, 'status_text'):
            self.status_text.append(
                f'NMP: saved {len(self.nmp_waypoints)} waypoints to:\n  {NMP_SAVED_WAYPOINTS_PATH}'
            )

    def start_nmp_planning(self):
        if self._nmp_plan_thread and self._nmp_plan_thread.isRunning():
            QMessageBox.warning(self, 'NMP', 'Planning already in progress.')
            return
        self._sync_nmp_waypoints_from_table()
        if len(self.nmp_waypoints) < 2:
            QMessageBox.warning(self, 'NMP', 'Need at least 2 waypoints.')
            return
        for i in range(len(self.nmp_waypoints) - 1):
            if self.nmp_waypoints[i][4] >= self.nmp_waypoints[i + 1][4]:
                QMessageBox.warning(
                    self, 'NMP',
                    f'Waypoint {i + 1} time must exceed waypoint {i} time.',
                )
                return
        phy = self._nmp_physics_dict()
        bounds = self._nmp_bounds_dict()
        params = {
            'dt': self.nmp_dt_spin.value(),
            'waypoints': list(self.nmp_waypoints),
            'm': phy['mass'],
            'I': (phy['Ixx'], phy['Iyy'], phy['Izz']),
            'r_thrust': (phy['r_thrust_x'], phy['r_thrust_y'], phy['r_thrust_z']),
            'bounds': bounds,
            'weights': {},
        }
        self.btn_nmp_plan.setEnabled(False)
        self.lbl_nmp_plan_status.setText('Planning…')
        self._nmp_plan_thread = NmpPlanningThread(params)
        self._nmp_plan_thread.finished.connect(self._nmp_planning_finished)
        self._nmp_plan_thread.error.connect(self._nmp_planning_error)
        self.status_text.append('NMP: starting flatness planning…')
        self._nmp_plan_thread.start()

    def _nmp_planning_error(self, message):
        self.btn_nmp_plan.setEnabled(True)
        self.lbl_nmp_plan_status.setText('Planning failed.')
        self.status_text.append(f'NMP error: {message}')
        QMessageBox.critical(self, 'NMP planning error', message)

    def _nmp_planning_finished(self, xs, us, timing_info):
        self.btn_nmp_plan.setEnabled(True)
        time_axis = None
        if timing_info.get('time_states') is not None:
            time_axis = np.asarray(timing_info['time_states'], dtype=float)
        elif timing_info.get('flat_outputs', {}).get('t') is not None:
            time_axis = np.asarray(timing_info['flat_outputs']['t'], dtype=float)
        phy = self._nmp_physics_dict()
        self.nmp_last_trajectory = {
            'xs': xs,
            'us': us,
            'dt': self.nmp_dt_spin.value(),
            'time_states': time_axis,
            'flat_outputs': timing_info.get('flat_outputs'),
            'flatness_physics': timing_info.get('flatness_physics') or {
                'mass': phy['mass'],
                'Ixx': phy['Ixx'],
                'Iyy': phy['Iyy'],
                'Izz': phy['Izz'],
                'r_thrust_z': phy['r_thrust_z'],
                'g': phy['g'],
            },
            'method_name': timing_info.get('method', 'Method 8'),
            'platform_id': self._current_nmp_platform_id(),
        }
        n_pts = len(xs) if xs else 0
        dur = float(time_axis[-1] - time_axis[0]) if time_axis is not None and len(time_axis) >= 2 else 0.0
        elapsed = float(timing_info.get('total_time', 0.0))
        self.lbl_nmp_plan_status.setText(
            f'Plan OK: {n_pts} samples, duration {dur:.2f} s, compute {elapsed:.2f} s.'
        )
        self._nmp_last_tracking_result = None
        self.status_text.append(
            f'NMP plan ready — {n_pts} points, {dur:.2f} s ({platform_label(self._current_nmp_platform_id())}).'
        )
        if hasattr(self, 'nmp_sim_dt_spin'):
            self._on_nmp_tracking_timing_changed()
        self._draw_nmp_plot_tab()
        if hasattr(self, 'plot_tabs'):
            for i in range(self.plot_tabs.count()):
                if self.plot_tabs.tabText(i) == 'NMP / Flatness':
                    self.plot_tabs.setCurrentIndex(i)
                    break

    def _current_nmp_tracking_ref_mode(self):
        if hasattr(self, 'nmp_ref_waypoint_radio') and self.nmp_ref_waypoint_radio.isChecked():
            return 'waypoint'
        return 'planned'

    def _current_nmp_plot_mode(self):
        if hasattr(self, 'nmp_plot_2d_radio') and self.nmp_plot_2d_radio.isChecked():
            return '2d'
        return 'all'

    def _on_nmp_plot_mode_changed(self, _checked=False):
        self._draw_nmp_plot_tab()

    def _nmp_flatness_physics_dict(self):
        """Live model params (mass/inertia/thrust point) from the Model section.

        Always reflects the current spinboxes, not a stale planned-trajectory
        snapshot, so waypoint-mode direct flight uses whatever model is
        currently configured.
        """
        phy = self._nmp_physics_dict()
        return {
            'mass': phy['mass'],
            'Ixx': phy['Ixx'],
            'Iyy': phy['Iyy'],
            'Izz': phy['Izz'],
            'r_thrust_z': phy['r_thrust_z'],
            'g': phy.get('g', 9.81),
        }

    def _create_nmp_tracking_timing_panel(self, parent_layout):
        """NMP-specific plant step, control period, and simulation horizon."""
        timing_group = QGroupBox('Tracking timing')
        grid = QGridLayout(timing_group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        cfg = dict(
            self._tracking_config.get('nmp_numerical_sim')
            or self._tracking_config.get('numerical_sim')
            or default_numerical_sim_config()
        )

        self.nmp_sim_dt_spin = QDoubleSpinBox()
        self.nmp_sim_dt_spin.setRange(0.001, 0.05)
        self.nmp_sim_dt_spin.setDecimals(4)
        self.nmp_sim_dt_spin.setSingleStep(0.001)
        self.nmp_sim_dt_spin.setToolTip('Plant integration step [s] for NMP tracking simulation.')
        self.nmp_sim_dt_spin.valueChanged.connect(self._on_nmp_tracking_timing_changed)

        self.nmp_control_dt_spin = QDoubleSpinBox()
        self.nmp_control_dt_spin.setRange(0.005, 0.2)
        self.nmp_control_dt_spin.setDecimals(4)
        self.nmp_control_dt_spin.setSingleStep(0.005)
        self.nmp_control_dt_spin.setToolTip('Controller update period [s] for both COM and ξ tracking.')
        self.nmp_control_dt_spin.valueChanged.connect(self._on_nmp_tracking_timing_changed)

        self.nmp_terminal_hold_spin = QDoubleSpinBox()
        self.nmp_terminal_hold_spin.setRange(0.0, 120.0)
        self.nmp_terminal_hold_spin.setDecimals(2)
        self.nmp_terminal_hold_spin.setSingleStep(0.5)
        self.nmp_terminal_hold_spin.setToolTip(
            'Extra hover time after the selected NMP reference ends. Ignored when Total duration > 0.'
        )
        self.nmp_terminal_hold_spin.valueChanged.connect(self._on_nmp_tracking_timing_changed)

        self.nmp_total_duration_spin = QDoubleSpinBox()
        self.nmp_total_duration_spin.setRange(0.0, 600.0)
        self.nmp_total_duration_spin.setDecimals(2)
        self.nmp_total_duration_spin.setSingleStep(1.0)
        self.nmp_total_duration_spin.setSpecialValueText('Auto (reference + hold)')
        self.nmp_total_duration_spin.setToolTip(
            'Fixed NMP tracking simulation length from t=0 [s]. 0 = automatic: '
            'selected reference duration + terminal hold.'
        )
        self.nmp_total_duration_spin.valueChanged.connect(self._on_nmp_tracking_timing_changed)

        self.lbl_nmp_control_hz = QLabel()
        self.lbl_nmp_control_hz.setStyleSheet('color: #555;')
        self.lbl_nmp_substeps = QLabel()
        self.lbl_nmp_substeps.setStyleSheet('color: #555;')
        self.lbl_nmp_sim_duration = QLabel()
        self.lbl_nmp_sim_duration.setStyleSheet('color: #555;')
        self.lbl_nmp_sim_duration.setWordWrap(True)

        grid.addWidget(QLabel('sim_dt [s]:'), 0, 0)
        grid.addWidget(self.nmp_sim_dt_spin, 0, 1)
        grid.addWidget(QLabel('control_dt [s]:'), 0, 2)
        grid.addWidget(self.nmp_control_dt_spin, 0, 3)
        grid.addWidget(QLabel('hold [s]:'), 1, 0)
        grid.addWidget(self.nmp_terminal_hold_spin, 1, 1)
        grid.addWidget(QLabel('total [s]:'), 1, 2)
        grid.addWidget(self.nmp_total_duration_spin, 1, 3)
        grid.addWidget(self.lbl_nmp_control_hz, 2, 0, 1, 2)
        grid.addWidget(self.lbl_nmp_substeps, 2, 2, 1, 2)
        grid.addWidget(self.lbl_nmp_sim_duration, 3, 0, 1, 4)

        parent_layout.addWidget(timing_group)
        self._nmp_sim_timing_guard = True
        self.nmp_sim_dt_spin.setValue(float(cfg.get('sim_dt', 0.005)))
        self.nmp_control_dt_spin.setValue(float(cfg.get('control_dt', 0.02)))
        self.nmp_terminal_hold_spin.setValue(float(cfg.get('terminal_hold_duration_s', 3.0)))
        self.nmp_total_duration_spin.setValue(float(cfg.get('total_duration_s', 0.0)))
        self._nmp_sim_timing_guard = False
        self._on_nmp_tracking_timing_changed()

    def _nmp_reference_duration_s(self):
        """Selected NMP reference span [s], used only for timing preview."""
        if self._current_nmp_tracking_ref_mode() == 'waypoint':
            self._sync_nmp_waypoints_from_table()
            if len(self.nmp_waypoints) >= 2:
                return max(
                    float(self.nmp_waypoints[-1][4]) - float(self.nmp_waypoints[0][4]),
                    0.0,
                )
            return None
        traj = getattr(self, 'nmp_last_trajectory', None)
        if traj and traj.get('xs') is not None:
            ts = traj.get('time_states')
            if ts is not None and len(ts) >= 2:
                arr = np.asarray(ts, dtype=float)
                return max(float(arr[-1] - arr[0]), 0.0)
            dt = traj.get('dt')
            xs = traj.get('xs')
            if dt is not None and xs is not None and len(xs) >= 2:
                return max(float((len(xs) - 1) * float(dt)), 0.0)
        return None

    def _on_nmp_tracking_timing_changed(self, _value=None):
        if getattr(self, '_nmp_sim_timing_guard', False):
            return
        sim_dt = float(self.nmp_sim_dt_spin.value())
        control_dt = float(self.nmp_control_dt_spin.value())
        if control_dt < sim_dt:
            self._nmp_sim_timing_guard = True
            self.nmp_control_dt_spin.setValue(sim_dt)
            self._nmp_sim_timing_guard = False
            control_dt = sim_dt
        ratio = max(1, int(round(control_dt / sim_dt)))
        aligned_control_dt = ratio * sim_dt
        if abs(aligned_control_dt - control_dt) > 1e-9:
            self._nmp_sim_timing_guard = True
            self.nmp_control_dt_spin.setValue(aligned_control_dt)
            self._nmp_sim_timing_guard = False
            control_dt = aligned_control_dt
        hz = 1.0 / control_dt if control_dt > 0 else 0.0
        self.lbl_nmp_control_hz.setText(f'Control rate: {hz:.1f} Hz')
        self.lbl_nmp_substeps.setText(f'Plant substeps per control update: {ratio}')

        hold = float(self.nmp_terminal_hold_spin.value())
        total = float(self.nmp_total_duration_spin.value())
        ref_s = self._nmp_reference_duration_s()
        if total > 0.0:
            dur_txt = f'Simulation length: {total:.2f} s (fixed total)'
        elif ref_s is not None:
            dur_txt = (
                f'Simulation length: {ref_s + hold:.2f} s '
                f'(reference {ref_s:.2f} s + hold {hold:.2f} s)'
            )
        else:
            dur_txt = (
                f'Simulation length: selected reference duration + {hold:.2f} s hold '
                f'(plan a trajectory or use waypoint mode)'
            )
        self.lbl_nmp_sim_duration.setText(dur_txt)
        self._tracking_config['nmp_numerical_sim'] = {
            'sim_dt': sim_dt,
            'control_dt': control_dt,
            'terminal_hold_duration_s': hold,
            'total_duration_s': total,
        }

    def _nmp_tracking_sim_params_for_sim(self):
        if hasattr(self, 'nmp_sim_dt_spin'):
            self._on_nmp_tracking_timing_changed()
        cfg = self._tracking_config.get('nmp_numerical_sim') or default_numerical_sim_config()
        return {
            'sim_dt': float(cfg.get('sim_dt', 0.005)),
            'control_dt': float(cfg.get('control_dt', 0.02)),
            'terminal_hold_duration_s': float(cfg.get('terminal_hold_duration_s', 3.0)),
            'total_duration_s': float(cfg.get('total_duration_s', 0.0)),
        }

    def _nmp_tracking_reference_pack(self):
        """Return arrays + metadata for the selected NMP tracking reference."""
        ref_mode = self._current_nmp_tracking_ref_mode()
        if ref_mode == 'waypoint':
            self._sync_nmp_waypoints_from_table()
            if len(self.nmp_waypoints) < 2:
                raise ValueError('Need at least 2 waypoints for direct flight.')
            for i in range(len(self.nmp_waypoints) - 1):
                if self.nmp_waypoints[i][4] >= self.nmp_waypoints[i + 1][4]:
                    raise ValueError(
                        f'Arrival time must increase between waypoints {i} and {i + 1}.',
                    )
            pack = build_waypoint_flight_reference(
                self.nmp_waypoints,
                self._nmp_flatness_physics_dict(),
                dt=self.nmp_dt_spin.value(),
                terminal_hold_s=0.0,
            )
            traj = {
                'xs': pack['xs'],
                'us': pack['us'],
                'time_states': pack['time_states'],
                'dt': pack['dt'],
                'flat_outputs': pack['flat_outputs'],
                'flatness_physics': pack['flatness_physics'],
                'platform_id': self._current_nmp_platform_id(),
                'method_name': pack['method_name'],
                'waypoint_mode': True,
            }
            return (
                np.asarray(pack['xs'], dtype=float),
                np.asarray(pack['us'], dtype=float) if pack.get('us') is not None else None,
                np.asarray(pack['time_states'], dtype=float),
                pack.get('flat_outputs'),
                pack.get('flatness_physics'),
                pack.get('x0'),
                traj,
                'waypoint',
            )

        traj = getattr(self, 'nmp_last_trajectory', None)
        if not traj or traj.get('xs') is None:
            raise ValueError('Plan a flat trajectory first, or switch to direct waypoint flight.')
        xs, us, time_states = self._nmp_trajectory_arrays_for_tracking()
        if xs is None:
            raise ValueError('No planned trajectory available.')
        return (
            xs, us, time_states,
            traj.get('flat_outputs'),
            traj.get('flatness_physics'),
            None,
            traj,
            'planned',
        )

    def _current_nmp_controller_id(self):
        if hasattr(self, 'nmp_ctrl_xi_radio') and self.nmp_ctrl_xi_radio.isChecked():
            return CONTROLLER_FLATNESS
        return CONTROLLER_PX4

    def _on_nmp_control_point_changed(self, _checked=False):
        if not hasattr(self, 'nmp_params_groups_layout'):
            return
        self._store_nmp_tracking_params_for_controller()
        self._rebuild_nmp_tracking_param_widgets()

    def _make_nmp_tracking_param_widget(self, spec, params, controller_id):
        w = self._make_tracking_param_widget(spec, params, controller_id)
        if spec.get('checkbox') and spec.get('key') == 'share_rp_gains':
            try:
                w.stateChanged.disconnect()
            except TypeError:
                pass
            w.stateChanged.connect(self._on_nmp_share_rp_changed)
        return w

    def _populate_nmp_tracking_param_group(self, grid, specs, params, controller_id):
        row = 0
        col_slot = 0
        for spec in specs:
            key = spec['key']
            if spec.get('full_width'):
                if col_slot == 1:
                    row += 1
                    col_slot = 0
                w = self._make_nmp_tracking_param_widget(spec, params, controller_id)
                if spec.get('checkbox'):
                    w.setText(spec['label'])
                    grid.addWidget(w, row, 0, 1, 4)
                else:
                    grid.addWidget(QLabel(spec['label']), row, 0)
                    grid.addWidget(w, row, 1, 1, 3)
                self._nmp_tracking_param_widgets[key] = w
                row += 1
                continue

            base_col = col_slot * 2
            grid.addWidget(QLabel(spec['label']), row, base_col)
            w = self._make_nmp_tracking_param_widget(spec, params, controller_id)
            grid.addWidget(w, row, base_col + 1)
            self._nmp_tracking_param_widgets[key] = w
            if col_slot == 1:
                row += 1
                col_slot = 0
            else:
                col_slot = 1
        if col_slot == 1:
            row += 1
        return row

    def _store_nmp_tracking_params_for_controller(self, controller_id=None):
        cid = controller_id or getattr(
            self, '_nmp_param_controller_id', self._current_nmp_controller_id()
        )
        if cid not in (CONTROLLER_PX4, CONTROLLER_FLATNESS):
            return
        platform_id = (
            getattr(self, '_cached_nmp_platform_id', None)
            or self._current_nmp_platform_id()
        )
        params_map = self._tracking_params_map_for(platform_id)
        stored = params_map.setdefault(cid, default_params_for(cid))
        for key, widget in getattr(self, '_nmp_tracking_param_widgets', {}).items():
            if isinstance(widget, QCheckBox):
                stored[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                stored[key] = int(widget.value())
            else:
                stored[key] = float(widget.value())
        if cid == CONTROLLER_PX4:
            params_map[cid] = migrate_px4_params(stored)

    def _rebuild_nmp_tracking_param_widgets(self):
        if not hasattr(self, 'nmp_params_groups_layout'):
            return
        while self.nmp_params_groups_layout.count():
            item = self.nmp_params_groups_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._nmp_tracking_param_widgets = {}

        cid = self._current_nmp_controller_id()
        self._nmp_param_controller_id = cid
        platform_id = self._current_nmp_platform_id()
        self._cached_nmp_platform_id = platform_id
        params_map = self._tracking_params_map_for(platform_id)
        params = params_map.setdefault(cid, default_params_for(cid))
        if cid == CONTROLLER_PX4:
            params.update(migrate_px4_params(params))
            params_map[cid] = params
        px4_extra = params if cid in (CONTROLLER_PX4, CONTROLLER_FLATNESS) else None
        for group in param_groups_for(cid, px4_params=px4_extra):
            box = QGroupBox(group.get('title', 'Parameters'))
            grid = QGridLayout(box)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(3)
            self._populate_nmp_tracking_param_group(
                grid, group.get('specs') or [], params, cid
            )
            self.nmp_params_groups_layout.addWidget(box)
        self.nmp_params_groups_layout.addStretch(1)
        self._refresh_tracking_params_platform_labels()
        QTimer.singleShot(0, self._refresh_tab_scroll_areas)

    def _on_nmp_share_rp_changed(self, _state=None):
        self._store_nmp_tracking_params_for_controller()
        self._rebuild_nmp_tracking_param_widgets()

    def _nmp_physics_for_tracking(self):
        phy = dict(self._nmp_physics_dict())
        phy['platform_id'] = self._current_nmp_platform_id()
        phy['rocket_geometry'] = self._nmp_rocket_geometry_dict()
        return phy

    def _collect_nmp_tracking_params(self):
        cid = self._current_nmp_controller_id()
        self._store_nmp_tracking_params_for_controller(cid)
        params = dict(default_params_for(cid))
        if hasattr(self, '_tracking_config'):
            saved = self._tracking_params_map_for(self._current_nmp_platform_id()).get(cid)
            if saved:
                params.update(saved)
        if hasattr(self, '_store_actuator_config'):
            self._store_actuator_config()
        if hasattr(self, '_actuator_params_for_sim'):
            params.update(self._actuator_params_for_sim())
        params.update(self._nmp_tracking_sim_params_for_sim())
        bounds = self._nmp_bounds_dict()
        params.update(bounds)
        return params

    def _nmp_trajectory_arrays_for_tracking(self):
        traj = getattr(self, 'nmp_last_trajectory', None)
        if not traj or traj.get('xs') is None:
            return None, None, None
        xs = np.asarray(traj['xs'], dtype=float)
        us = traj.get('us')
        if us is not None:
            us = np.asarray(us, dtype=float)
        ts = traj.get('time_states')
        if ts is not None and len(ts) == len(xs):
            t_list = np.asarray(ts, dtype=float)
        else:
            dt = float(traj.get('dt', 0.05))
            t_list = np.arange(len(xs), dtype=float) * dt
        if us is not None and us.shape[0] == xs.shape[0]:
            us = us[:-1]
        elif us is not None and us.shape[0] != max(xs.shape[0] - 1, 0):
            us = None
        return xs, us, t_list

    def run_nmp_tracking_sim(self):
        if self._nmp_tracking_sim_thread is not None and self._nmp_tracking_sim_thread.isRunning():
            return
        try:
            xs, us, time_states, flat_outputs, flatness_physics, x0, traj, ref_mode = (
                self._nmp_tracking_reference_pack()
            )
        except ValueError as exc:
            QMessageBox.warning(self, 'NMP', str(exc))
            return
        cid = self._current_nmp_controller_id()
        params = self._collect_nmp_tracking_params()
        ctrl_label = 'ξ (oscillation)' if cid == CONTROLLER_FLATNESS else 'COM'
        ref_label = 'waypoints' if ref_mode == 'waypoint' else 'flat plan'
        self.btn_nmp_run_tracking.setEnabled(False)
        self.lbl_nmp_tracking_status.setText(
            f'Running tracking sim — {ref_label}, PX4 cascade on {ctrl_label}…',
        )
        self.status_text.append(
            f'NMP: tracking sim ({ref_label}, {ctrl_label})…',
        )
        self._nmp_active_tracking_traj = traj
        self._nmp_tracking_sim_thread = TrackingSimulationThread(
            xs, us, time_states, cid, params, self._nmp_physics_for_tracking(),
            flat_outputs=flat_outputs, flatness_physics=flatness_physics, x0=x0,
        )
        self._nmp_tracking_sim_thread.finished.connect(self._on_nmp_tracking_sim_finished)
        self._nmp_tracking_sim_thread.error.connect(self._on_nmp_tracking_sim_error)
        self._nmp_tracking_sim_thread.start()

    def _on_nmp_tracking_sim_finished(self, result):
        self.btn_nmp_run_tracking.setEnabled(True)
        self._nmp_last_tracking_result = result
        self._last_tracking_result = result
        summary = tracking_summary_text(result)
        ctrl_label = 'ξ' if self._current_nmp_controller_id() == CONTROLLER_FLATNESS else 'COM'
        ref_label = (
            'waypoint flight'
            if self._current_nmp_tracking_ref_mode() == 'waypoint'
            else 'flat plan'
        )
        self.lbl_nmp_tracking_status.setText(
            f'{summary}  |  ref: {ref_label}  |  control: {ctrl_label}',
        )
        self.status_text.append(f'NMP tracking: {summary}')
        active_traj = getattr(self, '_nmp_active_tracking_traj', None) or self.nmp_last_trajectory
        if active_traj:
            self.last_trajectory = dict(active_traj)
        if self._display_panels_ready():
            self._draw_tracking_plot_tabs(result)
            self._draw_nmp_plot_tab()
            if hasattr(self, 'plot_tabs'):
                for i in range(self.plot_tabs.count()):
                    if self.plot_tabs.tabText(i) == 'NMP / Flatness':
                        self.plot_tabs.setCurrentIndex(i)
                        break

    def _on_nmp_tracking_sim_error(self, msg):
        self.btn_nmp_run_tracking.setEnabled(True)
        self.lbl_nmp_tracking_status.setText('Tracking simulation failed.')
        QMessageBox.critical(self, 'NMP tracking error', msg)
        self.status_text.append(f'NMP tracking error: {msg}')

    def _draw_nmp_plot_tab(self):
        if not hasattr(self, 'canvas_nmp'):
            return
        traj = getattr(self, '_nmp_active_tracking_traj', None) or getattr(self, 'nmp_last_trajectory', None)
        phy = None
        if traj:
            phy = traj.get('flatness_physics') or self._nmp_physics_dict()
        series = build_nmp_series(
            traj,
            phy=phy,
            tracking_result=getattr(self, '_nmp_last_tracking_result', None),
        )
        summary = ''
        if traj and series is not None:
            max_dx = float(np.max(np.abs(series['offset_x_meas']))) * 1000
            max_dy = float(np.max(np.abs(series['offset_y_meas']))) * 1000
            ctrl = 'ξ' if self._current_nmp_controller_id() == CONTROLLER_FLATNESS else 'COM'
            ref = (
                'waypoint'
                if traj.get('waypoint_mode') or self._current_nmp_tracking_ref_mode() == 'waypoint'
                else 'planned'
            )
            summary = (
                f'Platform: {platform_label(traj.get("platform_id", self._current_nmp_platform_id()))}  |  '
                f'plan max |x−ξ_x|={max_dx:.1f} mm  max |y−ξ_y|={max_dy:.1f} mm  |  '
                f'ref: {ref}  ctrl: {ctrl}'
            )
        draw_nmp_panels(
            nmp_axes_dict(self),
            series,
            summary=summary,
            mode=self._current_nmp_plot_mode(),
        )
        self.canvas_nmp.draw_idle()
        self._refresh_all_plot_layouts()

    def create_display_panel(self):
        """Create tabbed plot area: Overview / States / 3D / Metrics."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        fig_actions = QHBoxLayout()
        self.btn_save_figure = QPushButton('Save figure…')
        self.btn_save_figure.setToolTip('Export the active plot tab (PNG, PDF, or SVG)')
        self.btn_save_figure.clicked.connect(self.save_figure)
        self.btn_save_figure.setMaximumHeight(30)
        fig_actions.addWidget(self.btn_save_figure)
        fig_actions.addStretch(1)
        layout.addLayout(fig_actions)

        self.plot_tabs = QTabWidget()
        self.plot_tabs.setDocumentMode(True)

        # ── Tab 1: Overview (planning results, unchanged layout) ──
        self.fig = Figure(figsize=(12, 7))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(280, 220)
        gs = GridSpec(
            4, 4, figure=self.fig, height_ratios=[0.22, 1.0, 1.0, 1.0],
            hspace=0.32, wspace=0.26,
        )
        self.fig._tvc_gridspec_pads = {
            'left': 0.06, 'right': 0.98, 'top': 0.91, 'bottom': 0.06,
            'hspace': 0.32, 'wspace': 0.26,
        }
        self.fig.suptitle(
            'TVC Rocket Trajectory Optimization',
            fontsize=13, fontweight='bold', y=0.97,
        )
        self.ax_opt_info = self.fig.add_subplot(gs[0, :])
        self.ax_opt_info.axis('off')
        self.ax_3d = self.fig.add_subplot(gs[1, 0:2], projection='3d')
        self.ax_3d.set_xlabel('X (m)', fontsize=10)
        self.ax_3d.set_ylabel('Y (m)', fontsize=10)
        self.ax_3d.set_zlabel('Z (m)', fontsize=10)
        self.ax_3d.set_title('3D Position Trajectory', fontsize=11, fontweight='bold')
        self.ax_3d.grid(True, alpha=0.3)
        self.ax_cost = self.fig.add_subplot(gs[1, 2:4])
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        self.ax_pos = self.fig.add_subplot(gs[2, 0])
        self.ax_pos.set_xlabel('Time (s)', fontsize=9)
        self.ax_pos.set_ylabel('Position (m)', fontsize=9)
        self.ax_pos.set_title('Position', fontsize=10, fontweight='bold')
        self.ax_pos.grid(True, alpha=0.3)
        self.ax_vel = self.fig.add_subplot(gs[2, 1])
        self.ax_vel.set_xlabel('Time (s)', fontsize=9)
        self.ax_vel.set_ylabel('Velocity (m/s)', fontsize=9)
        self.ax_vel.set_title('Linear Velocity', fontsize=10, fontweight='bold')
        self.ax_vel.grid(True, alpha=0.3)
        self.ax_euler = self.fig.add_subplot(gs[2, 2])
        self.ax_euler.set_xlabel('Time (s)', fontsize=9)
        self.ax_euler.set_ylabel('Euler Angles (deg)', fontsize=9)
        self.ax_euler.set_title('Attitude (Euler)', fontsize=10, fontweight='bold')
        self.ax_euler.grid(True, alpha=0.3)
        self.ax_angvel = self.fig.add_subplot(gs[2, 3])
        self.ax_angvel.set_xlabel('Time (s)', fontsize=9)
        self.ax_angvel.set_ylabel('Angular Vel (°/s)', fontsize=9)
        self.ax_angvel.set_title('Angular Velocity', fontsize=10, fontweight='bold')
        self.ax_angvel.grid(True, alpha=0.3)
        self.ax_pitch = self.fig.add_subplot(gs[3, 0])
        self.ax_pitch.set_xlabel('Time (s)', fontsize=9)
        self.ax_pitch.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_pitch.set_title('TVC Pitch Angle', fontsize=10, fontweight='bold')
        self.ax_pitch.grid(True, alpha=0.3)
        self.ax_roll = self.fig.add_subplot(gs[3, 1])
        self.ax_roll.set_xlabel('Time (s)', fontsize=9)
        self.ax_roll.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_roll.set_title('TVC Roll Angle', fontsize=10, fontweight='bold')
        self.ax_roll.grid(True, alpha=0.3)
        self.ax_thrust = self.fig.add_subplot(gs[3, 2])
        self.ax_thrust.set_xlabel('Time (s)', fontsize=9)
        self.ax_thrust.set_ylabel('Thrust (N)', fontsize=9)
        self.ax_thrust.set_title('Thrust', fontsize=10, fontweight='bold')
        self.ax_thrust.grid(True, alpha=0.3)
        self.ax_yaw = self.fig.add_subplot(gs[3, 3])
        self.ax_yaw.set_xlabel('Time (s)', fontsize=9)
        self.ax_yaw.set_ylabel('Torque (N·m)', fontsize=9)
        self.ax_yaw.set_title('Yaw Torque', fontsize=10, fontweight='bold')
        self.ax_yaw.grid(True, alpha=0.3)

        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.addWidget(self.canvas)
        install_responsive_canvas(
            self.canvas, self.fig, base_width_px=1100, base_height_px=700,
            layout_mode='gridspec',
        )
        self.plot_tabs.addTab(overview_widget, 'Overview')

        # ── Tab 2: States (4×2, tracking ref vs sim) ──
        self.fig_states = Figure(figsize=(12, 8))
        self.canvas_states = FigureCanvas(self.fig_states)
        self.canvas_states.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_states.setMinimumSize(280, 220)
        gs_s = GridSpec(4, 2, figure=self.fig_states, hspace=0.30, wspace=0.22)
        self.fig_states._tvc_gridspec_pads = {
            'left': 0.07, 'right': 0.98, 'top': 0.93, 'bottom': 0.06,
            'hspace': 0.30, 'wspace': 0.22,
        }
        self.fig_states.suptitle('Tracking — States & Controls', fontsize=11, fontweight='bold', y=0.98)
        self.ax_trk_pos = self.fig_states.add_subplot(gs_s[0, 0])
        self.ax_trk_att = self.fig_states.add_subplot(gs_s[0, 1])
        self.ax_trk_vel = self.fig_states.add_subplot(gs_s[1, 0])
        self.ax_trk_angvel = self.fig_states.add_subplot(gs_s[1, 1])
        self.ax_trk_acc = self.fig_states.add_subplot(gs_s[2, 0])
        self.ax_trk_angacc = self.fig_states.add_subplot(gs_s[2, 1])
        self.ax_trk_gimbal = self.fig_states.add_subplot(gs_s[3, 0])
        self.ax_trk_thrust = self.fig_states.add_subplot(gs_s[3, 1])
        states_widget = QWidget()
        states_layout = QVBoxLayout(states_widget)
        states_layout.setContentsMargins(0, 0, 0, 0)
        states_series_row = QHBoxLayout()
        states_series_row.setContentsMargins(4, 2, 4, 0)
        states_series_row.addWidget(QLabel('Show:'))
        self.chk_states_show_plan = QCheckBox('plan')
        self.chk_states_show_sim = QCheckBox('sim')
        self.chk_states_show_cascade = QCheckBox('cascade')
        for chk in (
            self.chk_states_show_plan,
            self.chk_states_show_sim,
            self.chk_states_show_cascade,
        ):
            chk.setChecked(True)
            chk.setToolTip(
                'Toggle which series appear on the States plots '
                '(plan = reference, sim = plant, cascade = inner-loop setpoints).'
            )
            chk.stateChanged.connect(self._on_states_series_visibility_changed)
            states_series_row.addWidget(chk)
        states_series_row.addStretch(1)
        states_layout.addLayout(states_series_row)
        states_layout.addWidget(self.canvas_states)
        install_responsive_canvas(
            self.canvas_states, self.fig_states, base_width_px=1100, base_height_px=900,
            layout_mode='gridspec',
        )
        self._plot_tab_states = states_widget
        self.plot_tabs.addTab(states_widget, 'States')

        # ── Tab 3: 3D trajectory ──
        self.fig_3d_tab = Figure(figsize=(10, 8))
        self.canvas_3d_tab = FigureCanvas(self.fig_3d_tab)
        self.canvas_3d_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_3d_tab.setMinimumSize(280, 220)
        self.ax_3d_trk = self.fig_3d_tab.add_subplot(111, projection='3d')
        tab3_widget = QWidget()
        tab3_layout = QVBoxLayout(tab3_widget)
        tab3_layout.setContentsMargins(0, 0, 0, 0)
        tab3_layout.addWidget(self.canvas_3d_tab)
        install_responsive_canvas(self.canvas_3d_tab, self.fig_3d_tab, base_width_px=900, base_height_px=700)
        self.plot_tabs.addTab(tab3_widget, '3D Trajectory')

        # ── Tab 4: 3D tracking GIF (attitude arrow animation) ──
        gif_widget = QWidget()
        gif_layout = QVBoxLayout(gif_widget)
        gif_layout.setContentsMargins(4, 4, 4, 4)
        self.tracking_gif_status = QLabel(
            'Run numerical tracking simulation to generate the 3D animation.'
        )
        self.tracking_gif_status.setWordWrap(True)
        self.tracking_gif_status.setStyleSheet('color: #555;')
        gif_layout.addWidget(self.tracking_gif_status)
        gif_view_row = QHBoxLayout()
        gif_view_row.addWidget(QLabel('View:'))
        self.tracking_gif_3d_radio = QRadioButton('3D')
        self.tracking_gif_2d_radio = QRadioButton('2D (XZ)')
        self.tracking_gif_3d_radio.setChecked(True)
        self._tracking_gif_view_group = QButtonGroup(self)
        self._tracking_gif_view_group.addButton(self.tracking_gif_3d_radio, 0)
        self._tracking_gif_view_group.addButton(self.tracking_gif_2d_radio, 1)
        self.tracking_gif_3d_radio.toggled.connect(self._on_tracking_gif_view_changed)
        gif_view_row.addWidget(self.tracking_gif_3d_radio)
        gif_view_row.addWidget(self.tracking_gif_2d_radio)
        gif_view_row.addSpacing(16)
        self.chk_tracking_gif_realtime = QCheckBox('Real-time playback')
        self.chk_tracking_gif_realtime.setChecked(True)
        self.chk_tracking_gif_realtime.setToolTip(
            'Checked: GIF duration matches simulation time.\n'
            'Unchecked: play faster using the speed multiplier.'
        )
        self.chk_tracking_gif_realtime.stateChanged.connect(
            self._on_tracking_gif_playback_options_changed
        )
        gif_view_row.addWidget(self.chk_tracking_gif_realtime)
        gif_view_row.addWidget(QLabel('Speed:'))
        self.spin_tracking_gif_speed = QDoubleSpinBox()
        self.spin_tracking_gif_speed.setRange(1.0, 50.0)
        self.spin_tracking_gif_speed.setDecimals(1)
        self.spin_tracking_gif_speed.setSingleStep(1.0)
        self.spin_tracking_gif_speed.setValue(5.0)
        self.spin_tracking_gif_speed.setSuffix('×')
        self.spin_tracking_gif_speed.setToolTip(
            'Playback speed when Real-time is off (simulation time / this factor).'
        )
        self.spin_tracking_gif_speed.setEnabled(False)
        self.spin_tracking_gif_speed.valueChanged.connect(
            self._on_tracking_gif_playback_options_changed
        )
        gif_view_row.addWidget(self.spin_tracking_gif_speed)
        gif_view_row.addStretch(1)
        gif_layout.addLayout(gif_view_row)
        self.tracking_gif_label = QLabel()
        self.tracking_gif_label.setAlignment(Qt.AlignCenter)
        self.tracking_gif_label.setMinimumHeight(320)
        self.tracking_gif_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tracking_gif_label.setScaledContents(False)
        gif_layout.addWidget(self.tracking_gif_label, 1)
        gif_btn_row = QHBoxLayout()
        self.btn_save_tracking_gif = QPushButton('Save GIF…')
        self.btn_save_tracking_gif.setEnabled(False)
        self.btn_save_tracking_gif.clicked.connect(self.save_tracking_gif)
        self.btn_regenerate_tracking_gif = QPushButton('Regenerate GIF')
        self.btn_regenerate_tracking_gif.setToolTip(
            'Rebuild animation from the latest numerical tracking result.'
        )
        self.btn_regenerate_tracking_gif.clicked.connect(self.regenerate_tracking_gif)
        gif_btn_row.addWidget(self.btn_regenerate_tracking_gif)
        gif_btn_row.addWidget(self.btn_save_tracking_gif)
        gif_btn_row.addStretch(1)
        gif_layout.addLayout(gif_btn_row)
        self._plot_tab_gif = gif_widget
        self.plot_tabs.addTab(gif_widget, '3D traj GIF')

        # ── Tab 5: Optimization & tracking metrics ──
        self.fig_metrics = Figure(figsize=(12, 7))
        self.canvas_metrics = FigureCanvas(self.fig_metrics)
        self.canvas_metrics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_metrics.setMinimumSize(280, 220)
        gs_m = GridSpec(2, 2, figure=self.fig_metrics, hspace=0.30, wspace=0.22)
        self.fig_metrics._tvc_gridspec_pads = {
            'left': 0.07, 'right': 0.98, 'top': 0.93, 'bottom': 0.08,
            'hspace': 0.30, 'wspace': 0.22,
        }
        self.fig_metrics.suptitle('Optimization & Tracking Metrics', fontsize=11, fontweight='bold', y=0.98)
        self.ax_metrics_cost = self.fig_metrics.add_subplot(gs_m[0, 0])
        self.ax_metrics_pos_err = self.fig_metrics.add_subplot(gs_m[0, 1])
        self.ax_metrics_vel_err = self.fig_metrics.add_subplot(gs_m[1, 0])
        self.ax_metrics_info = self.fig_metrics.add_subplot(gs_m[1, 1])
        self.ax_metrics_info.axis('off')
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.addWidget(self.canvas_metrics)
        install_responsive_canvas(
            self.canvas_metrics, self.fig_metrics, base_width_px=1100, base_height_px=700,
            layout_mode='gridspec',
        )
        self.plot_tabs.addTab(metrics_widget, 'Metrics')

        # ── Tab: Stability margins (Bode / PM with vs without actuator) ──
        margins_widget = QWidget()
        margins_layout = QVBoxLayout(margins_widget)
        margins_layout.setContentsMargins(4, 4, 4, 4)

        margins_ctrl = QHBoxLayout()
        margins_ctrl.addWidget(QLabel('Axis:'))
        self.margins_axis_combo = QComboBox()
        self.margins_axis_combo.addItem('Pitch / X', AXIS_PITCH)
        self.margins_axis_combo.addItem('Roll / Y', 'roll')
        self.margins_axis_combo.addItem('Yaw / Z', 'yaw')
        self.margins_axis_combo.setToolTip(
            'Plots all four cascade loops for this axis.\n'
            'Pitch/Roll: horizontal via attitude.\n'
            'Yaw: rate/att use yaw; Vel/Pos use vertical Z + thrust τ.'
        )
        margins_ctrl.addWidget(self.margins_axis_combo)
        self.btn_margins_sync = QPushButton('Load from Tracking')
        self.btn_margins_sync.setToolTip(
            'Copy current Tracking controller gains + actuator τ into the spins below.\n'
            'Does not redraw Bode — click Update Bode afterwards.'
        )
        self.btn_margins_sync.clicked.connect(self._sync_margins_controls_from_tracking)
        margins_ctrl.addWidget(self.btn_margins_sync)
        self.btn_update_margins = QPushButton('Update Bode')
        self.btn_update_margins.setToolTip(
            'Recompute all four loop Bodes from the spins and push them back to Tracking.'
        )
        self.btn_update_margins.clicked.connect(self._refresh_stability_margins_tab)
        margins_ctrl.addWidget(self.btn_update_margins)
        margins_ctrl.addStretch(1)
        margins_layout.addLayout(margins_ctrl)

        show_row = QHBoxLayout()
        show_row.addWidget(QLabel('Show:'))
        self._margins_loop_show_cbs = {}
        for loop, label in (
            (LOOP_RATE, 'Rate (角速度)'),
            (LOOP_ATTITUDE, 'Attitude (角度)'),
            (LOOP_VELOCITY, 'Velocity (速度)'),
            (LOOP_POSITION, 'Position (位置)'),
        ):
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setToolTip(f'Show / hide the {label.split()[0]} Bode column')
            cb.stateChanged.connect(self._on_margins_loop_visibility_changed)
            self._margins_loop_show_cbs[loop] = cb
            show_row.addWidget(cb)
        show_row.addStretch(1)
        margins_layout.addLayout(show_row)

        # Editable gains + actuator τ used by the Bode (source of truth on this tab)
        self._margins_ctrl_guard = False
        gains_row = QHBoxLayout()
        self.lbl_margins_rate = QLabel('Rate PID:')
        gains_row.addWidget(self.lbl_margins_rate)
        self.spin_margins_kp_rate = QDoubleSpinBox()
        self.spin_margins_ki_rate = QDoubleSpinBox()
        self.spin_margins_kd_rate = QDoubleSpinBox()
        for spin, name, lo, hi, dec, default in (
            (self.spin_margins_kp_rate, 'Kp', 0.0, 100.0, 2, 15.0),
            (self.spin_margins_ki_rate, 'Ki', 0.0, 50.0, 2, 0.0),
            (self.spin_margins_kd_rate, 'Kd', 0.0, 20.0, 3, 0.0),
        ):
            spin.setRange(lo, hi)
            spin.setDecimals(dec)
            spin.setSingleStep(10 ** (-dec))
            spin.setValue(default)
            spin.setPrefix(f'{name} ')
            spin.setToolTip(f'Rate-loop {name} for the selected axis')
            spin.valueChanged.connect(self._on_margins_control_changed)
            gains_row.addWidget(spin)
        gains_row.addSpacing(12)
        self.lbl_margins_kp_att = QLabel('Att Kp:')
        gains_row.addWidget(self.lbl_margins_kp_att)
        self.spin_margins_kp_att = QDoubleSpinBox()
        self.spin_margins_kp_att.setRange(0.0, 50.0)
        self.spin_margins_kp_att.setDecimals(2)
        self.spin_margins_kp_att.setSingleStep(0.1)
        self.spin_margins_kp_att.setValue(6.5)
        self.spin_margins_kp_att.setToolTip(
            'Attitude P gain [1/deg] (SI effective gain = this value × err_rad).'
        )
        self.spin_margins_kp_att.valueChanged.connect(self._on_margins_control_changed)
        gains_row.addWidget(self.spin_margins_kp_att)
        gains_row.addStretch(1)
        margins_layout.addLayout(gains_row)

        outer_row = QHBoxLayout()
        self.lbl_margins_vel = QLabel('Vel PID:')
        outer_row.addWidget(self.lbl_margins_vel)
        self.spin_margins_kp_vel = QDoubleSpinBox()
        self.spin_margins_ki_vel = QDoubleSpinBox()
        self.spin_margins_kd_vel = QDoubleSpinBox()
        for spin, name, lo, hi, dec, default in (
            (self.spin_margins_kp_vel, 'Kp', 0.0, 50.0, 2, 1.8),
            (self.spin_margins_ki_vel, 'Ki', 0.0, 20.0, 2, 0.0),
            (self.spin_margins_kd_vel, 'Kd', 0.0, 10.0, 3, 0.0),
        ):
            spin.setRange(lo, hi)
            spin.setDecimals(dec)
            spin.setSingleStep(10 ** (-dec))
            spin.setValue(default)
            spin.setPrefix(f'{name} ')
            spin.setToolTip(f'Velocity-loop {name} (XY or Z depending on axis)')
            spin.valueChanged.connect(self._on_margins_control_changed)
            outer_row.addWidget(spin)
        outer_row.addSpacing(12)
        self.lbl_margins_kp_pos = QLabel('Pos Kp:')
        outer_row.addWidget(self.lbl_margins_kp_pos)
        self.spin_margins_kp_pos = QDoubleSpinBox()
        self.spin_margins_kp_pos.setRange(0.0, 20.0)
        self.spin_margins_kp_pos.setDecimals(2)
        self.spin_margins_kp_pos.setSingleStep(0.05)
        self.spin_margins_kp_pos.setValue(1.0)
        self.spin_margins_kp_pos.setToolTip('Position P gain (XY or Z depending on axis)')
        self.spin_margins_kp_pos.valueChanged.connect(self._on_margins_control_changed)
        outer_row.addWidget(self.spin_margins_kp_pos)
        outer_row.addStretch(1)
        margins_layout.addLayout(outer_row)

        act_row = QHBoxLayout()
        self.chk_margins_actuator = QCheckBox('With actuator lag')
        self.chk_margins_actuator.setChecked(True)
        self.chk_margins_actuator.setToolTip(
            'Dashed Bode uses first-order lag 1/(τs+1). Uncheck to compare only the ideal loop.'
        )
        self.chk_margins_actuator.stateChanged.connect(self._on_margins_control_changed)
        act_row.addWidget(self.chk_margins_actuator)
        self.lbl_margins_tau = QLabel('τ_gimbal [s]:')
        act_row.addWidget(self.lbl_margins_tau)
        self.spin_margins_tau = QDoubleSpinBox()
        self.spin_margins_tau.setRange(TAU_MIN, TAU_MAX)
        self.spin_margins_tau.setDecimals(3)
        self.spin_margins_tau.setSingleStep(0.005)
        self.spin_margins_tau.setValue(0.05)
        self.spin_margins_tau.setToolTip('Gimbal τ (pitch/roll) or yaw-torque τ (yaw axis).')
        self.spin_margins_tau.valueChanged.connect(self._on_margins_tau_changed)
        act_row.addWidget(self.spin_margins_tau)
        act_row.addWidget(QLabel('f_c [Hz]:'))
        self.spin_margins_fc = QDoubleSpinBox()
        self.spin_margins_fc.setRange(BW_MIN_HZ, BW_MAX_HZ)
        self.spin_margins_fc.setDecimals(2)
        self.spin_margins_fc.setSingleStep(0.1)
        self.spin_margins_fc.setValue(tau_to_bandwidth_hz(0.05))
        self.spin_margins_fc.setToolTip('−3 dB bandwidth; linked to τ = 1/(2π f_c).')
        self.spin_margins_fc.valueChanged.connect(self._on_margins_fc_changed)
        act_row.addWidget(self.spin_margins_fc)
        self.lbl_margins_tau_thrust = QLabel('τ_thrust [s]:')
        act_row.addWidget(self.lbl_margins_tau_thrust)
        self.spin_margins_tau_thrust = QDoubleSpinBox()
        self.spin_margins_tau_thrust.setRange(TAU_MIN, TAU_MAX)
        self.spin_margins_tau_thrust.setDecimals(3)
        self.spin_margins_tau_thrust.setSingleStep(0.005)
        self.spin_margins_tau_thrust.setValue(0.05)
        self.spin_margins_tau_thrust.setToolTip(
            'Thrust τ for vertical Vel/Pos (Yaw/Z axis only).'
        )
        self.spin_margins_tau_thrust.valueChanged.connect(self._on_margins_control_changed)
        act_row.addWidget(self.spin_margins_tau_thrust)
        self.lbl_margins_act_hint = QLabel(
            'Solid = no lag · Dashed = with τ  |  four loops shown together'
        )
        self.lbl_margins_act_hint.setStyleSheet('color: #555;')
        act_row.addWidget(self.lbl_margins_act_hint)
        act_row.addStretch(1)
        margins_layout.addLayout(act_row)

        self.fig_margins = Figure(figsize=(14, 7))
        self.canvas_margins = FigureCanvas(self.fig_margins)
        self.canvas_margins.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_margins.setMinimumSize(280, 220)
        gs_mg = GridSpec(
            2, 5, figure=self.fig_margins,
            width_ratios=[1.0, 1.0, 1.0, 1.0, 0.95],
            hspace=0.32, wspace=0.22,
        )
        self._margins_gridspec = gs_mg
        self.fig_margins._tvc_gridspec_pads = {
            'left': 0.05, 'right': 0.99, 'top': 0.90, 'bottom': 0.08,
            'hspace': 0.32, 'wspace': 0.22,
        }
        self.fig_margins.suptitle(
            'Stability margins — Rate / Att / Vel / Pos (solid=no lag, dashed=with τ)',
            fontsize=11, fontweight='bold', y=0.98,
        )
        self._margins_loop_axes = {}
        for col, loop in enumerate(LOOP_IDS):
            ax_mag = self.fig_margins.add_subplot(gs_mg[0, col])
            ax_phase = self.fig_margins.add_subplot(gs_mg[1, col], sharex=ax_mag)
            self._margins_loop_axes[loop] = {'ax_mag': ax_mag, 'ax_phase': ax_phase}
        self.ax_margins_info = self.fig_margins.add_subplot(gs_mg[:, 4])
        self.ax_margins_info.axis('off')
        # Keep legacy aliases pointing at Rate (for any old callers)
        self.ax_margins_mag = self._margins_loop_axes[LOOP_RATE]['ax_mag']
        self.ax_margins_phase = self._margins_loop_axes[LOOP_RATE]['ax_phase']
        margins_layout.addWidget(self.canvas_margins)
        install_responsive_canvas(
            self.canvas_margins, self.fig_margins, base_width_px=1300, base_height_px=720,
            layout_mode='gridspec',
        )
        self.margins_axis_combo.currentIndexChanged.connect(self._on_margins_loop_axis_changed)
        self._last_margins_result = None
        self.plot_tabs.addTab(margins_widget, 'Stability margins')
        draw_stability_margins_panels(self._margins_axes_dict(), None)
        # Pull Tracking values once widgets exist (deferred — Tracking panel may
        # still be initializing when this tab is built).
        QTimer.singleShot(0, self._sync_margins_controls_from_tracking)

        # ── Tab: NMP / flatness (3×2) ──
        self.fig_nmp = Figure(figsize=(12, 8))
        self.canvas_nmp = FigureCanvas(self.fig_nmp)
        self.canvas_nmp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_nmp.setMinimumSize(280, 220)
        gs_nmp = GridSpec(3, 2, figure=self.fig_nmp, hspace=0.38, wspace=0.28)
        self.fig_nmp._tvc_gridspec_pads = {
            'left': 0.07, 'right': 0.98, 'top': 0.92, 'bottom': 0.07,
            'hspace': 0.38, 'wspace': 0.28,
        }
        self.ax_nmp_position = self.fig_nmp.add_subplot(gs_nmp[0, 0])
        self.ax_nmp_angle = self.fig_nmp.add_subplot(gs_nmp[0, 1])
        self.ax_nmp_velocity = self.fig_nmp.add_subplot(gs_nmp[1, 0])
        self.ax_nmp_angvel = self.fig_nmp.add_subplot(gs_nmp[1, 1])
        self.ax_nmp_gimbal = self.fig_nmp.add_subplot(gs_nmp[2, 0])
        self.ax_nmp_thrust = self.fig_nmp.add_subplot(gs_nmp[2, 1])
        nmp_widget = QWidget()
        nmp_plot_layout = QVBoxLayout(nmp_widget)
        nmp_plot_layout.setContentsMargins(0, 0, 0, 0)
        nmp_plot_layout.addWidget(self.canvas_nmp)
        install_responsive_canvas(
            self.canvas_nmp, self.fig_nmp, base_width_px=1100, base_height_px=780,
            layout_mode='gridspec',
        )
        self.plot_tabs.addTab(nmp_widget, 'NMP / Flatness')
        self._draw_nmp_plot_tab()

        layout.addWidget(self.plot_tabs, 1)
        self.plot_tabs.currentChanged.connect(lambda _i: self._refresh_all_plot_layouts())

        self.iterations = []
        self.costs = []
        self.stops = []
        self.current_xs = None
        self.current_us = None
        self.segment_costs = {}
        self.segment_iterations = {}
        self._last_tracking_result = None
        self.current_segment_idx = 0
        self._last_figure_save_path = ''
        self._refresh_metrics_tab(opt_only=True)
        self._refresh_opt_info_display()
        return panel

    def _active_plot_figure_and_canvas(self):
        """Return (figure, canvas) for the currently selected plot tab."""
        idx = self.plot_tabs.currentIndex() if hasattr(self, 'plot_tabs') else 0
        mapping = (
            (self.fig, self.canvas),
            (self.fig_states, self.canvas_states),
            (self.fig_3d_tab, self.canvas_3d_tab),
            (None, None),  # GIF tab — no matplotlib figure
            (self.fig_metrics, self.canvas_metrics),
            (self.fig_nmp, self.canvas_nmp),
        )
        if 0 <= idx < len(mapping):
            fig, canvas = mapping[idx]
            if fig is not None:
                return fig, canvas
        return self.fig, self.canvas

    def _refresh_all_plot_layouts(self):
        """Reflow matplotlib panels after window resize."""
        pairs = (
            (self.fig, self.canvas),
            (self.fig_states, self.canvas_states),
            (self.fig_3d_tab, self.canvas_3d_tab),
            (self.fig_metrics, self.canvas_metrics),
            (getattr(self, 'fig_margins', None), getattr(self, 'canvas_margins', None)),
            (self.fig_nmp, self.canvas_nmp),
        )
        for fig, canvas in pairs:
            if fig is not None and canvas is not None:
                apply_responsive_layout(fig, canvas)

    def _cost_loggers_from_summary(self):
        histories = (self.opt_summary or {}).get('segment_cost_histories') or []
        if not histories:
            return None
        return [_SavedCostLogger(h) for h in histories]

    def _refresh_metrics_tab(self, opt_only=False):
        """Redraw Metrics tab (optimization cost + optional tracking errors)."""
        if not hasattr(self, 'ax_metrics_cost'):
            return
        result = None if opt_only else getattr(self, '_last_tracking_result', None)
        draw_tracking_metrics_panels(
            tracking_metrics_axes_dict(self),
            result,
            self.opt_summary,
            self._cost_loggers_from_summary(),
        )
        if hasattr(self, 'canvas_metrics'):
            self.canvas_metrics.draw_idle()
        self._refresh_all_plot_layouts()

    def _margins_selected_axis(self):
        axis = AXIS_PITCH
        if hasattr(self, 'margins_axis_combo'):
            data = self.margins_axis_combo.currentData()
            if data:
                axis = data
        return axis

    def _margins_selected_loop_axis(self):
        """Legacy (loop, axis) pair; loop is unused now (all four plotted)."""
        return LOOP_RATE, self._margins_selected_axis()

    def _margins_axes_dict(self):
        return {
            'ax_by_loop': getattr(self, '_margins_loop_axes', {}),
            'ax_mag': getattr(self, 'ax_margins_mag', None),
            'ax_phase': getattr(self, 'ax_margins_phase', None),
            'ax_info': getattr(self, 'ax_margins_info', None),
            'gridspec': getattr(self, '_margins_gridspec', None),
            'visible_loops': self._margins_loop_visibility(),
        }

    def _margins_loop_visibility(self):
        """Which cascade-loop Bode columns to show."""
        cbs = getattr(self, '_margins_loop_show_cbs', None) or {}
        return {
            loop: (bool(cbs[loop].isChecked()) if loop in cbs else True)
            for loop in LOOP_IDS
        }

    def _on_margins_loop_visibility_changed(self, _state=None):
        """Redraw last Bode result with updated column visibility (no recompute)."""
        if not hasattr(self, 'canvas_margins'):
            return
        result = getattr(self, '_last_margins_result', None)
        draw_stability_margins_panels(self._margins_axes_dict(), result)
        self.canvas_margins.draw_idle()
        self._refresh_all_plot_layouts()

    def _margins_primary_tau_key(self, axis=None):
        """τ edited by the main spin: gimbal (XY) or yaw-torque (yaw)."""
        if axis is None:
            axis = self._margins_selected_axis()
        return 'tau_yaw_torque' if axis == 'yaw' else 'tau_gimbal'

    def _update_margins_gain_visibility(self):
        """All cascade gains stay visible; τ_thrust only for Yaw/Z."""
        if not hasattr(self, 'spin_margins_kp_rate'):
            return
        axis = self._margins_selected_axis()
        for w in (
            getattr(self, 'lbl_margins_rate', None),
            self.spin_margins_kp_rate,
            self.spin_margins_ki_rate,
            self.spin_margins_kd_rate,
            self.lbl_margins_kp_att,
            self.spin_margins_kp_att,
            getattr(self, 'lbl_margins_vel', None),
            getattr(self, 'spin_margins_kp_vel', None),
            getattr(self, 'spin_margins_ki_vel', None),
            getattr(self, 'spin_margins_kd_vel', None),
            getattr(self, 'lbl_margins_kp_pos', None),
            getattr(self, 'spin_margins_kp_pos', None),
        ):
            if w is not None:
                w.setVisible(True)

        yaw = axis == 'yaw'
        if hasattr(self, 'lbl_margins_tau'):
            self.lbl_margins_tau.setText('τ_yaw [s]:' if yaw else 'τ_gimbal [s]:')
        if hasattr(self, 'spin_margins_tau'):
            self.spin_margins_tau.setToolTip(
                'Yaw-torque lag for rate/attitude.' if yaw
                else 'Gimbal lag for pitch/roll cascade (all four loops).'
            )
        thrust_vis = yaw
        if hasattr(self, 'lbl_margins_tau_thrust'):
            self.lbl_margins_tau_thrust.setVisible(thrust_vis)
        if hasattr(self, 'spin_margins_tau_thrust'):
            self.spin_margins_tau_thrust.setVisible(thrust_vis)

        on = self.chk_margins_actuator.isChecked()
        self.spin_margins_tau.setEnabled(on)
        self.spin_margins_fc.setEnabled(on)
        if hasattr(self, 'spin_margins_tau_thrust'):
            self.spin_margins_tau_thrust.setEnabled(on and thrust_vis)

    def _margins_tracking_params_snapshot(self):
        """Live Tracking PX4 gains + actuator settings (for Load from Tracking)."""
        cid = (
            self._current_tracking_controller_id()
            if hasattr(self, 'tracking_controller_combo') else CONTROLLER_PX4
        )
        if cid in (CONTROLLER_PX4, CONTROLLER_FLATNESS):
            return self._collect_tracking_params()
        params = dict(
            self._tracking_params_map_for(self._current_rocket_platform_id()).get(
                CONTROLLER_PX4, default_params_for(CONTROLLER_PX4)
            )
        )
        self._store_actuator_config()
        params.update(self._actuator_params_for_sim())
        return params

    def _sync_margins_controls_from_tracking(self):
        """Copy Tracking gains/τ into the Stability-margins spins (no Bode redraw)."""
        if not hasattr(self, 'spin_margins_kp_rate'):
            return
        axis = self._margins_selected_axis()
        raw = self._margins_tracking_params_snapshot()
        from controllers.px4_params import normalize_px4_params
        gains = normalize_px4_params(raw)
        if axis == 'roll':
            kp, ki, kd = gains['Kp_rate_roll'], gains['Ki_rate_roll'], gains['Kd_rate_roll']
            kp_att = gains['Kp_att_roll_deg']
            kp_v, ki_v, kd_v = gains['Kp_vel_xy'], gains['Ki_vel_xy'], gains['Kd_vel_xy']
            kp_pos = gains['Kp_pos_xy']
        elif axis == 'yaw':
            kp, ki, kd = gains['Kp_rate_yaw'], gains['Ki_rate_yaw'], gains['Kd_rate_yaw']
            kp_att = gains['Kp_att_yaw_deg']
            kp_v, ki_v, kd_v = gains['Kp_vel_z'], gains['Ki_vel_z'], gains['Kd_vel_z']
            kp_pos = gains['Kp_pos_z']
        else:
            kp, ki, kd = gains['Kp_rate_pitch'], gains['Ki_rate_pitch'], gains['Kd_rate_pitch']
            kp_att = gains['Kp_att_pitch_deg']
            kp_v, ki_v, kd_v = gains['Kp_vel_xy'], gains['Ki_vel_xy'], gains['Kd_vel_xy']
            kp_pos = gains['Kp_pos_xy']
        act = actuator_config_from_params(raw)
        tau_primary = float(act.get(self._margins_primary_tau_key(axis), 0.05))
        tau_thrust = float(act.get('tau_thrust', 0.05))
        enabled = bool(act.get('act_dyn_enable', False))

        self._margins_ctrl_guard = True
        try:
            self.spin_margins_kp_rate.setValue(float(kp))
            self.spin_margins_ki_rate.setValue(float(ki))
            self.spin_margins_kd_rate.setValue(float(kd))
            self.spin_margins_kp_att.setValue(float(kp_att))
            if hasattr(self, 'spin_margins_kp_vel'):
                self.spin_margins_kp_vel.setValue(float(kp_v))
                self.spin_margins_ki_vel.setValue(float(ki_v))
                self.spin_margins_kd_vel.setValue(float(kd_v))
            if hasattr(self, 'spin_margins_kp_pos'):
                self.spin_margins_kp_pos.setValue(float(kp_pos))
            self.chk_margins_actuator.setChecked(enabled or tau_primary > 0.0)
            self.spin_margins_tau.setValue(max(float(tau_primary), TAU_MIN))
            self.spin_margins_fc.setValue(tau_to_bandwidth_hz(float(self.spin_margins_tau.value())))
            if hasattr(self, 'spin_margins_tau_thrust'):
                self.spin_margins_tau_thrust.setValue(max(float(tau_thrust), TAU_MIN))
        finally:
            self._margins_ctrl_guard = False
        self._update_margins_gain_visibility()

    def _on_margins_loop_axis_changed(self, _index=None):
        # Reload axis-specific gains/τ into spins only; Bode waits for Update.
        self._sync_margins_controls_from_tracking()

    def _on_margins_control_changed(self, _value=None):
        if getattr(self, '_margins_ctrl_guard', False):
            return
        self._update_margins_gain_visibility()

    def _on_margins_tau_changed(self, _value=None):
        if getattr(self, '_margins_ctrl_guard', False):
            return
        self._margins_ctrl_guard = True
        try:
            tau = float(self.spin_margins_tau.value())
            self.spin_margins_fc.setValue(tau_to_bandwidth_hz(tau))
        finally:
            self._margins_ctrl_guard = False
        self._on_margins_control_changed()

    def _on_margins_fc_changed(self, _value=None):
        if getattr(self, '_margins_ctrl_guard', False):
            return
        self._margins_ctrl_guard = True
        try:
            tau = bandwidth_hz_to_tau(float(self.spin_margins_fc.value()))
            self.spin_margins_tau.setValue(tau)
        finally:
            self._margins_ctrl_guard = False
        self._on_margins_control_changed()

    def _margins_params_from_controls(self):
        """Build analyze params from the Stability-margins spins (all four loops)."""
        axis = self._margins_selected_axis()
        params = dict(self._margins_tracking_params_snapshot())
        kp = float(self.spin_margins_kp_rate.value())
        ki = float(self.spin_margins_ki_rate.value())
        kd = float(self.spin_margins_kd_rate.value())
        kp_att = float(self.spin_margins_kp_att.value())
        kp_v = float(self.spin_margins_kp_vel.value()) if hasattr(self, 'spin_margins_kp_vel') else 0.0
        ki_v = float(self.spin_margins_ki_vel.value()) if hasattr(self, 'spin_margins_ki_vel') else 0.0
        kd_v = float(self.spin_margins_kd_vel.value()) if hasattr(self, 'spin_margins_kd_vel') else 0.0
        kp_pos = float(self.spin_margins_kp_pos.value()) if hasattr(self, 'spin_margins_kp_pos') else 0.0
        tau = float(self.spin_margins_tau.value())
        use_act = bool(self.chk_margins_actuator.isChecked())

        if axis == 'roll':
            params.update({
                'Kp_rate_roll': kp, 'Ki_rate_roll': ki, 'Kd_rate_roll': kd,
                'Kp_att_roll_deg': kp_att,
                'Kp_rate_rp': kp, 'Ki_rate_rp': ki, 'Kd_rate_rp': kd,
                'Kp_att_rp_deg': kp_att,
                'Kp_vel_xy': kp_v, 'Ki_vel_xy': ki_v, 'Kd_vel_xy': kd_v,
                'Kp_pos_xy': kp_pos,
            })
        elif axis == 'yaw':
            params.update({
                'Kp_rate_yaw': kp, 'Ki_rate_yaw': ki, 'Kd_rate_yaw': kd,
                'Kp_att_yaw_deg': kp_att,
                'Kp_vel_z': kp_v, 'Ki_vel_z': ki_v, 'Kd_vel_z': kd_v,
                'Kp_pos_z': kp_pos,
            })
        else:
            params.update({
                'Kp_rate_pitch': kp, 'Ki_rate_pitch': ki, 'Kd_rate_pitch': kd,
                'Kp_att_pitch_deg': kp_att,
                'Kp_rate_rp': kp, 'Ki_rate_rp': ki, 'Kd_rate_rp': kd,
                'Kp_att_rp_deg': kp_att,
                'Kp_vel_xy': kp_v, 'Ki_vel_xy': ki_v, 'Kd_vel_xy': kd_v,
                'Kp_pos_xy': kp_pos,
            })
        params[self._margins_primary_tau_key(axis)] = tau
        if hasattr(self, 'spin_margins_tau_thrust'):
            params['tau_thrust'] = float(self.spin_margins_tau_thrust.value())
        params['act_dyn_enable'] = use_act
        params['_margins_axis'] = axis
        return params

    def _apply_margins_controls_to_tracking(self):
        """Push margins spins back into Tracking controller + actuator widgets."""
        if not hasattr(self, 'spin_margins_kp_rate'):
            return
        axis = self._margins_selected_axis()
        kp = float(self.spin_margins_kp_rate.value())
        ki = float(self.spin_margins_ki_rate.value())
        kd = float(self.spin_margins_kd_rate.value())
        kp_att = float(self.spin_margins_kp_att.value())
        kp_v = float(self.spin_margins_kp_vel.value()) if hasattr(self, 'spin_margins_kp_vel') else None
        ki_v = float(self.spin_margins_ki_vel.value()) if hasattr(self, 'spin_margins_ki_vel') else None
        kd_v = float(self.spin_margins_kd_vel.value()) if hasattr(self, 'spin_margins_kd_vel') else None
        kp_pos = float(self.spin_margins_kp_pos.value()) if hasattr(self, 'spin_margins_kp_pos') else None
        tau = float(self.spin_margins_tau.value())
        tau_thrust = (
            float(self.spin_margins_tau_thrust.value())
            if hasattr(self, 'spin_margins_tau_thrust') else None
        )
        use_act = bool(self.chk_margins_actuator.isChecked())

        params_map = self._tracking_params_map_for(self._current_rocket_platform_id())
        px4 = params_map.setdefault(CONTROLLER_PX4, default_params_for(CONTROLLER_PX4))
        flat = params_map.setdefault(CONTROLLER_FLATNESS, default_params_for(CONTROLLER_FLATNESS))
        for store in (px4, flat):
            if axis == 'roll':
                store.update({
                    'Kp_rate_roll': kp, 'Ki_rate_roll': ki, 'Kd_rate_roll': kd,
                    'Kp_att_roll_deg': kp_att,
                })
                if store.get('share_rp_gains', True):
                    store.update({
                        'Kp_rate_rp': kp, 'Ki_rate_rp': ki, 'Kd_rate_rp': kd,
                        'Kp_att_rp_deg': kp_att,
                        'Kp_rate_pitch': kp, 'Ki_rate_pitch': ki, 'Kd_rate_pitch': kd,
                        'Kp_att_pitch_deg': kp_att,
                    })
                if kp_v is not None:
                    store.update({
                        'Kp_vel_xy': kp_v, 'Ki_vel_xy': ki_v, 'Kd_vel_xy': kd_v,
                    })
                if kp_pos is not None:
                    store['Kp_pos_xy'] = kp_pos
            elif axis == 'yaw':
                store.update({
                    'Kp_rate_yaw': kp, 'Ki_rate_yaw': ki, 'Kd_rate_yaw': kd,
                    'Kp_att_yaw_deg': kp_att,
                })
                if kp_v is not None:
                    store.update({
                        'Kp_vel_z': kp_v, 'Ki_vel_z': ki_v, 'Kd_vel_z': kd_v,
                    })
                if kp_pos is not None:
                    store['Kp_pos_z'] = kp_pos
            else:
                store.update({
                    'Kp_rate_pitch': kp, 'Ki_rate_pitch': ki, 'Kd_rate_pitch': kd,
                    'Kp_att_pitch_deg': kp_att,
                })
                if store.get('share_rp_gains', True):
                    store.update({
                        'Kp_rate_rp': kp, 'Ki_rate_rp': ki, 'Kd_rate_rp': kd,
                        'Kp_att_rp_deg': kp_att,
                        'Kp_rate_roll': kp, 'Ki_rate_roll': ki, 'Kd_rate_roll': kd,
                        'Kp_att_roll_deg': kp_att,
                    })
                if kp_v is not None:
                    store.update({
                        'Kp_vel_xy': kp_v, 'Ki_vel_xy': ki_v, 'Kd_vel_xy': kd_v,
                    })
                if kp_pos is not None:
                    store['Kp_pos_xy'] = kp_pos

        cid = self._current_tracking_controller_id() if hasattr(self, 'tracking_controller_combo') else None
        if cid in (CONTROLLER_PX4, CONTROLLER_FLATNESS):
            self._rebuild_tracking_param_widgets()
        if hasattr(self, 'nmp_params_groups_layout'):
            self._rebuild_nmp_tracking_param_widgets()

        if hasattr(self, 'act_dyn_enable_cb'):
            self._act_dyn_updating = True
            try:
                self.act_dyn_enable_cb.setChecked(use_act)
                keys_vals = [(self._margins_primary_tau_key(axis), tau)]
                if axis == 'yaw' and tau_thrust is not None:
                    keys_vals.append(('tau_thrust', tau_thrust))
                act = self._tracking_config.setdefault(
                    'actuator', default_actuator_tracking_config()
                )
                act['enabled'] = use_act
                for key, val in keys_vals:
                    if key in getattr(self, '_act_dyn_tau_spins', {}):
                        self._act_dyn_tau_spins[key].setValue(val)
                        if key in self._act_dyn_bw_spins:
                            self._act_dyn_bw_spins[key].setValue(tau_to_bandwidth_hz(val))
                    act[key] = val
                self.act_dyn_lag_detail.setEnabled(use_act)
                self.act_dyn_lag_detail.setVisible(use_act)
            finally:
                self._act_dyn_updating = False
            self._store_actuator_config()

    def _refresh_stability_margins_tab(self, apply_to_tracking=True):
        """Recompute all four open-loop Bodes / PM from margins-tab spins."""
        if not hasattr(self, 'ax_margins_mag'):
            return
        axis = self._margins_selected_axis()
        try:
            if not hasattr(self, 'spin_margins_kp_rate'):
                params = self._margins_tracking_params_snapshot()
            else:
                params = self._margins_params_from_controls()
                if apply_to_tracking:
                    self._apply_margins_controls_to_tracking()
            phy = self._physics_dict_for_tracking()
            force_act = bool(params.get('act_dyn_enable', True))
            result = analyze_all_loops(
                phy, params, axis=axis, force_actuator=force_act,
            )
            self._last_margins_result = result
        except Exception as e:
            self._last_margins_result = None
            draw_stability_margins_panels(self._margins_axes_dict(), None)
            self.canvas_margins.draw_idle()
            if hasattr(self, 'status_text'):
                self.status_text.append(f'Stability margins error: {e}')
            return
        draw_stability_margins_panels(self._margins_axes_dict(), result)
        self.canvas_margins.draw_idle()
        self._refresh_all_plot_layouts()
        if hasattr(self, 'status_text'):
            parts = []
            for loop in LOOP_IDS:
                r = (result.get('by_loop') or {}).get(loop) or {}
                wo = r.get('without') or {}
                wa = r.get('with_actuator') or {}
                pm0, pm1 = wo.get('pm_deg', float('nan')), wa.get('pm_deg', float('nan'))
                if np.isfinite(pm0) and np.isfinite(pm1):
                    parts.append(f'{loop[0].upper()}:{pm0:.0f}→{pm1:.0f}°')
            self.status_text.append(
                f"Margins ({result.get('axis')}): " + '  '.join(parts)
            )

    def _states_series_visibility(self):
        """Which plan / sim / cascade series to draw on the States tab."""
        return {
            'show_plan': (
                bool(self.chk_states_show_plan.isChecked())
                if hasattr(self, 'chk_states_show_plan') else True
            ),
            'show_sim': (
                bool(self.chk_states_show_sim.isChecked())
                if hasattr(self, 'chk_states_show_sim') else True
            ),
            'show_cascade': (
                bool(self.chk_states_show_cascade.isChecked())
                if hasattr(self, 'chk_states_show_cascade') else True
            ),
        }

    def _on_states_series_visibility_changed(self, _state=None):
        result = getattr(self, '_last_tracking_result', None)
        if result is None or not hasattr(self, 'ax_trk_pos'):
            return
        result = self._enrich_tracking_result_with_loop_margins(result)
        plan = getattr(self, 'last_trajectory', None)
        draw_tracking_state_panels(
            tracking_state_axes_dict(self),
            result,
            plan,
            quat_to_euler_fn=quat_to_euler,
            **self._states_series_visibility(),
        )
        self.canvas_states.draw_idle()
        self._refresh_all_plot_layouts()

    def _enrich_tracking_result_with_loop_margins(self, result):
        """Attach continuous Bode / PM tags used in States panel titles."""
        if not result or not isinstance(result, dict):
            return result
        if result.get('loop_margins'):
            return result
        try:
            params = self._collect_tracking_params()
            phy = self._physics_dict_for_tracking()
            force_act = bool(params.get('act_dyn_enable', False))
            result['loop_margins'] = {
                'pitch': analyze_all_loops(
                    phy, params, axis='pitch', force_actuator=force_act,
                ),
                'yaw': analyze_all_loops(
                    phy, params, axis='yaw', force_actuator=force_act,
                ),
            }
        except Exception as e:
            if hasattr(self, 'status_text'):
                self.status_text.append(
                    f'States loop-margin titles skipped: {e}'
                )
        return result

    def _draw_tracking_plot_tabs(self, result):
        """Update States, 3D, and Metrics tabs after numerical tracking."""
        plan = getattr(self, 'last_trajectory', None)
        result = self._enrich_tracking_result_with_loop_margins(result)
        draw_tracking_state_panels(
            tracking_state_axes_dict(self),
            result,
            plan,
            quat_to_euler_fn=quat_to_euler,
            **self._states_series_visibility(),
        )
        draw_tracking_3d_panel(self.ax_3d_trk, result, plan)
        draw_tracking_metrics_panels(
            tracking_metrics_axes_dict(self),
            result,
            self.opt_summary,
            self._cost_loggers_from_summary(),
        )
        self.canvas_states.draw_idle()
        self.canvas_3d_tab.draw_idle()
        self.canvas_metrics.draw_idle()
        self._refresh_all_plot_layouts()
        self._start_tracking_gif_generation(result)

    def _current_tracking_gif_view_mode(self):
        if hasattr(self, 'tracking_gif_2d_radio') and self.tracking_gif_2d_radio.isChecked():
            return '2d'
        return '3d'

    def _tracking_gif_playback_speed(self):
        """1.0 = real-time; >1 accelerates when Real-time checkbox is off."""
        if (
            hasattr(self, 'chk_tracking_gif_realtime')
            and self.chk_tracking_gif_realtime.isChecked()
        ):
            return 1.0
        if hasattr(self, 'spin_tracking_gif_speed'):
            return float(self.spin_tracking_gif_speed.value())
        return 1.0

    def _on_tracking_gif_view_changed(self, _checked=False):
        if self._last_tracking_result is not None:
            self._start_tracking_gif_generation(self._last_tracking_result)

    def _on_tracking_gif_playback_options_changed(self, _value=None):
        if hasattr(self, 'spin_tracking_gif_speed') and hasattr(
            self, 'chk_tracking_gif_realtime'
        ):
            self.spin_tracking_gif_speed.setEnabled(
                not self.chk_tracking_gif_realtime.isChecked()
            )
        if self._last_tracking_result is not None:
            self._start_tracking_gif_generation(self._last_tracking_result)

    def _start_tracking_gif_generation(self, result=None):
        """Render tracking GIF in a background thread."""
        result = result or getattr(self, '_last_tracking_result', None)
        if result is None:
            return
        if self._tracking_gif_thread is not None and self._tracking_gif_thread.isRunning():
            # Options changed while rendering — rebuild once the current job finishes.
            self._tracking_gif_regen_pending = True
            return
        self._tracking_gif_regen_pending = False
        plan = getattr(self, 'last_trajectory', None)
        view_mode = self._current_tracking_gif_view_mode()
        speed = self._tracking_gif_playback_speed()
        label = '2D' if view_mode == '2d' else '3D'
        if speed <= 1.0 + 1e-9:
            timing = 'real-time'
        else:
            timing = f'{speed:.1f}× speed'
        self.tracking_gif_status.setText(f'Generating {label} tracking GIF ({timing})…')
        self.btn_regenerate_tracking_gif.setEnabled(False)
        self.btn_save_tracking_gif.setEnabled(False)
        if self._tracking_gif_movie is not None:
            self._tracking_gif_movie.stop()
            self._tracking_gif_movie = None
        self._tracking_gif_thread = TrackingGifThread(
            result, plan, DEFAULT_TRACKING_GIF_PATH,
            view_mode=view_mode, playback_speed=speed,
        )
        self._tracking_gif_thread.finished.connect(self._on_tracking_gif_finished)
        self._tracking_gif_thread.error.connect(self._on_tracking_gif_error)
        self._tracking_gif_thread.start()

    def _update_tracking_gif_scaled_size(self):
        movie = getattr(self, '_tracking_gif_movie', None)
        label = getattr(self, 'tracking_gif_label', None)
        if movie is None or label is None:
            return
        fw = movie.frameRect().width()
        fh = movie.frameRect().height()
        if fw <= 0 or fh <= 0:
            return
        lw, lh = label.width(), label.height()
        if lw <= 0 or lh <= 0:
            return
        scale = min(lw / fw, lh / fh)
        movie.setScaledSize(QSize(max(1, int(fw * scale)), max(1, int(fh * scale))))

    def _display_tracking_gif(self, path):
        if not path or not os.path.isfile(path):
            self.tracking_gif_status.setText(f'GIF not found: {path}')
            return
        self._tracking_gif_path = os.path.abspath(path)
        movie = QMovie(self._tracking_gif_path)
        if movie.isValid():
            if self._tracking_gif_movie is not None:
                self._tracking_gif_movie.stop()
            self._tracking_gif_movie = movie
            self.tracking_gif_label.setMovie(self._tracking_gif_movie)
            self._update_tracking_gif_scaled_size()
            self._tracking_gif_movie.start()
            speed = self._tracking_gif_playback_speed()
            if speed <= 1.0 + 1e-9:
                timing = 'real-time'
            else:
                timing = f'{speed:.1f}× speed'
            view = self._current_tracking_gif_view_mode().upper()
            self.tracking_gif_status.setText(
                f'{view} tracking GIF ({timing}) — red: body +Z (nose), '
                f'orange: thrust direction (length ∝ T)\n'
                f'{self._tracking_gif_path}'
            )
            self.btn_save_tracking_gif.setEnabled(True)
        else:
            self.tracking_gif_status.setText(f'Could not load GIF: {path}')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_tracking_gif_scaled_size()
        self._refresh_all_plot_layouts()

    def _on_tracking_gif_finished(self, path):
        self.btn_regenerate_tracking_gif.setEnabled(True)
        self._display_tracking_gif(path)
        if hasattr(self, 'status_text'):
            self.status_text.append(f'Tracking GIF saved: {path}')
        if getattr(self, '_tracking_gif_regen_pending', False):
            self._tracking_gif_regen_pending = False
            QTimer.singleShot(0, self._start_tracking_gif_generation)

    def _on_tracking_gif_error(self, msg):
        self.btn_regenerate_tracking_gif.setEnabled(True)
        self.tracking_gif_status.setText('GIF generation failed.')
        if hasattr(self, 'status_text'):
            self.status_text.append(f'Tracking GIF error: {msg.splitlines()[0]}')
        if getattr(self, '_tracking_gif_regen_pending', False):
            self._tracking_gif_regen_pending = False
            QTimer.singleShot(0, self._start_tracking_gif_generation)

    def regenerate_tracking_gif(self):
        if getattr(self, '_last_tracking_result', None) is None:
            QMessageBox.information(
                self, 'No tracking result',
                'Run numerical tracking simulation first.',
            )
            return
        self._start_tracking_gif_generation(self._last_tracking_result)

    def save_tracking_gif(self):
        src = getattr(self, '_tracking_gif_path', None)
        if not src or not os.path.isfile(src):
            QMessageBox.warning(self, 'No GIF', 'Generate the tracking GIF first.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save tracking GIF', src, 'GIF (*.gif);;All files (*)',
        )
        if not path:
            return
        if not path.lower().endswith('.gif'):
            path += '.gif'
        try:
            import shutil
            shutil.copy2(src, path)
            self.status_text.append(f'Saved tracking GIF to {path}')
        except OSError as e:
            QMessageBox.critical(self, 'Save failed', str(e))

    def save_figure(self):
        """Save the active plot tab figure to disk (or copy GIF on GIF tab)."""
        if hasattr(self, 'plot_tabs') and self.plot_tabs.currentWidget() is getattr(
            self, '_plot_tab_gif', None
        ):
            self.save_tracking_gif()
            return
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
        fig, canvas = self._active_plot_figure_and_canvas()
        try:
            canvas.draw()
            fig.savefig(path, dpi=150, bbox_inches='tight')
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

    def _current_optimization_context(self):
        """Active platform × trajectory × mode × method."""
        combo_index = (
            int(self.trajectory_preset_combo.currentIndex())
            if hasattr(self, 'trajectory_preset_combo')
            else 0
        )
        method_index = (
            int(self.method_combo.currentIndex())
            if hasattr(self, 'method_combo')
            else getattr(self, '_normal_method_index', METHOD_NORMAL_DEFAULT_INDEX)
        )
        return {
            'platform_id': self._current_rocket_platform_id(),
            'combo_index': combo_index,
            'mode_key': self._current_traj_opt_mode_key(),
            'method_index': method_index,
        }

    def _mode_cache_key(
        self, combo_index=None, mode_key=None, platform_id=None, method_index=None,
    ):
        ctx = self._current_optimization_context()
        if combo_index is None:
            combo_index = ctx['combo_index']
        if mode_key is None:
            mode_key = ctx['mode_key']
        if platform_id is None:
            platform_id = ctx['platform_id']
        if method_index is None:
            method_index = ctx['method_index']
        return optimization_cache_key(platform_id, combo_index, mode_key, method_index)

    def _restore_optimization_for_current_context(
        self, quiet=False, required_fingerprint=None,
    ):
        ctx = self._current_optimization_context()
        return self._restore_optimization_for_slot_and_mode(
            ctx['combo_index'],
            mode_key=ctx['mode_key'],
            platform_id=ctx['platform_id'],
            method_index=ctx['method_index'],
            quiet=quiet,
            required_fingerprint=required_fingerprint,
        )

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

    def _cache_results_for_slot_mode(
        self, combo_index, mode_key, platform_id=None, method_index=None,
    ):
        """Store in-memory results under platform × slot × mode × method."""
        if not hasattr(self, '_trajectory_mode_cache'):
            self._trajectory_mode_cache = {}
        ctx = self._current_optimization_context()
        if platform_id is None:
            platform_id = ctx['platform_id']
        if method_index is None:
            method_index = ctx['method_index']
        entry = self._clone_trajectory_for_cache(self.last_trajectory, self.opt_summary)
        if entry is not None:
            if not optimization_summary_matches_context(
                entry.get('opt_summary'), platform_id, mode_key, method_index,
            ):
                return
            self._trajectory_mode_cache[
                optimization_cache_key(platform_id, combo_index, mode_key, method_index)
            ] = entry

    def _cache_current_mode_results(self):
        """Keep in-memory results for the current optimization context."""
        ctx = self._current_optimization_context()
        self._cache_results_for_slot_mode(
            ctx['combo_index'],
            ctx['mode_key'],
            platform_id=ctx['platform_id'],
            method_index=ctx['method_index'],
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
            self._cache_results_for_slot_mode(
                combo_idx, old_mode_key,
                platform_id=self._current_rocket_platform_id(),
                method_index=self.method_combo.currentIndex(),
            )
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
            if not self._restore_optimization_for_current_context(quiet=True):
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
    
    def _method_params_for_platform(self, index):
        """Method defaults with thrust limits taken from the active rocket platform."""
        params = dict(self.DEFAULT_PARAMS.get(index, self.DEFAULT_PARAMS[0]))
        plat = default_constraints(self._current_rocket_platform_id())
        for key in ('T_min', 'T_max', 'tau_yaw_max', 'v_horizontal_max', 'v_vertical_max'):
            params[key] = plat[key]
        return params

    def on_method_changed(self, index):
        """Load parameter defaults when optimization method changes"""
        params = self._method_params_for_platform(index)
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
        self.T_min_thrust.setValue(params.get("T_min", 0.0))
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
            "Method 7 (Acados free-tf EXTERNAL)", "Method 8 (Differential flatness)",
        ]
        if hasattr(self, "status_text") and self.status_text is not None:
            self.status_text.append(
                f"Parameters loaded for {method_names[index] if index < len(method_names) else 'Unknown'}"
            )
        self._refresh_min_time_duration_group_visible(index)
        if not self._restore_optimization_for_current_context(quiet=True):
            self._clear_optimization_results(clear_plots=True)

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
        ctx = self._current_optimization_context()
        mode_label = 'Minimum time' if ctx['mode_key'] == 'min_time' else 'Normal'
        paths = resolve_trajectory_artifact_paths(
            ctx['combo_index'], ctx['mode_key'], ctx['platform_id'], ctx['method_index'],
        )
        label = trajectory_combo_label(ctx['combo_index'])
        self.trajectory_storage_path_label.setText(
            f'{label} / {platform_label(ctx["platform_id"])} / method {ctx["method_index"] + 1} '
            f'({mode_label})\n'
            f'  waypoints: {paths["json"]}\n'
            f'  CSV: {paths["csv"]}'
        )

    def _opt_info_empty_message(self):
        ctx = self._current_optimization_context()
        mode_label = 'Minimum time' if ctx['mode_key'] == 'min_time' else 'Normal'
        return (
            f'No saved result for {platform_label(ctx["platform_id"])} / '
            f'method {ctx["method_index"] + 1} ({mode_label}) — '
            f'run Start Optimization (cached automatically when inputs match).'
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
        self._cache_results_for_slot_mode(
            old_combo, old_mode_key,
            platform_id=self._current_rocket_platform_id(),
            method_index=self.method_combo.currentIndex(),
        )
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
            self._refresh_all_plot_layouts()

    def _clear_optimization_plots(self):
        """Clear trajectory / cost axes (keep titles)."""
        if not self._display_panels_ready():
            return
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

    def _apply_cached_trajectory_entry(
        self, entry, combo_index, mode_key, platform_id=None, method_index=None, quiet=False,
    ):
        self.last_trajectory = entry['last_trajectory']
        self.opt_summary = entry.get('opt_summary')
        ctx = self._current_optimization_context()
        if platform_id is None:
            platform_id = ctx['platform_id']
        if method_index is None:
            method_index = ctx['method_index']
        paths = resolve_trajectory_artifact_paths(
            combo_index, mode_key, platform_id, method_index,
        )
        if os.path.isfile(paths['csv']):
            self.last_csv_path = paths['csv']
        self._enable_trajectory_export_buttons(True)
        self._restore_plot_from_last_trajectory()
        if not quiet and hasattr(self, 'status_text'):
            label = trajectory_combo_label(combo_index)
            mode_label = 'Minimum time' if mode_key == 'min_time' else 'Normal'
            plat = platform_label(platform_id)
            iters = (self.opt_summary or {}).get('total_iters', '?')
            path_len = (self.opt_summary or {}).get('path_length_m', 0.0)
            method_idx = (self.opt_summary or {}).get('method', method_index)
            self.status_text.append(
                f'Loaded {label} / {plat} / method {int(method_idx) + 1} ({mode_label}): '
                f'{iters} iters, {path_len:.2f} m path.'
            )
        return True

    def _restore_optimization_for_slot_and_mode(
        self,
        combo_index,
        mode_key=None,
        platform_id=None,
        method_index=None,
        quiet=False,
        required_fingerprint=None,
    ):
        """Load in-memory or on-disk optimization for one full context."""
        combo_index = int(combo_index)
        ctx = self._current_optimization_context()
        mode_key = mode_key or ctx['mode_key']
        platform_id = platform_id if platform_id is not None else ctx['platform_id']
        method_index = int(method_index if method_index is not None else ctx['method_index'])
        cache_key = optimization_cache_key(platform_id, combo_index, mode_key, method_index)
        cached = self._trajectory_mode_cache.get(cache_key)
        if cached and cached.get('last_trajectory', {}).get('xs') is not None:
            summary = cached.get('opt_summary')
            if optimization_summary_matches_context(summary, platform_id, mode_key, method_index):
                fp = (summary or {}).get('input_fingerprint')
                if required_fingerprint is None or fp == required_fingerprint:
                    return self._apply_cached_trajectory_entry(
                        cached, combo_index, mode_key, platform_id, method_index, quiet=quiet,
                    )
            self._trajectory_mode_cache.pop(cache_key, None)
        return self._try_restore_saved_optimization(
            combo_index,
            mode_key=mode_key,
            platform_id=platform_id,
            method_index=method_index,
            quiet=quiet,
            required_fingerprint=required_fingerprint,
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
            'mode_key': self._current_traj_opt_mode_key(),
            'platform_id': self._current_rocket_platform_id(),
            'input_fingerprint': optimization_input_fingerprint(self.get_parameters()),
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

    def _display_panels_ready(self):
        """True once the matplotlib trajectory/cost axes exist."""
        return hasattr(self, 'ax_cost') and hasattr(self, 'ax_3d')

    def _restore_plot_from_last_trajectory(self):
        """Redraw all panels from ``self.last_trajectory``."""
        if not self._display_panels_ready():
            return
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
        draw_tracking_3d_panel(self.ax_3d_trk, self._last_tracking_result, traj)
        if hasattr(self, 'canvas_3d_tab'):
            self.canvas_3d_tab.draw_idle()
        self._refresh_metrics_tab(opt_only=(self._last_tracking_result is None))

    def _try_restore_saved_optimization(
        self,
        combo_index,
        json_data=None,
        mode_key=None,
        platform_id=None,
        method_index=None,
        quiet=False,
        required_fingerprint=None,
    ):
        """Load saved CSV/NPZ optimization for one platform × trajectory × mode × method."""
        combo_index = int(combo_index)
        ctx = self._current_optimization_context()
        mode_key = mode_key or ctx['mode_key']
        platform_id = platform_id if platform_id is not None else ctx['platform_id']
        method_index = int(method_index if method_index is not None else ctx['method_index'])
        paths = resolve_trajectory_artifact_paths(
            combo_index, mode_key, platform_id, method_index,
        )
        cache_key = optimization_cache_key(platform_id, combo_index, mode_key, method_index)
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
        summary = optimization_summary_lookup(json_data, platform_id, mode_key, method_index)
        if not summary or not optimization_summary_matches_context(
            summary, platform_id, mode_key, method_index,
        ):
            self._trajectory_mode_cache.pop(cache_key, None)
            self._clear_optimization_results(clear_plots=True)
            return False
        fp = summary.get('input_fingerprint')
        if required_fingerprint is not None and fp != required_fingerprint:
            self._trajectory_mode_cache.pop(cache_key, None)
            self._clear_optimization_results(clear_plots=True)
            return False
        loaded = self._load_trajectory_arrays_from_paths(paths, summary, mode_key)
        if loaded is None or loaded.get('xs') is None:
            self._trajectory_mode_cache.pop(cache_key, None)
            self._clear_optimization_results(clear_plots=True)
            return False
        try:
            self.opt_summary = dict(summary)
            self.last_trajectory = loaded
            if os.path.isfile(paths['csv']):
                self.last_csv_path = paths['csv']
            self._cache_results_for_slot_mode(
                combo_index, mode_key, platform_id, method_index,
            )
            return self._apply_cached_trajectory_entry(
                self._trajectory_mode_cache[cache_key],
                combo_index,
                mode_key,
                platform_id,
                method_index,
                quiet=quiet,
            )
        except (OSError, ValueError, KeyError) as e:
            if not quiet and hasattr(self, 'status_text'):
                self.status_text.append(f'Could not load saved trajectory: {e}')
            self._trajectory_mode_cache.pop(cache_key, None)
            self._clear_optimization_results(clear_plots=True)
            return False

    def _current_trajectory_csv_path(self):
        """CSV path for the current optimization context."""
        if not hasattr(self, 'trajectory_preset_combo'):
            return DEFAULT_TRAJ_CSV_PATH
        ctx = self._current_optimization_context()
        return resolve_trajectory_artifact_paths(
            ctx['combo_index'],
            ctx['mode_key'],
            ctx['platform_id'],
            ctx['method_index'],
        )['csv']

    def _persist_optimization_artifacts(self, quiet=False):
        """Write waypoints JSON + CSV + NPZ + summary for the current context."""
        self._sync_waypoints_from_table()
        if not self.last_trajectory or self.last_trajectory.get('xs') is None:
            return False
        payload = self._build_trajectory_csv_payload()
        if payload is None:
            return False

        ctx = self._current_optimization_context()
        combo_index = ctx['combo_index']
        mode_key = ctx['mode_key']
        platform_id = ctx['platform_id']
        method_index = ctx['method_index']
        paths = trajectory_artifact_paths(combo_index, mode_key, platform_id, method_index)
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
        summary['platform_id'] = platform_id
        summary['method'] = int(method_index)
        if not summary.get('input_fingerprint'):
            summary['input_fingerprint'] = optimization_input_fingerprint(self.get_parameters())
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

            existing = {}
            if os.path.isfile(paths['json']):
                try:
                    with open(paths['json'], 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                except (OSError, json.JSONDecodeError, TypeError):
                    existing = {}
            json_payload = merge_optimization_summary_into_json(
                existing, platform_id, mode_key, method_index, summary,
            )
            json_payload['name'] = label
            json_payload['waypoints'] = waypoints_to_json_list(self.waypoints)
            with open(paths['json'], 'w', encoding='utf-8') as f:
                json.dump(json_payload, f, indent=2, ensure_ascii=False, default=str)
        except OSError as e:
            if not quiet:
                QMessageBox.critical(self, 'Save failed', f'Could not write trajectory files:\n{e}')
            return False

        self.last_csv_path = paths['csv']
        self.default_traj_csv_path = paths['csv']
        self._cache_current_mode_results()
        self._refresh_trajectory_storage_path_label()
        self._refresh_opt_info_display()
        if hasattr(self, 'tracking_source_combo'):
            self._on_tracking_source_changed(self.tracking_source_combo.currentIndex())
        if not quiet and hasattr(self, 'status_text'):
            self.status_text.append(
                f'Saved {label} / {platform_label(platform_id)} / method {method_index + 1} '
                f'({mode_label}):\n'
                f'  JSON: {paths["json"]}\n'
                f'  CSV:  {paths["csv"]}'
            )
        return True

    def save_trajectory(self):
        """Save waypoints, optimized CSV, NPZ, and summary for the current context."""
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
        self._persist_optimization_artifacts(quiet=False)

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

    def _current_rocket_platform_id(self):
        if hasattr(self, 'rocket_real_radio') and self.rocket_real_radio.isChecked():
            return PLATFORM_REAL
        return PLATFORM_PROXY

    def _set_rocket_platform_radio(self, platform_id, block_signals=False):
        platform_id = normalize_platform_id(platform_id)
        if block_signals:
            self._rocket_platform_button_group.blockSignals(True)
        try:
            if platform_id == PLATFORM_REAL:
                self.rocket_real_radio.setChecked(True)
            else:
                self.rocket_proxy_radio.setChecked(True)
        finally:
            if block_signals:
                self._rocket_platform_button_group.blockSignals(False)

    def _snapshot_platform_settings(self):
        """Capture Physical + thrust-related constraint fields for the active platform."""
        return {
            'mass': self.mass.value(),
            'Ixx': self.Ixx.value(),
            'Iyy': self.Iyy.value(),
            'Izz': self.Izz.value(),
            'r_thrust_x': self.r_thrust_x.value(),
            'r_thrust_y': self.r_thrust_y.value(),
            'r_thrust_z': self.r_thrust_z.value(),
            'T_min': self.T_min_thrust.value(),
            'T_max': self.T_max.value(),
            'tau_yaw_max': self.tau_yaw_max.value(),
            'v_horizontal_max': self.v_horizontal_max.value(),
            'v_vertical_max': self.v_vertical_max.value(),
        }

    def _update_platform_spin_ranges(self, platform_id):
        ranges = physics_spin_ranges(platform_id)
        m_lo, m_hi, m_dec = ranges['mass']
        i_lo, i_hi, i_dec = ranges['inertia']
        r_lo, r_hi, r_dec = ranges['r_thrust']
        self.mass.setRange(m_lo, m_hi)
        self.mass.setDecimals(m_dec)
        for spin in (self.Ixx, self.Iyy, self.Izz):
            spin.setRange(i_lo, i_hi)
            spin.setDecimals(i_dec)
        for spin in (self.r_thrust_x, self.r_thrust_y, self.r_thrust_z):
            spin.setRange(r_lo, r_hi)
            spin.setDecimals(r_dec)

        cr = constraint_spin_ranges(platform_id)
        tmin_lo, tmin_hi, tmin_dec = cr['T_min']
        self.T_min_thrust.setRange(tmin_lo, tmin_hi)
        self.T_min_thrust.setDecimals(tmin_dec)
        t_lo, t_hi, t_dec = cr['T_max']
        self.T_max.setRange(t_lo, t_hi)
        self.T_max.setDecimals(t_dec)
        ty_lo, ty_hi, ty_dec = cr['tau_yaw_max']
        self.tau_yaw_max.setRange(ty_lo, ty_hi)
        self.tau_yaw_max.setDecimals(ty_dec)

    def _refresh_physics_platform_hint(self):
        if not hasattr(self, 'physics_platform_hint'):
            return
        pid = self._current_rocket_platform_id()
        self.physics_platform_hint.setText(
            f'Active platform: {platform_label(pid)} — {platform_description(pid)}'
        )
        if hasattr(self, 'rocket_platform_desc_label'):
            self.rocket_platform_desc_label.setText(platform_description(pid))

    def _apply_platform_settings(self, platform_id, settings=None):
        """Apply physical (and thrust-related constraint) defaults for a platform."""
        platform_id = normalize_platform_id(platform_id)
        self._rocket_platform_guard = True
        try:
            self._update_platform_spin_ranges(platform_id)
            physics = dict(default_physics(platform_id))
            constraints = dict(default_constraints(platform_id))
            if settings:
                settings = dict(settings)
                # Recover from older GUI spinbox cap (T_max range was 0–100 N).
                if (
                    platform_id == PLATFORM_REAL
                    and float(settings.get('T_max', constraints['T_max'])) <= 100.0 + 1e-6
                ):
                    settings['T_max'] = constraints['T_max']
                physics.update({k: settings[k] for k in physics if k in settings})
                constraints.update({k: settings[k] for k in constraints if k in settings})

            self.mass.setValue(float(physics['mass']))
            self.Ixx.setValue(float(physics['Ixx']))
            self.Iyy.setValue(float(physics['Iyy']))
            self.Izz.setValue(float(physics['Izz']))
            self.r_thrust_x.setValue(float(physics['r_thrust_x']))
            self.r_thrust_y.setValue(float(physics['r_thrust_y']))
            self.r_thrust_z.setValue(float(physics['r_thrust_z']))

            self.T_min_thrust.setValue(float(constraints['T_min']))
            self.T_max.setValue(float(constraints['T_max']))
            self.tau_yaw_max.setValue(float(constraints['tau_yaw_max']))
            self.v_horizontal_max.setValue(float(constraints['v_horizontal_max']))
            self.v_vertical_max.setValue(float(constraints['v_vertical_max']))
            self._refresh_physics_platform_hint()
        finally:
            self._rocket_platform_guard = False

    def _on_rocket_platform_changed(self, _button=None):
        if self._rocket_platform_guard:
            return
        previous_id = self._cached_platform_id
        self._platform_phys_cache[previous_id] = self._snapshot_platform_settings()

        new_id = self._current_rocket_platform_id()
        if previous_id != new_id:
            # Flush Tracking-tab gains into the previous platform slot before switching.
            self._store_tracking_params_for_controller(
                self._tracking_config.get('controller')
                or self._current_tracking_controller_id()
            )
        cached = self._platform_phys_cache.get(new_id)
        self._apply_platform_settings(new_id, settings=cached)
        self._cached_platform_id = new_id
        if previous_id != new_id:
            self._rebuild_tracking_param_widgets()
            self._refresh_tracking_params_platform_labels()
        if hasattr(self, 'status_text'):
            self.status_text.append(f'Rocket platform: {platform_label(new_id)}')
        if not self._restore_optimization_for_current_context(quiet=True):
            self._clear_optimization_results(clear_plots=True)

    def gui_config_to_dict(self):
        """Serialize GUI settings to a JSON-friendly dict."""
        preset_id = combo_index_to_trajectory_preset_id(
            self.trajectory_preset_combo.currentIndex()
        )
        cfg = {
            'version': GUI_PARAMS_VERSION,
            'rocket_platform': self._current_rocket_platform_id(),
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
            'T_min': self.T_min_thrust.value(),
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
            'tracking': self._tracking_config_to_dict() if hasattr(self, 'tracking_controller_combo') else None,
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

            if 'rocket_platform' in cfg:
                pid = normalize_platform_id(cfg['rocket_platform'])
                self._set_rocket_platform_radio(pid, block_signals=True)
                self._cached_platform_id = pid
                self._update_platform_spin_ranges(pid)

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
                          ('T_min', self.T_min_thrust), ('T_max', self.T_max), ('tau_yaw_max', self.tau_yaw_max),
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

            if 'tracking' in cfg and cfg['tracking']:
                self._apply_tracking_config(cfg['tracking'])

            if 'rocket_platform' in cfg:
                pid = normalize_platform_id(cfg['rocket_platform'])
                self._platform_phys_cache[pid] = self._snapshot_platform_settings()
                self._refresh_physics_platform_hint()
        finally:
            for combo in self._method_combos():
                combo.blockSignals(False)
        self._update_unified_checkbox_state(self.method_combo.currentIndex())
        self._refresh_min_time_duration_group_visible(self.method_combo.currentIndex())
        self._cached_platform_id = self._current_rocket_platform_id()
        self._refresh_physics_platform_hint()
        if hasattr(self, 'tracking_params_groups_layout'):
            self._rebuild_tracking_param_widgets()
        if hasattr(self, 'nmp_params_groups_layout'):
            self._rebuild_nmp_tracking_param_widgets()
        self._refresh_tracking_params_platform_labels()
        if self._current_rocket_platform_id() == PLATFORM_REAL:
            plat_def = default_constraints(PLATFORM_REAL)
            if self.T_max.value() <= 100.0 + 1e-6:
                self._update_platform_spin_ranges(PLATFORM_REAL)
                self.T_max.setValue(plat_def['T_max'])
            if self.T_min_thrust.value() < plat_def['T_min'] - 1e-6:
                self.T_min_thrust.setValue(plat_def['T_min'])

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
        t_min = float(self.T_min_thrust.value())
        t_max = float(self.T_max.value())
        if t_min > t_max:
            t_min = t_max
        bounds = {
            "th_p": (-th_p_max_rad, th_p_max_rad),
            "th_r": (-th_r_max_rad, th_r_max_rad),
            "T": (t_min, t_max),
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
            'rocket_platform': self._current_rocket_platform_id(),
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

        params = self.get_parameters()
        input_fp = optimization_input_fingerprint(params)
        if self._restore_optimization_for_current_context(
            quiet=True, required_fingerprint=input_fp,
        ):
            ctx = self._current_optimization_context()
            self.status_text.append(
                f'Loaded cached optimization — '
                f'{platform_label(ctx["platform_id"])} / '
                f'{trajectory_combo_label(ctx["combo_index"])} / '
                f'method {ctx["method_index"] + 1} (inputs unchanged).'
            )
            return

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
        self._refresh_all_plot_layouts()
    
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
            'T_min': self.T_min_thrust.value(),
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
        self._refresh_metrics_tab(opt_only=True)
        
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
        if time_axis is None and timing_info and timing_info.get('time_states') is not None:
            time_axis = np.asarray(timing_info['time_states'], dtype=float)
        elif time_axis is None and timing_info and timing_info.get('flat_outputs'):
            fo = timing_info['flat_outputs']
            if isinstance(fo, dict) and fo.get('t') is not None:
                time_axis = np.asarray(fo['t'], dtype=float)
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
            'flat_outputs': (timing_info or {}).get('flat_outputs'),
            'flatness_physics': (timing_info or {}).get('flatness_physics'),
        }
        self.opt_summary = self._make_optimization_summary(xs, us, all_loggers, timing_info)
        self._cache_current_mode_results()
        self._persist_optimization_artifacts(quiet=True)
        self._refresh_opt_info_display()
        self._enable_trajectory_export_buttons(True)
        if hasattr(self, 'num_sim_dt_spin'):
            self._on_numerical_sim_timing_changed()
        if hasattr(self, 'tracking_source_combo') and self.tracking_source_combo.currentIndex() == 0:
            self._on_tracking_source_changed(0)
        if self.method_combo.currentIndex() == 7 and hasattr(self, 'canvas_nmp'):
            self.nmp_last_trajectory = dict(self.last_trajectory)
            self.nmp_last_trajectory['platform_id'] = self._current_rocket_platform_id()
            if hasattr(self, 'lbl_nmp_plan_status'):
                self.lbl_nmp_plan_status.setText(
                    'Plan loaded from Trajectory tab (Method 8).'
                )
            self._draw_nmp_plot_tab()
    
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

    def _mission_waypoints_for_online_planner(self):
        """Mission targets from Trajectory tab (skip index-0 departure pose)."""
        self._sync_waypoints_from_table()
        if not self.waypoints:
            return []
        rows = [_normalize_waypoint_row(w) for w in self.waypoints]
        if len(rows) >= 2:
            return rows[1:]
        return rows

    def _encode_waypoints_for_online_planner(self):
        """Encode Trajectory-tab mission targets for online_planner (ENU x,y,z,yaw,time)."""
        mission = self._mission_waypoints_for_online_planner()
        if not mission:
            return ''
        parts = []
        for row in mission:
            x, y, z, yaw, t_arr = row
            parts.append(f'{x},{y},{z},{yaw},{t_arr}')
        return ';'.join(parts)

    def _update_online_planner_wp_info_label(self):
        if not hasattr(self, 'lbl_online_planner_wp_info'):
            return
        mission = self._mission_waypoints_for_online_planner()
        total = len(self.waypoints) if hasattr(self, 'waypoints') else 0
        if not mission:
            self.lbl_online_planner_wp_info.setText('WPs: (none from Trajectory)')
            return
        preview = ' | '.join(
            f'({r[0]:.1f},{r[1]:.1f},{r[2]:.1f})' for r in mission[:3]
        )
        if len(mission) > 3:
            preview += f' … +{len(mission) - 3}'
        self.lbl_online_planner_wp_info.setText(
            f'WPs: {len(mission)} targets (skip start; {total} in table) {preview}'
        )

    def _collect_online_planner_cost_config(self):
        """Acados cost / bounds / discretization from the Trajectory tab (Method 4 style)."""
        if not hasattr(self, 'w_p'):
            return None
        params = self.get_parameters()
        weights = dict(params.get('weights') or {})
        weights['acados_objective'] = 'tracking'
        return {
            'weights': weights,
            'terminal_weights': dict(params.get('terminal_weights') or {}),
            'bounds': dict(params.get('bounds') or {}),
            'dt': float(params.get('dt', 0.1)),
            'horizon_N': int(params.get('N', 20)),
            'max_iter': int(params.get('max_iter', 100)),
        }

    def _encode_online_planner_cost_json(self):
        """Serialize Trajectory-tab cost params for online_planner ROS parameters."""
        cfg = self._collect_online_planner_cost_config()
        if not cfg:
            return '', '', ''
        compact = json.dumps
        safe = _json_safe_for_fingerprint
        return (
            compact(safe(cfg['weights']), separators=(',', ':')),
            compact(safe(cfg['terminal_weights']), separators=(',', ':')),
            compact(safe(cfg['bounds']), separators=(',', ':')),
        )

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

    def _background_process_log_path(self, attr_name: str) -> str:
        log_dir = os.path.join(self._ros2_workspace_root(), 'logs', 'gui_background')
        os.makedirs(log_dir, exist_ok=True)
        slug = str(attr_name).lstrip('_')
        stamp = time.strftime('%Y%m%d_%H%M%S')
        return os.path.join(log_dir, f'{slug}_{stamp}.log')

    def _close_background_process_log(self, attr_name: str) -> None:
        fh = self._bg_proc_log_handles.pop(attr_name, None)
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass

    def _start_background_process(self, attr_name, cmd, label):
        """Start a detached process group; store handle on self.<attr_name>."""
        proc = getattr(self, attr_name, None)
        if proc is not None:
            if proc.poll() is None:
                QMessageBox.information(self, label, f'{label} is already running.')
                return
            setattr(self, attr_name, None)
            self._close_background_process_log(attr_name)
        log_path = self._background_process_log_path(attr_name)
        log_fh = None
        try:
            log_fh = open(log_path, 'w', encoding='utf-8', buffering=1)
            new_proc = subprocess.Popen(
                cmd,
                cwd=self._ros2_workspace_root(),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                text=True,
            )
        except OSError as e:
            if log_fh is not None:
                try:
                    log_fh.close()
                except OSError:
                    pass
            QMessageBox.critical(self, f'{label} failed', str(e))
            self.status_text.append(f'{label} failed: {e}')
            return
        self._bg_proc_log_handles[attr_name] = log_fh
        setattr(self, attr_name, new_proc)
        self.status_text.append(
            f'{label} started (pid {new_proc.pid})\n  log: {log_path}'
        )
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
        self._close_background_process_log(attr_name)
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
            sitl = (
                hasattr(self, 'tracking_sim_sitl_radio')
                and self.tracking_sim_sitl_radio.isChecked()
            )
            cid = self._current_tracking_controller_id()
            can_sitl = sitl and cid not in (CONTROLLER_MPC, CONTROLLER_ACADOS_NMPC)
            self.btn_start_px4_sitl.setEnabled(can_sitl and not tvc_running)
            self.btn_stop_px4_sitl.setEnabled(tvc_running)
        if hasattr(self, 'lbl_px4_sitl_status'):
            if tvc_running:
                self.lbl_px4_sitl_status.setText('Running')
                self.lbl_px4_sitl_status.setStyleSheet('color: #2e7d32; font-weight: bold;')
            else:
                self.lbl_px4_sitl_status.setText('Stopped')
                self.lbl_px4_sitl_status.setStyleSheet('color: #888;')
        if hasattr(self, 'btn_start_tracking'):
            sitl = (
                hasattr(self, 'tracking_sim_sitl_radio')
                and self.tracking_sim_sitl_radio.isChecked()
            )
            self.btn_start_tracking.setEnabled(sitl and not track_running)
            self.btn_stop_tracking.setEnabled(track_running)
        if hasattr(self, 'lbl_tracking_status'):
            if track_running:
                self.lbl_tracking_status.setText('Running')
                self.lbl_tracking_status.setStyleSheet('color: #2e7d32; font-weight: bold;')
            else:
                self.lbl_tracking_status.setText('Stopped')
                self.lbl_tracking_status.setStyleSheet('color: #888;')
        planner_running = self._online_planner_running() if hasattr(self, '_online_planner_running') else False
        sitl_running = self._sitl_proc_running() if hasattr(self, '_sitl_proc_running') else False
        planner_enabled = self._online_planner_enabled() if hasattr(self, '_online_planner_enabled') else False
        if hasattr(self, 'btn_online_planner_next_wp'):
            self.btn_online_planner_next_wp.setEnabled(sitl_running and planner_enabled)
        self._update_online_planner_diag_timer()
        self._update_sitl_nodes_timer()

    def _traj_player_proc_running(self) -> bool:
        proc = getattr(self, '_traj_player_proc', None)
        if proc is None:
            return False
        if proc.poll() is not None:
            setattr(self, '_traj_player_proc', None)
            return False
        return True

    def _px4_sitl_process_active(self) -> bool:
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'px4_sitl_default/bin/px4'],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def _sitl_monitored_node_specs(self):
        """(display label, ros node name) or (label, None) for non-ROS checks."""
        specs = [
            ('PX4', None),
            ('Gz bridge', '/ros_gz_bridge'),
            ('Vision', '/gz_vision_publisher'),
            ('RViz br.', '/px4_rviz_bridge'),
            ('RViz', '/rviz2'),
            ('Robot', '/robot_state_publisher'),
        ]
        if self._online_planner_enabled():
            specs.append(('Planner', '/online_planner'))
        specs.append(('Tracking', '/tvc_traj_player'))
        if self._current_tracking_controller_id() == CONTROLLER_LQR:
            specs.append(('LQR', '/lqr_px4_controller'))
        return specs

    def _poll_sitl_nodes_status(self):
        if not hasattr(self, 'lbl_sitl_nodes_status'):
            return
        active_nodes = set()
        result = self._ros2_run_in_workspace('ros2 node list', timeout=4.0)
        if result is not None and result.returncode == 0:
            for line in (result.stdout or '').splitlines():
                name = line.strip()
                if name:
                    active_nodes.add(name)
        px4_up = self._px4_sitl_process_active()
        chunks = []
        for label, node_name in self._sitl_monitored_node_specs():
            if node_name is None:
                up = px4_up
            else:
                up = node_name in active_nodes
            color = '#2e7d32' if up else '#888'
            mark = '●' if up else '○'
            chunks.append(f'<span style="color:{color}">{mark} {label}</span>')
        self.lbl_sitl_nodes_status.setText(' '.join(chunks))

    def _update_sitl_nodes_timer(self):
        if not hasattr(self, '_sitl_nodes_timer'):
            return
        sitl_running = self._sitl_proc_running()
        track_running = self._traj_player_proc_running()
        should_poll = sitl_running or track_running
        if should_poll:
            if not self._sitl_nodes_timer.isActive():
                self._sitl_nodes_timer.start()
            self._poll_sitl_nodes_status()
        else:
            self._sitl_nodes_timer.stop()
            if hasattr(self, 'lbl_sitl_nodes_status'):
                self.lbl_sitl_nodes_status.setText('Nodes: —')

    def _current_tracking_controller_id(self):
        if not hasattr(self, 'tracking_controller_combo'):
            return CONTROLLER_PX4
        idx = self.tracking_controller_combo.currentIndex()
        cid = self.tracking_controller_combo.itemData(idx)
        return cid if cid else CONTROLLER_PX4

    def _ensure_tracking_params_by_platform(self):
        """Ensure params_by_platform exists; keep flat params as a back-compat view."""
        by_plat = self._tracking_config.get('params_by_platform')
        if not isinstance(by_plat, dict) or not by_plat:
            by_plat = migrate_params_by_platform(self._tracking_config)
            self._tracking_config['params_by_platform'] = by_plat
        for pid in (PLATFORM_PROXY, PLATFORM_REAL):
            if pid not in by_plat or not isinstance(by_plat[pid], dict):
                by_plat[pid] = default_controller_params_map()
            for cid in CONTROLLER_IDS:
                if cid not in by_plat[pid] or not isinstance(by_plat[pid][cid], dict):
                    by_plat[pid][cid] = default_params_for(cid)
        # Active view used by older code paths / JSON consumers.
        active_pid = (
            self._current_rocket_platform_id()
            if hasattr(self, 'rocket_proxy_radio') else PLATFORM_PROXY
        )
        self._tracking_config['params'] = by_plat[normalize_platform_id(active_pid)]
        return by_plat

    def _tracking_params_map_for(self, platform_id):
        """Controller-params map for the given rocket platform."""
        by_plat = self._ensure_tracking_params_by_platform()
        pid = normalize_platform_id(platform_id)
        return by_plat.setdefault(pid, default_controller_params_map())

    def _refresh_tracking_params_platform_labels(self):
        """Show which platform the visible controller gains belong to."""
        if hasattr(self, 'tracking_params_group') and hasattr(self, 'rocket_proxy_radio'):
            pid = self._current_rocket_platform_id()
            title = f'Controller parameters ({platform_label(pid)})'
            set_title = getattr(self.tracking_params_group, '_tvc_set_title', None)
            if callable(set_title):
                set_title(title)
            if hasattr(self, 'lbl_tracking_params_platform'):
                self.lbl_tracking_params_platform.setText(
                    f'Editing gains for: {platform_label(pid)}. '
                    f'Proxy and Real keep separate parameter sets; '
                    f'switch Rocket platform to edit the other.'
                )
        if hasattr(self, 'nmp_params_group') and hasattr(self, 'nmp_proxy_radio'):
            nmp_pid = self._current_nmp_platform_id()
            nmp_title = f'Controller parameters ({platform_label(nmp_pid)})'
            set_nmp = getattr(self.nmp_params_group, '_tvc_set_title', None)
            if callable(set_nmp):
                set_nmp(nmp_title)
            if hasattr(self, 'lbl_nmp_params_platform'):
                self.lbl_nmp_params_platform.setText(
                    f'Editing gains for: {platform_label(nmp_pid)}. '
                    f'Shared with Tracking tab for the same platform.'
                )

    def _on_tracking_controller_changed(self, _index=None):
        self._store_tracking_params_for_controller(self._tracking_config.get('controller'))
        cid = self._current_tracking_controller_id()
        self._tracking_config['controller'] = cid
        self._rebuild_tracking_param_widgets()
        self._on_tracking_sim_mode_changed()

    def _on_tracking_sim_mode_changed(self, _btn=None):
        cid = self._current_tracking_controller_id()
        sitl = hasattr(self, 'tracking_sim_sitl_radio') and self.tracking_sim_sitl_radio.isChecked()
        self._tracking_config['sim_mode'] = SIM_SITL if sitl else SIM_NUMERICAL
        if hasattr(self, 'btn_run_numerical_tracking'):
            self.btn_run_numerical_tracking.setEnabled(not sitl)
        self._update_tracking_panel_visibility()
        cid = self._current_tracking_controller_id()
        if cid == CONTROLLER_MPC and sitl:
            self.btn_start_px4_sitl.setToolTip(
                'Linear MPC tracking is available in numerical simulation only.'
            )
        elif cid == CONTROLLER_ACADOS_NMPC and sitl:
            self.btn_start_px4_sitl.setToolTip(
                'Acados NMPC tracking is available in numerical simulation only.'
            )
        elif cid == CONTROLLER_LQR:
            self.btn_start_px4_sitl.setToolTip(
                'Launch PX4 SITL with external LQR node (launch_controller:=true).'
            )
        else:
            self.btn_start_px4_sitl.setToolTip(
                'Launch PX4 SITL + Gazebo; trajectory player sends setpoints to PX4 cascade.'
            )
        self._refresh_sim_button_states()

    def _store_tracking_params_for_controller(self, controller_id):
        if not controller_id or controller_id not in CONTROLLER_IDS:
            return
        platform_id = (
            getattr(self, '_cached_platform_id', None)
            or self._current_rocket_platform_id()
        )
        params_map = self._tracking_params_map_for(platform_id)
        if controller_id not in params_map:
            params_map[controller_id] = default_params_for(controller_id)
        stored = params_map[controller_id]
        for key, widget in self._tracking_param_widgets.items():
            if isinstance(widget, QCheckBox):
                stored[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                stored[key] = int(widget.value())
            else:
                stored[key] = float(widget.value())

    def _collect_tracking_params(self):
        cid = self._current_tracking_controller_id()
        self._store_tracking_params_for_controller(cid)
        self._store_actuator_config()
        self._store_numerical_sim_config()
        params = dict(
            self._tracking_params_map_for(self._current_rocket_platform_id()).get(
                cid, default_params_for(cid)
            )
        )
        params.update(self._actuator_params_for_sim())
        params.update(self._numerical_sim_params_for_sim())
        params.update(self._constraints_for_tracking())
        return params

    def _constraints_for_tracking(self):
        """Limits from Parameters → Constraints (shared with trajectory optimization)."""
        if not hasattr(self, 'th_p_max'):
            return {}
        return self._bounds_display_from_widgets()

    def _make_tracking_param_widget(self, spec, params, controller_id):
        key = spec['key']
        if spec.get('checkbox'):
            w = QCheckBox()
            w.setChecked(bool(params.get(key, spec['default'])))
            if controller_id == CONTROLLER_PX4 and key == 'share_rp_gains':
                w.stateChanged.connect(self._on_px4_share_rp_changed)
        elif spec.get('integer'):
            w = QSpinBox()
            w.setRange(int(spec['min']), int(spec['max']))
            w.setValue(int(params.get(key, spec['default'])))
        else:
            w = QDoubleSpinBox()
            w.setRange(float(spec['min']), float(spec['max']))
            w.setDecimals(int(spec.get('decimals', 3)))
            w.setSingleStep(10 ** (-int(spec.get('decimals', 3))))
            w.setValue(float(params.get(key, spec['default'])))
        return w

    def _populate_tracking_param_group(self, grid, specs, params, controller_id):
        row = 0
        col_slot = 0
        for spec in specs:
            key = spec['key']
            if spec.get('full_width'):
                if col_slot == 1:
                    row += 1
                    col_slot = 0
                w = self._make_tracking_param_widget(spec, params, controller_id)
                if spec.get('checkbox'):
                    w.setText(spec['label'])
                    grid.addWidget(w, row, 0, 1, 4)
                else:
                    grid.addWidget(QLabel(spec['label']), row, 0)
                    grid.addWidget(w, row, 1, 1, 3)
                self._tracking_param_widgets[key] = w
                row += 1
                continue

            base_col = col_slot * 2
            grid.addWidget(QLabel(spec['label']), row, base_col)
            w = self._make_tracking_param_widget(spec, params, controller_id)
            grid.addWidget(w, row, base_col + 1)
            self._tracking_param_widgets[key] = w
            if col_slot == 1:
                row += 1
                col_slot = 0
            else:
                col_slot = 1
        if col_slot == 1:
            row += 1
        return row

    def _rebuild_tracking_param_widgets(self):
        if not hasattr(self, 'tracking_params_groups_layout'):
            return
        while self.tracking_params_groups_layout.count():
            item = self.tracking_params_groups_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._tracking_param_widgets = {}

        cid = self._current_tracking_controller_id()
        platform_id = self._current_rocket_platform_id()
        params_map = self._tracking_params_map_for(platform_id)
        params = params_map.setdefault(cid, default_params_for(cid))
        if cid == CONTROLLER_PX4:
            params.update(migrate_px4_params(params))
            params_map[cid] = params
        px4_extra = params if cid == CONTROLLER_PX4 else None
        groups = param_groups_for(cid, px4_params=px4_extra)
        for group in groups:
            box = QGroupBox(group.get('title', 'Parameters'))
            grid = QGridLayout(box)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)
            self._populate_tracking_param_group(
                grid, group.get('specs') or [], params, cid
            )
            self.tracking_params_groups_layout.addWidget(box)
        self.tracking_params_groups_layout.addStretch(1)
        self._refresh_tracking_params_platform_labels()
        QTimer.singleShot(0, self._refresh_tab_scroll_areas)

    def _on_px4_share_rp_changed(self, _state=None):
        if self._current_tracking_controller_id() != CONTROLLER_PX4:
            return
        self._store_tracking_params_for_controller(CONTROLLER_PX4)
        self._rebuild_tracking_param_widgets()

    def _tracking_config_to_dict(self):
        self._store_tracking_params_for_controller(self._current_tracking_controller_id())
        self._store_actuator_config()
        self._store_numerical_sim_config()
        if hasattr(self, 'nmp_params_groups_layout'):
            self._store_nmp_tracking_params_for_controller()
        if hasattr(self, 'nmp_sim_dt_spin'):
            self._on_nmp_tracking_timing_changed()
        by_plat = self._ensure_tracking_params_by_platform()
        # Deep-copy so later UI edits do not mutate the saved snapshot.
        params_by_platform = {
            pid: {cid: dict(params) for cid, params in ctrl_map.items()}
            for pid, ctrl_map in by_plat.items()
        }
        active_params = params_map_for_platform(
            {'params_by_platform': params_by_platform},
            self._current_rocket_platform_id(),
        )
        out = {
            'controller': self._current_tracking_controller_id(),
            'sim_mode': (
                SIM_SITL
                if hasattr(self, 'tracking_sim_sitl_radio') and self.tracking_sim_sitl_radio.isChecked()
                else SIM_NUMERICAL
            ),
            'numerical_sim': dict(
                self._tracking_config.get('numerical_sim') or default_numerical_sim_config()
            ),
            'nmp_numerical_sim': dict(
                self._tracking_config.get('nmp_numerical_sim')
                or self._tracking_config.get('numerical_sim')
                or default_numerical_sim_config()
            ),
            'actuator': dict(self._tracking_config.get('actuator') or default_actuator_tracking_config()),
            'params_by_platform': params_by_platform,
            # Back-compat: flat params = currently selected Trajectory platform.
            'params': {cid: dict(p) for cid, p in active_params.items()},
            'enable_online_planner': self._online_planner_enabled(),
            'online_planner_rate_hz': self._online_planner_rate_hz(),
            'show_gazebo_gui': self._show_gazebo_gui_enabled(),
        }
        if hasattr(self, 'px4_tune_level_combo'):
            out['px4_tune'] = self._collect_px4_tune_config()
        return out

    def _apply_tracking_config(self, cfg):
        if not cfg:
            return
        by_plat = migrate_params_by_platform(cfg)
        self._tracking_config['params_by_platform'] = by_plat
        active_pid = (
            self._current_rocket_platform_id()
            if hasattr(self, 'rocket_proxy_radio') else PLATFORM_PROXY
        )
        self._tracking_config['params'] = by_plat[normalize_platform_id(active_pid)]
        actuator_cfg = cfg.get('actuator') or self._migrate_actuator_from_controller_params(cfg)
        self._tracking_config['actuator'] = dict(actuator_cfg)
        numerical_cfg = cfg.get('numerical_sim') or self._migrate_numerical_sim_from_controller_params(cfg)
        self._tracking_config['numerical_sim'] = dict(numerical_cfg)
        nmp_numerical_cfg = cfg.get('nmp_numerical_sim') or numerical_cfg
        self._tracking_config['nmp_numerical_sim'] = dict(nmp_numerical_cfg)
        controller = cfg.get('controller', CONTROLLER_PX4)
        if hasattr(self, 'tracking_controller_combo'):
            for i in range(self.tracking_controller_combo.count()):
                if self.tracking_controller_combo.itemData(i) == controller:
                    self.tracking_controller_combo.setCurrentIndex(i)
                    break
        sim_mode = cfg.get('sim_mode', SIM_NUMERICAL)
        if hasattr(self, 'tracking_sim_numerical_radio'):
            if sim_mode == SIM_SITL:
                self.tracking_sim_sitl_radio.setChecked(True)
            else:
                self.tracking_sim_numerical_radio.setChecked(True)
        self._tracking_config['controller'] = controller
        self._rebuild_tracking_param_widgets()
        if hasattr(self, 'nmp_params_groups_layout'):
            self._rebuild_nmp_tracking_param_widgets()
        self._refresh_tracking_params_platform_labels()
        self._apply_actuator_config(self._tracking_config.get('actuator'))
        self._apply_numerical_sim_config(self._tracking_config.get('numerical_sim'))
        if hasattr(self, 'nmp_sim_dt_spin'):
            self._nmp_sim_timing_guard = True
            self.nmp_sim_dt_spin.setValue(float(nmp_numerical_cfg.get('sim_dt', 0.005)))
            self.nmp_control_dt_spin.setValue(float(nmp_numerical_cfg.get('control_dt', 0.02)))
            self.nmp_terminal_hold_spin.setValue(
                float(nmp_numerical_cfg.get('terminal_hold_duration_s', 3.0))
            )
            self.nmp_total_duration_spin.setValue(float(nmp_numerical_cfg.get('total_duration_s', 0.0)))
            self._nmp_sim_timing_guard = False
            self._on_nmp_tracking_timing_changed()
        if cfg.get('px4_tune'):
            self._apply_px4_tune_config(cfg['px4_tune'])
        if hasattr(self, 'chk_enable_online_planner'):
            self.chk_enable_online_planner.blockSignals(True)
            self.chk_enable_online_planner.setChecked(
                bool(cfg.get('enable_online_planner', True))
            )
            self.chk_enable_online_planner.blockSignals(False)
            self._tracking_config['enable_online_planner'] = (
                self.chk_enable_online_planner.isChecked()
            )
        if hasattr(self, 'spin_online_planner_rate'):
            self.spin_online_planner_rate.blockSignals(True)
            self.spin_online_planner_rate.setValue(
                float(cfg.get('online_planner_rate_hz', 10.0))
            )
            self.spin_online_planner_rate.blockSignals(False)
            self._tracking_config['online_planner_rate_hz'] = (
                self._online_planner_rate_hz()
            )
        if hasattr(self, 'chk_show_gazebo_gui'):
            self.chk_show_gazebo_gui.blockSignals(True)
            self.chk_show_gazebo_gui.setChecked(
                bool(cfg.get('show_gazebo_gui', False))
            )
            self.chk_show_gazebo_gui.blockSignals(False)
            self._tracking_config['show_gazebo_gui'] = (
                self.chk_show_gazebo_gui.isChecked()
            )
        self._update_px4_tune_visibility()
        self._on_tracking_sim_mode_changed()

    def _update_tracking_params_file_label(self):
        if hasattr(self, 'tracking_params_file_label'):
            self.tracking_params_file_label.setText(
                f'Controller params file: {self.tracking_params_file_path}'
            )

    def save_tracking_params(self):
        try:
            with open(self.tracking_params_file_path, 'w', encoding='utf-8') as f:
                json.dump(self._tracking_config_to_dict(), f, indent=2, ensure_ascii=False)
            self.status_text.append(f'Saved tracking params to {self.tracking_params_file_path}')
        except OSError as e:
            QMessageBox.critical(self, 'Save failed', str(e))

    def load_tracking_params(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load tracking parameters', self.tracking_params_file_path,
            'JSON (*.json);;All files (*)',
        )
        if path:
            self._load_tracking_params_from_path(path, quiet=False)

    def _load_tracking_params_from_path(self, path, quiet=False):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except OSError as e:
            if not quiet:
                QMessageBox.critical(self, 'Load failed', str(e))
            return
        self._apply_tracking_config(cfg)
        self.tracking_params_file_path = os.path.abspath(path)
        self._update_tracking_params_file_label()
        if not quiet:
            self.status_text.append(f'Loaded tracking params from {path}')

    def _physics_dict_for_tracking(self):
        return {
            'mass': self.mass.value(),
            'Ixx': self.Ixx.value(),
            'Iyy': self.Iyy.value(),
            'Izz': self.Izz.value(),
            'r_thrust_x': self.r_thrust_x.value(),
            'r_thrust_y': self.r_thrust_y.value(),
            'r_thrust_z': self.r_thrust_z.value(),
            'g': 9.81,
            'platform_id': self._current_rocket_platform_id(),
        }

    def _flatness_physics_matches_tracking(self, flatness_physics, rtol=0.02):
        """True if GUI physics matches the snapshot stored with a Method-8 plan."""
        if not flatness_physics:
            return True
        cur = self._physics_dict_for_tracking()
        checks = (
            ('mass', flatness_physics.get('mass'), cur['mass']),
            ('Ixx', flatness_physics.get('Ixx'), cur['Ixx']),
            ('Iyy', flatness_physics.get('Iyy'), cur['Iyy']),
            ('Izz', flatness_physics.get('Izz'), cur['Izz']),
            ('r_thrust_z', flatness_physics.get('r_thrust_z'), cur['r_thrust_z']),
        )
        for _name, plan_v, gui_v in checks:
            if plan_v is None:
                continue
            plan_v = float(plan_v)
            gui_v = float(gui_v)
            tol = max(abs(plan_v) * rtol, 1e-6)
            if abs(plan_v - gui_v) > tol:
                return False
        return True

    def _trajectory_arrays_for_tracking(self):
        """Return (xs, us, time_states) from memory or CSV for numerical sim."""
        source = self.tracking_source_combo.currentIndex() if hasattr(self, 'tracking_source_combo') else 0
        if source == 1:
            path = self.tracking_csv_edit.text().strip()
            if not path or not os.path.isfile(path):
                return None, None, None
            return self._load_trajectory_arrays_from_csv(path)
        traj = getattr(self, 'last_trajectory', None)
        if traj and traj.get('xs') is not None:
            return (
                np.asarray(traj['xs'], dtype=float),
                np.asarray(traj['us'], dtype=float) if traj.get('us') is not None else None,
                np.asarray(traj['time_states'], dtype=float) if traj.get('time_states') is not None else None,
            )
        path = self._current_trajectory_csv_path()
        if os.path.isfile(path):
            return self._load_trajectory_arrays_from_csv(path)
        return None, None, None

    def _load_trajectory_arrays_from_csv(self, path):
        """Minimal CSV loader for tracking (planner ENU format)."""
        import csv
        t_list, rows, u_rows = [], [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                if line.lower().startswith('t,'):
                    header = [c.strip() for c in line.strip().split(',')]
                    break
            else:
                return None, None, None
            reader = csv.DictReader(f, fieldnames=header)
            for row in reader:
                try:
                    t_list.append(float(row['t']))
                    rows.append([
                        float(row['x']), float(row['y']), float(row['z']),
                        float(row['vx']), float(row['vy']), float(row['vz']),
                        float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz']),
                        float(row['wx']), float(row['wy']), float(row['wz']),
                    ])
                    if 'th_p_cmd' in row and 'T_cmd' in row:
                        u_rows.append([
                            float(row['th_p_cmd']), float(row['th_r_cmd']),
                            float(row['T_cmd']), float(row['tau_yaw_cmd']),
                        ])
                except (KeyError, ValueError):
                    continue
        if not rows:
            return None, None, None
        xs = np.asarray(rows, dtype=float)
        us = np.asarray(u_rows, dtype=float) if u_rows else None
        if us is not None and us.shape[0] == xs.shape[0]:
            us = us[:-1]
        elif us is not None and us.shape[0] != max(xs.shape[0] - 1, 0):
            us = None
        return xs, us, np.asarray(t_list, dtype=float)

    def run_numerical_tracking(self):
        xs, us, time_states = self._trajectory_arrays_for_tracking()
        if xs is None:
            QMessageBox.warning(
                self, 'No trajectory',
                'Optimize and save a trajectory first, or pick a valid CSV file.',
            )
            return
        if self._tracking_sim_thread is not None and self._tracking_sim_thread.isRunning():
            return
        cid = self._current_tracking_controller_id()
        if cid == CONTROLLER_ACADOS_NMPC and not acados_nmpc_available():
            QMessageBox.warning(
                self, 'Acados unavailable',
                'Nonlinear NMPC requires CasADi and a built acados installation.\n'
                'See TVC-traj-opt/README.md (Acados Installation).',
            )
            return
        params = self._collect_tracking_params()
        flat_outputs = None
        flatness_physics = None
        traj = getattr(self, 'last_trajectory', None)
        if traj:
            if traj.get('flat_outputs') is not None:
                flat_outputs = traj['flat_outputs']
            if traj.get('flatness_physics') is not None:
                flatness_physics = traj['flatness_physics']
        if (
            cid == CONTROLLER_FLATNESS
            and flatness_physics is not None
            and not self._flatness_physics_matches_tracking(flatness_physics)
        ):
            fp = flatness_physics
            cur = self._physics_dict_for_tracking()
            QMessageBox.warning(
                self,
                'Flatness physics mismatch',
                'Method 8 was planned with different physics than the Tracking tab '
                'spinboxes. ξ tracking will use the **planning** (m, I, l) for the '
                'flat output map, but the plant simulation still uses current GUI values.\n\n'
                f'Plan:  m={fp["mass"]:.3g} kg, r_thrust_z={fp["r_thrust_z"]:.3f} m\n'
                f'GUI:   m={cur["mass"]:.3g} kg, r_thrust_z={cur["r_thrust_z"]:.3f} m\n\n'
                'Re-select the rocket platform or re-run optimization to align parameters.',
            )
        self.btn_run_numerical_tracking.setEnabled(False)
        self.lbl_tracking_result.setText('Running numerical simulation…')
        self._tracking_sim_thread = TrackingSimulationThread(
            xs, us, time_states, cid, params, self._physics_dict_for_tracking(),
            flat_outputs=flat_outputs, flatness_physics=flatness_physics,
        )
        self._tracking_sim_thread.finished.connect(self._on_tracking_sim_finished)
        self._tracking_sim_thread.error.connect(self._on_tracking_sim_error)
        self._tracking_sim_thread.start()

    def _on_tracking_sim_finished(self, result):
        self.btn_run_numerical_tracking.setEnabled(True)
        self._last_tracking_result = result
        summary = tracking_summary_text(result)
        self.lbl_tracking_result.setText(summary)
        self.status_text.append(summary)
        if self._display_panels_ready():
            self._draw_tracking_plot_tabs(result)
            if hasattr(self, 'plot_tabs'):
                self.plot_tabs.setCurrentWidget(self._plot_tab_states)

    def _on_tracking_sim_error(self, msg):
        self.btn_run_numerical_tracking.setEnabled(True)
        self.lbl_tracking_result.setText('Simulation failed.')
        QMessageBox.critical(self, 'Tracking simulation error', msg)
        self.status_text.append(f'Tracking sim error: {msg}')

    def _online_planner_rate_hz(self) -> float:
        if hasattr(self, 'spin_online_planner_rate'):
            return float(self.spin_online_planner_rate.value())
        return float(self._tracking_config.get('online_planner_rate_hz', 10.0))

    def _online_planner_launch_extra_args(self):
        """ROS launch overrides for online_planner (platform + mission waypoints + cost)."""
        kw = sitl_launch_kwargs(self._current_rocket_platform_id())
        args = [
            self._launch_arg('rocket_platform', kw['rocket_platform']),
            self._launch_arg('replan_rate_hz', self._online_planner_rate_hz()),
        ]
        wp_str = self._encode_waypoints_for_online_planner()
        if wp_str:
            args.append(self._launch_arg('waypoints_enu', wp_str))
        cost_cfg = self._collect_online_planner_cost_config()
        if cost_cfg:
            w_json, tw_json, b_json = self._encode_online_planner_cost_json()
            if w_json:
                args.append(self._launch_arg('planner_weights_json', w_json))
            if tw_json:
                args.append(self._launch_arg('planner_terminal_weights_json', tw_json))
            if b_json:
                args.append(self._launch_arg('planner_bounds_json', b_json))
            args.append(self._launch_arg('dt', cost_cfg['dt']))
            args.append(self._launch_arg('horizon_N', cost_cfg['horizon_N']))
            args.append(self._launch_arg('max_iter', cost_cfg['max_iter']))
        return args

    def _sitl_proc_running(self) -> bool:
        proc = getattr(self, '_tvc_launch_proc', None)
        if proc is None:
            return False
        if proc.poll() is not None:
            setattr(self, '_tvc_launch_proc', None)
            return False
        return True

    def _online_planner_proc_alive(self) -> bool:
        proc = getattr(self, '_online_planner_proc', None)
        if proc is None:
            return False
        if proc.poll() is not None:
            setattr(self, '_online_planner_proc', None)
            return False
        return True

    def _online_planner_node_active(self) -> bool:
        """True if /online_planner is registered in the ROS graph."""
        result = self._ros2_run_in_workspace('ros2 node list', timeout=4.0)
        if result is None or result.returncode != 0:
            return False
        return '/online_planner' in (result.stdout or '')

    def _online_planner_running(self) -> bool:
        """Launch process alive, or ROS node present (e.g. after GUI restart)."""
        if self._online_planner_proc_alive():
            return True
        return self._online_planner_node_active()

    def _ros2_run_in_workspace(self, shell_cmd: str, timeout: float = 2.0):
        """Run a one-shot shell command with workspace ROS env."""
        ws = self._ros2_workspace_root()
        setup = os.path.join(ws, 'install', 'setup.bash')
        if os.path.isfile(setup):
            cmd = ['bash', '-lc', f'source "{setup}" && {shell_cmd}']
        else:
            cmd = ['bash', '-lc', shell_cmd]
        try:
            return subprocess.run(
                cmd,
                cwd=ws,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    def _parse_ros2_string_echo(self, stdout: str) -> str:
        for line in (stdout or '').splitlines():
            stripped = line.strip()
            if stripped.startswith('data:'):
                value = stripped.split('data:', 1)[1].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                return value
        return ''

    def _set_online_planner_param(self, name: str, value) -> bool:
        if isinstance(value, str):
            escaped = value.replace("'", "'\\''")
            val_str = f"'{escaped}'"
        else:
            val_str = str(value)
        result = self._ros2_run_in_workspace(
            f'ros2 param set /online_planner {name} {val_str}',
            timeout=3.0,
        )
        return result is not None and result.returncode == 0

    def _push_online_planner_config(self):
        """Push Trajectory-tab waypoints and cost params to a running planner."""
        if not self._online_planner_running():
            return
        wp_str = self._encode_waypoints_for_online_planner()
        if wp_str:
            ok_wp = self._set_online_planner_param('waypoints_enu', wp_str)
            n = len(self._mission_waypoints_for_online_planner())
            if ok_wp:
                self.status_text.append(
                    f'Online planner: pushed {n} mission waypoint(s) from Trajectory tab.'
                )
            else:
                self.status_text.append(
                    'Online planner: failed to push waypoints_enu param.'
                )
        else:
            self.status_text.append(
                'Online planner: Trajectory tab has no mission waypoints to push.'
            )

        cost_cfg = self._collect_online_planner_cost_config()
        if cost_cfg:
            w_json, tw_json, b_json = self._encode_online_planner_cost_json()
            ok_cost = True
            if w_json:
                ok_cost &= self._set_online_planner_param('planner_weights_json', w_json)
            if tw_json:
                ok_cost &= self._set_online_planner_param(
                    'planner_terminal_weights_json', tw_json,
                )
            if b_json:
                ok_cost &= self._set_online_planner_param('planner_bounds_json', b_json)
            ok_cost &= self._set_online_planner_param('dt', cost_cfg['dt'])
            ok_cost &= self._set_online_planner_param('horizon_N', cost_cfg['horizon_N'])
            ok_cost &= self._set_online_planner_param('max_iter', cost_cfg['max_iter'])
            w = cost_cfg['weights']
            if ok_cost:
                self.status_text.append(
                    'Online planner: synced Trajectory cost '
                    f'(p={w.get("p")}, v={w.get("v")}, R={w.get("R")}, '
                    f'k_term={w.get("terminal_cost_multiplier")}, '
                    f'dt={cost_cfg["dt"]}, N={cost_cfg["horizon_N"]}).'
                )
            else:
                self.status_text.append(
                    'Online planner: failed to push some cost/discretization params.'
                )
        self._update_online_planner_wp_info_label()

    def _push_online_planner_waypoints(self):
        """Backward-compatible alias."""
        self._push_online_planner_config()

    def _invoke_advance_waypoint_service(self):
        result = self._ros2_run_in_workspace(
            'ros2 service call /online_planner/advance_waypoint std_srvs/srv/Trigger',
            timeout=10.0,
        )
        if result is None or result.returncode != 0:
            err = ''
            if result is not None:
                err = (result.stderr or result.stdout or '').strip()
            self.status_text.append(
                'Online planner: failed to call advance_waypoint service.'
                + (f' {err}' if err else '')
            )
            return
        out = result.stdout or ''
        if 'success=True' in out.replace(' ', ''):
            msg = ''
            for line in out.splitlines():
                if 'message:' in line:
                    msg = line.split('message:', 1)[1].strip().strip("'\"")
                    break
            detail = f' -> {msg}' if msg else ''
            self.status_text.append(
                f'Online planner: advanced to next waypoint (cyclic){detail}.'
            )
        else:
            self.status_text.append(
                f'Online planner: advance_waypoint rejected.\n{out.strip()}'
            )

    def _online_planner_advance_retry(self):
        if self._online_planner_running():
            self._invoke_advance_waypoint_service()
            return
        self._online_planner_advance_retries_left -= 1
        if self._online_planner_advance_retries_left > 0:
            QTimer.singleShot(2000, self._online_planner_advance_retry)
            return
        self.status_text.append(
            'Online planner: start timed out. '
            'Check logs/gui_background/online_planner_proc_*.log'
        )

    def _online_planner_advance_waypoint(self):
        if not self._online_planner_enabled():
            QMessageBox.information(
                self, 'Online planner',
                '请先勾选 “Online safety planner (RViz)”。',
            )
            return
        if not self._sitl_proc_running():
            QMessageBox.information(
                self, 'Online planner',
                '请先点击 Start SITL。',
            )
            return
        if not self._online_planner_running():
            self.status_text.append(
                'Online planner 未在运行（已勾选但未启动或已退出），正在自动启动…'
            )
            self.start_online_planner()
            self._online_planner_advance_retries_left = 10
            QTimer.singleShot(2000, self._online_planner_advance_retry)
            return
        self._invoke_advance_waypoint_service()

    def _on_online_planner_rate_changed(self, value: float):
        self._tracking_config['online_planner_rate_hz'] = float(value)
        if self._online_planner_running():
            ok = self._set_online_planner_param('replan_rate_hz', float(value))
            if not ok:
                self.status_text.append(
                    f'Online planner: failed to set replan_rate_hz={value:.1f}'
                )

    def _parse_online_planner_rate_from_diag(self, diag: str):
        """Return (actual_hz or None, target_hz or None) from diagnostics prefix."""
        import re
        m = re.search(r'rate=([\d.]+|—)/([\d.]+)Hz', diag or '')
        if not m:
            return None, None
        actual = None if m.group(1) == '—' else float(m.group(1))
        target = float(m.group(2))
        return actual, target

    def _update_online_planner_actual_hz_label(self, diag: str):
        if not hasattr(self, 'lbl_online_planner_actual_hz'):
            return
        actual, target = self._parse_online_planner_rate_from_diag(diag)
        if target is None:
            self.lbl_online_planner_actual_hz.setText('Actual: —')
            self.lbl_online_planner_actual_hz.setStyleSheet(
                'color: #888; font-family: monospace; font-size: 11px;'
            )
            return
        if actual is None:
            self.lbl_online_planner_actual_hz.setText(f'Actual: — / {target:.1f} Hz')
            self.lbl_online_planner_actual_hz.setStyleSheet(
                'color: #888; font-family: monospace; font-size: 11px;'
            )
            return
        ratio = actual / target if target > 1e-6 else 0.0
        if ratio >= 0.85:
            color = '#2e7d32'
        elif ratio >= 0.5:
            color = '#f9a825'
        else:
            color = '#c62828'
        import re
        busy_m = re.search(r'busy (\d+)%', diag or '')
        busy_suffix = f' (busy {busy_m.group(1)}%)' if busy_m else ''
        self.lbl_online_planner_actual_hz.setText(
            f'Actual: {actual:.1f} / {target:.1f} Hz{busy_suffix}'
        )
        self.lbl_online_planner_actual_hz.setStyleSheet(
            f'color: {color}; font-family: monospace; font-size: 11px; font-weight: bold;'
        )

    def _poll_online_planner_diagnostics(self):
        if not hasattr(self, 'lbl_online_planner_diag'):
            return
        if not self._online_planner_running():
            if (
                self._online_planner_enabled()
                and self._sitl_proc_running()
            ):
                self.lbl_online_planner_diag.setText(
                    'Planner: not running (auto-starts ~8s after SITL, or click Next WP)'
                )
            else:
                self.lbl_online_planner_diag.setText('Planner: stopped')
            self.lbl_online_planner_diag.setStyleSheet('color: #888;')
            self._update_online_planner_actual_hz_label('')
            return
        result = self._ros2_run_in_workspace(
            'ros2 topic echo /online_planner/diagnostics --once',
            timeout=2.0,
        )
        if result is None or result.returncode != 0:
            self.lbl_online_planner_diag.setText('Planner: waiting for diagnostics…')
            self.lbl_online_planner_diag.setStyleSheet('color: #888;')
            return
        diag = self._parse_ros2_string_echo(result.stdout)
        if not diag:
            self.lbl_online_planner_diag.setText('Planner: waiting for diagnostics…')
            self.lbl_online_planner_diag.setStyleSheet('color: #888;')
            return
        self.lbl_online_planner_diag.setText(f'Planner: {diag}')
        self.lbl_online_planner_diag.setStyleSheet('color: #9cf;')
        self._update_online_planner_actual_hz_label(diag)

    def _update_online_planner_diag_timer(self):
        if not hasattr(self, '_online_planner_diag_timer'):
            return
        if self._online_planner_running():
            if not self._online_planner_diag_timer.isActive():
                self._online_planner_diag_timer.start()
            self._poll_online_planner_diagnostics()
        else:
            self._online_planner_diag_timer.stop()
            if hasattr(self, 'lbl_online_planner_diag'):
                if (
                    self._online_planner_enabled()
                    and self._sitl_proc_running()
                ):
                    self.lbl_online_planner_diag.setText(
                        'Planner: not running (auto-starts ~8s after SITL, or click Next WP)'
                    )
                else:
                    self.lbl_online_planner_diag.setText('Planner: stopped')
                self.lbl_online_planner_diag.setStyleSheet('color: #888;')
            self._update_online_planner_actual_hz_label('')

    def _online_planner_enabled(self) -> bool:
        return (
            hasattr(self, 'chk_enable_online_planner')
            and self.chk_enable_online_planner.isChecked()
        )

    def _on_online_planner_checkbox_changed(self, state):
        enabled = bool(state)
        self._tracking_config['enable_online_planner'] = enabled
        tvc_running = (
            self._tvc_launch_proc is not None and self._tvc_launch_proc.poll() is None
        )
        if not tvc_running:
            self._update_online_planner_diag_timer()
            return
        if enabled:
            self.start_online_planner()
        else:
            self.stop_online_planner()
        self._update_online_planner_diag_timer()

    def start_online_planner(self):
        """Launch online_planner for SITL safety monitoring / RViz preview."""
        if not self._online_planner_enabled():
            return
        self._sync_waypoints_from_table()
        self._update_online_planner_wp_info_label()
        wp_str = self._encode_waypoints_for_online_planner()
        if not wp_str:
            self.status_text.append(
                'Online planner: Trajectory tab has no mission waypoints '
                '(need ≥2 rows: start + target). Launch may use defaults.'
            )
        else:
            n = len(self._mission_waypoints_for_online_planner())
            cost_cfg = self._collect_online_planner_cost_config()
            cost_note = ''
            if cost_cfg:
                w = cost_cfg['weights']
                cost_note = (
                    f'; cost p={w.get("p")}, v={w.get("v")}, R={w.get("R")}, '
                    f'k_term={w.get("terminal_cost_multiplier")}'
                )
            self.status_text.append(
                f'Online planner: {n} mission waypoint(s) from Trajectory tab '
                f'({len(self.waypoints)} rows, skipping start pose){cost_note}.'
            )
        extra = self._online_planner_launch_extra_args()
        self._start_background_process(
            '_online_planner_proc',
            self._ros2_shell_command('online_planner.launch.py', extra),
            'Online planner (online_planner.launch.py)',
        )
        QTimer.singleShot(2000, self._push_online_planner_config)
        self._update_online_planner_diag_timer()

    def stop_online_planner(self):
        """Stop online_planner background launch."""
        self._stop_background_process('_online_planner_proc', 'Online planner')
        self._update_online_planner_diag_timer()

    def start_px4_sitl_for_tracking(self):
        """Start PX4 SITL with controller choice from Tracking tab."""
        cid = self._current_tracking_controller_id()
        if cid in (CONTROLLER_MPC, CONTROLLER_ACADOS_NMPC):
            kind = 'Linear MPC' if cid == CONTROLLER_MPC else 'Acados NMPC'
            QMessageBox.information(
                self, f'{kind} SITL',
                f'{kind} tracking is supported in numerical simulation only.\n'
                'Switch to Numerical simulation or choose PX4 / LQR for SITL.',
            )
            return
        self.start_px4_sitl()

    def _show_gazebo_gui_enabled(self) -> bool:
        return (
            hasattr(self, 'chk_show_gazebo_gui')
            and self.chk_show_gazebo_gui.isChecked()
        )

    def _on_show_gazebo_gui_changed(self, state):
        self._tracking_config['show_gazebo_gui'] = bool(state)

    def _sitl_launch_extra_args(self):
        """ROS launch overrides for the active rocket platform and tracking controller."""
        kw = sitl_launch_kwargs(self._current_rocket_platform_id())
        args = [self._launch_arg('rocket_platform', kw['rocket_platform'])]
        launch_ctrl = kw.get('launch_controller')
        cid = self._current_tracking_controller_id() if hasattr(self, 'tracking_controller_combo') else None
        if cid == CONTROLLER_LQR:
            launch_ctrl = 'true'
        elif cid == CONTROLLER_PX4:
            launch_ctrl = 'false'
        if launch_ctrl is not None:
            args.append(self._launch_arg('launch_controller', launch_ctrl))
        args.append(self._launch_arg('gz_gui', str(self._show_gazebo_gui_enabled()).lower()))
        return args

    def start_px4_sitl(self):
        """Launch TVC SITL stack (PX4 + Gazebo; external LQR optional)."""
        self._sync_waypoints_from_table()
        self._update_online_planner_wp_info_label()
        platform_id = self._current_rocket_platform_id()
        extra = self._sitl_launch_extra_args()
        self._start_background_process(
            '_tvc_launch_proc',
            self._ros2_shell_command('tvc.launch.py', extra),
            'PX4 SITL (tvc.launch.py)',
        )
        self.status_text.append(
            f'PX4 SITL starting ({platform_label(platform_id)}) — '
            f'PX4 built-in control; Gazebo model follows the selected platform.'
            + ('' if self._show_gazebo_gui_enabled() else ' Gazebo: headless (RViz only).')
        )
        if self._online_planner_enabled():
            # Wait for SITL stack + EKF/vision before first replan.
            QTimer.singleShot(8000, self.start_online_planner)

    def stop_px4_sitl(self):
        """Stop tvc.launch.py process group."""
        self.stop_online_planner()
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
        if self._online_planner_running():
            QTimer.singleShot(1500, self._push_online_planner_config)
        self._refresh_sim_button_states()

    def stop_tracking_node(self):
        """Stop tvc_traj_player and clear all trajectory layers in RViz."""
        self._stop_background_process('_traj_player_proc', 'Tracking node')
        self.clear_rviz_trajectory_display(quiet=True, include_planned=True)
        self.status_text.append('RViz planned and executed trajectories cleared.')

    def closeEvent(self, event):
        """Terminate background ROS2 launches when the GUI exits."""
        for attr, label in (
            ('_traj_player_proc', 'Tracking node'),
            ('_online_planner_proc', 'Online planner'),
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
                self._close_background_process_log(attr)
        super().closeEvent(event)

    def optimization_error(self, error_msg):
        """Optimization error"""
        QMessageBox.critical(self, 'Error', f'Error during optimization:\n{error_msg}')
        self.status_text.append(f'Error: {error_msg}')
        self.run_btn.setEnabled(True)


def _load_app_icon():
    """Load application window icon from assets/icon/rocket.jpg."""
    if not os.path.isfile(APP_ICON_PATH):
        return None
    icon = QIcon(APP_ICON_PATH)
    return icon if not icon.isNull() else None


def _apply_app_icon(app=None, window=None):
    """Set the rocket icon on the QApplication and/or main window."""
    icon = _load_app_icon()
    if icon is None:
        return
    if app is not None:
        app.setWindowIcon(icon)
    if window is not None:
        window.setWindowIcon(icon)


def run_gui(argv=None) -> int:
    """Create Qt application, show main window, run event loop."""
    if argv is not None:
        sys.argv = list(argv)
    app = QApplication(sys.argv)
    _apply_app_icon(app=app)
    window = MainWindow()
    _apply_app_icon(window=window)
    window.show()
    return int(app.exec_())


def main() -> int:
    """Bootstrap acados/runtime, then launch the GUI."""
    from tvc_runtime import bootstrap

    bootstrap()
    return run_gui()


if __name__ == '__main__':
    raise SystemExit(main())
