# -*- coding: utf-8 -*-
"""
Rocket platform definitions for TVC trajectory optimization.

* **proxy** — small validation platform (current hardware).
* **real** — full-scale 20 kg real platform (cylinder + nose cone + landing gear model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models" / "tvc"

PLATFORM_PROXY = "proxy"
PLATFORM_REAL = "real"
PLATFORM_IDS = (PLATFORM_PROXY, PLATFORM_REAL)

# Legacy id used in earlier GUI saves.
_PLATFORM_ALIASES = {"flight": PLATFORM_REAL}

# --- Real platform geometry (uniform solid cylinder for inertia) ---
REAL_MASS_KG = 20.0
REAL_THRUST_MIN_N = 150.0
REAL_THRUST_MAX_N = 250.0
REAL_BODY_RADIUS_M = 0.08
REAL_BODY_LENGTH_M = 1.40
REAL_CONE_HEIGHT_M = 0.35
REAL_LEG_COUNT = 4


def uniform_solid_cylinder_inertia(m: float, r: float, length: float) -> tuple[float, float, float]:
    """Principal inertias (Ixx, Iyy, Izz) for a uniform cylinder aligned with body z."""
    i_trans = (1.0 / 12.0) * m * (3.0 * r * r + length * length)
    i_axial = 0.5 * m * r * r
    return i_trans, i_trans, i_axial


def _real_physics() -> Dict[str, float]:
    m = REAL_MASS_KG
    r = REAL_BODY_RADIUS_M
    L = REAL_BODY_LENGTH_M
    ixx, iyy, izz = uniform_solid_cylinder_inertia(m, r, L)
    # Thrust point at bottom of body (TVC gimbal), CG near cylinder centre.
    r_thrust_z = -0.5 * L - 0.02
    return {
        "mass": m,
        "Ixx": ixx,
        "Iyy": iyy,
        "Izz": izz,
        "r_thrust_x": 0.0,
        "r_thrust_y": 0.0,
        "r_thrust_z": r_thrust_z,
    }


def _proxy_physics() -> Dict[str, float]:
    return {
        "mass": 0.6,
        "Ixx": 0.02,
        "Iyy": 0.02,
        "Izz": 0.01,
        "r_thrust_x": 0.0,
        "r_thrust_y": 0.0,
        # Matches DIST_COM_2_THRUST in tvc_params.yaml (the deployed/tuned LQR
        # controller) and the gimbal/nozzle chain in models/tvc/tvc.urdf; the
        # old -0.2 placed the thrust point inside the body, above its bottom
        # face at -0.5.
        "r_thrust_z": -0.5693,
    }


_PLATFORM_CONSTRAINTS: Dict[str, Dict[str, float]] = {
    PLATFORM_PROXY: {
        "T_min": 0.0,
        "T_max": 25.0,
        "tau_yaw_max": 1.0,
        "v_horizontal_max": 1.0,
        "v_vertical_max": 3.0,
    },
    PLATFORM_REAL: {
        "T_min": REAL_THRUST_MIN_N,
        "T_max": REAL_THRUST_MAX_N,
        "tau_yaw_max": 8.0,
        "v_horizontal_max": 3.0,
        "v_vertical_max": 5.0,
    },
}

# Typical thrust command quantization for numerical tracking simulation.
_PLATFORM_THRUST_RESOLUTION_N: Dict[str, float] = {
    PLATFORM_PROXY: 0.5,
    PLATFORM_REAL: 10.0,
}


def default_thrust_quantization_resolution(platform_id: str) -> float:
    """Default thrust step size [N] when simulating discrete thrust commands."""
    pid = normalize_platform_id(platform_id)
    return float(_PLATFORM_THRUST_RESOLUTION_N.get(pid, 0.5))


_PLATFORM_META: Dict[str, Dict[str, Any]] = {
    PLATFORM_PROXY: {
        "label": "Proxy (validation)",
        "description": "Small TVC validation platform (~0.6 kg).",
        "urdf": "tvc_simple.urdf",
        "physics": _proxy_physics(),
        "physics_ranges": {
            "mass": (0.01, 10.0, 3),
            "inertia": (0.0001, 1.0, 4),
            "r_thrust": (-1.0, 1.0, 3),
        },
        "constraint_ranges": {
            "T_min": (0.0, 50.0, 1),
            "T_max": (0.0, 50.0, 2),
            "tau_yaw_max": (0.0, 5.0, 2),
        },
    },
    PLATFORM_REAL: {
        "label": "Real platform (20 kg)",
        "description": (
            f"Real vehicle ({REAL_MASS_KG:.0f} kg, thrust "
            f"{REAL_THRUST_MIN_N:.0f}–{REAL_THRUST_MAX_N:.0f} N). "
            f"Optimization uses cylinder r={REAL_BODY_RADIUS_M:.2f} m, L={REAL_BODY_LENGTH_M:.2f} m; "
            f"RViz/Gazebo SITL use the same simplified primitive visual as proxy."
        ),
        "urdf": "tvc_real.urdf",
        "physics": _real_physics(),
        "physics_ranges": {
            "mass": (0.5, 80.0, 2),
            "inertia": (0.001, 50.0, 3),
            "r_thrust": (-2.5, 2.5, 3),
        },
        "constraint_ranges": {
            "T_min": (0.0, 500.0, 1),
            "T_max": (0.0, 500.0, 1),
            "tau_yaw_max": (0.0, 20.0, 2),
        },
    },
}


def normalize_platform_id(platform_id: Optional[str]) -> str:
    if platform_id is None:
        return PLATFORM_PROXY
    pid = str(platform_id).strip().lower()
    pid = _PLATFORM_ALIASES.get(pid, pid)
    if pid in _PLATFORM_META:
        return pid
    return PLATFORM_PROXY


def platform_label(platform_id: str) -> str:
    return str(_PLATFORM_META[normalize_platform_id(platform_id)]["label"])


def platform_description(platform_id: str) -> str:
    return str(_PLATFORM_META[normalize_platform_id(platform_id)]["description"])


def default_physics(platform_id: str) -> Dict[str, float]:
    pid = normalize_platform_id(platform_id)
    return dict(_PLATFORM_META[pid]["physics"])


def default_constraints(platform_id: str) -> Dict[str, float]:
    pid = normalize_platform_id(platform_id)
    return dict(_PLATFORM_CONSTRAINTS[pid])


def physics_spin_ranges(platform_id: str) -> Dict[str, Any]:
    pid = normalize_platform_id(platform_id)
    return dict(_PLATFORM_META[pid]["physics_ranges"])


def constraint_spin_ranges(platform_id: str) -> Dict[str, Any]:
    pid = normalize_platform_id(platform_id)
    return dict(_PLATFORM_META[pid]["constraint_ranges"])


def urdf_path(platform_id: str) -> Path:
    pid = normalize_platform_id(platform_id)
    name = str(_PLATFORM_META[pid]["urdf"])
    return MODELS_DIR / name


# PX4 SITL / ROS2 stack launch arguments (must stay in sync with tvc_controller launch).
_SITL_LAUNCH: Dict[str, Dict[str, str]] = {
    PLATFORM_PROXY: {
        "rocket_platform": PLATFORM_PROXY,
        "launch_controller": "false",
        "px4_sim_model": "tvc",
        "config_file": "tvc_params.yaml",
        "robot_urdf": "tvc.urdf",
        "gz_odometry_topic": "/model/tvc_0/odometry",
        "gz_bridge_config": "bridge.yaml",
    },
    PLATFORM_REAL: {
        "rocket_platform": PLATFORM_REAL,
        "launch_controller": "false",
        "px4_sim_model": "tvc_real",
        "px4_sys_autostart": "6004",
        "px4_gz_model_pose": "0,0,0.35",
        "config_file": "tvc_params_real.yaml",
        "robot_urdf": "tvc_real.urdf",
        "gz_odometry_topic": "/model/tvc_real_0/odometry",
        "gz_bridge_config": "bridge_real.yaml",
    },
}


def sitl_launch_kwargs(platform_id: str) -> Dict[str, str]:
    """Keyword arguments for ``ros2 launch tvc_controller tvc.launch.py``."""
    pid = normalize_platform_id(platform_id)
    return dict(_SITL_LAUNCH[pid])


def rocket_visual_geometry(platform_id: str) -> Dict[str, float]:
    """
    Body-frame rocket stick dimensions [m] for GIF / 3D plots (COM at origin).

    Matches optimization URDF / Gazebo primitive sizes.
    """
    pid = normalize_platform_id(platform_id)
    if pid == PLATFORM_REAL:
        return {
            'body_bottom_z': -0.5 * REAL_BODY_LENGTH_M,
            'nose_tip_z': 0.5 * REAL_BODY_LENGTH_M + REAL_CONE_HEIGHT_M,
            'fin_z': -0.15,
            'shaft_lw': 5.5,
            'fin_lw': 3.2,
        }
    # Proxy — tvc_simple.urdf cylinder L=1.0 m
    return {
        'body_bottom_z': -0.5,
        'nose_tip_z': 0.55,
        'fin_z': -0.10,
        'shaft_lw': 4.0,
        'fin_lw': 2.4,
    }
