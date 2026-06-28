#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared runtime setup for TVC-traj-opt (scripts path + acados library preload).

Used by the main GUI entry and by benchmark / CLI tools that need Acados without
starting Qt.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")


def project_root() -> str:
    return _ROOT_DIR


def scripts_dir() -> str:
    return _SCRIPTS_DIR


def ensure_scripts_on_path() -> str:
    """Add ``scripts/`` to ``sys.path`` for solver module imports."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    return _SCRIPTS_DIR


def find_acados_root() -> Optional[str]:
    """Return the acados source dir (which contains ``lib/libacados.so``)."""
    candidate = os.environ.get("ACADOS_SOURCE_DIR")
    if candidate and os.path.isdir(os.path.join(candidate, "lib")):
        return candidate

    try:
        import acados_template  # type: ignore

        pkg_dir = os.path.dirname(os.path.abspath(acados_template.__file__))
        walk = pkg_dir
        for _ in range(4):
            walk = os.path.dirname(walk)
            if os.path.isdir(os.path.join(walk, "lib")) and os.path.isfile(
                os.path.join(walk, "lib", "libacados.so")
            ):
                return walk
    except Exception:
        pass

    for fallback in (
        os.path.expanduser("~/Documents/GitHub/acados"),
        os.path.expanduser("~/acados"),
        "/opt/acados",
    ):
        if os.path.isfile(os.path.join(fallback, "lib", "libacados.so")):
            return fallback

    return None


def preload_acados_libs() -> None:
    """Preload acados shared libraries with RTLD_GLOBAL (see acados docs)."""
    acados_root = find_acados_root()
    if acados_root is None:
        return

    lib_dir = os.path.join(acados_root, "lib")
    os.environ.setdefault("ACADOS_SOURCE_DIR", acados_root)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = (
            lib_dir + (os.pathsep + ld_path if ld_path else "")
        )

    mode = ctypes.RTLD_GLOBAL
    for libname in (
        "libblasfeo.so",
        "libhpipm.so",
        "libqpOASES_e.so",
        "libosqp.so",
    ):
        path = os.path.join(lib_dir, libname)
        if not os.path.isfile(path):
            continue
        try:
            ctypes.CDLL(path, mode=mode)
        except OSError as e:
            print(
                f"[tvc_runtime] WARNING: failed to preload {path}: {e}",
                file=sys.stderr,
            )


def bootstrap() -> str:
    """Prepare ``sys.path`` and acados for solver imports."""
    ensure_scripts_on_path()
    preload_acados_libs()
    return _SCRIPTS_DIR
