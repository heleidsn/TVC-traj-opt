#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVC rocket trajectory optimization — project main entry (GUI).

This module only launches the UI. Optimization methods live under ``scripts/`` and are
invoked from the GUI (Methods 1–7):

- Method 1: ``scripts/tvc_traj_opt.py`` (Crocoddyl custom calcDiff)
- Methods 2–3: ``scripts/tvc_traj_opt_pinocchio.py`` (FDDP / BoxFDDP)
- Methods 4–5–7: ``scripts/tvc_traj_opt_acados.py`` (Acados: tracking / min-time / free-tf)
- Method 6: ``scripts/tvc_traj_opt_acados_min_time.py`` (Spannagl FFTF)

Usage::

    python run_tvc_traj_opt.py

After installation::

    tvc-traj-opt
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional


def _ensure_scripts_on_path() -> str:
    """Add ``scripts/`` to ``sys.path`` so imports match ``python scripts/...``."""
    root = os.path.dirname(os.path.abspath(__file__))
    scripts = os.path.join(root, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return scripts


def _find_acados_root() -> Optional[str]:
    """Return the acados source dir (which contains ``lib/libacados.so``).

    Looks at, in order: ``ACADOS_SOURCE_DIR``, then walks up from the
    ``acados_template`` Python package, then a few well-known fallback
    locations under ``~``.
    """
    candidate = os.environ.get("ACADOS_SOURCE_DIR")
    if candidate and os.path.isdir(os.path.join(candidate, "lib")):
        return candidate

    try:
        import acados_template  # type: ignore

        pkg_dir = os.path.dirname(os.path.abspath(acados_template.__file__))
        # acados_template lives at <acados>/interfaces/acados_template/acados_template
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


def _preload_acados_libs() -> None:
    """Preload acados shared libraries with RTLD_GLOBAL.

    The acados-generated C code (and the Python wrapper) calls
    ``dlopen("libqpOASES_e.so")`` by basename. The dynamic linker only
    consults ``LD_LIBRARY_PATH`` *once* at process startup, so setting it
    via ``os.environ`` from Python is too late. Loading each library
    explicitly with its absolute path and ``RTLD_GLOBAL`` makes the
    symbols globally available, so subsequent basename ``dlopen`` calls
    succeed without ``LD_LIBRARY_PATH``.
    """
    acados_root = _find_acados_root()
    if acados_root is None:
        # No acados available – Crocoddyl-based methods (1-3) still work.
        return

    lib_dir = os.path.join(acados_root, "lib")
    os.environ.setdefault("ACADOS_SOURCE_DIR", acados_root)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(os.pathsep):
        # Keep env consistent for any subprocess we might spawn later (e.g.
        # the acados code generator), even though it does not help the
        # current process's dlopen.
        os.environ["LD_LIBRARY_PATH"] = (
            lib_dir + (os.pathsep + ld_path if ld_path else "")
        )

    # Preload all dependencies of libacados.so with ``RTLD_GLOBAL`` so their
    # symbols are visible to the later load of libacados.so (which
    # acados_template performs internally). The dynamic linker caches
    # ``LD_LIBRARY_PATH`` at process startup, so updating ``os.environ`` is
    # too late.
    #
    # The set is derived from ``ldd libacados.so``:
    #   libblasfeo.so.0, libhpipm.so, libqpOASES_e.so, libosqp.so
    # ``LINSYS_SOLVER_NAME`` (referenced by libacados) is defined in libosqp,
    # so libosqp **must** be preloaded too – otherwise libacados fails to
    # resolve that symbol when acados_template dlopens it.
    mode = ctypes.RTLD_GLOBAL
    preload_order = (
        "libblasfeo.so",
        "libhpipm.so",
        "libqpOASES_e.so",
        "libosqp.so",
    )
    for libname in preload_order:
        path = os.path.join(lib_dir, libname)
        if not os.path.isfile(path):
            continue
        try:
            ctypes.CDLL(path, mode=mode)
        except OSError as e:
            # Best-effort: report once so the user can see what failed, but do
            # not abort the GUI (non-acados methods still work).
            print(
                f"[run_tvc_traj_opt] WARNING: failed to preload {path}: {e}",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    """
    Launch the trajectory optimization GUI.

    Parameters
    ----------
    argv : optional
        Qt command-line arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        Qt exit code.
    """
    if argv is not None:
        sys.argv = list(argv)
    _ensure_scripts_on_path()
    _preload_acados_libs()
    from tvc_traj_opt_gui import run_gui

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
